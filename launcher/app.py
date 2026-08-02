"""Training-run orchestration on Modal: one launcher, one worker, one lease.

Implements training_run_spec_kv.md with the smallest number of moving parts --
a single module, three substrates, and no state anywhere else:

    Dict   = who may write, now        (the lease; fresh reads, atomic per key)
    Modal  = who is alive, now         (FunctionCall state; queried, never written)
    Volume = what happened             (config, checkpoints, ledger, logs)

`launch` is the mutex: max_containers=1 with single-input concurrency, so every
launch/resume request is executed alone, in order. It is the only writer of the
Dict -- a worker can never promote, renew, or release itself, so there is no
unlock path to get wrong. `train` is the worker: it reads the lease and compares
it against its own call id (the "fence"), and the one place that check gates is
the checkpoint boundary -- no confirmed MATCH, no shared write. That single gate
is what makes "at most one writer per run folder" a mechanism rather than a
convention.

Deliberately simpler than the spec, in ways that are noted at each site:
  - the worker here is a toy counter (§"the work" below), so the protocol can be
    exercised end-to-end in seconds with no GPU. Real training swaps out one
    function; the fence/gate/commit structure around it does not change.
  - no config_sha256, no checkpoint pruning, no advisory between-boundary checks,
    and no "compute one interval ahead while blocked" optimization.
  - grants are not mirrored to the Volume. The spec keeps a write-only receipt of
    every grant for audit, since the Dict holds only the current one and expires
    after 7 idle days; the cost is a commit per launch, and what it buys back is
    only history. Each attempt still has its own log folder, and every ledger row
    still carries the call id that produced it.

Takeover clears the key *before* spawning rather than overwriting it after, which
the spec does not spell out and which a live run proved is load-bearing -- see the
comment at the L5/L6 branch.

Usage:
    modal deploy launcher/app.py                       # the mutex must be the deployed app
    modal run launcher/app.py --name alpha             # start a run
    modal run launcher/app.py --run-id 2026...._alpha  # resume it (idempotent)
"""

import json
import time
from contextlib import contextmanager
from datetime import datetime, UTC
from pathlib import Path

import modal
from modal.exception import Error, FunctionTimeoutError, OutputExpiredError

APP_NAME = "training-launcher"
VOLUME_NAME = "test-volume"  # flip to "LLM-pretraining" once the real trainer is wired in
DICT_NAME = "training-leases"

RUNS = Path("/storage/runs")  # volume mount path, as seen inside the containers

FENCE_IN_RETRIES = 3  # covers the microscopic spawn-before-put window plus transients
FENCE_IN_BACKOFF = 2.0  # seconds between fence-in retries
GATE_RETRIES = 5  # boundary patience on an indeterminate lease read
GATE_BACKOFF = 5.0  # seconds; the gate sleeps GATE_BACKOFF * attempt (linear, no jitter needed at N=5)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
leases = modal.Dict.from_name(DICT_NAME, create_if_missing=True)

# The toy worker needs nothing but the stdlib. A real trainer swaps this for the
# uv_sync image from modal_lab/run.ipynb -- only this line changes.
image = modal.Image.debian_slim(python_version="3.12")


# --------------------------------------------------------------------------------------
# liveness: the single source of truth for "is that call still running?"
# --------------------------------------------------------------------------------------


async def status(call):
    """Map every outcome of `call.get(timeout=0)` to a state string.

    Verified against modal 1.5.2 (`_functions.py:poll_function` +
    `_utils/function_utils.py:_process_result`). Ordering matters:
    OutputExpiredError/FunctionTimeoutError subclass *modal's* TimeoutError,
    which is NOT the builtin -- so `except TimeoutError` catches only "running",
    and those two must be caught before the generic Error branch. Any modal
    upgrade must re-check that assumption; it is the one version-fragile thing
    in this file.

    There is deliberately one implementation, async: a second sync copy would
    drift, and drift here means misclassifying a live worker as dead.
    """
    try:
        await call.get.aio(timeout=0)
        return "done"
    except TimeoutError:  # builtin -- still executing
        return "running"
    except OutputExpiredError:  # result garbage-collected; outcome unknowable now
        return "expired"
    except FunctionTimeoutError:  # exceeded its own timeout=
        return "timed_out"
    except Error:  # any other modal-side failure, including cancellation
        return "failed"
    except Exception:  # the worker raised -> re-raised here on get()
        return "crashed"


# --------------------------------------------------------------------------------------
# the lease
# --------------------------------------------------------------------------------------

MATCH, MISMATCH, UNKNOWN = "match", "mismatch", "unknown"


class LeaseLost(Exception):
    """This attempt can no longer prove it owns the run, so it must stop writing.

    An exception rather than a return value because the check lives inside a
    context manager the training loop enters: raising is what lets a denied gate
    unwind a loop this module does not own, without the trainer having to know a
    lease exists. `_attempt` is the only thing that catches it.
    """


def lease_key(run_id: str) -> str:
    return f"lease:{run_id}"


def fence(run_id: str, my_call_id: str) -> str:
    """One fresh Dict read, three outcomes -- never two.

    MATCH is a positive fact and so is MISMATCH: the key holding *another* id is
    the server asserting someone else was deliberately granted this run, and the
    only correct response is to stop immediately (retrying a "no" is hoping the
    truth changes). UNKNOWN -- absent key, or the call raised -- is silence, not
    denial: its causes are a Dict outage, a 7-day expiry, or a worker that was
    never granted at all, none of which mean "someone else owns this". Routing
    absence to MISMATCH would let one Dict blip fell the whole fleet; routing it
    to MATCH would let a partitioned zombie write. Bounded patience is the only
    policy whose errors are conservative in both directions.
    """
    try:
        grant = leases.get(lease_key(run_id))
    except Exception:
        return UNKNOWN
    if grant is None:
        return UNKNOWN
    return MATCH if grant["call_id"] == my_call_id else MISMATCH


# --------------------------------------------------------------------------------------
# the volume: what happened
# --------------------------------------------------------------------------------------


def run_dir(run_id: str) -> Path:
    return RUNS / run_id


def read_config(run_id: str) -> dict | None:
    """The run's immutable definition, or None if it is missing/unparseable --
    which is the launcher's cue to quarantine the folder rather than guess."""
    try:
        return json.loads((run_dir(run_id) / "config.json").read_text())
    except (OSError, ValueError):
        return None


def checkpoint_step(run_id: str) -> int:
    """Step of the newest committed checkpoint; 0 if there is none.

    Progress is read off the checkpoint *files*, not the ledger, because
    checkpoints are the only thing the commit boundary makes durable -- the
    ledger rows between boundaries live in a container that may never come back.
    Naming follows transformer.util.checkpoint_path, so the real trainer's
    checkpoints are already readable here unchanged.
    """
    files = sorted((run_dir(run_id) / "checkpoints").glob("step_*.obj"))
    return int(files[-1].stem.split("_")[1]) if files else 0


def is_complete(run_id: str, config: dict) -> bool:
    return checkpoint_step(run_id) >= config["total_steps"]


def mint_run_id(name: str) -> str:
    """`{YYYYMMDDTHHMMSS}_{name}` -- the folder name is the run's immutable
    primary key.
    """
    now = datetime.now(UTC)
    return f"{now:%Y%m%dT%H%M%S}_{name}"


def utc() -> str:
    """JS-friendly UTC timestamp, matching transformer.util's ledger rows."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------------------
# the launcher: the only writer of the Dict, and the only mutual exclusion we have
# --------------------------------------------------------------------------------------


@app.function(image=image, volumes={"/storage": volume}, max_containers=1, scaledown_window=60, timeout=300)
async def launch(run_id: str | None = None, name: str = "run", config: dict | None = None) -> dict:
    """Start a new run, or resume an existing one. Blind retries are always safe.

    `max_containers=1` plus single-input concurrency means Modal runs every one
    of these sequentially in one container: the read-decide-write sequences below
    (read the lease, ask Modal if that call is alive, then grant) are not atomic
    in any substrate, and this is the only true mutual-exclusion primitive
    available -- the Volume can't provide one (check-then-act over stale
    snapshots) and neither can the Dict (no compare-and-swap on value).

    That serialization only holds within one *app instance*, so every caller must
    reach the deployed function via `Function.from_name` -- an ephemeral
    `modal run` of this module gets its own container pool and its own mutex.
    If that rule is broken, two protections remain: put-if-absent picks exactly
    one winner among concurrent fresh grants, and the worker's fence bounds any
    takeover race to one dropped checkpoint interval.

    Returns {"run_id", "reason", ...} -- "reason" is always a plain string saying
    what was decided, because a no-op and a spawn look identical from the caller
    otherwise.
    """
    await volume.reload.aio()  # this container is long-lived; its snapshot is stale by default

    # L1 -- fresh run: create the folder and its immutable config, then spawn,
    # then grant. Until the lease lands the worker sees no key at all, which is
    # UNKNOWN, so it retries FENCE_IN_RETRIES times before giving up.
    #
    # The config commit must precede the spawn so the worker's mount
    # can see it; the call id does not exist before the spawn, so the grant
    # cannot precede it either. That ordering is forced, not chosen.
    if run_id is None:
        run_id = mint_run_id(name)
        if run_dir(run_id).exists():
            # if this folder has already been created - abort launch.
            return {"run_id": run_id, "reason": "id already exists -- refusing to overwrite it", "call_id": None}
        (run_dir(run_id) / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir(run_id) / "config.json").write_text(json.dumps(config or {}, indent=2))
        await volume.commit.aio()

        call = await train.spawn.aio(run_id)
        granted = await leases.put.aio(
            lease_key(run_id), {"call_id": call.object_id, "granted_ts": time.time(), "attempt": 1}, skip_if_exists=True
        )
        if not granted:
            # L2 -- someone else holds the lease, so abort this path. Unreachable
            # in practice; the worker we just spawned self-evicts on its own fence.
            return {"run_id": run_id, "reason": "lost a race for a freshly minted id", "call_id": None}

        return {"run_id": run_id, "reason": "spawned", "call_id": call.object_id}

    # L7 -- no readable config: refuse. A run whose definition is gone is
    # quarantined, never guessed at.
    config = read_config(run_id)
    if config is None:
        return {"run_id": run_id, "reason": "no readable config.json -- orphaned", "call_id": None}

    # L3 -- complete: decided from the Volume alone. A finished run must read
    # finished forever, and Modal garbage-collects call outcomes eventually, so
    # neither the Dict nor Modal is consulted here.
    if is_complete(run_id, config):
        return {"run_id": run_id, "reason": "already complete", "call_id": None}

    # check if a lease exists - it might be a failed or paused run. tells you if
    # anyone else ever worked on this folder, does not mean active. If there is a
    # running legitimate owner, this is how we find it. Abort if someone else is
    # already working on it
    grant = await leases.get.aio(lease_key(run_id))
    if grant is not None:
        state = await status(modal.FunctionCall.from_id(grant["call_id"]))
        if state == "running":
            return {"run_id": run_id, "reason": "already running", "call_id": grant["call_id"]}

    # L5/L6 -- if we made this far this means this folder is available. Either start
    # a new run or takeover from an expired/old lease. First clear (pop) the old
    # lease if exists - then assign a lease to this function with updated attempt
    # number.
    #
    # The pop must come before the spawn, not after. Between spawn and put the key
    # still names the dead attempt, and a warm container boots fast enough to read
    # it -- an occupied key is an explicit MISMATCH, so the new worker would kill
    # itself over a lease that is milliseconds from being its own. Clearing first
    # makes that window an absent key, which is UNKNOWN, which fence-in rides out.

    await leases.pop.aio(lease_key(run_id), None)
    attempt = (grant["attempt"] + 1) if grant else 1
    call = await train.spawn.aio(run_id)
    await leases.put.aio(lease_key(run_id), {"call_id": call.object_id, "granted_ts": time.time(), "attempt": attempt})

    resumed_from = checkpoint_step(run_id)
    return {
        "run_id": run_id,
        "reason": f"spawned attempt {attempt} from step {resumed_from}",
        "call_id": call.object_id,
    }


# --------------------------------------------------------------------------------------
# the worker: computes freely, commits only with a confirmed lease
# --------------------------------------------------------------------------------------


@app.function(image=image, volumes={"/storage": volume}, timeout=3600, retries=0)
def train(run_id: str) -> dict:
    # if the worker raises a timeout error which comes up through call.get()
    # this can't be distinguished from modal's own call.get(timeout=0) which
    # is how we tell a function is running. This function catches every exception
    # that appears in the worker and raises them as RuntimeErrors which clears
    # the ambiguity and a raised timeout error is guaranteed to come from modal.
    """One attempt at one run. Wrapped whole, because an escaping builtin
    TimeoutError would be classified `running` by status() forever, pinning a
    dead run at in_progress and blocking every future resume. Re-raising as
    RuntimeError guarantees every escape classifies as `crashed`.
    """
    try:
        return _attempt(run_id)
    except Exception as e:
        raise RuntimeError(f"attempt failed for {run_id}: {e!r}") from e


def _attempt(run_id: str) -> dict:
    # The one reload, and it must be the very first thing: a container is reused
    # across calls, so a warm worker inherits a volume snapshot taken before this
    # run's folder even existed -- it would read no config and no checkpoints, and
    # "no checkpoints" means "restart from step 0", which is how a stale snapshot
    # turns into silently discarded training. It has to precede the first local
    # write (the log folder below), since reload resolves the whole mount.
    # Nothing after this needs it: the worker never reads another process's data
    # once it owns the run, so the entire class of reload-ordering bugs stops here.
    volume.reload()

    my_id = modal.current_function_call_id()
    rdir = run_dir(run_id)
    logdir = rdir / "logs" / my_id
    logdir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        """Append-only, and private to this attempt: the folder is named by a
        globally unique call id, so no two attempts can collide here and even an
        evicted attempt can safely flush its final log without holding the lease.
        """
        line = f"{utc()} {msg}"
        with open(logdir / "attempt.log", "a") as f:
            f.write(line + "\n")
        print(line)

    def log_exit(reason: str) -> dict:
        """Record why this attempt is leaving and publish that record. The caller
        does the actual leaving, by returning what this hands back.

        The commit is the point: attempt.log is container-local until then, so an
        attempt that aborts before its first checkpoint would otherwise vanish
        without saying why. It is also leaseless, which is only safe because the
        sole dirty file in the container is attempt.log, in a folder no other
        attempt can write -- commit granularity is the container, not the path,
        which is the rule the `pending` note in the loop exists to respect.
        """
        log(f"EXIT {reason}")
        volume.commit()
        return {"run_id": run_id, "reason": reason}

    @contextmanager
    def holding_lease(label: str, tries: int = GATE_RETRIES, backoff: float = GATE_BACKOFF, commit: bool = False):
        """Run an arbitrary block only while this attempt provably owns the run,
        and optionally make what it wrote durable.

            with holding_lease("fence-in", FENCE_IN_RETRIES, FENCE_IN_BACKOFF):
                ...                                  # just needs to be the owner
            with holding_lease(f"step {step}", commit=True):
                write_checkpoint(...)                # ... and wants it to survive

        Enter is the gate, and it is the whole protocol: a mismatch raises at once,
        because that is a fresh, positive "someone else owns this" and retrying a
        no is hoping the truth changes. Only silence -- absent key, or the read
        itself failed -- is worth waiting on, and only for a bounded time. Patience
        is safe against a legitimate takeover: the launcher gates takeovers on
        Modal, and a worker that has merely lost the Dict is still `running` there,
        so its lease cannot move while we wait.

        Exit commits, when asked. That commit is the only irreversible act in the
        worker -- everything the block wrote before it is container-local, private,
        and discarded when the container dies. So a denied gate costs work, never
        history, and `commit=False` is genuinely free.

        If the block raises, the commit is skipped and the exception propagates: a
        half-written boundary is not a boundary.
        """
        for i in range(1, tries + 1):
            verdict = fence(run_id, my_id)
            if verdict == MATCH:
                break
            if verdict == MISMATCH:
                raise LeaseLost(f"{label}: another attempt holds this run")
            if i == tries:
                raise LeaseLost(f"{label}: indeterminate after {tries} tries -- ownership never confirmed")
            log(f"{label}: indeterminate, retry {i}/{tries}")
            time.sleep(backoff * i)  # linear; no jitter needed at this fleet size

        yield

        if commit:
            volume.commit()
            log(f"committed {label}")

    log(f"boot call_id={my_id}")

    # Ledger rows are held in memory, not appended to train.jsonl as they happen.
    # That looks like a needless buffer and is not: volume.commit() has *container*
    # granularity, not per-path, so the exit commit that flushes this attempt's
    # private log would carry any dirty shared file out with it -- a superseded
    # worker publishing an interval it was explicitly denied. Keeping the rows off
    # disk until the gate has said MATCH is what makes "no confirmed lease, no
    # shared write" true of the mechanism rather than of the intent. Losing them on
    # the way out is not a cost; discarding the unconfirmed interval is the point.
    pending: list[dict] = []

    try:
        # Fence-in. Its retries cover the launcher's spawn-before-put window plus
        # transport transients; nothing here is written, so nothing needs committing.
        with holding_lease("fence-in", FENCE_IN_RETRIES, FENCE_IN_BACKOFF):
            config = read_config(run_id)
            if config is None:
                return log_exit("config.json missing or unparseable")

            total_steps = config["total_steps"]
            save_every = config["save_every"]

            # Resume from the checkpoint files, not the ledger: the checkpoint is
            # the training state, the ledger is for display.
            step = checkpoint_step(run_id)
            log(f"fence-in MATCH; resuming at step {step}/{total_steps}")

        while step < total_steps:
            step += 1
            do_work(config)  # <- the only line a real trainer replaces
            pending.append({"step": step, "ts": utc(), "call_id": my_id})

            if step % save_every and step != total_steps:
                continue

            # Everything durable about this interval lands as one unit: the
            # checkpoint first, so a committed row naming it always resolves.
            with holding_lease(f"boundary at step {step}/{total_steps}", commit=True):
                write_checkpoint(rdir, step, config)
                with open(rdir / "train.jsonl", "a") as f:
                    f.writelines(json.dumps(row) + "\n" for row in pending)
                pending.clear()
    except LeaseLost as e:
        return log_exit(str(e))

    return log_exit(f"finished at step {step}/{total_steps}")


# --------------------------------------------------------------------------------------
# the work itself -- the toy stand-in for training
# --------------------------------------------------------------------------------------


def do_work(config: dict) -> None:
    """One step. A real trainer does a forward/backward here."""
    time.sleep(config.get("step_seconds", 1.0))


def write_checkpoint(rdir: Path, step: int, config: dict) -> None:
    """tmp + rename, so no reader can ever observe a torn checkpoint: rename is
    atomic on the container filesystem, and the volume only ever uploads the
    final name. Write-once and monotonic -- checkpoints are never rewritten.

    Naming matches transformer.util.checkpoint_path (`step_{step:010d}.obj`), so
    swapping torch.save in here is the whole change on the real-training path.
    No pruning: the trainer owns its own checkpoint hygiene.
    """
    out = rdir / "checkpoints" / f"step_{step:010d}.obj"
    tmp = out.with_suffix(".obj.tmp")
    tmp.write_text(json.dumps({"step": step, "seed": config.get("seed")}))
    tmp.rename(out)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


@app.local_entrypoint()
def main(run_id: str = "", name: str = "run", total_steps: int = 20, save_every: int = 5, step_seconds: float = 1.0):
    """Drive the *deployed* launcher -- never a local spawn.

    `Function.from_name` is what keeps the mutex intact: calling `launch.remote`
    from this ephemeral app instance would run it in a second container pool that
    does not serialize against the deployed one.
    """
    launcher = modal.Function.from_name(APP_NAME, "launch")
    if run_id:
        print(launcher.remote(run_id=run_id))
    else:
        config = {"total_steps": total_steps, "save_every": save_every, "step_seconds": step_seconds, "seed": 0}
        print(launcher.remote(name=name, config=config))
