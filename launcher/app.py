"""Run long jobs on Modal so that crashes, cancels, and restarts are all safe.

A job lives in one folder on a Volume. Restarting picks up from the last
checkpoint, asking twice does nothing, and two workers can never write the same
folder -- enforced, not assumed.

Each system does only what it is good at:

    Dict   = who may write, now    (the lease -- atomic per key, always fresh)
    Modal  = who is alive, now     (call state -- we ask, never write)
    Volume = what happened         (config, checkpoints, ledger, logs)

The parts:

    launch    start or resume a run; safe to call repeatedly. Spawns workers.
    work      one worker -- one container, one call id, one go at one run.
    JOB       what this deployment runs; the toy counter stands in for training.
    etl       runs the notebooks/pipeline workflow to build the data a run trains
              on. No lease: a DAG of file targets is already idempotent.

Logging lives in logs.py -- a logger named for the call id, handlers answering to
that name only, so a handler outliving its call cannot write into the next run.

The guarantee: `launch` is single-container and the only writer of the Dict, so a
worker can never promote itself. A worker checks the lease against its own call
id, and that gates exactly one thing -- `lease(commit=True)`. Nothing it writes is
visible to anyone until that commit.

Simplifications, each explained where it happens: no config hash, no checkpoint
pruning, no audit copy of each grant, one retry policy for every lease check.

Usage:
    modal deploy launcher/app.py                       # deploy (do this first)
    modal run launcher/app.py --name alpha             # start a run
    modal run launcher/app.py --run-id 2026...._alpha  # resume it
    modal run launcher/app.py::prepare --run-id 2026....  # build that run's data
"""

import json
import re
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import modal
from modal.exception import Error, FunctionTimeoutError, OutputExpiredError

# Jobs and the shared run-folder layout live in jobs.py. The dependency runs one
# way (app -> jobs), so there is no cycle and Aborted has a single definition.
from jobs import Aborted, Count, mint_run_id, load_config, run_dir

# Logging lives in logs.py: a logger named for this call, and handlers that
# answer to that name only. A worker takes one and passes it on.
from logs import release_worker_logger, worker_logger

APP_NAME = "training-launcher"
VOLUME_NAME = "test-volume"  # flip to "LLM-pretraining" once the real trainer is wired in
DICT_NAME = "training-leases"

# Where the pipeline lives on each side. PROJECT_ROOT comes from __file__, not
# config.py: only launcher/ is on the path in a container, and this is read while
# describing the image, never inside one.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = PROJECT_ROOT / "notebooks" / "pipeline"  # the Snakefile and its modules, locally
PIPELINE = "/pipeline"  # the same folder in the image, read-only
STORAGE = "/storage"  # the Volume: where the workflow's artifacts and its metadata land

LEASE_RETRIES = 5  # how many times an indeterminate lease read is worth re-asking
LEASE_BACKOFF = 5.0  # seconds x try number -> 5, 10, 15, 20: ~50s of patience in total

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
leases = modal.Dict.from_name(DICT_NAME, create_if_missing=True)

# The toy worker needs nothing but the stdlib; a real trainer swaps in a uv_sync
# image, and only this line changes.
base_image = modal.Image.debian_slim(python_version="3.12")

# app.py imports both at module load, so every image needs them or the container
# dies on import. add_local_* must come last in a chain: Modal adds these at
# container startup rather than baking them in, so any build step precedes them.
LOCAL_MODULES = ("jobs", "logs")

image = base_image.add_local_python_source(*LOCAL_MODULES)

# The dashboard needs a web server, and its own module shipped alongside this one.
web_image = base_image.pip_install("fastapi[standard]").add_local_python_source(*LOCAL_MODULES, "dashboard")

# The ETL needs the real project environment: uv_sync installs the lockfile's deps
# (snakemake, joblib, ...) and `transformer` rides along because the Snakefile
# imports it. The Snakefile and its profile come as a plain directory since
# add_local_python_source only ships importable modules. data/, runs/ and
# .snakemake/ are excluded -- they are the workflow's output and its requests,
# which live on the Volume, and a laptop's copy would plant a stale spec and stale
# artifacts snakemake would trust.
etl_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=PROJECT_ROOT)
    .add_local_python_source(*LOCAL_MODULES, "transformer")
    .add_local_dir(PIPELINE_DIR, PIPELINE, ignore=["data", "runs", ".snakemake", "**/__pycache__"])
)

# --------------------------------------------------------------------------------------
# liveness: the single source of truth for "is that call still running?"
# --------------------------------------------------------------------------------------


async def status(call):
    """Is this call still running? Ask Modal, without blocking.

    Checked against modal 1.5.2, and the one version-fragile thing in the file:
    OutputExpiredError and FunctionTimeoutError subclass *modal's* TimeoutError,
    not the builtin, so they must precede the general Error branch and `except
    TimeoutError` catches only "still running".

    Keep this as the only copy -- a second one drifts, and drift here means calling
    a live worker dead.
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




class LeaseLost(Aborted):
    """We can no longer prove we own this run, so we must stop writing.

    Raised rather than returned: the check sits inside a context manager the job's
    loop enters, so raising unwinds that loop without the job knowing a lease
    exists.
    """


def lease_key(run_id: str) -> str:
    return f"lease:{run_id}"


def fence(run_id: str, my_call_id: str) -> tuple[str, dict | None]:
    """Do we own this run? One fresh Dict read, three answers -- never two.

    The grant comes back with the verdict: the read already fetched it, and callers
    want to name *who* holds the run, not just whether we do.

    - MATCH / MISMATCH are real answers. Someone else's id means stop at once;
      asking again is just hoping it changes.
    - UNKNOWN (no key, or the read failed) is silence, not a no -- a Dict outage, a
      7-day expiry, or a worker never granted anything. Denying on silence takes
      down the fleet on one blip; granting on it lets a zombie write. So callers
      wait a bounded time, then give up.
    """
    try:
        grant = leases.get(lease_key(run_id))
    except Exception:
        return UNKNOWN, None
    if grant is None:
        return UNKNOWN, None
    return (MATCH if grant["call_id"] == my_call_id else MISMATCH), grant


# --------------------------------------------------------------------------------------
# the volume: what happened
# --------------------------------------------------------------------------------------
#
# run_dir, load_config and mint_run_id -- the shared folder layout -- live in
# jobs.py, the leaf both sides import from.


# --------------------------------------------------------------------------------------
# the launcher: the only writer of the Dict, and the only mutual exclusion we have
# --------------------------------------------------------------------------------------


@app.function(image=image, volumes={"/storage": volume}, max_containers=1, scaledown_window=60, timeout=300)
async def launch(run_id: str | None = None, name: str = "run", config: dict | None = None) -> dict:
    """Start a new run, or resume one. Calling it repeatedly is always safe.

    `max_containers=1` is the only real lock available: read the lease, ask Modal
    if that call is alive, then grant is a sequence no single system does
    atomically (the Volume checks stale snapshots, the Dict has no compare-and-swap).

    It holds only within one deployed app, so always reach this through
    `Function.from_name` -- a `modal run` of this module gets its own pool and its
    own lock. If that slips: put-if-absent picks one winner among new runs, and a
    worker's fence caps a takeover race at one dropped interval.

    Always returns a "reason", or a no-op and a fresh spawn look identical.
    """
    await volume.reload.aio()  # this container is long-lived, so its snapshot is stale by default

    # L1 -- new run: create folder + config, spawn, then grant. The order is forced:
    # config must be committed before the spawn or the worker's mount misses it, and
    # the call id does not exist until spawn returns. In the gap the worker sees no
    # key, reads UNKNOWN, and waits.
    if run_id is None:
        run_id = mint_run_id(name)
        if run_dir(run_id).exists():
            # config.json is written once and every resume runs under it, so never
            # write through an existing one.
            return {"run_id": run_id, "reason": "id already exists -- refusing to overwrite it", "call_id": None}
        run_dir(run_id).mkdir(parents=True)  # the job owns whatever lives inside
        (run_dir(run_id) / "config.json").write_text(json.dumps(config or {}, indent=2))
        await volume.commit.aio()

        call = await work.spawn.aio(run_id)
        granted = await leases.put.aio(
            lease_key(run_id), {"call_id": call.object_id, "granted_ts": time.time(), "attempt": 1}, skip_if_exists=True
        )
        if not granted:
            # L2 -- someone grabbed this id first. Unreachable in practice; the
            # worker we spawned sees the mismatch and bows out.
            return {"run_id": run_id, "reason": "lost a race for a freshly minted id", "call_id": None}

        return {"run_id": run_id, "reason": "spawned", "call_id": call.object_id}

    # L7 -- a run whose definition is gone gets quarantined, never guessed at.
    config = load_config(run_id)
    if config is None:
        return {"run_id": run_id, "reason": "no readable config.json -- orphaned", "call_id": None}

    # L3 -- finished? The job decides from committed files alone. A finished run
    # must read as finished forever, and Modal forgets old call outcomes, so ask
    # neither Modal nor the Dict. An unrecognised config is quarantined like an
    # unreadable one.
    try:
        done, total = JOB.progress(run_dir(run_id), config)
    except Aborted as e:
        return {"run_id": run_id, "reason": f"{e} -- orphaned", "call_id": None}
    if done >= total:
        return {"run_id": run_id, "reason": "already complete", "call_id": None}

    # L4 -- someone on it? The lease says who was last *granted* the run, not who is
    # running it -- a lease pointing at a dead worker is the normal state of
    # anything waiting to resume. Ask Modal about that call before believing it.
    grant = await leases.get.aio(lease_key(run_id))
    if grant is not None:
        state = await status(modal.FunctionCall.from_id(grant["call_id"]))
        if state == "running":
            return {"run_id": run_id, "reason": "already running", "call_id": grant["call_id"]}

    # L5/L6 -- the folder is free: take over a dead lease, or adopt one with none.
    # Clear before spawning, because between spawn and grant the key still holds the
    # dead worker's id -- a warm container can read it, see "someone else owns this"
    # and kill itself over a lease milliseconds from being its own. An empty key
    # instead just means "wait".

    await leases.pop.aio(lease_key(run_id), None)
    # An ordinal, not an entity: attempt 2 *at* the run. The worker making it is
    # named by call id; only the count lives here.
    attempt = (grant["attempt"] + 1) if grant else 1
    call = await work.spawn.aio(run_id)
    await leases.put.aio(
        lease_key(run_id),
        {"call_id": call.object_id, "granted_ts": time.time(), "attempt": attempt},
    )

    return {
        "run_id": run_id,
        "reason": f"spawned attempt {attempt} from {done}/{total}",
        "call_id": call.object_id,
    }


# --------------------------------------------------------------------------------------
# the worker: computes freely, commits only with a confirmed lease
# --------------------------------------------------------------------------------------


class Lease:
    """What a job writes through: check the lease, and gate commits on it.

    Nothing here grants anything -- the lease is the Dict entry and only `launch`
    ever writes one. This is the reading half: compare that grant against the call
    id of the worker asking, and refuse to commit without a match. Two surfaces
    over that one check, so proving ownership need not pretend to be a block:

        lease.confirm("fence-in")        # just prove the grant still names us
        with lease(label, commit=True):  # prove it, run the block, commit
            write_checkpoint(...)

    A job holds one of these and nothing else of the worker's -- no identity, no
    Dict. `call_id` is public because a ledger stamps rows with it; `logger` is the
    worker's own, so checks land in the same file as everything else. Retry
    settings live here, so callers ask for permission, not for a number of tries.
    """

    def __init__(
        self, run_id: str, call_id: str, logger, tries: int = LEASE_RETRIES, backoff: float = LEASE_BACKOFF
    ):
        self.run_id = run_id
        self.call_id = call_id
        self.logger = logger
        self.tries = tries
        self.backoff = backoff

    def confirm(self, label: str = "lease") -> None:
        """Prove the grant still names us, or raise LeaseLost.

        Someone else's id raises at once; only UNKNOWN (see `fence`) is worth
        waiting out, and only briefly. Waiting cannot cost us the run -- the
        launcher checks Modal for liveness before reassigning, and a worker that
        merely lost the Dict still looks alive there.

        Passes are logged too: afterwards only the log tells a boundary that
        committed from one that was merely allowed to.
        """
        for i in range(1, self.tries + 1):
            verdict, grant = fence(self.run_id, self.call_id)
            if verdict == MATCH:
                self.logger.info(f"{label}: lease held (attempt {grant['attempt']}, try {i}/{self.tries})")
                return
            if verdict == MISMATCH:
                raise LeaseLost(f"{label}: another worker holds this run ({grant['call_id']})")
            if i == self.tries:
                raise LeaseLost(f"{label}: indeterminate after {self.tries} tries -- ownership never confirmed")
            self.logger.info(f"{label}: indeterminate, retry {i}/{self.tries}")
            time.sleep(self.backoff * i)  # linear; no jitter needed at this fleet size

    @contextmanager
    def __call__(self, label: str = "lease", commit: bool = False):
        """Prove ownership, run the block, then commit if asked.

        Callable rather than a named method so a job reads as `with self.lease(...)`.

        The commit is the only irreversible thing a worker does -- everything before
        it dies with the container, so being denied costs work, never history. A
        raising block skips it.

        Not mutual exclusion at an instant: the check and the commit hit two
        services and cannot be atomic, so the lease can move mid-block. The
        guarantee is weaker and sufficient -- nobody commits without having proved
        ownership since their last commit. The gap is as wide as the block is slow
        and costs at most one interval done twice (checkpoints overwrite per step,
        the ledger tolerates repeats). Shorten the block if that ever matters.
        """
        self.confirm(label)

        yield

        if commit:
            self.logger.info(f"committing {label}")
            volume.commit()


@app.function(image=image, volumes={"/storage": volume}, timeout=3600, retries=0)
def work(run_id: str) -> dict:
    """One worker: prove ownership, hand the lease to the job, record why we stopped.

    The container is the worker, so this function is the whole of one: it sets up
    its own logging, builds the lease callback the job writes through, runs the
    job, and records why it stopped. Nothing here knows what the job does -- not
    its config, not its layout, not what "done" means to it. And the job never
    learns a worker exists; it enters a context manager when it wants something to
    survive, and a denied gate raises straight through its own loop.

    The whole body is wrapped because of one ambiguity: Modal signals "still
    running" by raising a builtin TimeoutError, and it also re-raises whatever the
    worker raised. So a worker that dies on a socket or dataloader timeout -- both
    are builtin TimeoutError since Python 3.10 -- would read as healthy forever,
    and the launcher would refuse to resume it. Re-raising as RuntimeError
    guarantees every failure reads as a failure.

    The log capture sits inside that wrapper, so a crash is written to the run's
    own file -- while the handler is still attached -- before the failure is
    reworded on the way out.
    """
    try:
        volume.reload() # fresh view
        call_id = modal.current_function_call_id()
        logdir = run_dir(run_id) / "logs" / call_id
        logdir.mkdir(parents=True, exist_ok=True)
        logger = worker_logger(call_id, logdir / "worker.log")
        # lease allows us to check ownership of the folder.
        lease = Lease(run_id, call_id, logger)
        try:
            logger.info(f"boot call_id={call_id}")
            try:
                # context manager format - does not really hold ownership - 
                # rather it checks before starting the code inside 
                # and optionally commits at the end
                with lease("fence-in",commit=False):
                    job = JOB(lease, run_id, logger)
                    job.run()
            except Aborted as e:
                # A clean stop, not a crash: the job was handed a folder it cannot
                # run, or was denied the lease and left without writing.
                reason = str(e)
            else:
                # Ask the job's progress rather than trust a return value.
                done, total = JOB.progress(job.rdir, job.config)
                reason = f"finished at {done}/{total}"

            # The tidy way out: say why we are leaving, publish it, and hand the
            logger.info(f"EXIT {reason}")
            volume.commit()
            return {"run_id": run_id, "reason": reason}
        except BaseException as e:
            logger.exception("EXIT crashed: %r", e)
            raise
        finally:
            release_worker_logger()
    except Exception as e:
        raise RuntimeError(f"worker failed for {run_id}: {e!r}") from e


# What this deployment runs. One name, so the launcher and the worker cannot
# disagree: the launcher calls JOB.progress to see if a folder is finished, the
# worker builds a JOB to do the work. Swapping in a real trainer is this line
# plus the image.
# TODO: This will generalize so that it will pick up the job from config of the folder.
JOB = Count


# --------------------------------------------------------------------------------------
# the dashboard
# --------------------------------------------------------------------------------------


# scaledown_window has to be longer than the page's poll interval, or the container
# goes idle *between* polls and every single refresh pays a cold start -- which is
# what "the dashboard feels slow" turns out to mean. 60s keeps it warm through a
# session and for a minute after.
#
# The cost is deploy latency: a warm container keeps serving the ASGI app it was
# built with, so a UI change can take up to a minute to appear. When iterating on
# the page, drop this to 2 (Modal's floor) to see changes within ~6s.
@app.function(image=web_image, volumes={"/storage": volume}, timeout=60, scaledown_window=60)
@modal.asgi_app()
def dashboard():
    """Serve the run list. All the drawing lives in dashboard.py.

    That module is imported here, inside the function, for two reasons: it imports
    from this one, so a module-level import would be circular; and it is only ever
    needed in the container, where fastapi is installed.
    """
    from dashboard import build_web_app

    return build_web_app()


# --------------------------------------------------------------------------------------
# the ETL: the pipeline workflow, run against the Volume
# --------------------------------------------------------------------------------------


@app.function(image=etl_image, volumes={STORAGE: volume}, cpu=4, timeout=3600, max_containers=1)
def etl(run_id: str) -> dict:
    """Build what one run needs, with the Volume as snakemake's data directory.

    `run_id` is the same key `launch` uses -- one run, one folder, one primary
    key from ETL through training. One hardcoded command, shelling out to the real
    snakemake rather than reimplementing any part of it. Three of its arguments
    are the interesting ones:

        --directory /storage   the working directory, so .snakemake/ (the metadata
                               that decides what needs rebuilding) survives the
                               container instead of dying with it
        --config volume=...    where artifacts go. The Snakefile defaults this to
                               its own folder, which here is a read-only image
                               mount, so it has to be redirected at the Volume.
        --config run_id=...    whose request to build. The Snakefile has no default
                               for it: guessing would build another run's spec.

    The first two are the whole reason a container can do incremental builds at
    all: with either one missing, every invocation starts from an empty tree and
    rebuilds the world.

    What to build is read from the Volume too -- /storage/runs/{run_id}/run.yaml,
    which the Snakefile locates from those same two values. It is not in the
    image, so changing what a run asks for is an upload rather than a redeploy:

        modal volume put test-volume run.yaml /runs/{run_id}/run.yaml --force

    and it has to be there before the call, or snakemake raises on the missing
    file. Artifacts land under /storage/data/, addressed by uid and shared with
    every other run -- asking for an encoder another run already fit costs
    nothing.

    No lease, unlike `work`. This is a different shape of job -- a DAG that is
    idempotent by construction, where a crashed run leaves finished outputs in
    place and re-running resumes from them. What it does need is to not race
    itself, and `max_containers=1` gives that the same way it does for `launch`.
    Snakemake's own directory lock is the backstop: a container killed mid-run (a
    cancel, a timeout) leaves that lock behind on the Volume, and the next call
    refuses to start until someone clears it -- which right now means adding
    --unlock to the command below and redeploying.
    """
    # run_id is the one thing interpolated into a shell command below, and it also
    # becomes a path segment. Both reasons to insist it is a bare name -- which
    # every id `mint_run_id` produces already is.
    assert re.fullmatch(r"[A-Za-z0-9._-]+", run_id), f"run_id {run_id!r} is not a bare name"

    volume.reload()  # containers are reused; start from what previous runs committed

    # The target goes first, before any option: `--config` takes a variable-length
    # list of key=value pairs, so a target after it would be swallowed as another
    # pair and rejected as `Invalid config definition`.
    command = (
        "snakemake all"
        f" --snakefile {PIPELINE}/Snakefile"
        f" --directory {STORAGE}"
        f" --workflow-profile {PIPELINE}/profiles/default"
        f" --config volume={STORAGE} run_id={run_id}"
    )
    print(f"$ {command}")

    # shell=True so what runs is exactly the line printed above. Bare `snakemake`
    # resolves because uv_sync puts its venv's bin on PATH.
    result = subprocess.run(command, shell=True)

    # Commit either way. A failed workflow still finished some of its jobs, and
    # those outputs plus the metadata describing them are exactly what makes the
    # next run skip them -- discarding that is what would make failure costly.
    # (snakemake deletes the output of the job that failed, so nothing half-written
    # is being kept here.)
    volume.commit()

    if result.returncode:
        raise RuntimeError(f"snakemake exited {result.returncode}")
    return {"run_id": run_id, "target": "all", "returncode": result.returncode}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


@app.local_entrypoint()
def main(run_id: str = "", name: str = "run", total_steps: int = 20, save_every: int = 5, step_seconds: float = 1.0):
    """Start or resume a run through the deployed launcher.

    `Function.from_name` is what keeps the lock intact. Calling `launch.remote`
    from this throwaway app instance would run it in a second container pool that
    doesn't serialize against the deployed one.
    """
    launcher = modal.Function.from_name(APP_NAME, "launch")
    if run_id:
        # A resume passes no config at all: config.json was written once, at
        # creation, and every worker since runs under it.
        print(launcher.remote(run_id=run_id))
    else:
        config = JOB.make_config(total_steps=total_steps, save_every=save_every, step_seconds=step_seconds)
        print(launcher.remote(name=name, config=config))


@app.local_entrypoint()
def prepare(run_id: str):
    """Build the data one run needs, on Modal.

        modal run launcher/app.py::prepare --run-id 2026...._alpha

    The same run_id `main` starts a run under: its request is read from
    /storage/runs/{run_id}/run.yaml, which has to be on the Volume first.

    Unlike `main`, this calls `etl.remote` on the local app rather than reaching
    for the deployed one. `main` needs `Function.from_name` because `launch`'s
    single-container lock only means anything within one deployment; the workflow
    has no such lock to protect, so running it from here is the same job.
    """
    print(etl.remote(run_id=run_id))
