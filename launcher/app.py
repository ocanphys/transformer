"""Run long jobs on Modal so that crashes, cancels, and restarts are all safe.

A job lives in one folder on a Volume. Restarting picks up from the last
checkpoint, asking twice does nothing, and two workers can never write the same
folder -- enforced, not assumed.

Each system does only what it is good at:

    Dict   = who may write, now    (the lease -- atomic per key, always fresh)
    Modal  = who is alive, now     (call state -- we ask, never write)
    Volume = what happened         (config, checkpoints, ledger, logs)

The parts:

    launch    start or resume a run; safe to call repeatedly. Builds the run's data,
              then spawns workers.
    work      one worker -- one container, one call id, one go at one run.
    JOB       what this deployment runs; Train, on the data the ETL built.
    etl       runs the notebooks/pipeline workflow to build the data a run trains
              on. No lease: a DAG of file targets is already idempotent.

A run is one folder and one file: runs/{run_id}/config.json, written once at
creation from the config dict in etl_train_pipeline.ipynb. Everything else in the
folder is derived -- the request the ETL resolved, the data it gathered, the
checkpoints the worker wrote.

Logging lives in logs.py -- a logger named for the call id, handlers answering to
that name only, so a handler outliving its call cannot write into the next run.

The guarantee: `launch` is single-container and the only writer of the Dict, so a
worker can never promote itself. A worker checks the lease against its own call
id, and that gates exactly one thing -- `lease(commit=True)`. Nothing it writes is
visible to anyone until that commit.

Simplifications, each explained where it happens: no config hash, no checkpoint
pruning, no audit copy of each grant, one retry policy for every lease check.

Usage:
    modal deploy launcher/app.py                          # deploy (do this first)
    etl_train_pipeline.ipynb                              # start a run: it owns the config
    modal run launcher/app.py --run-id 2026...._alpha     # resume it
    modal run launcher/app.py::prepare --run-id 2026....  # rebuild that run's data
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import modal
from modal.exception import Error, FunctionTimeoutError, OutputExpiredError

# Jobs and the shared run-folder layout live in jobs.py. The dependency runs one
# way (app -> jobs), so there is no cycle and Aborted has a single definition.
from jobs import Aborted, Train, encoder_record, mint_run_id, load_config, run_dir, split_bin

# Logging lives in logs.py: a logger named for this call, and handlers that
# answer to that name only. `work` and `etl` each take one and pass it on.
from logs import call_logger, release_call_logger

APP_NAME = "training-launcher"
VOLUME_NAME = "test-volume"  # flip to "LLM-pretraining" once the real trainer is wired in
DICT_NAME = "training-leases"

# Where the pipeline lives on each side. PROJECT_ROOT comes from __file__, not
# config.py: only launcher/ is on fthe path in a container, and this is read while
# describing the image, never inside one.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_DIR = PROJECT_ROOT / "notebooks" / "pipeline"  # the Snakefile and its modules, locally
PIPELINE = "/pipeline"  # the same folder in the image, read-only
STORAGE = "/storage"  # the Volume: where the workflow's artifacts and its metadata land

LEASE_RETRIES = 5  # how many times an indeterminate lease read is worth re-asking
LEASE_BACKOFF = 5.0  # seconds x try number -> 5, 10, 15, 20: ~50s of patience in total

# What `work` trains on. Must be a GPU the configs' model_params.device asks for:
# set this to None and device to "cpu" in the config to exercise the wiring without
# paying for one -- both have to change together or torch raises at construction.
GPU = "T4"

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
# applog rides along because the dashboard is the only thing that calls it: it asks
# Modal for this app's own logs on demand and renders them, so there is no
# collector, no second container, and nothing stored.
web_image = base_image.pip_install("fastapi[standard]").add_local_python_source(
    *LOCAL_MODULES, "applog", "dashboard"
)

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

# The trainer needs the same project environment as the ETL, minus the pipeline
# directory -- it reads its data out of the run folder, not out of the workflow --
# plus wandb_run, which `Train.run` imports lazily.
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=PROJECT_ROOT)
    .add_local_python_source(*LOCAL_MODULES, "wandb_run", "transformer")
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

        # The data this config asks for, before anyone trains on it. Committed first,
        # because the ETL reads that config.json from its own mount.
        #
        # Blocking, and inside the lock: a first build of a new encoder holds this
        # container for as long as it takes, and every other launch waits. What makes
        # that affordable is that it is the only time it happens -- artifacts are
        # addressed by uid, so a config asking for anything already built comes back
        # in seconds with "Nothing to be done". Creation only: config.json is written
        # once, so a resume's request cannot have changed, and a bin that went missing
        # under one is a real error the worker should report rather than paper over.
        await etl.remote.aio(run_id)

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
        return {"run_id": run_id, "reason": "unreadable -- no config.json to run under", "call_id": None}

    # L3 -- finished? The job decides from committed files alone. A finished run
    # must read as finished forever, and Modal forgets old call outcomes, so ask
    # neither Modal nor the Dict. An unrecognised config is quarantined like an
    # unreadable one.
    try:
        done, total = JOB.progress(run_dir(run_id), config)
    except Aborted as e:
        return {"run_id": run_id, "reason": f"unreadable -- {e}", "call_id": None}
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
    await etl.remote.aio(run_id)
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


@app.function(
    image=train_image,
    volumes={"/storage": volume},
    gpu=GPU,
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=3600,
    retries=0,
    single_use_containers=True,
)
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
    reworded on the way out. The one exception is the reload below, for a reason
    that cannot be designed around; see there.

    `single_use_containers=True` is what makes "the container is the worker" true
    rather than aspirational. Containers are reused by default, and a worker leaves
    this one holding open memmaps on its run's train.bin/valid.bin -- a cancelled or
    crashed attempt never gets to close them, which is exactly the case that needs
    to survive. The next input to land here would then fail on the reload below,
    because a volume cannot be swapped while files under it are open. One input per
    container means every attempt starts with nothing open, at the price of a cold
    start per attempt -- nothing, against a training run.
    """
    try:
        # A fresh view of what previous attempts committed. First, and before anything
        # opens a file on the volume: reload replaces the mount and refuses outright
        # while any file under it is open.
        #
        # That ordering is forced, and it is why a failure here is the one thing that
        # never reaches the run's own worker.log -- the log file lives on this volume,
        # so opening it before the reload would itself be what blocks the reload. This
        # failure appears in the container log only. `modal app logs` is where to look.
        volume.reload()
        call_id = modal.current_function_call_id()
        logdir = run_dir(run_id) / "logs" / call_id
        logdir.mkdir(parents=True, exist_ok=True)
        logger = call_logger(call_id, logdir / "worker.log")
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
                # Ask the job's progress rather than trust a return value. The stored
                # config, not a resolved one: `progress` is a classmethod the launcher
                # also calls with nothing but a config.json in hand.
                done, total = JOB.progress(job.rdir, job.config_dict)
                reason = f"finished at {done}/{total}"

            # The tidy way out: say why we are leaving, publish it, and hand the
            logger.info(f"EXIT {reason}")
            volume.commit()
            return {"run_id": run_id, "reason": reason}
        except BaseException as e:
            logger.exception("EXIT crashed: %r", e)
            raise
        finally:
            release_call_logger()
    except Exception as e:
        raise RuntimeError(f"worker failed for {run_id}: {e!r}") from e


# What this deployment runs. One name, so the launcher and the worker cannot
# disagree: the launcher calls JOB.progress to see if a folder is finished, the
# worker builds a JOB to do the work.
#
# `jobs.Count` is the toy that exercises the lease machinery without a GPU. Going
# back to it means importing it here as well as naming it, and minting its flat
# {total_steps, save_every} config by hand -- nothing writes one any more.
# TODO: This will generalize so that it will pick up the job from config of the folder.
JOB = Train


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
# One container, many requests. Without this a container serves a single input at a
# time, so a poll that overlaps the previous one can only be answered by starting
# another container -- which cold-starts, which is slow, which causes more overlap.
# Measured before adding it: 1,639 requests spread over **89 containers**, with
# `/rows` at p50 538ms, p90 1.5s, and 61 requests slower than the 3s poll interval.
#
# Everything here waits on the network -- a volume reload, Modal's liveness calls,
# a log fetch -- so one container can hold many of them at once for free. `def`
# handlers still go to FastAPI's threadpool, which is why applog takes a lock
# around its archive writes.
@modal.concurrent(max_inputs=50)
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


def run_logged(command: str, log) -> int:
    """Run `command`, put every line it writes into `log`, and return its exit code.

    `subprocess.run` inherits this process's stdout, which reaches Modal's container
    log and nowhere else. Reading the pipe line by line instead puts snakemake's own
    account of the DAG -- which rules ran, why, and what a failure said -- into the
    run's own folder, where the dashboard merges it with the worker's log. It still
    reaches the container log too: `log`'s second handler is stdout.

    stderr is folded into stdout because snakemake writes its progress there and the
    interleaving is the point -- one stream keeps the order things happened in.
    PYTHONUNBUFFERED stops the child's stdout from block-buffering behind the pipe,
    which would otherwise deliver a whole twenty-minute build in one lump at exit.
    """
    process = subprocess.Popen(
        command,
        shell=True,
        text=True,
        bufsize=1,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    # Blank lines are kept: snakemake's job tables and rule blocks are readable
    # because of them, and a bare timestamp is a cheap price for that.
    for line in process.stdout:
        log.info(line.rstrip())
    return process.wait()


def gather_run_data(run_id: str, log) -> None:
    """Put this run's data in its own folder: train.bin, valid.bin, encoder.json.

    PROVISIONAL, and the seam between a pipeline that builds one bin per source and
    a trainer that wants one per split. Not a snakemake rule, so it is a deletion
    rather than an untangling once a rule produces one shared bin per (encoder,
    split) -- this function and its one call site go, and `jobs.split_bin` with
    them. The cost meanwhile is a copy of the tokens per run.

    Runs inside the ETL container, after the DAG is complete and before its commit,
    so a worker never sees a run folder whose bins are half-joined. Imports the
    pipeline's own etl.py from the read-only mount rather than reimplementing where
    a bin lives -- the same module the Snakefile uses.
    """
    sys.path.insert(0, PIPELINE)
    import etl as pipeline

    rdir = run_dir(run_id)
    data = f"{STORAGE}/data"

    # From the request rather than the config: it carries the encoder's address,
    # already derived and validated by the workflow that just ran.
    request = pipeline.read_request(rdir)
    uid = request["encoder"]["encoder_uid"]

    for split in ("train", "valid"):
        bins = [pipeline.encoded_bin(data, uid, uid_) for uid_ in request["sources"][f"{split}_sources"]]
        target = pipeline.concat_bins(bins, split_bin(rdir, split))
        log.info(f"{split}.bin  {len(bins)} sources -> {target.stat().st_size:,} bytes")

    # What encoded those tokens, beside them, so a job can check vocab_size without
    # knowing anything about where encoders live.
    shutil.copyfile(pipeline.encoder_config(data, uid), encoder_record(rdir))


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

    # Before anything creates a directory. `mkdir(parents=True)` under a run_id that
    # is not a run would mint a folder the dashboard then lists forever as
    # "unreadable"; the workflow would fail on the missing config a minute later
    # anyway. This fails first, and leaves nothing behind.
    if load_config(run_id) is None:
        raise RuntimeError(f"no readable config.json for {run_id} -- nothing to build")

    # The same folder a worker logs into, named for this call, with the actor in the
    # filename. That is the whole of what puts the ETL in the run's merged log view:
    # `dashboard.read_logs` globs the folder and sorts on the timestamp column, so an
    # `etl.log` beside a `worker.log` interleaves with no further plumbing.
    #
    # After the reload, never before -- reload refuses while a file on the volume is
    # open, and this opens one.
    call_id = modal.current_function_call_id()
    logdir = run_dir(run_id) / "logs" / call_id
    logdir.mkdir(parents=True, exist_ok=True)
    log = call_logger(call_id, logdir / "etl.log")

    returncode = None
    try:
        log.info(f"etl call_id={call_id} run_id={run_id}")

        # The target goes first, before any option: `--config` takes a variable-length
        # list of key=value pairs, so a target after it would be swallowed as another
        # pair and rejected as `Invalid config definition`.
        #
        # --nocolor because this output is now a file as well as a terminal: snakemake
        # suppresses colour behind a pipe on its own, and the flag makes that a
        # guarantee rather than something observed once.
        command = (
            "snakemake all"
            f" --snakefile {PIPELINE}/Snakefile"
            f" --directory {STORAGE}"
            f" --workflow-profile {PIPELINE}/profiles/default"
            " --nocolor"
            f" --config volume={STORAGE} run_id={run_id}"
        )
        log.info(f"$ {command}")

        # shell=True so what runs is exactly the line logged above. Bare `snakemake`
        # resolves because uv_sync puts its venv's bin on PATH.
        returncode = run_logged(command, log)

        # Only once the DAG is complete: a partial build would concatenate whichever
        # bins happen to exist into a file the trainer would read as authoritative.
        if not returncode:
            gather_run_data(run_id, log)

        log.info(f"EXIT {'finished' if not returncode else f'snakemake exited {returncode}'}")

        # Commit either way. A failed workflow still finished some of its jobs, and
        # those outputs plus the metadata describing them are exactly what makes the
        # next run skip them -- discarding that is what would make failure costly.
        # (snakemake deletes the output of the job that failed, so nothing half-written
        # is being kept here.) The log written above rides along on the same commit.
        volume.commit()
    except BaseException as e:
        # While the handler is still attached, so the run's own folder records the
        # crash rather than only the container log.
        log.exception("EXIT crashed: %r", e)
        raise
    finally:
        # Correctness here, not tidiness: this container is reused and every call
        # starts with `volume.reload()`, which fails outright while a file under the
        # mount is open. A leaked handler would break the *next* ETL call.
        release_call_logger()

    # After the log is closed and the commit has landed -- the nonzero exit is already
    # recorded as an EXIT line, so this only needs to make the call fail.
    if returncode:
        raise RuntimeError(f"snakemake exited {returncode}")
    return {"run_id": run_id, "target": "all", "returncode": returncode}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


@app.local_entrypoint()
def main(run_id: str):
    """Resume a run through the deployed launcher.

        modal run launcher/app.py --run-id 2026...._alpha

    Resume only, and it passes no config at all: config.json was written once, at
    creation, and every worker since runs under it. Starting a *new* run means
    supplying that config, which lives in etl_train_pipeline.ipynb -- so the
    notebook calls `launch` directly rather than this being able to mint one.

    `Function.from_name` is what keeps the lock intact. Calling `launch.remote`
    from this throwaway app instance would run it in a second container pool that
    doesn't serialize against the deployed one.
    """
    launcher = modal.Function.from_name(APP_NAME, "launch")
    print(launcher.remote(run_id=run_id))


@app.local_entrypoint()
def prepare(run_id: str):
    """Rebuild the data one run needs, on Modal.

        modal run launcher/app.py::prepare --run-id 2026...._alpha

    `launch` already does this for every new run, so this is for doing it again on
    its own -- after clearing an artifact, or to see the workflow's output without
    starting a worker. Reads /storage/runs/{run_id}/config.json, which has to be on
    the Volume already.

    Unlike `main`, this calls `etl.remote` on the local app rather than reaching
    for the deployed one. `main` needs `Function.from_name` because `launch`'s
    single-container lock only means anything within one deployment; the workflow
    has no such lock to protect, so running it from here is the same job.
    """
    print(etl.remote(run_id=run_id))
