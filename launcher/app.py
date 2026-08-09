"""Run long jobs on Modal so that crashes, cancels, and restarts are all safe.

A job lives in one folder on a Volume. Restarting it picks up from the last
checkpoint; asking twice does nothing the second time; and two workers can never
write the same folder. That last guarantee is enforced, not assumed.

Three systems, each used only for what it is actually good at:

    Dict   = who may write, now    (the lease -- always fresh, atomic per key)
    Modal  = who is alive, now     (call state -- we ask, never write)
    Volume = what happened         (config, checkpoints, ledger, logs)

Logging is not one of the three. `logs.py` owns it -- handlers, formatter, and
the stdout capture that puts a job's print() in the same file as the protocol
lines -- because it is process-wide interpreter state, not launcher logic.

Three parts, and only the last is specific to what you are running:

    launch    start or resume a run. Safe to call repeatedly -- it spawns workers.
    work      one worker: prove ownership, run the job, record why it stopped.
              A worker is one container with one call id, having one go at one
              run. `Worker` is its identity, `work` is what it does.
    JOB       the class this deployment runs; the toy counter here stands in
              for training.

`etl` sits beside all of that rather than inside it: it runs the snaketl
snakemake workflow in a container to build the data a run trains on. It needs no
lease because a DAG of file targets is already idempotent -- the guarantee the
lease buys is the one thing snakemake gives for free.

How the guarantee works. `launch` runs one-at-a-time (max_containers=1) and is
the only thing that ever writes the Dict, so a worker can never promote or
release itself. A worker compares the lease against its own call id, and that
check gates exactly one thing: `lease(commit=True)`. Nothing a worker writes is
visible to anyone until that commit, so an unconfirmed worker can compute all it
likes and still change nothing.

Known simplifications, each explained where it happens: no config hash, no
checkpoint pruning, no audit copy of each grant, and one shared retry policy for
every lease check.

Usage:
    modal deploy launcher/app.py                       # deploy (do this first)
    modal run launcher/app.py --name alpha             # start a run
    modal run launcher/app.py --run-id 2026...._alpha  # resume it
    modal run launcher/app.py::prepare --targets main  # build the data for one split
"""

import json
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import modal
from modal.exception import Error, FunctionTimeoutError, OutputExpiredError

# The jobs, and the run-folder substrate both sides share, live in jobs.py. The
# dependency runs one way (app -> jobs; jobs never imports app), so there is no
# cycle, and Aborted has one definition -- the class app.py catches here is the
# same one Count raises there.
from jobs import Aborted, Count, mint_run_id, load_config, run_dir

# All the process-wide logging state -- handlers, formatter, stdout capture --
# lives in logs.py. A worker holds one WorkerLog and nothing else of it.
from logs import WorkerLog

APP_NAME = "training-launcher"
VOLUME_NAME = "test-volume"  # flip to "LLM-pretraining" once the real trainer is wired in
DICT_NAME = "training-leases"

# Where the snaketl workflow lives on each side. PROJECT_ROOT is derived from
# __file__ rather than imported from config.py because only launcher/ is on the
# path in a container -- and it is a local-only value anyway, read while the
# image is being described, never inside one.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAKETL = "/snaketl"  # the workflow definition, read-only, shipped with the image
STORAGE = "/storage"  # the Volume: where the workflow's artifacts and its metadata land

LEASE_RETRIES = 5  # how many times an indeterminate lease read is worth re-asking
LEASE_BACKOFF = 5.0  # seconds x try number -> 5, 10, 15, 20: ~50s of patience in total

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
leases = modal.Dict.from_name(DICT_NAME, create_if_missing=True)

# The toy worker needs nothing but the stdlib. A real trainer swaps this for the
# uv_sync image from modal_lab/run.ipynb -- only this line changes.
base_image = modal.Image.debian_slim(python_version="3.12")

# jobs.py and logs.py ride along on every image: app.py imports both at module
# load, so they must be present in any container that imports app.py -- launch,
# work, dashboard, and etl alike. Miss one and the container fails on import,
# before any of this code gets a chance to run.
# add_local_* has to come last in a chain (Modal adds these files at container
# startup rather than baking them in, so editing jobs.py never rebuilds the image);
# any build step like pip_install must therefore run before it.
LOCAL_MODULES = ("jobs", "logs")

image = base_image.add_local_python_source(*LOCAL_MODULES)

# The dashboard needs a web server, and its own module shipped alongside this one.
web_image = base_image.pip_install("fastapi[standard]").add_local_python_source(*LOCAL_MODULES, "dashboard")

# The ETL container runs the snaketl workflow, so unlike the toy worker it needs
# the real project environment: uv_sync installs pyproject.toml's dependencies
# (snakemake, joblib, regex, ...) from the lockfile, and `transformer` itself
# rides along as local source because the Snakefile imports from it directly.
#
# The Snakefile, its config.yaml and its profile come over as a plain directory:
# add_local_python_source only ships importable modules, and none of these are.
# data/ and .snakemake/ are excluded because they are the workflow's *output* --
# the container builds those on the Volume, and shipping a laptop's copy would
# both bloat the mount and plant stale artifacts where snakemake would trust them.
etl_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=PROJECT_ROOT)
    .add_local_python_source(*LOCAL_MODULES, "transformer")
    .add_local_dir(PROJECT_ROOT / "snaketl", SNAKETL, ignore=["data", ".snakemake", "**/__pycache__"])
)

# --------------------------------------------------------------------------------------
# liveness: the single source of truth for "is that call still running?"
# --------------------------------------------------------------------------------------


async def status(call):
    """Is this call still running? Ask Modal, without blocking.

    Checked against modal 1.5.2. The catch order matters: OutputExpiredError and
    FunctionTimeoutError both subclass *modal's* TimeoutError, which is not the
    builtin -- so `except TimeoutError` catches only "still running", and those
    two must come before the general Error branch. Re-check this on any modal
    upgrade; it is the one version-fragile thing in the file.

    Keep this as the only copy. A second implementation drifts, and drift here
    means calling a live worker dead.
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

    Raised rather than returned because the check sits inside a context manager
    the job's own loop enters. Raising unwinds that loop from the inside, so the
    job never needs to know a lease exists.
    """


def lease_key(run_id: str) -> str:
    return f"lease:{run_id}"


def fence(run_id: str, my_call_id: str) -> tuple[str, dict | None]:
    """Do we own this run? One fresh Dict read, three answers -- never two.

    Returns the grant alongside the verdict. The read already fetched it, and
    callers want to say *who* holds the run, not just whether we do -- which
    attempt number we are, or which call beat us -- and that should not cost a
    second round-trip to find out.

    MATCH and MISMATCH are both real answers: a key holding someone else's id is
    the server telling us they were granted this run, so we stop at once (asking
    again is just hoping the answer changes).

    UNKNOWN -- no key, or the read failed -- is silence, not a no. It happens
    during a Dict outage, after a 7-day expiry, or for a worker that was never
    granted anything. Treating silence as "denied" would take down the whole
    fleet on one blip; treating it as "granted" would let a disconnected zombie
    write. So we wait a bounded time and then give up, which is the only policy
    that errs safely in both directions.
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
# run_dir, load_config, and mint_run_id -- the folder layout every run shares --
# live in jobs.py, the leaf both the launcher here and the jobs there import from.


# --------------------------------------------------------------------------------------
# the launcher: the only writer of the Dict, and the only mutual exclusion we have
# --------------------------------------------------------------------------------------


@app.function(image=image, volumes={"/storage": volume}, max_containers=1, scaledown_window=60, timeout=300)
async def launch(run_id: str | None = None, name: str = "run", config: dict | None = None) -> dict:
    """Start a new run, or resume one. Calling it repeatedly is always safe.

    `max_containers=1` makes Modal run these one at a time, in one container, and
    that is the only real lock available here. The steps below read the lease, ask
    Modal whether that call is alive, then grant -- a sequence no single system
    can do atomically. The Volume cannot lock (it checks against stale snapshots)
    and neither can the Dict (no compare-and-swap).

    That only holds within one deployed app, so always reach this through
    `Function.from_name`. A `modal run` of this module gets its own container pool
    and its own lock. If that slips, two safety nets remain: put-if-absent picks
    one winner among simultaneous new runs, and a worker's own fence limits any
    takeover race to one dropped checkpoint interval.

    Always returns a "reason" string, because otherwise a no-op and a fresh spawn
    look the same to the caller.
    """
    await volume.reload.aio()  # this container is long-lived, so its snapshot is stale by default

    # L1 -- new run: create the folder and its config, spawn, then grant.
    #
    # That order is forced, not chosen. The config has to be committed before the
    # spawn or the worker's mount won't see it, and the call id doesn't exist
    # until the spawn returns, so the grant can't come first either. In the gap
    # the worker sees no key at all, which reads as UNKNOWN, so it waits.
    if run_id is None:
        run_id = mint_run_id(name)
        if run_dir(run_id).exists():
            # Someone already owns this name. config.json is written once and
            # every resume runs under it, so never write through an existing one.
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
            # worker we just spawned will see the mismatch and bow out by itself.
            return {"run_id": run_id, "reason": "lost a race for a freshly minted id", "call_id": None}

        return {"run_id": run_id, "reason": "spawned", "call_id": call.object_id}

    # L7 -- no readable config. A run whose definition is gone gets quarantined,
    # never guessed at.
    config = load_config(run_id)
    if config is None:
        return {"run_id": run_id, "reason": "no readable config.json -- orphaned", "call_id": None}

    # L3 -- finished? The job decides, from committed files alone. A finished run
    # must read as finished forever, and Modal eventually forgets old call
    # outcomes, so we ask neither Modal nor the Dict here. A config this job does
    # not understand is quarantined the same way an unreadable one is.
    try:
        done, total = JOB.progress(run_dir(run_id), config)
    except Aborted as e:
        return {"run_id": run_id, "reason": f"{e} -- orphaned", "call_id": None}
    if done >= total:
        return {"run_id": run_id, "reason": "already complete", "call_id": None}

    # L4 -- someone already working on it? The lease says who was last granted the
    # run, which is not the same as who is running it: a lease pointing at a dead
    # worker is the normal state of anything waiting to resume. So ask Modal about
    # that specific call before believing it.
    grant = await leases.get.aio(lease_key(run_id))
    if grant is not None:
        state = await status(modal.FunctionCall.from_id(grant["call_id"]))
        if state == "running":
            return {"run_id": run_id, "reason": "already running", "call_id": grant["call_id"]}

    # L5/L6 -- the folder is free. Take over from a dead lease, or adopt one that
    # has none. Clear the old key, spawn, then grant.
    #
    # Clearing before the spawn matters. Between spawn and grant the key still
    # holds the dead worker's id, and a warm container can boot fast enough to
    # read it -- an occupied key means "someone else owns this", so the new worker
    # would kill itself over a lease that is milliseconds from being its own.
    # Clearing first makes that gap an empty key, which just means "wait".

    await leases.pop.aio(lease_key(run_id), None)
    # An ordinal, not an entity: this is attempt 2 *at* the run, and the worker
    # making it is named by call id. Only the count lives here.
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


class Worker:
    """Who this worker is and where it logs.

    A worker is one container having one go at one run, and its call id -- unique
    forever, minted by the spawn that created it -- is its whole identity. That
    id is what the lease names, what the log folder is called, and what the ledger
    stamps rows with, so "which worker" always has exactly one answer.

    `work` is the body that goes with this identity; the two are one thing split
    only because a class is how you hold state across a `with`.

    Jobs never get one of these. They get what `bind_lease()` returns --
    permission, without the identity behind it -- because the only thing a job
    needs from a worker is the right to make its writes durable.
    """

    def __init__(self, run_id: str):
        # Refresh our view of the Volume, first thing and once only.
        #
        # Containers are reused, so a warm worker starts out holding a snapshot
        # from before this run's folder existed. It would find no config and no
        # progress -- and "no progress" means "start over", which is how a stale
        # snapshot quietly throws away finished work. This has to come before our
        # first local write, because reload can implicitly commit dirty files.
        #
        # Nothing later needs another one: once we own the run, we never read
        # anyone else's data.
        #
        # This runs before fence-in, which looks backwards but isn't. Fencing only
        # touches the Dict, so the two are unrelated; the order decides only how
        # old our folder snapshot is when we read it -- one Dict round-trip's
        # worth. It also cannot make that read authoritative: a superseded worker
        # that passed its gate before the takeover may still commit afterwards,
        # and that commit hasn't happened yet when we reload. See bind_lease.
        volume.reload()

        self.run_id = run_id
        self.call_id = modal.current_function_call_id()
        self.rdir = run_dir(run_id)
        self.logdir = self.rdir / "logs" / self.call_id
        self.logdir.mkdir(parents=True, exist_ok=True)

        # Redirects this whole container's logging into this worker's own file,
        # and keeps it there until release(). Constructed here, after
        # volume.reload(): reloading fails while a file on the volume is open,
        # and this opens one.
        #
        self._log = WorkerLog(self.logdir / "worker.log")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        """The other way out: something threw, so log_exit never ran.

        The traceback is written here, while the file handler is still attached --
        otherwise the one event most worth having a record of is the one event
        that never reaches the record. Worded as an EXIT so every worker ends
        with exactly one, however it ended.

        Nothing is committed on this path. Modal commits mounted volumes when a
        container stops, so the line lands anyway, and choosing to commit here
        would publish whatever else the job left dirty mid-failure.
        """
        if exc_type is not None:
            self._log.crashed(exc, (exc_type, exc, tb))
        self._log.release()
        return False  # never swallow: work() still needs to see it and re-raise

    def log(self, msg: str) -> None:
        """Write one line to this worker's log. See logs.WorkerLog.log."""
        self._log.log(msg)

    def log_exit(self, reason: str) -> dict:
        """Say why we are leaving, publish it, and hand the caller its return
        value. The caller does the actual leaving.

        There are two ways out of a worker and this is the tidy one, reached
        when we decided to stop: finished, denied, or handed a folder we cannot
        run. The other is __exit__, for when something threw.

        The commit is leaseless, which is safe only because the log is the one
        dirty file -- commits publish the whole container, not one path. A job
        that writes shared files continuously, instead of buffering to a
        checkpoint the way `Count` does, has to deal with that; see `pending`.
        """
        self.log(f"EXIT {reason}")
        volume.commit()
        return {"run_id": self.run_id, "reason": reason}

    def bind_lease(self, tries: int = LEASE_RETRIES, backoff: float = LEASE_BACKOFF):
        """Tie this worker to the lease `launch` granted it, and gate commits on it.

        Nothing here grants anything -- the lease is the Dict entry and only
        `launch` ever writes one. This is the reading half: check that grant
        against our own call id, and refuse to commit without it. Two surfaces
        over that one check, so proving ownership need not pretend to be a block:

            lease.confirm("fence-in")        # just prove the grant still names us
            with lease(label, commit=True):  # prove it, run the block, commit
                write_checkpoint(...)

        A closure so a job can hold it and nothing else of ours; `lease.call_id`
        rides along only because a ledger stamps rows with who wrote them. Retry
        settings are captured here, so callers ask for permission rather than for
        a number of tries.
        """

        def confirm(label: str = "lease") -> None:
            """Prove the grant still names us, or raise LeaseLost.

            Someone else's id is a real answer, so it raises at once; only
            UNKNOWN (see `fence`) is worth waiting out, and only briefly. Waiting
            cannot cost us the run: the launcher checks Modal for liveness before
            reassigning, and a worker that has merely lost the Dict still looks
            alive there.

            Passes are logged too -- afterwards, only the log distinguishes a
            boundary that committed from one that was merely allowed to.
            """
            for i in range(1, tries + 1):
                verdict, grant = fence(self.run_id, self.call_id)
                if verdict == MATCH:
                    self.log(f"{label}: lease held (attempt {grant['attempt']}, try {i}/{tries})")
                    return
                if verdict == MISMATCH:
                    raise LeaseLost(f"{label}: another worker holds this run ({grant['call_id']})")
                if i == tries:
                    raise LeaseLost(f"{label}: indeterminate after {tries} tries -- ownership never confirmed")
                self.log(f"{label}: indeterminate, retry {i}/{tries}")
                time.sleep(backoff * i)  # linear; no jitter needed at this fleet size

        @contextmanager
        def lease(label: str = "lease", commit: bool = False):
            """Prove ownership, run the block, then commit if asked.

            The commit is the only irreversible thing a worker does: everything
            before it dies with the container, so being denied costs work, never
            history. A raising block skips it -- a half-written checkpoint is not
            a checkpoint.

            This is not mutual exclusion at an instant. The check and the commit
            hit two different services and cannot be atomic, so the lease can move
            mid-block. The guarantee is weaker and sufficient: nobody commits
            without having proved ownership since their last commit. The gap is as
            wide as the block is slow, and costs at most one interval done twice
            (checkpoints overwrite per step, the ledger tolerates repeats).
            Shorten the block if that ever matters.
            """
            confirm(label)

            yield

            if commit:
                self.log(f"committing {label}")
                volume.commit()

        lease.confirm = confirm
        lease.call_id = self.call_id
        return lease


@app.function(image=image, volumes={"/storage": volume}, timeout=3600, retries=0)
def work(run_id: str) -> dict:
    """One worker: prove ownership, hand the lease to the job, record why we stopped.

    This is the body of a worker, and `Worker` is its identity -- there is no
    third thing between them. Nothing here knows what the job does: not its
    config, not its layout, not what "done" means to it. And the job never learns
    a lease exists; it enters a context manager when it wants something to
    survive, and a denied gate raises straight through its own loop.

    The whole body is wrapped because of one ambiguity: Modal signals "still
    running" by raising a builtin TimeoutError, and it also re-raises whatever the
    worker raised. So a worker that dies on a socket or dataloader timeout -- both
    are builtin TimeoutError since Python 3.10 -- would read as healthy forever,
    and the launcher would refuse to resume it. Re-raising as RuntimeError
    guarantees every failure reads as a failure.

    The `with` sits inside that wrapper, so a crash still runs `Worker.__exit__`
    -- which writes the traceback while the log file is still attached -- before
    the failure is reworded on the way out.
    """
    try:
        # A context manager because of what it installs, not what it holds: log
        # capture is process-wide state, and a reused container must never carry
        # one worker's handlers into the next run.
        with Worker(run_id) as worker:
            worker.log(f"boot call_id={worker.call_id}")

            # The launcher granted this run's lease when it spawned us; this ties
            # it to our call id. The one thing a job needs from us, under the name
            # it will use it by.
            lease = worker.bind_lease()

            try:
                # Prove ownership before doing anything at all, including reading
                # the job's own config. The retries here cover the launcher's gap
                # between spawn and grant, plus ordinary network hiccups. Nothing
                # is written, so there is nothing to commit.
                lease.confirm("fence-in")
                job = JOB(lease, run_id)
                job.run()
            except Aborted as e:
                return worker.log_exit(str(e))

            # Ask the job's progress rather than trust a return value. A job is
            # just code that runs once it holds the lease; requiring it to report
            # a step count would force every future job to keep one. This is the
            # same call the launcher makes over the same committed files, so both
            # read one truth.
            done, total = JOB.progress(job.rdir, job.config)
            return worker.log_exit(f"finished at {done}/{total}")
    except Exception as e:
        raise RuntimeError(f"worker failed for {run_id}: {e!r}") from e


# What this deployment runs. One name, so the launcher and the worker cannot
# disagree: the launcher calls JOB.progress to see if a folder is finished, the
# worker builds a JOB to do the work. Swapping in a real trainer is this line
# plus the image.
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
# the ETL: the snaketl workflow, run against the Volume
# --------------------------------------------------------------------------------------


@app.function(image=etl_image, volumes={STORAGE: volume}, cpu=4, timeout=3600, max_containers=1)
def etl(targets: list[str] | None = None, flags: list[str] | None = None, unlock: bool = False) -> dict:
    """Run `snakemake` in a container, with the Volume as its data directory.

    Same targets and flags as the local command -- `etl(["main"], ["-n"])` is
    `snakemake main -n` -- because this shells out to the real snakemake rather
    than reimplementing any part of it. Two arguments are supplied here and are
    not yours to pass:

        --directory /storage   the working directory, so .snakemake/ (the metadata
                               that decides what needs rebuilding) survives the
                               container instead of dying with it
        --config volume=...    where artifacts go. The Snakefile defaults this to
                               its own folder, which here is a read-only image
                               mount, so it has to be redirected at the Volume.

    Those two are the whole reason a container can do incremental builds at all:
    with either one missing, every invocation starts from an empty tree and
    rebuilds the world.

    What to build is read from the Volume too -- /storage/data/config.yaml, which
    the Snakefile locates from that same `volume` value. It is not in the image,
    so changing which sources exist is an upload rather than a redeploy:

        modal volume put test-volume snaketl/data/config.yaml /data/config.yaml --force

    and it has to be there before the first call, or snakemake raises on the
    missing configfile.

    No lease, unlike `work`. This is a different shape of job -- a DAG that is
    idempotent by construction, where a crashed run leaves finished outputs in
    place and re-running resumes from them. What it does need is to not race
    itself, and `max_containers=1` gives that the same way it does for `launch`.
    Snakemake's own directory lock is the backstop, and it is why `unlock` exists:
    a container killed mid-run (a cancel, a timeout) leaves that lock behind on
    the Volume, and the next call refuses to start until someone clears it.
    """
    volume.reload()  # containers are reused; start from what previous runs committed

    # Targets go first, before any option. `--config` takes a variable-length list
    # of key=value pairs, so anything non-dash that follows it is swallowed as
    # another pair -- a trailing target becomes `Invalid config definition`. Ending
    # the line with --config, and putting the only positional arguments up front,
    # is what keeps that from depending on whether `flags` happens to be empty.
    argv = [
        sys.executable,
        "-m",
        "snakemake",
        *(targets or []),
        "--snakefile",
        f"{SNAKETL}/Snakefile",
        "--directory",
        STORAGE,
        "--workflow-profile",
        f"{SNAKETL}/profiles/default",
        *(["--unlock"] if unlock else []),
        *(flags or []),
        "--config",
        f"volume={STORAGE}",
    ]
    print(f"$ {' '.join(argv)}")
    result = subprocess.run(argv)

    # Commit either way. A failed workflow still finished some of its jobs, and
    # those outputs plus the metadata describing them are exactly what makes the
    # next run skip them -- discarding that is what would make failure costly.
    # (snakemake deletes the output of the job that failed, so nothing half-written
    # is being kept here.)
    volume.commit()

    if result.returncode:
        raise RuntimeError(f"snakemake exited {result.returncode}")
    return {"targets": targets or ["all"], "returncode": result.returncode}


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
def prepare(targets: str = "", flags: str = "", unlock: bool = False):
    """Run the snaketl workflow on Modal.

        modal run launcher/app.py::prepare                          # everything
        modal run launcher/app.py::prepare --targets "main split"   # named targets
        modal run launcher/app.py::prepare --flags "-n"             # dry run
        modal run launcher/app.py::prepare --unlock                 # clear a stale lock

    Targets and flags arrive as one string each and are split on whitespace,
    because a `modal run` entrypoint only takes scalars. That rules out any
    argument containing a space -- none of ours do, and the alternative is
    quoting rules nobody wants to learn for a wrapper this thin.

    Unlike `main`, this calls `etl.remote` on the local app rather than reaching
    for the deployed one. `main` needs `Function.from_name` because `launch`'s
    single-container lock only means anything within one deployment; the workflow
    has no such lock to protect, so running it from here is the same job.
    """
    print(etl.remote(targets=targets.split(), flags=flags.split(), unlock=unlock))
