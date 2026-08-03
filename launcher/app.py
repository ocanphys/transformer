"""Run long jobs on Modal so that crashes, cancels, and restarts are all safe.

A job lives in one folder on a Volume. Restarting it picks up from the last
checkpoint; asking twice does nothing the second time; and two workers can never
write the same folder. That last guarantee is enforced, not assumed.

Three systems, each used only for what it is actually good at:

    Dict   = who may write, now    (the lease -- always fresh, atomic per key)
    Modal  = who is alive, now     (call state -- we ask, never write)
    Volume = what happened         (config, checkpoints, ledger, logs)

Three parts, and only the last is specific to what you are running:

    launch    start or resume a run. Safe to call repeatedly.
    work      one attempt: prove ownership, run the job, record why it stopped.
    JOB       the class this deployment runs; the toy counter here stands in
              for training.

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
"""

import json
import logging
import sys
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

LEASE_RETRIES = 5  # how many times an indeterminate lease read is worth re-asking
LEASE_BACKOFF = 5.0  # seconds x try number -> 5, 10, 15, 20: ~50s of patience in total

# What lands in an attempt's log. Root sits at LOG_LEVEL so anything anyone logs
# is captured by default; LOG_LEVELS pins the chatty libraries down so their
# noise cannot bury the run's own story. Add a name here when something new gets
# loud, or set one to DEBUG when you need to see inside it.
LOG_LEVEL = logging.INFO
LOG_LEVELS = {
    "modal": logging.WARNING,
    "modal-client": logging.WARNING,
    "grpclib": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "urllib3": logging.WARNING,
    "asyncio": logging.WARNING,
    "filelock": logging.WARNING,
}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
leases = modal.Dict.from_name(DICT_NAME, create_if_missing=True)

# The toy worker needs nothing but the stdlib. A real trainer swaps this for the
# uv_sync image from modal_lab/run.ipynb -- only this line changes.
image = modal.Image.debian_slim(python_version="3.12")

# The dashboard needs a web server, and its own module shipped alongside this one.
web_image = image.pip_install("fastapi[standard]").add_local_python_source("dashboard")

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# logging: everything an attempt does, in the attempt's own file
# --------------------------------------------------------------------------------------


class AttemptFormatter(logging.Formatter):
    """`<iso-timestamp> <message>`, in UTC.

    The first column is a contract, not a style choice: the dashboard merges every
    attempt's log for a run by sorting on it, and that only works because the
    timestamps are one fixed-width UTC format. `converter = time.gmtime` is what
    keeps them UTC even on a machine that thinks otherwise.

    Our own lines stay bare. Anything from another library gets its level and
    logger name in front, so foreign noise is identifiable at a glance without
    cluttering the run's own story.
    """

    converter = time.gmtime

    def __init__(self):
        super().__init__(fmt="%(asctime)s.%(msecs)03dZ %(prefix)s%(message)s", datefmt="%Y-%m-%dT%H:%M:%S")

    def format(self, record):
        record.prefix = "" if record.name == __name__ else f"{record.levelname} {record.name}: "
        return super().format(record)


def _remove_attempt_handlers(root: logging.Logger) -> None:
    """Drop and close any handler an earlier attempt left behind. Identified by a
    marker attribute rather than by identity, so this works even when the attempt
    that installed them is long gone -- which is exactly the case that matters,
    since a leftover handler points at a previous run's file."""
    for handler in list(root.handlers):
        if getattr(handler, "_attempt_handler", False):
            root.removeHandler(handler)
            handler.close()


class StdoutToLog:
    """Sends `print()` through the logger instead of straight to the console.

    Line-buffered, because print writes its text and its newline separately and a
    half-line is not a log record. Nothing is written to the real stream here --
    the logger's own console handler does that, so each line still reaches Modal's
    logs exactly once, and there is no way to loop back into ourselves.

    Only stdout is taken. tqdm and friends write to stderr, and their carriage-
    return redraws would turn a training log into tens of thousands of fragments.
    """

    def __init__(self, log):
        self.log = log
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        while "\n" in self.buffer:
            line, _, self.buffer = self.buffer.partition("\n")
            if line.strip():
                self.log(line)
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False


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


class Aborted(Exception):
    """Stop this attempt, but it is not a crash -- leaving is the right answer.

    `_attempt` catches this and logs a clean exit. Anything else that escapes is
    a bug and gets reported as one. Jobs raise this when handed a folder they
    cannot run.
    """


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


def run_dir(run_id: str) -> Path:
    return RUNS / run_id


def read_config(run_id: str) -> dict | None:
    """The run's immutable definition, or None if it is missing/unparseable --
    which is the launcher's cue to quarantine the folder rather than guess."""
    try:
        return json.loads((run_dir(run_id) / "config.json").read_text())
    except (OSError, ValueError):
        return None


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
    config = read_config(run_id)
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
    # holds the dead attempt's id, and a warm container can boot fast enough to
    # read it -- an occupied key means "someone else owns this", so the new worker
    # would kill itself over a lease that is milliseconds from being its own.
    # Clearing first makes that gap an empty key, which just means "wait".

    await leases.pop.aio(lease_key(run_id), None)
    attempt = (grant["attempt"] + 1) if grant else 1
    call = await work.spawn.aio(run_id)
    await leases.put.aio(lease_key(run_id), {"call_id": call.object_id, "granted_ts": time.time(), "attempt": attempt})

    return {
        "run_id": run_id,
        "reason": f"spawned attempt {attempt} from {done}/{total}",
        "call_id": call.object_id,
    }


# --------------------------------------------------------------------------------------
# the worker: computes freely, commits only with a confirmed lease
# --------------------------------------------------------------------------------------


class Attempt:
    """One attempt at one run: who this container is and where it logs.

    Jobs never get one of these. They get `make_lease()` -- permission, without
    the identity behind it -- because the only thing a job needs from an attempt
    is the right to make its writes durable.
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
        # and that commit hasn't happened yet when we reload. See make_lease.
        volume.reload()

        self.run_id = run_id
        self.call_id = modal.current_function_call_id()
        self.rdir = run_dir(run_id)
        self.logdir = self.rdir / "logs" / self.call_id
        self.logdir.mkdir(parents=True, exist_ok=True)
        self._capture_logging()

    def _capture_logging(self) -> None:
        """Point everything this container logs at this attempt's own file.

        Handlers go on the *root* logger, so any code anywhere -- ours, a job's, a
        library's -- is captured without being asked to cooperate. A job holds only
        a lease and has no route back to us, so this is what lets it log at all:
        `logging.getLogger(__name__)` from anywhere lands in the right file.

        Two handlers, on purpose: the file is the durable record that ships with
        the run folder, the console keeps `modal app logs` working. print() is
        routed through the logger rather than the console handler's stream, so it
        appears in both, once each, and cannot loop back into itself.

        Old handlers of ours are torn off first. Containers are reused, and a
        leftover handler still points at the *previous* run's file -- which would
        quietly write this attempt's lines into another run's folder.

        Must run after volume.reload(): reloading fails while a file on the volume
        is open, and this opens one.
        """
        root = logging.getLogger()
        _remove_attempt_handlers(root)

        self._file_handler = logging.FileHandler(self.logdir / "attempt.log")
        self._console_handler = logging.StreamHandler(sys.stdout)
        for handler in (self._file_handler, self._console_handler):
            handler.setFormatter(AttemptFormatter())
            handler._attempt_handler = True  # so a later attempt can find and drop it
            root.addHandler(handler)

        root.setLevel(LOG_LEVEL)
        for name, level in LOG_LEVELS.items():
            logging.getLogger(name).setLevel(level)
        logging.captureWarnings(True)  # warnings.warn -> the log, not the void

        self._stdout = sys.stdout
        sys.stdout = StdoutToLog(self.log)

    def _release_logging(self) -> None:
        """Undo it, in the reverse order. Restoring stdout first means a late
        print during teardown still has somewhere real to go."""
        sys.stdout = self._stdout
        _remove_attempt_handlers(logging.getLogger())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        """The other way out: something threw, so log_exit never ran.

        The traceback is written here, while the file handler is still attached --
        otherwise the one event most worth having a record of is the one event
        that never reaches the record. Worded as an EXIT so every attempt ends
        with exactly one, however it ended.

        Nothing is committed on this path. Modal commits mounted volumes when a
        container stops, so the line lands anyway, and choosing to commit here
        would publish whatever else the job left dirty mid-failure.
        """
        if exc_type is not None:
            logger.error("EXIT crashed: %r", exc, exc_info=(exc_type, exc, tb))
        self._release_logging()
        return False  # never swallow: work() still needs to see it and re-raise

    def log(self, msg: str) -> None:
        """Write one line to this attempt's log.

        Sugar over the module logger, so protocol lines and everything else land
        in the same file through the same path. The folder is named by a call id
        that is unique forever, so attempts can never overwrite each other's logs,
        and even a superseded one can safely flush its last words.
        """
        logger.info(msg)

    def log_exit(self, reason: str) -> dict:
        """Say why we are leaving, publish it, and hand the caller its return
        value. The caller does the actual leaving.

        There are two ways out of an attempt and this is the tidy one, reached
        when we decided to stop: finished, denied, or handed a folder we cannot
        run. The other is __exit__, for when something threw.

        The commit is leaseless, which is safe only because attempt.log is the one
        dirty file -- commits publish the whole container, not one path. A job
        that writes shared files continuously, instead of buffering to a
        checkpoint the way `Count` does, has to deal with that; see `pending`.
        """
        self.log(f"EXIT {reason}")
        volume.commit()
        return {"run_id": self.run_id, "reason": reason}

    def make_lease(self, tries: int = LEASE_RETRIES, backoff: float = LEASE_BACKOFF):
        """Hand out permission to write, as two ways of using one check:

            lease.confirm("fence-in")     # just prove we own the run
            with lease(commit=True):      # prove it, then make the block durable

        `confirm` exists so proving ownership doesn't have to pretend to be a
        block. A `with` whose exit does nothing is just a function call with extra
        indentation, and it reads like a second gate when only one ever commits.

        A closure, not a method, so it can travel alone: a job holds `lease` and
        nothing else of ours. `lease.call_id` comes along because a job keeping a
        ledger needs to stamp rows with who wrote them, and that is the only thing
        about us it needs to know.

        Retry settings are fixed here and captured, so callers ask for permission
        rather than for a number of tries. One setting covers both uses; if
        fence-in ever wants its own, call make_lease again with other arguments.
        """

        def confirm(label: str = "lease") -> None:
            """Prove we still own the run, or raise LeaseLost.

            Someone else's id raises immediately -- that is a real answer, and
            asking again is just hoping it changes. Only silence is worth waiting
            out, and only briefly. Waiting is safe even if a takeover is genuinely
            underway: the launcher checks Modal before taking a run, and a worker
            that has merely lost the Dict still looks alive there, so nobody can
            take our lease while we wait.

            Every check is logged, including the ones that pass. A boundary that
            committed and a boundary that was allowed to commit are different
            facts, and after the event only the log can tell them apart -- an
            ownership check that quietly succeeded leaves nothing else behind.
            """
            for i in range(1, tries + 1):
                verdict, grant = fence(self.run_id, self.call_id)
                if verdict == MATCH:
                    self.log(f"{label}: lease held (attempt {grant['attempt']}, try {i}/{tries})")
                    return
                if verdict == MISMATCH:
                    raise LeaseLost(f"{label}: another attempt holds this run ({grant['call_id']})")
                if i == tries:
                    raise LeaseLost(f"{label}: indeterminate after {tries} tries -- ownership never confirmed")
                self.log(f"{label}: indeterminate, retry {i}/{tries}")
                time.sleep(backoff * i)  # linear; no jitter needed at this fleet size

        @contextmanager
        def lease(label: str = "lease", commit: bool = False):
            """Prove we own the run, run the block, then commit if asked.

                with lease(f"step {step}", commit=True):
                    write_checkpoint(...)

            The commit is the only irreversible thing a worker does. Everything
            written before it exists only in this container and disappears with
            it, so being denied costs work, never history. If the block raises,
            the commit is skipped -- a half-written checkpoint is not a
            checkpoint.

            This does not mean "only one writer at any instant". The check and the
            commit talk to two different services and cannot be made atomic, so
            the lease can move while the block runs, and any worker can land a
            commit it was authorized for a moment ago. The real guarantee is
            weaker but enough: nobody commits without having proved ownership
            since their last commit.

            That gap is as wide as the block is slow, which ties it to checkpoint
            size -- the toy writes a few bytes, but torch.save of a large model
            plus the upload holds it open for seconds. Widening it corrupts
            nothing: checkpoints are written once per step number, so a redone
            interval overwrites itself from the same starting point, and the
            ledger tolerates repeated steps. The cost is one interval done twice.
            If that ever matters, shorten the block -- write the file outside the
            gate and let the gated part do only the rename and commit.
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
    """One attempt at one run.

    Everything is wrapped because of one ambiguity: Modal signals "still running"
    by raising a builtin TimeoutError, and it also re-raises whatever the worker
    raised. So a worker that dies on a socket or dataloader timeout -- both are
    builtin TimeoutError since Python 3.10 -- would read as healthy forever, and
    the launcher would refuse to resume it. Re-raising as RuntimeError guarantees
    every failure reads as a failure.
    """
    try:
        return _attempt(run_id)
    except Exception as e:
        raise RuntimeError(f"attempt failed for {run_id}: {e!r}") from e


def _attempt(run_id: str) -> dict:
    """Prove ownership, hand the lease to the job, record why we stopped.

    Nothing here knows what the job does -- not its config, not its layout, not
    what "done" means to it. And the job never learns a lease exists: it just
    enters a context manager when it wants something to survive, and a denied
    gate raises straight through its own loop.
    """
    # A context manager because of what it installs, not what it holds: log capture
    # is process-wide state, and a reused container must never carry one attempt's
    # handlers into the next run.
    with Attempt(run_id) as attempt:
        attempt.log(f"boot call_id={attempt.call_id}")

        # The one thing a job needs from us, under the name it will use it by.
        lease = attempt.make_lease()

        try:
            # Prove ownership before doing anything at all, including reading the
            # job's own config. The retries here cover the launcher's gap between
            # spawn and grant, plus ordinary network hiccups. Nothing is written, so
            # there is nothing to commit.
            lease.confirm("fence-in")
            job = JOB(lease, run_id)
            job.run()
        except Aborted as e:
            return attempt.log_exit(str(e))

        # Ask the job's progress rather than trust a return value. A job is just
        # code that runs once it holds the lease; requiring it to report a step
        # count would force every future job to keep one. This is the same call the
        # launcher makes over the same committed files, so both read one truth.
        done, total = JOB.progress(job.rdir, job.config)
        return attempt.log_exit(f"finished at {done}/{total}")


# --------------------------------------------------------------------------------------
# the job -- the toy stand-in for training, and the shape any other job follows
# --------------------------------------------------------------------------------------


class Count:
    """Counts to total_steps, saving a checkpoint every save_every.

    A job is a class over one run folder, and it is only two things: code that
    runs once the lease is held, and a way to report how far the folder got.

    That split shows up in the method types. `run` is an instance method because
    it needs the lease. It takes nothing and returns nothing, which is what makes
    it safe to call twice: where to start comes from the folder, and what got done
    comes from the folder afterwards. So a job can be any code at all, rather than
    something obliged to count steps. `make_config` and `progress` are
    classmethods because the launcher calls them too, with no attempt and no lease
    in hand.

    Shaped like training on purpose, so the seam is proven rather than assumed:
    step-numbered checkpoints named the way transformer.util names them, an
    append-only ledger, and progress read from committed files.
    """

    def __init__(self, lease, run_id: str):
        # A lease and a folder is everything a job gets. No Attempt, so no route
        # to identity, logging, or the Dict; the one useful fact, which call is
        # writing, rides along on the lease.
        #
        # The job reads its own config, since it is the only thing that can say
        # whether a given config.json is one it can run. Both failures below are
        # Aborted rather than crashes: a folder this job cannot run is one to
        # leave alone.
        self.lease = lease
        self.run_id = run_id
        self.rdir = run_dir(run_id)
        config = read_config(run_id)
        if config is None:
            raise Aborted("config.json missing or unparseable")
        try:
            self.total_steps = config["total_steps"]
            self.save_every = config["save_every"]
        except KeyError as e:
            raise Aborted(f"config.json is missing {e}") from None
        self.config = config

        # Ledger rows wait in memory instead of going straight to train.jsonl.
        # This looks like a pointless buffer and is the opposite: it is what makes
        # the whole guarantee true.
        #
        # Two things publish files we did not choose to publish. Our own exit
        # commit covers the container, not one path. And Modal commits every
        # mounted volume automatically when a container exits -- see
        # modal/_runtime/user_code_imports.py, a finally block outside all
        # lifecycle handlers -- so anything dirty on local disk is published even
        # if we never call commit at all, and even if we were superseded.
        #
        # So "no lease, no durable write" cannot be enforced by gating commits.
        # It is enforced by not writing shared files to disk until the gate has
        # already said yes. Rows held in memory die with the container, which is
        # exactly what should happen to an interval we were denied.
        #
        # This is the thing to carry over when wiring a real trainer: run_training
        # appends to train.jsonl every step, so those rows would survive on exit.
        # Either buffer them the way this does, or truncate the file back to its
        # last committed size before leaving.
        self.pending: list[dict] = []

    @classmethod
    def make_config(cls, total_steps: int, save_every: int, step_seconds: float = 1.0, seed: int = 0) -> dict:
        """Build the config this job needs, in one place.

        Kept here so the keys `run` reads and the keys a launch writes cannot
        drift apart -- which matters because config.json is written once and every
        later resume runs under it.
        """
        return {"total_steps": total_steps, "save_every": save_every, "step_seconds": step_seconds, "seed": seed}

    @classmethod
    def progress(cls, rdir: Path, config: dict) -> tuple[int, int]:
        """How far this folder got, read from the checkpoint files.

        Checkpoints are the only thing a commit makes durable -- ledger rows
        written between them live in a container that may never come back. The
        naming matches transformer.util.checkpoint_path, so a real trainer's
        checkpoints are already readable here.

        Raises Aborted if the config is not one this job understands. Callers scan
        whole directories of folders, some of which were written by other tools
        entirely, and "not mine" has to be an answer rather than a crash.
        """
        if "total_steps" not in config:
            raise Aborted("config.json is not for this job (no total_steps)")
        files = sorted((rdir / "checkpoints").glob("step_*.obj"))
        done = int(files[-1].stem.split("_")[1]) if files else 0
        return done, config["total_steps"]

    def run(self) -> None:
        """Carry this folder to total_steps, committing every save_every.

        Takes nothing and returns nothing, which is what makes it safe to call
        again: the starting point comes from the folder, not from an argument, and
        what got done comes from `progress` afterwards, not from a return value.
        Same reasoning as the launcher -- no caller is trusted to say where the
        work is up to.
        """
        step, _ = self.progress(self.rdir, self.config)
        logger.info("counting from %d to %d", step, self.total_steps)

        while step < self.total_steps:
            step += 1
            time.sleep(self.config.get("step_seconds", 1.0))  # a real trainer's forward/backward
            self.pending.append({"step": step, "ts": utc(), "call_id": self.lease.call_id})

            if step % self.save_every and step != self.total_steps:
                continue

            # Everything durable about this interval lands together, checkpoint
            # first, so a committed row never names a file that isn't there.
            with self.lease(f"boundary at step {step}/{self.total_steps}", commit=True):
                self.write_checkpoint(self.rdir, step, self.config)
                with open(self.rdir / "train.jsonl", "a") as f:
                    f.writelines(json.dumps(row) + "\n" for row in self.pending)
                self.pending.clear()

    @staticmethod
    def write_checkpoint(rdir: Path, step: int, config: dict) -> None:
        """Write a checkpoint, via a temp file and a rename.

        Rename is atomic on the local filesystem, so the final name only ever
        exists complete and no reader can catch a half-written file. Dropping
        torch.save in here is the whole change for real training -- and note that
        transformer.util.save_checkpoint writes straight to the final path, which
        would need this same treatment.
        """
        out = rdir / "checkpoints" / f"step_{step:010d}.obj"
        out.parent.mkdir(parents=True, exist_ok=True)  # the job owns its own layout
        tmp = out.with_suffix(".obj.tmp")
        tmp.write_text(json.dumps({"step": step, "seed": config.get("seed")}))
        tmp.rename(out)


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
        # creation, and every attempt since runs under it.
        print(launcher.remote(run_id=run_id))
    else:
        config = JOB.make_config(total_steps=total_steps, save_every=save_every, step_seconds=step_seconds)
        print(launcher.remote(name=name, config=config))
