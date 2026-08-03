import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, UTC
from pathlib import Path

logger = logging.getLogger(__name__)
RUNS = Path("/storage/runs")  # volume mount path, as seen inside the containers

class Aborted(Exception):
    """Stop this attempt, but it is not a crash -- leaving is the right answer.

    `_attempt` catches this and logs a clean exit. Anything else that escapes is
    a bug and gets reported as one. Jobs raise this when handed a folder they
    cannot run.
    """

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
# the contract every job satisfies
# --------------------------------------------------------------------------------------


class Job(ABC):
    """The contract every job satisfies -- and all the launcher and worker rely on.

    A job is two things over one folder: code that runs once the lease is held
    (`run`), and a way to read how far the folder got from committed files alone
    (`progress`). `make_config` writes the definition both will later run under.

    The method kinds carry the split. `run` is an instance method -- it needs the
    lease. `progress` and `make_config` are classmethods, because the launcher
    calls them with no attempt and no lease in hand: to decide a folder is finished
    before spawning anyone, and to mint a config at the CLI.

    Subclassing is enforcement, not decoration: a job missing `run` cannot be
    instantiated, so `JOB = SomeJob` fails at deploy, not mid-attempt.

    The durability rule every job lives under. A commit is the only irreversible
    thing a worker does, and it publishes the whole container, not one path -- so
    do not write a shared file to disk until the lease has already said yes. Modal
    also auto-commits every mounted volume when a container exits (a finally block
    in modal/_runtime/user_code_imports.py, outside all lifecycle handlers), so
    anything dirty on local disk is published even if you never call commit, and
    even if you were superseded. The safe pattern is to hold un-owned writes in
    memory and touch shared files only inside `with self.lease(..., commit=True)`;
    see `Count` for one way to do it.
    """

    def __init__(self, lease, run_id: str):
        # A lease and a folder is everything a job gets. No Attempt, so no route to
        # identity, logging, or the Dict; the one useful fact -- which call is
        # writing -- rides along on the lease.
        #
        # The config read is shared because the failure is shared: a folder whose
        # config.json is gone is one no job can run, and leaving it alone is
        # Aborted, not a crash. The job reads its own config because it is the only
        # thing that can say whether a given config.json is one it understands --
        # which keys it needs, and how to validate them, is `configure`.
        self.lease = lease
        self.run_id = run_id
        self.rdir = run_dir(run_id)
        config = read_config(run_id)
        if config is None:
            raise Aborted("config.json missing or unparseable")
        self.config = config
        self.configure(config)

    def configure(self, config: dict) -> None:
        """Pull out and validate the keys this job needs, and set up per-job state.

        Default: nothing -- a job with no required config need not override. Raise
        Aborted on a missing key so a folder this job cannot run is left alone
        rather than crashed on.
        """

    @abstractmethod
    def run(self) -> None:
        """Carry this folder forward while the lease holds.

        Takes and returns nothing, which is what makes an attempt safe to repeat:
        where to start comes from the folder (`progress`), not an argument, and
        what got done comes from the folder afterwards, not a return value. So a
        job can be any code at all, rather than something obliged to count steps.
        """

    @classmethod
    @abstractmethod
    def progress(cls, rdir: Path, config: dict) -> tuple[int, int]:
        """(done, total) for this folder, read from committed files only.

        Raise Aborted if `config` is not one this job understands. Callers scan
        whole directories of folders, some written by other tools entirely, so
        "not mine" has to be an answer rather than a crash.
        """

    @classmethod
    @abstractmethod
    def make_config(cls, **kwargs) -> dict:
        """Build the config.json this job runs under.

        Written once at creation; every later resume runs under it, so the keys
        `run` and `progress` read can never drift from the keys a launch writes --
        which is why minting the config lives here, beside the code that reads it.
        """


# --------------------------------------------------------------------------------------
# Count: the reference job, a toy stand-in for training
# --------------------------------------------------------------------------------------


class Count(Job):
    """Counts to total_steps, saving a checkpoint every save_every.

    The reference job, shaped like training on purpose so the seam is proven
    rather than assumed: step-numbered checkpoints named the way transformer.util
    names them, an append-only ledger, and progress read from committed files.
    """

    def configure(self, config: dict) -> None:
        # The keys this job runs under, or Aborted: a folder missing them is one
        # Count cannot run, so leave it alone rather than crash on it.
        try:
            self.total_steps = config["total_steps"]
            self.save_every = config["save_every"]
        except KeyError as e:
            raise Aborted(f"config.json is missing {e}") from None

        # Ledger rows wait here instead of going straight to train.jsonl. Per the
        # durability rule in Job: rows written to local disk are published on exit
        # whether or not we still hold the lease, so an interval we were denied
        # would survive. Held in memory, a denied interval dies with the container
        # -- exactly what should happen to it. `run` flushes these only inside the
        # gated boundary, so a row reaches train.jsonl only once its checkpoint has.
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

