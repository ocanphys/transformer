"""The jobs the worker runs, and the folder layout they share.

WHAT A JOB IS. Code that carries one run folder forward while a lease is held.
`app.py` names one of these classes as `JOB` and needs exactly four things of it,
which is the whole contract -- no base class states it, because two classes
implementing four members do not need a third class to agree:

    JOB(lease, run_id)              construct: read config.json, resume or start
    job.run()                       carry the folder forward; takes and returns nothing
    JOB.progress(rdir, config)      (done, total) from committed files alone
    JOB.make_config(**kwargs)       mint the config.json a launch writes

`run` is an instance method because it needs the lease. `progress` and
`make_config` are classmethods because the launcher calls them with no attempt
and no lease in hand: to decide a folder is finished before spawning anyone, and
to mint a config at the CLI. `progress` raises Aborted rather than crashing on a
config it does not understand -- callers scan whole directories of folders, some
written by other tools entirely, so "not mine" has to be an answer.

`run` returns nothing on purpose, and that is what makes an attempt safe to
repeat: where to start comes from the folder, and what got done comes from
`progress` reading the folder afterwards. No caller is trusted to say where the
work is up to.

THE DURABILITY RULE EVERY JOB LIVES UNDER. A commit is the only irreversible
thing a worker does, and it publishes the whole container, not one path -- so do
not write a shared file to disk until the lease has already said yes. Modal also
auto-commits every mounted volume when a container exits (a finally block in
modal/_runtime/user_code_imports.py, outside all lifecycle handlers), so anything
dirty on local disk is published even if you never call commit, and even if you
were superseded. The safe pattern is to hold un-owned writes in memory -- see
`pending` in either job -- and touch shared files only inside
`with self.lease(..., commit=True)`.

Both jobs therefore have the same three-line shape at a boundary, written out at
each site rather than shared through a base class, because seeing the ordering is
the point:

    with self.lease(label, commit=True):
        <write the checkpoint>          # first, so a committed row never
        <append the buffered rows>      # names a file that isn't there

STDLIB AT IMPORT. `app.py` and `dashboard.py` import this module, and their
containers have no torch, numpy, or wandb. Every heavy import is therefore lazy,
inside the method that needs it.
"""

import json
import logging
import time
from datetime import datetime, UTC
from pathlib import Path

logger = logging.getLogger(__name__)

# The volume mount, as seen inside the containers, and the three trees on it. A
# config's train_path/valid_path are relative to VOLUME, not to a run folder --
# the same convention transformer.util.run_training reads them under, so one
# config.json resolves identically whether it runs here or there:
#     VOLUME/data/{dataset}/bin/{tokenizer_uid}/{split}.bin   input, from the ETL
#     VOLUME/tokenizers/{tokenizer_uid}/config.json           input, from the ETL
#     VOLUME/runs/{run_id}/                                   output, this job's
VOLUME = Path("/storage")
RUNS = VOLUME / "runs"


class Aborted(Exception):
    """Stop this attempt, but it is not a crash -- leaving is the right answer.

    `_attempt` catches this and logs a clean exit. Anything else that escapes is
    a bug and gets reported as one. Jobs raise this when handed a folder they
    cannot run.
    """


def run_dir(run_id: str) -> Path:
    return RUNS / run_id


def load_config(run_id: str) -> dict | None:
    """load the config.json and convert it to a dict"""
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


def checkpoint_path(rdir: Path, step: int) -> Path:
    """Where step N's checkpoint lives. Same name transformer.util gives it, so a
    run's checkpoints are readable by either.

    A function of (rdir, step) rather than a method, because `progress` needs it
    with no instance in hand -- the launcher asks how far a folder got without
    constructing anything.
    """
    return rdir / "checkpoints" / f"step_{step:010d}.obj"


def latest_checkpoint_path(rdir: Path) -> Path | None:
    """The newest checkpoint in `rdir`, or None for a folder that has none.

    Zero-padded step numbers make the lexical sort the numeric one.
    """
    checkpoints = sorted((rdir / "checkpoints").glob("step_*.obj"))
    return checkpoints[-1] if checkpoints else None


def committed_step(rdir: Path) -> int:
    """The step of the newest checkpoint, or 0 -- how far this folder actually
    got, from committed files alone.

    Both jobs' `progress` is this plus their own idea of the total, which is the
    only part that differs between them.
    """
    latest = latest_checkpoint_path(rdir)
    return int(latest.stem.split("_")[1]) if latest else 0


# --------------------------------------------------------------------------------------
# Train: the real one
# --------------------------------------------------------------------------------------


class ETL:
    """downloads text and BPE tokenizes.
     
       """
    def __init__(self, lease, run_id: str):
        """"""
        pass

    def download(self):
        pass


class Train:
    """Trains a TransformerLM to training.total_iterations, checkpointing every
    training.save_every.   
    """

    def __init__(self, lease, run_id: str):
        # A lease and a unique folder name {run_id} is everything a job gets.
        self.lease = lease
        self.run_id = run_id
        self.rdir = run_dir(run_id)

        self.config_dict = load_config(run_id)
        if self.config_dict is None:
            raise Aborted("config.json missing or unparseable")

        # Ledger rows wait here instead of going straight to train.jsonl, keyed by
        # the step they describe. Per the durability rule at the top of this file:
        # rows written to local disk are published on container/worker exit whether or
        # not we still hold the lease, so an interval we were denied would
        # survive. Held in memory, a denied interval dies with the container --
        # exactly what should happen to it.
        self.pending: dict[int, dict] = {}

        # Where `write_progress` puts the live step. Under logs/<call_id>/, this
        # attempt's private namespace, so it needs no lease and cannot collide
        # with another attempt's. `Attempt` has already made the folder; the
        # mkdir is for anyone constructing a job outside a worker.
        self.progress_file = self.rdir / "logs" / self.lease.call_id / "progress.json"
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)

        self.configure(self.config_dict)

    def write_progress(self) -> None:
        """Overwrite this attempt's progress file with where the loop is now.

        The last step only, not a history -- train.jsonl is the history, and this
        answers "is it moving?" for a human watching the dashboard.

        Free of the two things that make the loop slow: `self.step` is a Python
        int, so nothing reads the device, and there is no commit here. Modal
        mounts every volume with allow_background_commits, so it publishes dirty
        files on its own schedule -- which is exactly the guarantee this wants and
        would be wrong to rely on for anything durable.
        """
        self.progress_file.write_text(json.dumps({"step": self.step, "total": self.total_steps, "ts": utc()}))

    def record_step(self, step: int, **values) -> None:
        """Buffer what we learned at `step`, to be flushed at the next checkpoint.
        values might be tensors living on the GPU - so they may not be known at the
        time when we "record" them into self.pending - 

        we dont' call `.item()` on these every step since this would sync the device 
        every step. They are turned into numbers once, in `flush_pending` during a
        commit with a lease.
        """
        if step not in self.pending:
            self.pending[step] = {"ts": utc(), "call_id": self.lease.call_id}
        self.pending[step].update(values)

    def save_checkpoint(self, keep_optimizer_history: bool = False) -> None:
        """Write the checkpoint file for `step`.

        Step 0 is the initial checkpoint, right after model/optimizer
        construction and before any training has happened; step N is the state
        after the Nth training step has fully completed.

        attempt: which continuous execution produced this checkpoint -- 1 for a
        fresh run, incremented by 1 every time `configure` picks up from an
        existing checkpoint instead of starting at step 0. Persisted so a run's
        history can be split back into its distinct attempts later.

        - FOR NOW WE ARE SAVING THE OPTIMIZER ON EACH CHECKPOINT.
        #TODO Adam's optimizer state roughly doubles checkpoint size and is only ever
        needed from the latest checkpoint to resume training. So by default, once
        the new checkpoint is written, the previous latest checkpoint's optimizer
        state is stripped (its model weights are kept). Pass keep_optimizer_history=True
        to preserve full optimizer state in every checkpoint.
        """

        # prev = latest_checkpoint_path(self.rdir)
        out = checkpoint_path(self.rdir, self.step)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "step": self.step,
                "attempt": self.attempt,
                "seed": self.seed,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            out,
        )
        ## TODO: deal with optimizer state

    def flush_pending(self) -> dict[int, dict]:
        """Append self.pending to train.jsonl, then clear it.

        `.item()` blocks until the GPU finishes everything queued ahead of the
        value, so it happens here rather than in the loop -- once per boundary,
        where we are about to block on a checkpoint write and a commit anyway.
        Scalars only, since that is what `.item()` takes; non-tensors pass
        through.

        Called inside the lease and after the checkpoint is written, so a
        committed row never names a step whose checkpoint is missing. Rows come
        back for `train` to hand to W&B without reading them again.
        """
        if not self.pending:
            return {}

        # The first .item() is the one real sync
        for row in self.pending.values():
            for key, value in row.items():
                if isinstance(value, self.torch.Tensor):
                    row[key] = value.item()

        rows = self.pending
        with open(self.rdir / "train.jsonl", "a") as f:
            for step in sorted(rows):
                f.write(json.dumps({"step": step, **rows[step]}) + "\n")
        self.pending = {}  # a fresh buffer; `rows` keeps the old one for the caller
        return rows

    def validate_data(self, config: dict) -> None:
        """Fail before training on a config whose data bins are missing, or whose
        vocab_size disagrees with the tokenizer that produced them. Lifted from
        transformer.util.run_training.

        These paths hang off VOLUME, not off the run folder: the bins and the
        tokenizer are shared ETL output that every run reads, so a config records
        them relative to the volume root and resolves the same way here as in
        run_training. Raises Aborted rather than the bare
        FileNotFoundError/ValueError run_training used: a run that cannot proceed
        is one to leave alone, not crash-loop.
        """
        train_path = VOLUME / config["training"]["train_path"]
        valid_path = VOLUME / config["training"]["valid_path"]
        if not train_path.exists():
            raise Aborted(f"train_path not found: {train_path}")
        if not valid_path.exists():
            raise Aborted(f"valid_path not found: {valid_path}")

        # tokenizer-specific: tokenizer_uid is train_path's parent dir name
        # (data/{dataset_name}/bin/{tokenizer_uid}/{split}.bin) -- the single source
        # of truth for which tokenizer produced these bins.
        tokenizer_uid = train_path.parent.name
        tokenizer_config = json.loads((VOLUME / "tokenizers" / tokenizer_uid / "config.json").read_text())
        if tokenizer_config["vocab_size"] != config["model_params"]["vocab_size"]:
            raise Aborted(
                f"model_params.vocab_size ({config['model_params']['vocab_size']}) does not match "
                f"tokenizer {tokenizer_uid!r}'s vocab_size ({tokenizer_config['vocab_size']})"
            )

    def get_batch(self, data, batch_seed: tuple):
        """point to the memmap ndarray"""
        """batch-seed is a tuple like (self.seed, self.step, ...) """
        np = self.np  # numpy from conf
        high = len(data) - self.context_length
        rng = np.random.default_rng(batch_seed)
        starts = rng.integers(0, high, size=self.batch_size)
        # cast on the numpy side - Embedding needs long
        inputs = np.stack([data[i : i + self.context_length] for i in starts]).astype(np.int64)
        targets = np.stack([data[i + 1 : i + self.context_length + 1] for i in starts]).astype(np.int64)
        batch = {
            "inputs": self.torch.from_numpy(inputs).to(self.device),
            "labels": self.torch.from_numpy(targets).to(self.device),
        }
        return batch

    def forward_pass(self, batch: dict) -> tuple:
        """One forward pass over `batch`: returns (loss, metrics).

        Split out from the step loop so the graph-attached loss and the numbers we
        report come from a single place -- `run` backprops the first and hands the
        second to the ledger without ever recomputing either. Metrics are plain
        floats, not tensors: anything held past `loss.backward()` would otherwise
        pin the whole autograd graph for that step.

        Untyped past `dict`/`tuple` on purpose: the tensor types are torch's, and
        torch is imported lazily in `configure` so this module stays stdlib at
        import. Concretely --
            batch:   a `get_batch` result, "inputs" and "labels", both
                     (batch_size, context_length) int64 on self.device
            returns: (torch.Tensor scalar loss, dict[str, float])
        """
        loss_function = self.torch.nn.CrossEntropyLoss()
        logits = self.model(batch["inputs"])
        # CrossEntropyLoss wants (N, C, ...) -- classes on dim 1, not last
        loss = loss_function(logits.transpose(1, 2), batch["labels"])
        # we keep these on GPU and not call .item() because this synchronizes GPU and CPU
        with (
            self.torch.no_grad()
        ):  # any new tensor operations in that context manager will not have the computational graph attached to it.
            metrics = {
                "loss": loss.detach(),  # we need a tensor operation to clear the loss to save it as a metric.
                "perplexity": loss.exp(),
            }
        return loss, metrics

    def evaluate(self, number_of_eval_batches=20):
        self.model.eval()
        with self.torch.inference_mode():
            val_losses = []
            for k in range(number_of_eval_batches):
                val_loss, _ = self.forward_pass(self.get_batch(self.valid_data, (self.seed, self.step, k)))
                val_losses.append(val_loss)
            mean_val_loss = self.torch.stack(val_losses).mean()
        self.model.train()
        metrics = {"val_loss": mean_val_loss}
        return metrics

    def configure(self, config: dict) -> None:
        # Data must exist and match the tokenizer before we build anything heavy.
        self.validate_data(config)

        # Heavy deps, lazy so jobs.py stays stdlib at import (launch/dashboard have
        # no torch). self.torch/self.np are used by the other methods too.
        import torch
        import numpy as np
        from transformer.util import resolve_config

        self.torch = torch
        self.np = np

        # The keys this job runs under, or Aborted -- same as Count. Nothing
        # checks them up front: the read is the check, and the KeyError it raises
        # already names the key that was missing, which is the whole message.
        # resolve_config is inside because it reads the config too -- model_class
        # and the rest -- so a folder that isn't ours fails here like any other
        # missing key, rather than as the one crash that got out.
        try:
            # convert from a dictionary of strings to values with torch objects
            self.config = resolve_config(config)
            c = self.config

            self.seed = c["seed"]
            self.device = c["model_params"]["device"]
            self.context_length = c["model_params"]["context_length"]
            self.lr_schedule_fn = c["lr_schedule_fn"]
            self.lr_schedule_params = c["lr_schedule_params"]

            tc = c["training"]
            self.total_steps = tc["total_iterations"]
            self.save_every = tc["save_every"]
            self.val_every = tc["val_every"]
            self.batch_size = tc["batch_size"]
            self.max_norm = tc["max_norm"]
            self.gpu_check_every = tc.get("gpu_check_every")  # optional -> .get
        except KeyError as e:
            raise Aborted(f"config.json is missing {e}") from None

        self.train_data = np.memmap(VOLUME / tc["train_path"], dtype=np.uint16, mode="r")
        self.valid_data = np.memmap(VOLUME / tc["valid_path"], dtype=np.uint16, mode="r")

        # build model + optimizer from the resolved config; seed first, so init
        # weights are reproducible (must happen before construction).
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.model = c["model_class"](**c["model_params"])  # device/dtype applied at construction
        self.optimizer = c["optimizer_class"](self.model.parameters(), **c["optimizer_params"])

        # fresh run -> save the step-0 checkpoint under the lease; resume -> load the
        # latest. step 0 is the initial state, before any training step runs.
        if latest_checkpoint_path(self.rdir) is None:
            self.step = 0
            self.attempt = 1
            # Nothing is buffered yet, so this boundary is the checkpoint alone.
            # self.step/self.attempt are set just above -- save_checkpoint reads them.
            with self.lease(f"boundary at step 0/{self.total_steps}", commit=True):
                self.save_checkpoint()
            logger.info("Train: fresh run %s (attempt 1, %d total steps)", self.rdir.name, self.total_steps)
        else:
            checkpoint = torch.load(latest_checkpoint_path(self.rdir), map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.step = checkpoint["step"]
            self.attempt = checkpoint["attempt"] + 1
            logger.info(
                "Train: resuming %s from step %d/%d (attempt %d)",
                self.rdir.name,
                self.step,
                self.total_steps,
                self.attempt,
            )

    def clip_gradients(self, max_norm):
        total_norm = self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
        ## TODO: this syncs the GPU and CPU every step at the if level too. This is not good.
        if total_norm > max_norm:
            logger.info(
                "run_training: clipped large gradients at step %d (attempt %d) -- total norm %.4f > max %.4f",
                self.step,
                self.attempt,
                total_norm.item(),
                max_norm,
            )

    def train(self):
        try:
            if self.device == "cuda" and self.torch.cuda.is_available():
                self.torch.cuda.reset_peak_memory_stats()
            while self.step < self.total_steps:
                self.step += 1
                self.model.train()  # sets self.training = True for all module components
                batch = self.get_batch(self.train_data, (self.seed, self.step))
                # lr_schedule_fn returns a single scalar lr, applied uniformly to
                lr = self.lr_schedule_fn(self.step, **self.lr_schedule_params)
                # every param group (there's only ever one lr in this implementation)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                self.optimizer.zero_grad()
                loss, metrics = self.forward_pass(batch)
                loss.backward()
                self.clip_gradients(self.max_norm)
                self.optimizer.step()
                ## add eval metrics
                if self.step % self.val_every == 0:
                    metrics.update(self.evaluate(20))

                # Everything we learned this step, buffered as-is, in one row.
                # These are all still device tensors -- reading one here would
                # sync the GPU every step; they come across together at the
                # boundary.
                self.record_step(self.step, lr=lr, **metrics)
                self.write_progress()  # a local file overwrite; no lease, no commit, no sync

                # The last step is a boundary whatever save_every says: a run that
                # reached total_steps without a checkpoint there would look
                # unfinished to `progress` and be resumed forever.
                if self.step % self.save_every == 0 or self.step == self.total_steps:
                    label = f"boundary at step {self.step}/{self.total_steps} (attempt {self.attempt})"
                    # Entering proves we still own the run, exiting commits. If we
                    # were superseded, entry raises LeaseLost and neither the
                    # checkpoint nor the rows are written -- which unwinds this
                    # loop as a clean stop, and the buffered rows die with the
                    # container, which is what should happen to an interval we no
                    # longer owned. Checkpoint first, so a committed row never
                    # names a file that isn't there.
                    with self.lease(label, commit=True):
                        self.save_checkpoint()
                        rows = self.flush_pending()
                    logger.info("Train: %s for %s", label, self.rdir.name)

                    # W&B after the commit, and outside the gate: these rows are
                    # durable now, so W&B can never hold an interval the ledger
                    # doesn't have, and a burst of log calls never widens the
                    # window between the ownership check and the commit. Logging
                    # only at boundaries is also what keeps W&B's step monotonic
                    # across attempts -- wandb refuses to log to a step it has
                    # already passed, and an attempt that died mid-interval sent
                    # nothing, so the one that resumes starts exactly where W&B is.
                    for step in sorted(rows):
                        # ts and call_id are the ledger's provenance, not metrics.
                        self.wb.log({k: v for k, v in rows[step].items() if k not in ("ts", "call_id")}, step=step)
        except BaseException as exc:
            # Purely for visibility -- deliberately re-raised unchanged, not
            # swallowed. `step` is whatever was in progress when this fired,
            # not necessarily a completed/checkpointed one. Closing W&B is not
            # this method's job: `run` opened it and `run` ends it, on both paths.
            logger.warning(
                "Train: loop for %s exited before finishing, at step %d/%d (%.1f%%) -- %r",
                self.rdir.name,
                self.step,
                self.total_steps,
                100 * self.step / self.total_steps,
                exc,
            )
            raise

    @classmethod
    def progress(cls, rdir: Path, config: dict) -> tuple[int, int]:
        """How far this folder got, read from the checkpoint files.

        Checkpoints are the only thing a commit makes durable -- ledger rows
        written between them live in a container that may never come back. The
        naming matches transformer.util.checkpoint_path, so a run's checkpoints
        are readable by either.

        Raises Aborted if the config is not one this job understands -- the read
        itself is what finds that out. Callers scan whole directories of folders,
        some of which were written by other tools entirely, so "not mine" has to
        be an answer rather than a crash: TypeError comes along with KeyError
        because a foreign config.json can have anything at all under "training",
        not just a dict with the wrong keys.
        """
        try:
            total = config["training"]["total_iterations"]
        except (KeyError, TypeError):
            raise Aborted("config.json is not for this job (no training.total_iterations)") from None
        return committed_step(rdir), total

    def run(self) -> None:
        """Carry this folder to total_iterations, checkpointing every save_every.

        Takes nothing and returns nothing, which is what makes it safe to call
        again: the starting point comes from the folder -- `configure` already
        loaded the latest checkpoint into self.step -- not from an argument, and
        what got done comes from `progress` afterwards, not from a return value.

        The loop itself is `train`; this is the seam the worker calls. W&B's whole
        lifetime is here too -- opened on the line below, ended on both ways out.
        """
        # Not in `configure`: everything that can go wrong before training starts
        # -- a config we cannot read, a model that will not fit, a checkpoint that
        # will not load, a lease already lost at step 0 -- would otherwise leave an
        # initialized W&B run behind, heartbeating as if it were training, in a
        # container that is about to die. Nothing is opened until the folder is
        # ours and the loop is about to run, so there is nothing to clean up on
        # any of those paths.
        #
        # Lazy import, same reason as configure's: the launch and dashboard
        # containers import this module and have no wandb.
        from wandb_run import WandbRun

        # The stored config, not the resolved one: wandb wants the JSON-safe form
        # it can show as columns, which is exactly what config.json holds.
        self.wb = WandbRun(self.run_id, self.config_dict)

        logger.info("Train: %s from step %d to %d", self.rdir.name, self.step, self.total_steps)
        try:
            self.train()
        except BaseException:
            # Say so now rather than leaving wandb to notice a missing heartbeat
            # minutes later. Note this also fires when we were superseded, which
            # is a clean stop -- it marks a run another attempt may already be
            # training as failed.
            self.wb.fail()
            raise
        self.wb.finish(self.rdir)


# --------------------------------------------------------------------------------------
# Count: the reference job, a toy stand-in for training
# --------------------------------------------------------------------------------------


class Count:
    """Counts to total_steps, saving a checkpoint every save_every.

    The toy stand-in: shaped like `Train` on purpose, so the launcher's lease,
    takeover, and resume machinery can be exercised end to end without a GPU --
    step-numbered checkpoints named the way transformer.util names them, an
    append-only ledger, and progress read from committed files.

    It shares no code with `Train` and inherits nothing. The four members
    `app.py` calls are the only thing the two have in common, plus a boundary
    that is three lines in both and reads the same in both.
    """

    def __init__(self, lease, run_id: str):
        self.lease = lease
        self.run_id = run_id
        self.rdir = run_dir(run_id)

        self.config_dict = load_config(run_id)
        if self.config_dict is None:
            raise Aborted("config.json missing or unparseable")

        # The keys this job runs under, or Aborted: a folder missing them is one
        # Count cannot run, so leave it alone rather than crash on it.
        try:
            self.total_steps = self.config_dict["total_steps"]
            self.save_every = self.config_dict["save_every"]
        except KeyError as e:
            raise Aborted(f"config.json is missing {e}") from None

        # Rows wait in memory until a boundary; see the durability rule at the top
        # of this file. This is the thing to carry over when wiring a real trainer
        # -- run_training appended to train.jsonl every step, so those rows would
        # survive on container exit whether or not the lease still held. Either
        # buffer them the way this does, or truncate the file back to its last
        # committed size before leaving.
        self.pending: dict[int, dict] = {}

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
        written between them live in a container that may never come back.

        Raises Aborted if the config is not one this job understands. Callers scan
        whole directories of folders, some of which were written by other tools
        entirely, and "not mine" has to be an answer rather than a crash.
        """
        if "total_steps" not in config:
            raise Aborted("config.json is not for this job (no total_steps)")
        return committed_step(rdir), config["total_steps"]

    def run(self) -> None:
        """Carry this folder to total_steps, committing every save_every.

        Takes nothing and returns nothing, which is what makes it safe to call
        again: the starting point comes from the folder, not from an argument, and
        what got done comes from `progress` afterwards, not from a return value.
        Same reasoning as the launcher -- no caller is trusted to say where the
        work is up to.
        """
        step, _ = self.progress(self.rdir, self.config_dict)
        logger.info("counting from %d to %d", step, self.total_steps)

        while step < self.total_steps:
            step += 1
            time.sleep(self.config_dict.get("step_seconds", 1.0))  # a real trainer's forward/backward
            # A real trainer's metrics; here the row is just when and who.
            self.pending[step] = {"ts": utc(), "call_id": self.lease.call_id}

            if step % self.save_every and step != self.total_steps:
                continue

            # Entering proves we still own the run, exiting commits. Checkpoint
            # first, so a committed row never names a file that isn't there; if we
            # were superseded, entry raises LeaseLost and neither is written.
            label = f"boundary at step {step}/{self.total_steps}"
            with self.lease(label, commit=True):
                self.save_checkpoint(step)
                self.flush_pending()
            logger.info("Count: %s for %s", label, self.rdir.name)

    def flush_pending(self) -> None:
        """Append the buffered rows to train.jsonl and drop them.

        Called only from inside the lease, and only after the checkpoint file is
        written, so a committed row never describes a step whose checkpoint isn't
        there.
        """
        with open(self.rdir / "train.jsonl", "a") as f:
            f.writelines(json.dumps({"step": step, **self.pending[step]}) + "\n" for step in sorted(self.pending))
        self.pending.clear()

    def save_checkpoint(self, step: int) -> None:
        """Write the checkpoint file for `step`.

        Via a temp file and a rename: rename is atomic on the local filesystem,
        so the final name only ever exists complete and no reader can catch a
        half-written file -- see `Train.save_checkpoint` for the torch version,
        and note that transformer.util.save_checkpoint writes straight to the
        final path, which would need this same treatment.
        """
        out = checkpoint_path(self.rdir, step)
        out.parent.mkdir(parents=True, exist_ok=True)  # the job owns its own layout
        tmp = out.with_suffix(".obj.tmp")
        tmp.write_text(json.dumps({"step": step, "seed": self.config_dict.get("seed")}))
        tmp.rename(out)
