import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
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
    memory and touch shared files only from inside `checkpoint`, which is the one
    place this class opens the lease; see `Count` for one way to do it.
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
        self.config_dict = load_config(run_id)
        if self.config_dict is None:
            raise Aborted("config.json missing or unparseable")

        # Ledger rows wait here instead of going straight to train.jsonl, keyed by
        # the step they describe. Per the durability rule above: rows written to
        # local disk are published on exit whether or not we still hold the lease,
        # so an interval we were denied would survive. Held in memory, a denied
        # interval dies with the container -- exactly what should happen to it.
        # `checkpoint` drains this, so a row reaches train.jsonl only once the
        # checkpoint it belongs behind has.
        #
        # Set before `configure` so a job can record from there, and so no
        # subclass has to remember to create it.
        self.pending: dict[int, dict] = {}

        # Which keys this job needs, and how to validate them, is `configure` -- so a
        # job whose config differs (Count's flat total_steps vs Train's training.*)
        # isn't forced through the base. A missing key there raises Aborted the same way.
        self.configure(self.config_dict)

    def checkpoint_path(self, step: int) -> Path:
        return self.rdir / "checkpoints" / f"step_{step:010d}.obj"

    def latest_checkpoint_path(self) -> Path | None:
        checkpoints = sorted((self.rdir / "checkpoints").glob("step_*.obj"))
        return checkpoints[-1] if checkpoints else None

    def record(self, step: int, **values) -> None:
        """Buffer what we learned at `step`, to be flushed at the next checkpoint.

        One row per step, merged across calls, so a step's loss and the val_loss
        computed after it land as one row rather than two. `ts` is stamped on
        first sight of the step -- when the step happened, not when it was
        flushed -- and `call_id` records which attempt produced it.

        Values are stored exactly as handed over, which is the point for GPU
        metrics: a 0-dim cuda tensor costs a few bytes to keep and nothing to
        store, while `.item()` on it would sync the device every step. They are
        turned into numbers once, in `materialize`, at the boundary.
        """
        self.pending.setdefault(step, {"ts": utc(), "call_id": self.lease.call_id}).update(values)

    def materialize(self) -> dict[int, dict]:
        """`self.pending` with every value JSON-encodable, ready to write.

        The default assumes it already is. A job buffering device tensors
        overrides this to bring them across in one go -- see `Train`.
        """
        return dict(self.pending)

    def flush_pending(self) -> dict[int, dict]:
        """Append the buffered rows to train.jsonl, drop them, and return them.

        Only ever called from inside `checkpoint`, and only after the checkpoint
        file is written, so a committed row never describes a step whose
        checkpoint isn't there.

        Returns what it wrote so a job with somewhere else to send the same
        numbers -- W&B, in `Train` -- can do it without materializing them a
        second time. Empty when there was nothing buffered.
        """
        if not self.pending:
            return {}
        rows = self.materialize()
        with open(self.rdir / "train.jsonl", "a") as f:
            f.writelines(json.dumps({"step": step, **rows[step]}) + "\n" for step in sorted(rows))
        self.pending.clear()
        return rows

    def checkpoint(self, write: Callable[[], None], message: str, *args) -> None:
        """Publish a boundary: under the lease, run `write`, then flush the ledger.

        The split is the point. *When* a checkpoint may become durable is the
        same question for every job -- prove we still own the run, write, commit
        -- so it is answered once, here. *What* gets written is the job's own
        business: `write` is a no-argument callable closing over whatever that
        job needs (Train's torch.save, Count's temp-file rename).

        Draining `pending` afterwards is part of the same answer, not the job's
        choice: the rows describing an interval and the checkpoint ending it are
        durable together or not at all, and in that order, so a committed row
        never names a step whose checkpoint is missing.

        `message` and `*args` are a logger-style pair, formatted once and used as
        the one label for the boundary: the lease logs its ownership check under
        it, the commit is logged under it, and it is what we log here -- so a
        checkpoint that committed and a checkpoint that was merely allowed to
        commit read as the same event in the attempt log, told apart by which
        lines follow.

        If we were superseded, entering the lease raises LeaseLost and `write` is
        never called, which unwinds the job's loop as a clean stop -- and the
        buffered rows die unwritten with the container, which is what should
        happen to an interval we no longer owned. If `write` raises, the commit
        is skipped: a half-written checkpoint is not a checkpoint.
        """
        label = message % args if args else message
        with self.lease(label, commit=True):
            write()
            self.flush_pending()
            logger.info("%s: %s", type(self).__name__, label)

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


class Train(Job):
    """Trains a TransformerLM to training.total_iterations, checkpointing every
    training.save_every.

    The config is transformer.util's, unchanged and nested -- model_class,
    model_params, optimizer_*, lr_schedule_*, and a training block -- because a
    real run's definition is what it is, and forcing it flat here would only mean
    translating it back before use. `make_config` mints that shape; `configure`
    and `progress` each find out it is missing by reading it.
    """

    def save_checkpoint(self, step: int, attempt: int, keep_optimizer_history: bool = False) -> None:
        """Write the checkpoint file for `step` -- Train's half of `Job.checkpoint`.

        Step 0 is the initial checkpoint, right after model/optimizer
        construction and before any training has happened; step N is the state
        after the Nth training step has fully completed.

        attempt: which continuous execution produced this checkpoint -- 1 for a
        fresh run, incremented by 1 every time `configure` picks up from an
        existing checkpoint instead of starting at step 0. Persisted so a run's
        history can be split back into its distinct attempts later.

        Adam's optimizer state roughly doubles checkpoint size and is only ever
        needed from the latest checkpoint to resume training. So by default, once
        the new checkpoint is written, the previous latest checkpoint's optimizer
        state is stripped (its model weights are kept). Pass keep_optimizer_history=True
        to preserve full optimizer state in every checkpoint.

        Nothing here touches the lease: this only ever runs as the `write` handed
        to `Job.checkpoint`, which has already proved ownership and will commit
        after it returns.
        """

        # prev = self.latest_checkpoint_path()

        out = self.checkpoint_path(step)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "step": step,
                "attempt": attempt,
                "seed": self.seed,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            out,
        )
        ## TODO: deal with optimizer state

    def flush_pending(self) -> dict[int, dict]:
        """The ledger, then W&B -- one interval's rows, materialized once.

        W&B is fed from what `Job.flush_pending` just wrote rather than from
        `pending`, so the numbers cost exactly one device transfer no matter how
        many places they end up: `materialize` already brought them across, and
        nothing here touches a tensor.

        Logging here, at the boundary, is also what keeps W&B's step monotonic
        across attempts -- and wandb refuses to log to a step it has already
        passed. Rows reach both the ledger and W&B only at a committed boundary,
        so an attempt that dies mid-interval sent neither, and the attempt that
        resumes from the last checkpoint re-runs those steps against a W&B run
        whose step is exactly that checkpoint. The one gap: if the commit itself
        fails after this returns, W&B keeps rows the ledger never got, and the
        resumed attempt's numbers for those steps are dropped as backwards.

        This runs inside the lease block, so it widens the window between the
        ownership check and the commit -- by a burst of save_every buffered
        `wandb.log` calls, each a local handoff to wandb's own process rather
        than a network round-trip. Widening it costs at most a repeated
        interval; if it ever matters, log from the loop instead, off the return
        value of `checkpoint`.
        """
        rows = super().flush_pending()
        # `run` opens self.wb, and one boundary happens before it does: the step-0
        # checkpoint in `configure`. Nothing has been recorded by then, so `rows`
        # is empty there and the loop below never looks for a W&B run.
        for step in sorted(rows):
            # ts and call_id are the ledger's provenance, not metrics -- W&B gets
            # the numbers, keyed by the step they belong to.
            self.wb.log({k: v for k, v in rows[step].items() if k not in ("ts", "call_id")}, step=step)
        return rows

    def materialize(self) -> dict[int, dict]:
        """Bring the buffered tensors to the host, in one transfer for the whole
        interval.

        This is what holding the addresses bought. Every metric `record` took is
        still a 0-dim tensor sitting in device memory, unread -- no `.item()`
        anywhere in the step loop, so nothing forced the GPU and CPU into
        lockstep while training. Here, once per checkpoint, they are stacked into
        a single tensor and moved across in one shot: one sync per boundary
        rather than one per metric per step.

        Stacking requires the held tensors to agree in shape and device, which is
        the contract for anything handed to `record`: scalars, on one device. A
        mismatch raises here, at the boundary, rather than writing a bad row.
        Non-tensors (lr, ts, call_id) pass through untouched.
        """
        torch = self.torch
        held = [
            (step, key, value)
            for step, row in self.pending.items()
            for key, value in row.items()
            if isinstance(value, torch.Tensor)
        ]
        rows = {
            step: {key: value for key, value in row.items() if not isinstance(value, torch.Tensor)}
            for step, row in self.pending.items()
        }
        if held:
            # .float() first: bf16/fp16 metrics would otherwise come across at
            # their training precision, which is not the precision we measured in.
            values = torch.stack([value for *_, value in held]).float().cpu().tolist()
            for (step, key, _), value in zip(held, values):
                rows[step][key] = value
        return rows

    def validate_data(self, config: dict) -> None:
        """Fail before training on a config whose data bins are missing, or whose
        vocab_size disagrees with the tokenizer that produced them. Lifted from
        transformer.util.run_training.

        Paths hang off self.rdir, same base as the memmaps above. Raises Aborted
        rather than the bare FileNotFoundError/ValueError run_training used: a run
        that cannot proceed is one to leave alone, not crash-loop.
        """
        train_path = self.rdir / config["training"]["train_path"]
        valid_path = self.rdir / config["training"]["valid_path"]
        if not train_path.exists():
            raise Aborted(f"train_path not found: {train_path}")
        if not valid_path.exists():
            raise Aborted(f"valid_path not found: {valid_path}")

        # tokenizer-specific: tokenizer_uid is train_path's parent dir name
        # (data/{dataset_name}/bin/{tokenizer_uid}/{split}.bin) -- the single source
        # of truth for which tokenizer produced these bins.
        tokenizer_uid = train_path.parent.name
        tokenizer_config = json.loads((self.rdir / "tokenizers" / tokenizer_uid / "config.json").read_text())
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

        self.train_data = np.memmap(self.rdir / tc["train_path"], dtype=np.uint16, mode="r")
        self.valid_data = np.memmap(self.rdir / tc["valid_path"], dtype=np.uint16, mode="r")

        # build model + optimizer from the resolved config; seed first, so init
        # weights are reproducible (must happen before construction).
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.model = c["model_class"](**c["model_params"])  # device/dtype applied at construction
        self.optimizer = c["optimizer_class"](self.model.parameters(), **c["optimizer_params"])

        # fresh run -> save the step-0 checkpoint under the lease; resume -> load the
        # latest. step 0 is the initial state, before any training step runs.
        if self.latest_checkpoint_path() is None:
            self.step = 0
            self.attempt = 1
            self.checkpoint(
                lambda: self.save_checkpoint(step=0, attempt=1),
                "boundary at step 0/%d -- fresh run %s (attempt 1)",
                self.total_steps,
                self.rdir.name,
            )
        else:
            checkpoint = torch.load(self.latest_checkpoint_path(), map_location=self.device)
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

                # Everything we learned this step, buffered as-is. metrics are
                # still device tensors -- reading one here would sync the GPU
                # every step; they come across together in `materialize`.
                self.record(self.step, lr=lr, **metrics)
                if self.step % self.val_every == 0:
                    self.record(self.step, **self.evaluate(20))

                # The last step is a boundary whatever save_every says: a run that
                # reached total_steps without a checkpoint there would look
                # unfinished to `progress` and be resumed forever.
                if self.step % self.save_every == 0 or self.step == self.total_steps:
                    self.checkpoint(
                        lambda: self.save_checkpoint(self.step, self.attempt),
                        "boundary at step %d/%d (%.1f%%), attempt %d",
                        self.step,
                        self.total_steps,
                        100 * self.step / self.total_steps,
                        self.attempt,
                    )
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
    def make_config(
        cls,
        *,
        model_class,
        model_params: dict,
        optimizer_class,
        optimizer_params: dict,
        lr_schedule_fn,
        lr_schedule_params: dict,
        training: dict,
        seed: int = 0,
        description: str = "",
        metadata: dict | None = None,
    ) -> dict:
        """Build the config.json this job runs under, in the form it is stored in.

        Kept here so the keys `configure` reads and the keys a launch writes
        cannot drift apart -- which matters because config.json is written once,
        at creation, and every later resume runs under it.

        The blocks are named arguments rather than one `config` dict so a launch
        that forgets one fails here, at the caller, naming what is missing --
        rather than in a container, after the folder and its write-once
        config.json already exist.

        The class and function references, a torch.dtype in model_params, and the
        betas tuple cannot survive json.dumps, so `serialize_config` turns them
        into "module.qualname" strings, a bare dtype name, and a list. `configure`
        reverses exactly that with resolve_config -- the two are inverses and live
        beside each other for the same reason these keys live beside `configure`.

        metadata is the wandb block, passed through untouched; omitted entirely
        when None, which is what WandbRun reads as "this run doesn't log to W&B".
        """
        # Lazy, like configure's imports: this is called from a laptop or a
        # notebook, but jobs.py is also imported by the launch and dashboard
        # containers, which have no torch.
        from transformer.util import serialize_config

        config = {
            "description": description,
            "seed": seed,
            "model_class": model_class,
            "model_params": model_params,
            "optimizer_class": optimizer_class,
            "optimizer_params": optimizer_params,
            "lr_schedule_fn": lr_schedule_fn,
            "lr_schedule_params": lr_schedule_params,
            "training": training,
        }
        if metadata is not None:
            config["metadata"] = metadata
        return serialize_config(config)

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
        files = sorted((rdir / "checkpoints").glob("step_*.obj"))
        done = int(files[-1].stem.split("_")[1]) if files else 0
        return done, total

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

        # Nothing else to set up: the ledger buffer is `Job.pending`, created for
        # every job. This is the thing to carry over when wiring a real trainer --
        # run_training appended to train.jsonl every step, so those rows would
        # survive on container exit whether or not the lease still held. Either
        # buffer them through `record` the way this does, or truncate the file
        # back to its last committed size before leaving.

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
        step, _ = self.progress(self.rdir, self.config_dict)
        logger.info("counting from %d to %d", step, self.total_steps)

        while step < self.total_steps:
            step += 1
            time.sleep(self.config_dict.get("step_seconds", 1.0))  # a real trainer's forward/backward
            self.record(step)  # a real trainer's metrics; here the row is just the step

            if step % self.save_every and step != self.total_steps:
                continue

            # `write` runs inside `checkpoint`, before the loop moves on, so the
            # closure over `step` never sees a later value.
            self.checkpoint(lambda: self.save_checkpoint(step), "boundary at step %d/%d", step, self.total_steps)

    def save_checkpoint(self, step: int) -> None:
        """Write the checkpoint file for `step` -- Count's half of `Job.checkpoint`.

        Via a temp file and a rename: rename is atomic on the local filesystem,
        so the final name only ever exists complete and no reader can catch a
        half-written file -- see `Train.save_checkpoint` for the torch version,
        and note that transformer.util.save_checkpoint writes straight to the
        final path, which would need this same treatment.

        The ledger is not this method's business. `Job.checkpoint` flushes the
        buffered rows right after this returns, still inside the lease.
        """
        out = self.checkpoint_path(step)
        out.parent.mkdir(parents=True, exist_ok=True)  # the job owns its own layout
        tmp = out.with_suffix(".obj.tmp")
        tmp.write_text(json.dumps({"step": step, "seed": self.config_dict.get("seed")}))
        tmp.rename(out)
