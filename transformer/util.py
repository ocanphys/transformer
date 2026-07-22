import importlib
import json
import logging
import os
import traceback
import torch
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import requests
from itertools import islice
from tqdm import tqdm
from transformer.tokenizer import Tokenizer


logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging for notebooks / scripts.

    Idempotent: calling multiple times only updates the level. Safe to call
    at the top of every notebook cell that may be re-run.

    Args:
        level: Log level (e.g. logging.INFO, logging.DEBUG).
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    root.setLevel(level)


def download_and_concat(
    urls: list[str], output_path: str, separator: str = "\n"
) -> Path:
    """
    Download text files from URLs and concatenate them into a single file.

    Args:
        urls: List of URLs pointing to plain text files.
        output_path: Path (including filename) where the combined file will be saved.
        separator: String inserted between files (default: newline).

    Returns:
        Path object of the written file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for i, url in enumerate(urls):
            logger.info("[%d/%d] downloading %s", i + 1, len(urls), url)
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            if i > 0:
                f.write(separator)
            f.write(r.text)

    logger.info("wrote %s (%s bytes)", out, f"{out.stat().st_size:,}")


def textfile_to_tokens_as_binary(
    source_text, binary_target, tokenizer: Tokenizer, binary_file_mode="wb"
):
    """
    converts a text file into a raw binary file that can be used as memmap
    for training - we are using uint16 which supports vocab size 2^16 max
    source = "data/combined.txt"
    target = "data/train.bin"
    textfile_to_tokens_as_binary(source_text=source, binary_target=target)
    """

    def batched(it, n):
        it = iter(it)
        while batch := list(islice(it, n)):
            yield batch

    total_bytes = os.path.getsize(source_text)
    # iterable that returns lines from the text file
    with (
        open(source_text, "r") as source_file,
        tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            desc=f"tokenizing {source_text}",
        ) as pbar,
    ):

        def tracked_lines():
            for line in source_file:
                pbar.update(len(line.encode("utf-8")))
                yield line

        # encode line by line and returns an iterator (lazy) of tokens
        token_stream = tokenizer.encode_iterable(tracked_lines())
        with open(binary_target, binary_file_mode) as target_file:
            for chunk in batched(token_stream, 1 << 20):
                np.array(chunk, dtype=np.uint16).tofile(target_file)


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    seed: int,
    step: int,
    device: torch.device | str = "cpu",
):
    high = len(data) - context_length
    rng = np.random.default_rng((seed, step))
    starts = rng.integers(0, high, size=batch_size)
    # cast on the numpy side - Embedding needs long
    inputs = np.stack([data[i : i + context_length] for i in starts]).astype(np.int64)
    targets = np.stack([data[i + 1 : i + context_length + 1] for i in starts]).astype(
        np.int64
    )
    return torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)


def seed_everything(seed: int) -> None:
    """Seed every RNG that affects model initialization (weights in every
    Linear/Embedding/MultiHeadAttention/SwiGLU), on both CPU and CUDA."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(description: str, seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    d = Path("runs") / f"{timestamp}_{description}_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint_path(run_directory: Path, iteration: int) -> Path:
    return run_directory / "checkpoints" / f"step_{iteration:010d}.obj"


def latest_checkpoint_path(run_directory: Path) -> Path | None:
    checkpoints = sorted((run_directory / "checkpoints").glob("step_*.obj"))
    return checkpoints[-1] if checkpoints else None


def import_ref(obj: type | Callable) -> str:
    """Encode a class or top-level function as an importable "module.qualname"
    string, so it round-trips through config.json."""
    return f"{obj.__module__}.{obj.__qualname__}"


def resolve_import_ref(ref):
    """Inverse of import_ref. Passes through anything that isn't a string
    (e.g. already a live class/function, as when config is freshly built)."""
    if not isinstance(ref, str):
        return ref
    module_name, _, qualname = ref.rpartition(".")
    return getattr(importlib.import_module(module_name), qualname)


def resolve_config(config: dict) -> dict:
    """Normalize a raw config dict -- whether freshly built with live class/
    function refs and torch types, or just read back from config.json with
    string-encoded ones -- into one that's ready to use directly everywhere:
    config["model_class"]/config["optimizer_class"]/config["lr_schedule_fn"]
    as real classes/functions, config["model_params"]["dtype"] as a
    torch.dtype, config["optimizer_params"]["betas"] as a tuple.
    """
    resolved = dict(config)
    resolved["model_class"] = resolve_import_ref(config["model_class"])
    resolved["optimizer_class"] = resolve_import_ref(config["optimizer_class"])
    resolved["lr_schedule_fn"] = resolve_import_ref(config["lr_schedule_fn"])

    model_params = dict(config["model_params"])
    dtype = model_params.get("dtype")
    model_params["dtype"] = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    resolved["model_params"] = model_params

    optimizer_params = dict(config["optimizer_params"])
    if "betas" in optimizer_params:
        optimizer_params["betas"] = tuple(optimizer_params["betas"])
    resolved["optimizer_params"] = optimizer_params

    return resolved


def strip_optimizer_state(checkpoint_file: Path) -> None:
    """Rewrite a checkpoint file in place, dropping its optimizer state while
    keeping the model weights. No-op if the file has no optimizer state.
    """
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    if checkpoint["optimizer"] is None:
        return
    checkpoint["optimizer"] = None
    torch.save(checkpoint, checkpoint_file)


def save_checkpoint(
    model, optimizer, iteration, seed, run_directory, keep_optimizer_history=False
):
    """Save a new checkpoint under `run_directory`.

    Adam's optimizer state roughly doubles checkpoint size and is only ever
    needed from the latest checkpoint to resume training. So by default, once
    the new checkpoint is written, the previous latest checkpoint's optimizer
    state is stripped (its model weights are kept). Pass keep_optimizer_history=True
    to preserve full optimizer state in every checkpoint.
    """
    run_directory = Path(run_directory)
    prev = latest_checkpoint_path(run_directory)

    out = checkpoint_path(run_directory, iteration)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "seed": seed,
            "model": model.state_dict() if model is not None else None,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
        },
        out,
    )

    if not keep_optimizer_history and prev is not None:
        strip_optimizer_state(prev)


def append_log(
    run_directory: Path, step: int, loss: float, val_loss: float | None, lr: float
) -> None:
    # Z-suffixed UTC timestamp: parses directly with JS `new Date(...)` and
    # converts unambiguously to any viewer's local timezone in a UI.
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )
    with open(run_directory / "train.jsonl", "a") as f:
        f.write(
            json.dumps(
                {
                    "step": step,
                    "loss": loss,
                    "val_loss": val_loss,
                    "lr": lr,
                    "timestamp": timestamp,
                }
            )
            + "\n"
        )


def write_run_summary(run_directory: Path, config: dict, final_iteration: int) -> None:
    """Write summary.json capturing the run's final state. Reads the full
    train.jsonl history (which spans every resume of this run) rather than
    just what happened in the current call, so stats stay accurate across
    interrupted/resumed runs.
    """
    run_directory = Path(run_directory)
    log_file = run_directory / "train.jsonl"
    rows = (
        [json.loads(line) for line in log_file.read_text().splitlines()]
        if log_file.exists()
        else []
    )
    val_rows = [r for r in rows if r["val_loss"] is not None]
    total_iterations = config["training"]["total_iterations"]

    best_train_row = min(rows, key=lambda r: r["loss"], default=None)
    best_val_row = min(val_rows, key=lambda r: r["val_loss"], default=None)

    summary = {
        "run_directory": str(run_directory),
        "description": config.get("description"),
        "seed": config.get("seed"),
        "final_iteration": final_iteration,
        "total_iterations": total_iterations,
        "completed": final_iteration >= total_iterations,
        "final_train_loss": rows[-1]["loss"] if rows else None,
        "final_val_loss": val_rows[-1]["val_loss"] if val_rows else None,
        "best_train_loss": best_train_row["loss"] if best_train_row else None,
        "best_train_loss_step": best_train_row["step"] if best_train_row else None,
        "best_val_loss": best_val_row["val_loss"] if best_val_row else None,
        "best_val_loss_step": best_val_row["step"] if best_val_row else None,
        "started_at": rows[0]["timestamp"] if rows else None,
        "ended_at": rows[-1]["timestamp"] if rows else None,
    }
    (run_directory / "summary.json").write_text(json.dumps(summary, indent=2))


def run_training(
    config_or_run_dir: dict | Path,
    save_on_exit: bool = True,
    keep_optimizer_history: bool = False,
):
    """Pass a config dict to always start a brand-new run, or an existing run's
    Path (as returned by a previous call) to continue it -- safe to call again
    after a crash/interrupt, and a no-op once total_iterations is reached.

    save_on_exit: when True (default), a checkpoint is written in `finally` even
    if the loop is interrupted or raises, so a later call can resume from the
    last completed step. Set False to skip that save (only the periodic
    `save_every` checkpoints will exist).

    keep_optimizer_history: when False (default), each new checkpoint strips
    the optimizer state from the previous one (keeping its model weights) to
    save disk space, since only the latest checkpoint's optimizer state is
    ever needed to resume training. Set True to keep full optimizer state in
    every checkpoint.
    """
    if isinstance(config_or_run_dir, dict):
        config = config_or_run_dir
        rdir = make_run_dir(
            config["description"], config["seed"]
        )  # always a fresh folder
        serializable = {
            **config,
            "model_class": import_ref(config["model_class"]),
            "optimizer_class": import_ref(config["optimizer_class"]),
            "lr_schedule_fn": import_ref(config["lr_schedule_fn"]),
        }
        (rdir / "config.json").write_text(json.dumps(serializable, indent=2))
        save_checkpoint(
            None,
            None,
            iteration=0,
            seed=config["seed"],
            run_directory=rdir,
            keep_optimizer_history=keep_optimizer_history,
        )
    else:
        rdir = Path(config_or_run_dir)
        config = json.loads((rdir / "config.json").read_text())

    config = resolve_config(config)
    train_cfg = config["training"]
    device = config["model_params"]["device"]
    seed = config["seed"]
    lr_schedule_fn = config["lr_schedule_fn"]
    lr_schedule_params = config["lr_schedule_params"]

    checkpoint = torch.load(latest_checkpoint_path(rdir), map_location=device)
    iteration = checkpoint["iteration"]

    if checkpoint["model"] is None:
        seed_everything(seed)  # reproducible init weights

    model = config["model_class"](**config["model_params"])  # fails loudly on model/config mismatch; already on `device`
    optimizer = config["optimizer_class"](model.parameters(), **config["optimizer_params"])

    if checkpoint["model"] is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    train_data = np.memmap(train_cfg["train_path"], dtype=np.uint16, mode="r")
    valid_data = np.memmap(train_cfg["valid_path"], dtype=np.uint16, mode="r")
    loss_function = torch.nn.CrossEntropyLoss()
    context_length = config["model_params"]["context_length"]

    pbar = tqdm(
        range(iteration, train_cfg["total_iterations"]),
        desc=str(rdir),
        colour="white",
    )
    # `step` is defined BEFORE the try so the except/finally below can always read
    # it, even if we're interrupted or error out on the very first iteration (before
    # the loop variable is otherwise assigned). `iteration - 1` means "no step
    # completed yet", which the `step >= iteration` guards below rely on.
    step = iteration - 1
    try:
        for step in pbar:
            model.train()
            inputs, targets = get_batch(
                train_data,
                train_cfg["batch_size"],
                context_length,
                seed=seed,
                step=step,
                device=device,
            )
            # lr_schedule_fn returns a single scalar lr, applied uniformly to
            # every param group (there's only ever one lr in this implementation)
            lr = lr_schedule_fn(step, **lr_schedule_params)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs.transpose(1, 2), targets)
            loss.backward()
            optimizer.step()
            if device == "mps":
                # optimizer.step()'s in-place param updates are queued
                # asynchronously on MPS; without this, the eval forward below
                # can read partially-written weights and produce degenerate
                # (near-uniform) output.
                torch.mps.synchronize()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

            val_loss = None
            if step % train_cfg["val_every"] == 0:
                model.eval()
                with torch.no_grad():
                    vi, vt = get_batch(
                        valid_data,
                        train_cfg["batch_size"],
                        context_length,
                        seed=seed,
                        step=step,
                        device=device,
                    )
                    val_loss = loss_function(model(vi).transpose(1, 2), vt).item()

            append_log(rdir, step, loss.item(), val_loss, lr)

            if (step + 1) % train_cfg["save_every"] == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    step + 1,
                    seed,
                    rdir,
                    keep_optimizer_history=keep_optimizer_history,
                )
    except BaseException as exc:
        # Catch EVERYTHING here -- Ctrl-C (KeyboardInterrupt), device/runtime errors
        # (e.g. an MPS/CUDA RuntimeError), and ordinary bugs all match, because we
        # catch BaseException (the ROOT of the hierarchy), not just Exception.
        # We log + print the exception but deliberately do NOT re-raise: that stops
        # it propagating, so the `finally` below can hand back the most recent model.
        # NOTE: this intentionally hides the failure from the caller -- the log/print
        # here is the ONLY signal that the run ended early instead of completing.
        logging.exception(
            "run_training: training loop exited via exception at step %d", step
        )
        print(
            f"[run_training] exception at step {step}: {exc!r} "
            f"-- returning most recent checkpoint"
        )
        traceback.print_exc()
    finally:
        # Runs on EVERY exit path: normal loop completion OR the caught exception
        # above. The `return` at the end of this block is the function's single exit
        # for all of those paths -- a `return` inside `finally` is exactly what makes
        # "always hand back the latest model, no matter what" work.
        if save_on_exit and step >= iteration:  # at least one step actually ran
            save_checkpoint(
                model,
                optimizer,
                step + 1,
                seed,
                rdir,
                keep_optimizer_history=keep_optimizer_history,
            )
        final_iteration = step + 1 if step >= iteration else iteration
        write_run_summary(rdir, config, final_iteration)
        # `return` inside `finally`: the function's exit for every path (clean finish
        # or caught exception). Because it lives in `finally`, it also swallows any
        # exception that were somehow still pending -- intended here, but the reason a
        # stray failure can vanish silently, hence the logging/print in `except`.
        return model, optimizer, rdir


class LiveLossPlot:
    """Context manager that draws a live loss curve in a Jupyter notebook.

    Usage:
        with LiveLossPlot(every=10) as plot:
            for iteration in range(n_steps):
                ...
                plot.log(loss.item(), iteration)

    Requires `%matplotlib widget` (ipympl) in the notebook for live updates.
    """

    def __init__(
        self,
        every: int = 10,
        figsize: tuple[int, int] = (10, 4),
        title: str = "Training loss (live)",
        text_color: str = "white",
    ):
        self.every = every
        self.figsize = figsize
        self.title = title
        self.text_color = text_color
        self.iterations: list[int] = []
        self.losses: list[float] = []
        self.val_steps: list[int] = []
        self.val_losses: list[float] = []

    def __enter__(self):
        import matplotlib.pyplot as plt
        from IPython.display import display

        # ioff prevents auto-display so we only render via the display handle
        with plt.ioff():
            self.fig, self.ax = plt.subplots(figsize=self.figsize)

        # transparent background + text_color for ticks/labels to match a dark (VS Code) theme
        self.fig.patch.set_alpha(0)
        self.ax.patch.set_alpha(0)
        for spine in self.ax.spines.values():
            spine.set_color(self.text_color)
        self.ax.tick_params(colors=self.text_color)
        self.ax.xaxis.label.set_color(self.text_color)
        self.ax.yaxis.label.set_color(self.text_color)
        self.ax.title.set_color(self.text_color)

        (self.line,) = self.ax.plot([], [], label="train")
        (self.val_line,) = self.ax.plot(
            [], [], color="tab:orange", marker="o", label="val"
        )
        self.ax.set_xlabel("iteration")
        self.ax.set_ylabel("loss")
        self.ax.set_title(self.title)
        self.ax.grid(True, alpha=0.3, color=self.text_color)
        legend = self.ax.legend(loc="upper right")
        legend.get_frame().set_alpha(0)
        for text in legend.get_texts():
            text.set_color(self.text_color)
        self._dh = display(
            self.fig, display_id=True
        )  # reserve an output slot we can overwrite
        return self

    def log(self, loss: float, iteration: int) -> None:
        self.iterations.append(iteration)
        self.losses.append(loss)
        if (len(self.losses) - 1) % self.every == 0:
            self._redraw()

    def log_val(self, loss: float, iteration: int) -> None:
        """Record a validation loss at the given training iteration."""
        self.val_steps.append(iteration)
        self.val_losses.append(loss)
        self._redraw()

    def _redraw(self) -> None:
        self.line.set_data(self.iterations, self.losses)
        self.val_line.set_data(self.val_steps, self.val_losses)
        self.ax.relim()
        self.ax.autoscale_view()
        self._dh.update(self.fig)

    def __exit__(self, exc_type, exc_val, exc_tb):
        import matplotlib.pyplot as plt

        del exc_type, exc_val, exc_tb
        self._redraw()  # final paint so the last few steps show up
        plt.close(self.fig)  # prevent a duplicate render at cell end
        return False


class LMDataLoader(torch.utils.data.IterableDataset):
    """Infinite random-batch loader over a uint16 token memmap."""

    def __init__(
        self,
        path: str,
        batch_size: int,
        context_length: int,
        seed: int = 0,
        step: int = 0,
        device: torch.device | str = "cpu",
    ):
        self.path = path
        self.batch_size = batch_size
        self.context_length = context_length
        self.seed = seed
        self.step = step
        self.device = device

    def __iter__(self):
        data = np.memmap(self.path, dtype=np.uint16, mode="r")
        while True:
            yield get_batch(
                data,
                self.batch_size,
                self.context_length,
                seed=self.seed,
                step=self.step,
                device=self.device,
            )
            self.step += 1
