import importlib
import json
import time
import logging
import os
import torch
import wandb
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import requests
from itertools import islice
from tqdm import tqdm
from transformer.tokenizer import Tokenizer, train_bpe


logger = logging.getLogger(__name__)


def utc_formatter(fields: str = "%(name)s %(levelname)s %(message)s") -> logging.Formatter:
    """A formatter whose first column is this project's timestamp contract.

    `2026-08-12T18:30:00.123Z`, UTC, fixed width, always column one. Every log in
    this project starts that way -- `launcher/logs.py` for a Modal call,
    `launcher/applog.py` for Modal's own stream -- which is what lets any two of
    them be merged by sorting the raw strings, with no parsing and no timezone to
    reason about.

    Those two define the same format independently, and that is deliberate rather
    than sloppy: the launcher's containers do not have this package installed, so
    there is nothing to share. The contract is the shape, not an object.

    `converter` is set on the instance rather than on `logging.Formatter`, so
    asking for UTC here cannot silently restamp every other log in the process.
    """
    formatter = logging.Formatter(fmt=f"%(asctime)s.%(msecs)03dZ {fields}", datefmt="%Y-%m-%dT%H:%M:%S")
    formatter.converter = time.gmtime
    return formatter


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging for notebooks / scripts.

    Idempotent: calling multiple times only updates the level. Safe to call
    at the top of every notebook cell that may be re-run.

    Timestamps are UTC, in the same shape a run's log file on the Volume uses, so
    a line read in a notebook and a line read out of `runs/{run_id}/logs/` can be
    lined up against each other without converting anything.

    Args:
        level: Log level (e.g. logging.INFO, logging.DEBUG).
    """
    root = logging.getLogger()
    if not root.handlers:
        # Not basicConfig: the UTC converter lives on a formatter instance, and
        # basicConfig only takes a format string.
        handler = logging.StreamHandler()
        handler.setFormatter(utc_formatter())
        root.addHandler(handler)
    root.setLevel(level)


def download_and_concat(urls: list[str], output_path: str, volume: Path, separator: str = "\n") -> Path:
    """
    Download text files from URLs and concatenate them into a single file.

    Args:
        urls: List of URLs pointing to plain text files.
        output_path: Path (including filename) where the combined file will be saved.
        separator: String inserted between files (default: newline).

    Returns:
        Path object of the written file.
    """
    out = volume / Path(output_path)
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


def prepare_tokenizer(
    tokenizer_uid: str,
    vocab_size: int,
    special_tokens: list[str],
    raw_text_path: str,
    volume: Path,
) -> Path:
    """Train a BPE tokenizer and write tokenizers/{tokenizer_uid}/tokenizer.joblib
    (vocab + merges) and config.json (special_tokens, vocab_size, raw_text_path),
    loadable back via Tokenizer.from_files(tokenizer_uid, volume).

    raw_text_path is relative to volume (e.g. "data/train.txt"), same as everything
    else volume-scoped -- so the same call reproduces identically regardless of
    which volume (local or the Modal Volume mount) it's run against.

    Always retrains and overwrites -- no check for an existing tokenizer_uid.
    """
    vocab, merges = train_bpe(str(volume / raw_text_path), vocab_size, special_tokens)

    tokenizer_dir = volume / "tokenizers" / tokenizer_uid
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump((vocab, merges), tokenizer_dir / "tokenizer.joblib")

    tokenizer_config = {
        "special_tokens": special_tokens,
        "vocab_size": vocab_size,
        "raw_text_path": str(raw_text_path),
    }
    (tokenizer_dir / "config.json").write_text(json.dumps(tokenizer_config, indent=2))
    # return tokenizer_dir


def textfile_to_tokens_as_binary(source_text, binary_target, tokenizer: Tokenizer, volume: Path, binary_file_mode="wb"):
    """
    converts a text file into a raw binary file that can be used as memmap
    for training - we are using uint16 which supports vocab size 2^16 max
    all filepaths are relative to volume/
    source = "data/combined.txt"
    target = "data/train.bin"
    textfile_to_tokens_as_binary(source_text=source, binary_target=target)
    """

    source_text = volume / source_text
    binary_target = volume / binary_target

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
    targets = np.stack([data[i + 1 : i + context_length + 1] for i in starts]).astype(np.int64)
    return torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)


def seed_everything(seed: int) -> None:
    """Seed every RNG that affects model initialization (weights in every
    Linear/Embedding/MultiHeadAttention/SwiGLU), on both CPU and CUDA."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(description: str, seed: int, volume: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    d = volume / "runs" / f"{timestamp}_{description}_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint_path(run_directory: Path, step: int) -> Path:
    return run_directory / "checkpoints" / f"step_{step:010d}.obj"


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


def serialize_config(config: dict) -> dict:
    """Inverse of resolve_config: turn a live config -- real classes/functions, a
    torch.dtype, a tuple of betas -- into a JSON-safe one, ready for json.dumps() ->
    config.json. The three class/function refs become importable "module.qualname"
    strings, the dtype a bare name like "float32", and betas a list; everything else
    passes through untouched.

    Only the fields resolve_config resolves are converted, so the two are exact
    inverses -- serialize_config then resolve_config (or the round-trip through
    config.json) yields an equivalent live config. Like its inverse it is forgiving:
    values already in serialized form pass through, so calling it on an
    already-serialized config is a no-op rather than an error.
    """

    def ref(obj):  # already a "module.qualname" string (re-serializing) -> leave it
        return obj if isinstance(obj, str) else import_ref(obj)

    serialized = dict(config)
    serialized["model_class"] = ref(config["model_class"])
    serialized["optimizer_class"] = ref(config["optimizer_class"])
    serialized["lr_schedule_fn"] = ref(config["lr_schedule_fn"])

    model_params = dict(config["model_params"])
    dtype = model_params.get("dtype")
    model_params["dtype"] = str(dtype).removeprefix("torch.") if isinstance(dtype, torch.dtype) else dtype
    serialized["model_params"] = model_params

    optimizer_params = dict(config["optimizer_params"])
    if "betas" in optimizer_params:
        optimizer_params["betas"] = list(optimizer_params["betas"])
    serialized["optimizer_params"] = optimizer_params

    return serialized


def strip_optimizer_state(checkpoint_file: Path) -> None:
    """Rewrite a checkpoint file in place, dropping its optimizer state while
    keeping the model weights. No-op if the file has no optimizer state.
    """
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    if checkpoint["optimizer"] is None:
        return
    checkpoint["optimizer"] = None
    torch.save(checkpoint, checkpoint_file)


def save_checkpoint(model, optimizer, step, attempt, seed, run_directory, keep_optimizer_history=False):
    """Save a new checkpoint under `run_directory`, at `step` (step 0 is the
    initial checkpoint, right after model/optimizer construction and before
    any training has happened; step N is the state after the Nth training
    step has fully completed).

    attempt: which continuous execution produced this checkpoint -- 1 for a
    fresh run, incremented by 1 every time run_training picks up from an
    existing checkpoint instead of starting at step 0. Persisted so a run's
    history can be split back into its distinct attempts later.

    Adam's optimizer state roughly doubles checkpoint size and is only ever
    needed from the latest checkpoint to resume training. So by default, once
    the new checkpoint is written, the previous latest checkpoint's optimizer
    state is stripped (its model weights are kept). Pass keep_optimizer_history=True
    to preserve full optimizer state in every checkpoint.
    """
    run_directory = Path(run_directory)
    prev = latest_checkpoint_path(run_directory)

    out = checkpoint_path(run_directory, step)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "attempt": attempt,
            "seed": seed,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        out,
    )

    if not keep_optimizer_history and prev is not None and prev != out:
        strip_optimizer_state(prev)


def gpu_stats(device) -> dict | None:
    """Snapshot of GPU telemetry: device name, utilization, power draw,
    temperature, clock rate, peak memory allocated + device total + percent
    used. None if device isn't CUDA.

    Resets peak-memory tracking after reading it, so the next call's
    max_memory_allocated_gb reflects only the interval since now, not since
    process start -- call this on some fixed cadence (see gpu_check_every)
    rather than freely, or that interval stops meaning anything consistent.
    """
    if not str(device).startswith("cuda"):
        return None
    total_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1.0e9
    max_allocated_gb = torch.cuda.max_memory_allocated(device) / 1.0e9
    stats = {
        "gpu_type": torch.cuda.get_device_name(device),
        "utilization": torch.cuda.utilization(device),
        "power_draw": torch.cuda.power_draw(device),
        "temperature": torch.cuda.temperature(device),
        "clock_rate": torch.cuda.clock_rate(device),
        "max_memory_allocated_gb": max_allocated_gb,
        "total_memory_gb": total_memory_gb,
        "memory_percent_used": 100 * max_allocated_gb / total_memory_gb,
    }
    torch.cuda.reset_peak_memory_stats(device)
    return stats


def write_run_summary(run_directory: Path, config: dict, final_step: int, volume: Path) -> None:
    """Write summary.json capturing the run's final state. Reads the full
    train.jsonl history (which spans every resume of this run) rather than
    just what happened in the current call, so stats stay accurate across
    interrupted/resumed runs.

    run_directory is stored relative to volume, same as everything else
    volume-scoped -- never an absolute path.
    """
    run_directory = Path(run_directory)
    log_file = run_directory / "train.jsonl"
    rows = [json.loads(line) for line in log_file.read_text().splitlines()] if log_file.exists() else []
    val_rows = [r for r in rows if r["val_loss"] is not None]
    total_iterations = config["training"]["total_iterations"]

    best_train_row = min(rows, key=lambda r: r["loss"], default=None)
    best_val_row = min(val_rows, key=lambda r: r["val_loss"], default=None)
    # Peak across every gpu_check_every snapshot this run logged (each one only
    # covers its own interval, since gpu_stats() resets peak-tracking every call).
    max_memory_allocated_gb = max(
        (r["max_memory_allocated_gb"] for r in rows if "max_memory_allocated_gb" in r), default=None
    )

    # gpu_type/total_memory_gb are static hardware facts -- one fresh snapshot
    # here is enough, no need to aggregate them across rows like the peak above.
    device = str(config.get("model_params", {}).get("device", ""))
    final_gpu_stats = gpu_stats(device)
    gpu_type = final_gpu_stats["gpu_type"] if final_gpu_stats else None
    gpu_total_memory_gb = final_gpu_stats["total_memory_gb"] if final_gpu_stats else None

    summary = {
        "run_directory": str(run_directory.relative_to(volume)),
        "description": config.get("description"),
        "seed": config.get("seed"),
        "final_step": final_step,
        "total_iterations": total_iterations,
        "completed": final_step >= total_iterations,
        "final_train_loss": rows[-1]["loss"] if rows else None,
        "final_val_loss": val_rows[-1]["val_loss"] if val_rows else None,
        "best_train_loss": best_train_row["loss"] if best_train_row else None,
        "best_train_loss_step": best_train_row["step"] if best_train_row else None,
        "best_val_loss": best_val_row["val_loss"] if best_val_row else None,
        "best_val_loss_step": best_val_row["step"] if best_val_row else None,
        "started_at": rows[0]["timestamp"] if rows else None,
        "ended_at": rows[-1]["timestamp"] if rows else None,
        "wall_clock_time": str(
            datetime.fromisoformat(rows[-1]["timestamp"]) - datetime.fromisoformat(rows[0]["timestamp"])
        )
        if rows
        else None,
        "max_memory_allocated_gb": max_memory_allocated_gb,
        "gpu_type": gpu_type,
        "gpu_total_memory_gb": gpu_total_memory_gb,
    }
    (run_directory / "summary.json").write_text(json.dumps(summary, indent=2))


def run_training(
    config_or_run_dir: dict | Path,
    volume: Path,
    wandb_kwargs: dict | None = None,
    keep_optimizer_history: bool = False,
):
    """Pass a config dict to always start a brand-new run, or an existing run's
    Path *relative to volume* (e.g. "runs/20260722T073953_baseline4layer_0") to
    continue it -- safe to call again, and a no-op once total_iterations is
    reached.

    Steps only, no separate "iteration" concept: step 0 is the initial
    checkpoint (model/optimizer state right after construction, before any
    training happens); step N is the state after the Nth training step has
    fully completed. Resuming always continues from (latest checkpoint's
    step) + 1.

    No try/except around the training loop: a checkpoint is only ever written
    once a step has fully finished, so if training is interrupted (Ctrl-C, a
    crash, anything) partway through a step, nothing is saved for that
    in-progress step and nothing claims it completed. Resuming restarts at the
    last real checkpoint -- no silently-skipped steps, no half-applied weight
    update ever persisted as if it were done. The tradeoff: an interrupted
    call raises instead of always handing back a model.

    Every checkpoint and every train.jsonl row also carries `attempt`: 1 for a
    fresh run, incremented by 1 each time this function picks up from an
    existing checkpoint instead of starting at step 0. A run's full history
    (across every crash/resume) lives in one train.jsonl, so `attempt` is what
    lets you cleanly split it back into its distinct continuous executions
    afterward, e.g. to sanity-check that loss didn't jump at an attempt
    boundary.

    config["training"]["train_path"]/["valid_path"] must likewise be relative to
    volume (e.g. "data/{dataset_name}/bin/{tokenizer_uid}/train.bin"), not
    absolute -- so the same config.json reproduces identically whether run
    against the local volume or the Modal Volume mount. Nothing in this
    function is stored as an absolute path.

    tokenizer_uid isn't a separate config field -- it's read off train_path's
    parent directory name (the {tokenizer_uid} path segment above), since that's
    already the single source of truth for which tokenizer produced these bins;
    duplicating it into config would just be a second place for it to go stale.
    For a fresh run (config is a dict), this is validated before anything else
    runs: train_path/valid_path must exist under volume, and
    config["model_params"]["vocab_size"] must match that tokenizer's own
    config.json vocab_size -- otherwise this raises immediately rather than
    starting a doomed run.

    wandb_kwargs: when not None (e.g. {} or {"project": ..., "tags": [...]}),
    also logs to Weights & Biases in addition to train.jsonl (requires
    WANDB_API_KEY in the environment -- e.g. via a Modal Secret for runs on
    Modal). Passed straight into wandb.init(...); project/name/tags/entity/etc
    are entirely the caller's choice. id and resume="allow" are supplied by
    this function (rdir.name, so every attempt of the same run appends to one
    wandb run, same as train.jsonl) unless wandb_kwargs overrides them.

    keep_optimizer_history: when False (default), each new checkpoint strips
    the optimizer state from the previous one (keeping its model weights) to
    save disk space, since only the latest checkpoint's optimizer state is
    ever needed to resume training. Set True to keep full optimizer state in
    every checkpoint.
    """
    if isinstance(config_or_run_dir, dict):
        config = config_or_run_dir

        train_path = volume / config["training"]["train_path"]
        valid_path = volume / config["training"]["valid_path"]
        if not train_path.exists():
            raise FileNotFoundError(f"train_path not found: {train_path}")
        if not valid_path.exists():
            raise FileNotFoundError(f"valid_path not found: {valid_path}")

        ## < -This part is tokenizer specific.
        tokenizer_uid = train_path.parent.name  # data/{dataset_name}/bin/{tokenizer_uid}/{split}.bin
        tokenizer_config = json.loads((volume / "tokenizers" / tokenizer_uid / "config.json").read_text())
        if tokenizer_config["vocab_size"] != config["model_params"]["vocab_size"]:
            raise ValueError(
                f"model_params.vocab_size ({config['model_params']['vocab_size']}) does not match "
                f"tokenizer {tokenizer_uid!r}'s vocab_size ({tokenizer_config['vocab_size']})"
            )
        ## ->

        rdir = make_run_dir(config["description"], config["seed"], volume)  # always a fresh folder
        serializable = {
            **config,
            "model_class": import_ref(config["model_class"]),
            "optimizer_class": import_ref(config["optimizer_class"]),
            "lr_schedule_fn": import_ref(config["lr_schedule_fn"]),
        }
        (rdir / "config.json").write_text(json.dumps(serializable, indent=2))
        config_json = serializable
        start_step = 0
    else:
        rdir = volume / Path(config_or_run_dir)  # config_or_run_dir is relative to volume
        config = json.loads((rdir / "config.json").read_text())
        config_json = config  # already the JSON form -- resolve_config below rebinds `config`, doesn't mutate this
        start_step = None  # resolved below, from the latest checkpoint

    config = resolve_config(config)
    train_cfg = config["training"]
    total_iterations = train_cfg["total_iterations"]
    device = config["model_params"]["device"]
    seed = config["seed"]
    lr_schedule_fn = config["lr_schedule_fn"]
    lr_schedule_params = config["lr_schedule_params"]
    gpu_check_every = train_cfg.get("gpu_check_every")  # in steps; None disables GPU telemetry entirely

    # Named after rdir, so this call's logs are scoped to this run by
    # construction -- no shared handler to add/remove, no risk of a later
    # run_training call in the same process writing into this run's file.
    #
    # events.log is this function's own record and predates the Modal launcher; on
    # that path `launcher/jobs.Train` does the training and writes
    # `runs/{run_id}/logs/{call_id}/worker.log` instead, so nothing on Modal ever
    # writes this file. It keeps the shared timestamp contract anyway, so a local
    # run's log still merges against everything else.
    run_logger = logging.getLogger(f"{__name__}.{rdir.name}")
    run_logger.setLevel(logging.INFO)
    events_handler = logging.FileHandler(rdir / "events.log")
    events_handler.setFormatter(utc_formatter())
    run_logger.addHandler(events_handler)

    if start_step == 0:
        seed_everything(seed)  # reproducible init weights -- must happen before construction below

    model = config["model_class"](
        **config["model_params"]
    )  # fails loudly on model/config mismatch; already on `device`
    optimizer = config["optimizer_class"](model.parameters(), **config["optimizer_params"])

    if start_step == 0:
        attempt = 1
        save_checkpoint(model, optimizer, 0, attempt, seed, rdir, keep_optimizer_history=keep_optimizer_history)
        run_logger.info(
            "run_training: starting fresh run %s (attempt %d, %d total steps)", rdir.name, attempt, total_iterations
        )
    else:
        checkpoint = torch.load(latest_checkpoint_path(rdir), map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        attempt = checkpoint["attempt"] + 1  # picking up from a checkpoint -- a new attempt
        run_logger.info(
            "run_training: resuming %s from step %d/%d (attempt %d)",
            rdir.name,
            start_step,
            total_iterations,
            attempt,
        )

    if wandb_kwargs is not None:
        wandb.init(
            **{
                "id": rdir.name,
                "resume": "allow",
                "config": config_json,
                **wandb_kwargs,  # caller wins on any key it explicitly sets
            }
        )
        # Redoing steps after resuming from an older checkpoint than the last
        # step actually reached (checkpoints only save every save_every steps,
        # but every step gets logged) would otherwise violate wandb's own
        # monotonically-increasing step counter and get silently dropped. Log
        # `step` as a plain metric instead and use it as the x-axis, so wandb's
        # internal counter (which only ever advances) is decoupled from ours
        # (which can legitimately repeat across attempts).
        # FOR NOW lets identify. If we recompute an older point, wandb will ignore it.
        # #TODO: ADD A CLEAN CHECK FOR THIS.
        # wandb.define_metric("step")
        # wandb.define_metric("*", step_metric="step")

    train_data = np.memmap(volume / train_cfg["train_path"], dtype=np.uint16, mode="r")
    valid_data = np.memmap(volume / train_cfg["valid_path"], dtype=np.uint16, mode="r")
    loss_function = torch.nn.CrossEntropyLoss()
    # #TODO: this is also language/transformer specific - compactify eventually.
    context_length = config["model_params"]["context_length"]
    if start_step < total_iterations:
        pbar = tqdm(
            range(start_step + 1, total_iterations + 1),
            desc=str(rdir),
            colour="white",
        )
        try:
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            for step in pbar:
                # set self.training = True for all modules / submodule specific behaviour is controlled
                model.train()
                # pull a batch and return as tensors living on the device memory.
                inputs, targets = get_batch(
                    train_data,
                    train_cfg["batch_size"],
                    context_length,
                    seed=seed,
                    step=step,
                    device=device,
                )
                # lr_schedule_fn returns a single scalar lr, applied uniformly to
                lr = lr_schedule_fn(step, **lr_schedule_params)
                # every param group (there's only ever one lr in this implementation)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs.transpose(1, 2), targets)
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if total_norm > 1.0:
                    run_logger.info(
                        "run_training: clipped large gradients at step %d (attempt %d) -- total norm %.4f > max 1.0",
                        step,
                        attempt,
                        total_norm.item(),
                    )

                optimizer.step()
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

                # Z-suffixed UTC timestamp: parses directly with JS `new Date(...)` and
                # converts unambiguously to any viewer's local timezone in a UI.
                row = {
                    "step": step,
                    "attempt": attempt,
                    "loss": loss.item(),
                    "val_loss": val_loss,
                    "lr": lr,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                }
                if gpu_check_every and step % gpu_check_every == 0:
                    stats = gpu_stats(device)
                    if stats is not None:
                        row.update(stats)
                with open(rdir / "train.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")

                if wandb_kwargs is not None:
                    wandb.log(row, step=step)

                if step % train_cfg["save_every"] == 0:
                    save_checkpoint(
                        model, optimizer, step, attempt, seed, rdir, keep_optimizer_history=keep_optimizer_history
                    )
                    run_logger.info(
                        "run_training: checkpoint saved at step %d/%d (%.1f%%) for %s",
                        step,
                        total_iterations,
                        100 * step / total_iterations,
                        rdir.name,
                    )
        except BaseException as exc:
            # Purely for visibility -- deliberately re-raised unchanged, not
            # swallowed. `step` is whatever was in progress when this fired,
            # not necessarily a completed/checkpointed one.
            run_logger.warning(
                "run_training: loop for %s exited before finishing, at step %d/%d (%.1f%%) -- %r",
                rdir.name,
                step,
                total_iterations,
                100 * step / total_iterations,
                exc,
            )
            if wandb_kwargs is not None:
                # exit_code!=0 marks the run "Failed" immediately, instead of
                # leaving it to wandb's passive crashed-heartbeat detection.
                wandb.finish(exit_code=1)
            raise

        # always persist the true final step, even if total_iterations isn't a
        # clean multiple of save_every -- so a normal, uninterrupted finish
        # never leaves trained steps unpersisted.
        if total_iterations % train_cfg["save_every"] != 0:
            save_checkpoint(
                model,
                optimizer,
                total_iterations,
                attempt,
                seed,
                rdir,
                keep_optimizer_history=keep_optimizer_history,
            )
            run_logger.info(
                "run_training: checkpoint saved at step %d/%d (100.0%%) for %s",
                total_iterations,
                total_iterations,
                rdir.name,
            )

    run_logger.info("run_training: writing summary for %s", rdir.name)
    write_run_summary(rdir, config, total_iterations, volume)

    if wandb_kwargs is not None:
        # base_path=str(rdir) makes the path relative to rdir -- just "summary.json",
        # so it lands flat at the run's files root instead of wandb mirroring the
        # full absolute directory structure into a nested folder.
        wandb.save(
            str(rdir / "summary.json"), base_path=str(rdir)
        )  # must happen before finish() -- can't upload to a closed run
        wandb.finish()

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
        (self.val_line,) = self.ax.plot([], [], color="tab:orange", marker="o", label="val")
        self.ax.set_xlabel("iteration")
        self.ax.set_ylabel("loss")
        self.ax.set_title(self.title)
        self.ax.grid(True, alpha=0.3, color=self.text_color)
        legend = self.ax.legend(loc="upper right")
        legend.get_frame().set_alpha(0)
        for text in legend.get_texts():
            text.set_color(self.text_color)
        self._dh = display(self.fig, display_id=True)  # reserve an output slot we can overwrite
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
