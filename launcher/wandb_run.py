"""Weights & Biases logging for one training run, or a no-op when disabled.

Kept in its own module so `jobs.py` stays stdlib at import time: nothing here is
imported by the launch/dashboard containers unless a job actually asks for it, and
even then the heavy `wandb` package is pulled lazily (inside `__init__`) only when
a run is enabled. So importing this module is free; only constructing an enabled
`WandbRun` reaches for `wandb`.
"""

from pathlib import Path


class WandbRun:
    """W&B logging for one run, or a no-op when the run doesn't ask for it.

    A run's `config["metadata"]` block is yours to define freely. On init it is
    split: the keys that are genuinely wandb.init fields (project, name, notes,
    tags, group, ...) are passed as native arguments, so they populate wandb's
    built-in columns (Name, Tags, Notes, Project) rather than a nested
    `metadata.tags` config entry; everything else you put in metadata is logged as
    a flat config column, so nothing is dropped and an unrecognized key never
    crashes wandb.init. The `wandb_kwargs` argument is an additional dict of the
    same kind, empty by default, merged on top; it wins on any collision.

    Enabled iff the config declares a `metadata` block (or a non-empty override is
    passed). That single decision lives in `self.on`, so the training loop just
    calls `wb.log(...)` unconditionally and the on/off rule sits in one place.
    """

    def __init__(self, run_id: str, config: dict, wandb_kwargs: dict | None = None):
        metadata = config.get("metadata")  # yours to define freely, or None -> no wandb
        self.on = metadata is not None or bool(wandb_kwargs)
        if not self.on:
            return
        import wandb  # lazy: only an enabled run in the training container reaches this

        self.wandb = wandb
        # Split metadata (with wandb_kwargs merged on top, winning): the keys
        # wandb.init knows natively become real wandb fields; the rest are logged as
        # flat config columns -- so an unknown key is recorded, not a TypeError.
        native_fields = {"project", "entity", "name", "notes", "tags", "group", "job_type", "mode"}
        meta = {**(metadata or {}), **(wandb_kwargs or {})}
        native = {k: v for k, v in meta.items() if k in native_fields}
        custom = {k: v for k, v in meta.items() if k not in native_fields}

        # Config's own keys minus the metadata block, plus metadata's non-native
        # keys lifted to the root -- all flat columns in the runs table.
        logged = {k: v for k, v in config.items() if k != "metadata"}
        logged.update(custom)
        # id=run_id + resume="allow" so every attempt of the same run appends to one
        # wandb run; native metadata is spread as init args -> native wandb fields.
        self.wandb.init(**{"id": run_id, "resume": "allow", "config": logged, **native})

    def log(self, row: dict, step: int) -> None:
        if self.on:
            self.wandb.log(row, step=step)

    def fail(self) -> None:
        """Mark the run Failed immediately, instead of leaving it to wandb's
        passive crashed-heartbeat detection."""
        if self.on:
            self.wandb.finish(exit_code=1)

    def finish(self, rdir: Path) -> None:
        if self.on:
            # base_path=str(rdir) uploads summary.json flat at the run's files
            # root, rather than mirroring the full directory structure.
            self.wandb.save(str(rdir / "summary.json"), base_path=str(rdir))
            self.wandb.finish()
