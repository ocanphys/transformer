# modal_lab

Barebones pipeline for running the `transformer` package's PyTorch training/eval on Modal.

## Directory layout

```
data/{dataset_name}/
├── train.txt                          # raw text
└── valid.txt                          # raw text

tokenizers/{tokenizer_uid}/
├── tokenizer.joblib                   # vocab + merges
└── config.json                        # special_tokens, vocab_size, raw_text_path (what it was trained on)

data/{dataset_name}/bin/{tokenizer_uid}/
├── train.bin                          # train.txt encoded with this tokenizer
└── valid.bin                          # valid.txt encoded with this tokenizer
```

Tokenizers are prepared independently of any dataset folder — `tokenizers/{tokenizer_uid}/config.json` is the single source of truth for that tokenizer's `special_tokens`, `vocab_size`, and the raw text it was trained on. A run references a tokenizer only by its `{tokenizer_uid}`; a single tokenizer can be reused across multiple datasets/runs.

## Preparation stage

Tokenizers are built explicitly, ahead of time — not lazily inside a run:

1. Pick `vocab_size`, `special_tokens`, and a raw text path.
2. Train the tokenizer, assign it a `{tokenizer_uid}`, and write `tokenizers/{tokenizer_uid}/tokenizer.joblib` + `config.json`.

This removes the earlier race condition around lazily building a tokenizer mid-run — the artifact already exists by the time any run references it.

## Config file

One `config.json` per run holds:

- `model_params` — transformer architecture (existing schema from `transformer/util.py`), including `vocab_size`
- `optimizer_params` — optimizer + LR schedule (existing schema from `transformer/util.py`)
- `tokenizer_uid` — a single reference to `tokenizers/{tokenizer_uid}/`. No tokenizer settings (vocab size, special tokens, raw text path) are duplicated into the run config — they're only ever read from the tokenizer's own `config.json`.

## Pipeline behavior

1. Load `tokenizers/{tokenizer_uid}/config.json` (the tokenizer's specification: `special_tokens`, `vocab_size`, `raw_text_path`) and `tokenizer.joblib`, using the `tokenizer_uid` named in the run config.
2. Validate the run: if `model_params.vocab_size` doesn't equal `vocab_size` from the tokenizer's `config.json`, raise an exception immediately — don't start training against a mismatched tokenizer.
3. Proceed with training using `model_params` + `optimizer_params`.


## Open questions / risks

1. **Modal Volume placement** — `data/`, `tokenizers/`, and run output directories all need to live on a Modal Volume the remote container mounts. Not yet specified: which volume, mount path, and how local files get uploaded there. Also remember the Volume write-hang issue from `first-app.ipynb` — any new directory (e.g. `bin/{tokenizer_uid}/`) must be created explicitly before writing into it, since missing-parent-directory writes didn't fail fast.
2. **Run/checkpoint output location** — existing local runs write `config.json` + run summaries into per-run directories (`transformer/util.py`). Not yet specified whether Modal runs follow the same convention, or where those directories live relative to the Volume.
3. **`{tokenizer_uid}` generation scheme** — not yet specified how a uid is assigned (manual name, counter, hash of `vocab_size` + `special_tokens` + raw text path, etc.), or how collisions/reuse are handled if the preparation step is run again with identical settings.
4. **wandb step logging on resume, reverted for now** — `run_training` (`transformer/util.py`) logs every step to wandb via `wandb.log(row, step=step)`, but checkpoints only save every `save_every` steps. So resuming from an older checkpoint than the last step actually reached redoes some steps that wandb already has records for, and `step=` forces those redone points to violate wandb's monotonically-increasing step requirement — wandb silently drops them and prints a warning. A fix (`wandb.define_metric("step")` + `wandb.define_metric("*", step_metric="step")`, then dropping `step=` from `wandb.log`, decoupling wandb's own internal counter from our `step` field) was tried and reverted for now — revisit later. Two things to fix properly when we do:
   - The warning currently reappears on *every* run/resume that has this overlap, not just once.
   - The redo-consistency check (comparing every recomputed step's loss against `train.jsonl`'s existing rows) does a full linear scan of the whole file on every run start — redundant and grows unbounded with run history; only the overlapping tail actually needs checking.
