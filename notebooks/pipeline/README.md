# The pipeline, and what it costs

A training config names sources and describes an encoder. Everything else --
which files exist, what depends on what, what is already built -- is derived.

```
etl_train_pipeline.ipynb                                   Snakefile
  config  --serialize_config()-->  runs/{run_id}/config.json  --read_request()--> DAG
                                        ^                     |
                                        |                     v
  sources.yaml (source catalog: uid -> urls)          runs/{run_id}/run.yaml
```

Two files feed the workflow, they are different kinds of thing, and they sit at
different levels for that reason. `sources.yaml` is a catalog: what a source uid
*means*, hand-maintained, adding an entry builds nothing. It lives beside the
Snakefile because it describes the pipeline, not a tree. `runs/{run_id}/config.json`
is one whole run, written once, of which this workflow reads only the `encoder` and
`sources` blocks. So `--config volume=X run_id=Y` picks the tree and the run within
it while the catalog stays with the workflow -- on Modal the volume carries what to
build, the image carries what a uid means.

There is exactly one per-run input. `run.yaml` is an *output*: the request as the
Snakefile resolved it, written back as a record by `rule run_spec`. Nothing reads
it, so it cannot drift from the config the way the hand-written copy it replaced
could.

`run_id` is the same key the launcher runs under (`launcher/jobs.py` names the
same `runs/{run_id}/` folder), so one identifier covers a run from its ETL to its
checkpoints. There is no default: a request lives in exactly one folder, and
guessing which would build somebody else's run.

What a run *asks for* is per-run; what gets *built* is not. Artifacts are
addressed by content -- source uid, encoder uid -- and live under `data/`, beside
`runs/` rather than inside one, so two runs wanting the same encoder share its
bins instead of each building a copy.

## How the DAG gets built

Three rules, no encoder or split named in any of them:

```
sources.yaml urls → sources/{source_uid}/content.txt
                     |  fit_sources          |  train + valid
                     v                       v
   encoders/{encoder_uid}/encoder.joblib → encoders/{encoder_uid}/{source_uid}.bin
```

`rule all` asks for this run's bins. Each bin declares the encoder as an input,
the encoder declares its fit sources as inputs, and a source has no inputs at
all -- its raw material is a url, which is why it is a leaf. Snakemake walks
that backwards and builds only what it reaches.

The dependency is *dynamic* in one specific place: `fit_encoder`'s input is a
lambda over `wildcards.encoder_uid`, so the set of `content.txt` files it needs
is not known until a path is requested. That is what removes the hand-written
edge -- `snakemake train` is enough to download two books, fit a tokenizer on
one of them, and encode both.

## The address

```python
encoder_uid = f"{kind}-{sha256({kind, params, sorted(fit_sources)})[:8]}"
```

Derived, never typed. Two runs asking for the same encoder land in the same
directory and share one fit; change a special token and the address moves, so
the bins under the old one cannot be silently reused. The bins live *under* the
encoder that produced them, so a `.bin` cannot be read as something it is not.

The payload is *only* what defines a tokenizer -- the `encoder` block's `kind`,
`params` and `fit_sources`. Nothing about the model reaches it: `d_model`,
`num_layers`, lr, batch size, seed, `total_steps` and the wandb metadata all
change without moving the address, so a sweep over model shapes fits one encoder
and encodes one set of bins. The fields are listed explicitly rather than
hashing the block whole, which is also why a `description` key added beside them
does not move the address and why `encoder_uid` can be re-derived from a spec
that already carries one. `vocab_size` is written twice -- once for the
embedding table, once for the fit -- and only the encoder's copy is hashed; the
notebook asserts they agree rather than deriving one from the other.

`kind` selects an implementation from `encoders.py`, which is where every
encoder-specific thing lives: what the params mean, what the artifact holds,
what class does the encoding. The workflow calls `encoders.fit()` and
`encoders.encode()` and knows nothing else -- a second kind is one entry in
`KINDS` and no change to the rules. Each encoder's `config.json` is what
`encode_source` reads, not `run.yaml`, so bins under an older encoder still
build.

## Strengths

- **Nothing to keep in sync.** There is no list of encoders or splits to edit
  alongside the config. Delete a source from `train_sources` and the DAG is
  smaller on the next run; no file says otherwise.
- **Leakage is a parse-time error.** Train/valid overlap and a holdout in the
  fit set are asserted before a single byte is fetched, and the message names
  the offenders. This is the check that is invisible once it is a `.bin`.
- **Sharing is automatic and safe.** Content addressing means reuse happens when
  the definitions match and never when they do not -- the usual failure (a
  hand-named `bpe-5k` directory holding a tokenizer fit on something else)
  cannot be expressed.
- **Layout defined once.** `etl.py` is imported by the notebook that writes the
  spec and the Snakefile that reads it, so the writer and the reader cannot
  disagree about where a file goes.
- **Staleness is checksum-backed.** Snakemake records a sha256 per input, so a
  refetch returning identical bytes does not cascade a refit. Only real byte
  changes propagate: a fit source invalidates the encoder and everything under
  it, a non-fit source only its own bin.
- **Artifacts are write-once, and a changed source is not retroactive.** Outputs
  are `protected()` (chmod `r--r--r--`, so nothing *can* rewrite them) and
  inputs are `ancient()` (so nothing *tries*). Edit a source and rerun: nothing
  happens, no error, no refit -- verified, byte-identical bins and an untouched
  encoder. The change shows up only in what is built afterwards, so deleting one
  bin and rebuilding re-encodes it from the new bytes while its neighbours keep
  the old. Missing files are unaffected by any of this: a source that does not
  exist is still downloaded, and a new source still encodes under the existing
  encoder.

## Simplifications

Deliberate cuts, each with a reason and a cost:

- **One encoder per run.** `encoder_definition()` raises for any uid other than
  the current run's, because a hash cannot be inverted back into the config that
  produced it. Cost: no sweep over encoders in a single DAG -- rerun the notebook
  cell per encoder.
- **A split is a query, not an artifact.** `train`/`valid` are just two lists of
  sources; nothing is concatenated or copied per split. Cost: the trainer has to
  read several memmaps instead of one file.
- **`uint16`, everywhere, implicitly.** `textfile_to_tokens_as_binary` hardcodes
  it and readers assume it. Fine at `vocab_size: 5000`.
- **Asserts instead of a schema.** Validation is ten lines in `etl.py`, run both
  in the notebook and at workflow parse time.
- **`run:` blocks, not scripts.** The rules call library functions in
  snakemake's own process. Simple, and it means jobs cannot be given a conda or
  container environment.
- **Sources are expendable once encoded.** `ancient()` means an existing bin
  never looks at its source, so deleting `sources/{uid}/content.txt` is a valid
  way to reclaim disk and the bin stays usable. Cost: a plain run reports
  "nothing to do" on a tree that is missing sources, and only says otherwise
  when something actually needs one -- `snakemake {uid}` fetches it back.

## Weaknesses

- **The address covers the recipe, not the data.** `encoder_uid` hashes source
  *names*, not their bytes, so editing a url leaves the uid meaning a different
  corpus than it did. Freezing keeps that from corrupting anything already
  built, but it does not record it: nothing on disk says which bytes a bin came
  from.
- **A directory can hold bins from different vintages.** The flip side of the
  freeze. Change a source, add another source to the run, and the new bin is
  encoded from the new bytes while the old ones are not -- one `encoder_uid`,
  two versions of the corpus, no marker saying so. Deliberate, and the reason a
  per-bin record of what it was encoded from would be worth having.
- **Nothing is collected.** Encoder directories accumulate, and bins for sources
  dropped from a run are never deleted -- only `rm -rf` shrinks the tree.
  Harmless, but disk grows monotonically with the number of configs tried.
- **Staleness is unreportable.** `ancient()` suppresses the question, so
  `snakemake -n` can no longer tell you that a bin no longer matches its source
  -- it says "nothing to do" either way. Answering that needs a comparison the
  workflow does not make.
- **Editing a rule cannot rebuild anything.** `rerun-triggers` drops `code` in
  the profile, because a rule-body edit would otherwise ask to rewrite a
  protected file and come back as an error. Fix a bug in the encoding and
  nothing regenerates until you delete what it produced.
- **A partially written bin can still be read.** Protection closes the overwrite
  case; it does not make creation atomic. A trainer that opens a bin while it is
  first being written maps a truncated file. A temp path plus `os.replace` would
  close that; the workflow does not.
- **No `.meta.json`.** `n_tokens`, `dtype` and document offsets are not
  recorded. `doc_offsets` in particular is only obtainable at encode time, so
  document-boundary masking later means re-encoding everything.
- **`config.json` is a parse-time input, not a tracked one.** The shape of the DAG
  is read out of it before any rule runs, but only `run_spec` declares it as an
  input -- so editing a run's config changes what the DAG *is* without invalidating
  any encoder or bin. That is what `encoder_uid` being a hash covers: a changed
  encoder gets a different address rather than a stale artifact.
- **The splits are joined outside the DAG.** `etl.concat_bins`, called by
  `launcher/app.py`'s `gather_run_data`, copies a run's per-source bins into
  `runs/{run_id}/train.bin` and `valid.bin` because the trainer wants one file per
  split. Provisional and deliberately separable: the bytes are duplicated per run,
  snakemake does not know those files exist, and the sources abut with no separator
  token at the seams, so a batch straddling one mixes two documents. Replacing it
  with a rule that produces one shared bin per (encoder, split) deletes that
  function and `jobs.split_bin`.
- **Two ways to load an encoder in the repo.** This workflow loads artifacts
  explicitly from their paths; `Tokenizer.from_files` still looks in
  `volume/tokenizers/{uid}`, which this layout does not produce. Nothing in
  `launcher/` reads that path any more.
- **Catalog uids share a namespace with rule names.** Every source becomes a
  target so `snakemake odyssey` works; an assert rejects a uid that would shadow
  `all`, `encoder`, `train`, `valid` or `run_spec`. It works, but it is a namespace
  hack.
