"""What one training run needs built, and where it lands.

Imported by both ends: the launcher writes runs/{run_id}/config.json and the
Snakefile derives its request from it with these functions, so the layout and the
encoder's address are defined once.

    <workflow>/sources.yaml                   the catalog: uid -> urls, tags
    {volume}/runs/{run_id}/config.json        the run, entire -- the only input
    {volume}/runs/{run_id}/run.yaml           what the DAG was asked to build

config.json is the whole of a run and is written once. run.yaml is derived from
it, by the DAG, as a record: nothing reads it back, so it can never disagree with
the config the way a hand-written copy could.

A run's request is per-run; what it builds is not. Artifacts are addressed by
content (source uid, encoder uid), so two runs asking for the same encoder share
its bins instead of each building a copy -- which is why data/ sits beside runs/
rather than inside one.

Layout under data/ (everything but sources/ is derived and safe to rm -rf):

    sources/{source_uid}/content.txt          raw text, EOS-separated
    encoders/{encoder_uid}/encoder.joblib     the fitted encoder
    encoders/{encoder_uid}/config.json        what it was fit from
    encoders/{encoder_uid}/{source_uid}.bin   that source, encoded by it
"""

import hashlib
import json
import shutil
from pathlib import Path

import yaml

# The catalog, beside the Snakefile: about the pipeline, not about one tree.
CATALOG_FILE = "sources.yaml"
# The run itself, written once by whoever starts it. The only per-run input.
CONFIG_FILE = "config.json"
# What the DAG was asked to build, derived from CONFIG_FILE. A record, not an input.
RUN_FILE = "run.yaml"
# Where run folders live under a volume. launcher/jobs.py names the same folder
# from the other side, where the volume is always /storage.
RUNS_DIR = "runs"


def encoder_uid(encoder: dict) -> str:
    """Content address of an encoder: {kind}-{first 8 hex of sha256}.

    The payload is spelled out field by field so a key that does not change the
    artifact cannot change its address -- including encoder_uid itself, which
    validate() re-derives from a spec that already carries one. fit_sources is
    sorted because their order does not affect the fit; params is not, because
    special_tokens is a list whose order sets the token ids.
    """
    payload = {
        "kind": encoder["kind"],
        "params": encoder["params"],
        "fit_sources": sorted(encoder["fit_sources"]),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"{encoder['kind']}-{digest[:8]}"


def spec(config: dict) -> dict:
    """A training config's `encoder` and `sources` blocks, verbatim, plus the
    encoder's derived address. Verbatim so there is no second vocabulary to keep
    in sync and `params` can carry whatever a kind needs."""
    return {
        "encoder": {**config["encoder"], "encoder_uid": encoder_uid(config["encoder"])},
        "sources": config["sources"],
    }


def validate(spec: dict, catalog: dict) -> None:
    """Reject a request the DAG would happily build the wrong thing from.

    The overlap rules are policy: an encoder that saw a validation source has the
    holdout in its merge table, and a validation loss measured against training
    text measures nothing.
    """
    encoder = spec["encoder"]
    train = spec["sources"]["train_sources"]
    valid = spec["sources"]["valid_sources"]
    fit = encoder["fit_sources"]

    assert train, "no train_sources"
    assert valid, "no valid_sources"
    assert fit, "encoder has no fit_sources"

    unknown = sorted((set(fit) | set(train) | set(valid)) - set(catalog))
    assert not unknown, f"not in the source catalog ({CATALOG_FILE}): {unknown}"

    # Named, because with several sources a side "there is an overlap" leaves you
    # diffing lists by eye.
    overlap = sorted(set(train) & set(valid))
    assert not overlap, f"train/valid overlap: {overlap}"
    leaked = sorted(set(valid) & set(fit))
    assert not leaked, f"valid sources in the encoder's fit set: {leaked}"

    # A uid that no longer matches its definition means a hand-edited run.yaml.
    expected = encoder_uid(encoder)
    assert encoder["encoder_uid"] == expected, (
        f"encoder_uid {encoder['encoder_uid']} does not match this definition ({expected})"
    )


def read_catalog(workflow_dir: Path) -> dict:
    """The catalog's sources, uid -> {urls, tags}. workflow_dir holds the
    Snakefile, this module and sources.yaml, and is passed rather than derived
    from __file__ so the caller says which pipeline it means."""
    return yaml.safe_load((Path(workflow_dir) / CATALOG_FILE).read_text())["sources"]


def run_dir(volume: Path, run_id: str) -> Path:
    """One run's folder: where its request lives and where the trainer writes.
    Both halves of a run_id's folder are named here, so the launcher and this
    pipeline cannot disagree about where a request was left."""
    return Path(volume) / RUNS_DIR / run_id


def read_config(rdir: Path) -> dict:
    """The run's config.json: the whole of what it is, and the only per-run input.

    Missing is an error, not an empty workflow -- a run that builds nothing and
    reports success is the worse failure.
    """
    path = Path(rdir) / CONFIG_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- write the run's config there before building it")
    return json.loads(path.read_text())


def write_config(config: dict, rdir: Path) -> Path:
    """Write config.json into a run folder, creating it.

    JSON-safe input only: pass a live notebook config through
    transformer.util.serialize_config first. Used by the notebook for local runs;
    on Modal the launcher writes this file from the dict it was handed.
    """
    path = Path(rdir) / CONFIG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))
    return path


def read_request(rdir: Path) -> dict:
    """This run's ETL request: config.json's `encoder` and `sources` blocks, plus
    the encoder's derived address. The only thing the DAG reads about a run.

    `spec` does the extracting and touches nothing else, so this reads a config
    written for training without knowing anything about training.
    """
    return spec(read_config(rdir))


def write_spec(spec: dict, rdir: Path) -> Path:
    """Write run.yaml into the run's own folder: a record of what the DAG was asked
    to build, derived from config.json. Nothing reads it back."""
    path = Path(rdir) / RUN_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# derived from {CONFIG_FILE} by {Path(__file__).name} -- a record, not an input\n"
    path.write_text(header + yaml.safe_dump(spec, sort_keys=False))
    return path


def source_content(data: Path, source_uid: str) -> Path:
    return Path(data) / "sources" / source_uid / "content.txt"


def encoder_dir(data: Path, encoder_uid: str) -> Path:
    return Path(data) / "encoders" / encoder_uid


def encoder_artifact(data: Path, encoder_uid: str) -> Path:
    return encoder_dir(data, encoder_uid) / "encoder.joblib"


def encoder_config(data: Path, encoder_uid: str) -> Path:
    return encoder_dir(data, encoder_uid) / "config.json"


def encoded_bin(data: Path, encoder_uid: str, source_uid: str) -> Path:
    """encoded sources live under the encoder that encoded them"""
    return encoder_dir(data, encoder_uid) / f"{source_uid}.bin"


def concat_bins(sources: list[Path], target: Path) -> Path:
    """Join encoded bins end to end into one file.

    PROVISIONAL, and the one thing here that lives outside the DAG. The pipeline
    builds one bin per source; a trainer wants one per split. Doing it here rather
    than as a rule means the bins are copied per run and snakemake does not know
    they exist -- both deliberate, so replacing this with a rule that produces one
    shared bin per (encoder, split) is a deletion rather than an untangling.

    Byte concatenation, so it is dtype-agnostic as long as every input shares one:
    they do, all written by the same encoder. Two consequences of joining encoded
    text rather than raw text: there is no separator token at the seams (the DAG's
    EOS only joins urls *within* a source), so a batch straddling one mixes two
    documents.

    Writes to a temp path and renames, so a trainer memory-mapping the target never
    sees a half-built file. Skipped when the target is already newer than every
    source, which is what makes calling it on every launch cheap.
    """
    sources = [Path(s) for s in sources]
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        newest = max(s.stat().st_mtime for s in sources)
        if target.stat().st_mtime >= newest:
            return target

    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "wb") as out:
        for source in sources:
            with open(source, "rb") as f:
                shutil.copyfileobj(f, out)
    tmp.rename(target)
    return target
