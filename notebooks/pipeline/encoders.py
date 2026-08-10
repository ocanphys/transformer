"""Encoder kinds: how to fit one, and how to encode a source with it.

The workflow calls fit() and encode() and knows only the `kind` string in the
config. Everything specific to a kind -- what its params mean, what its artifact
holds, what class does the encoding -- stops here. A new kind is one entry in
KINDS.
"""

from pathlib import Path

import joblib

from transformer.tokenizer import Tokenizer, train_bpe
from transformer.util import textfile_to_tokens_as_binary


def _fit_bpe(params, sources: list[str], artifact: str) -> None:
    """params are train_bpe's keyword arguments: vocab_size, special_tokens."""
    # train_bpe pretokenizes each file and sums the counts, so a list fits on
    # them as one corpus without concatenating first.
    vocab, merges = train_bpe(sources, **params)
    joblib.dump((vocab, merges), artifact)


def _load_bpe(params, artifact: str) -> Tokenizer:
    vocab, merges = joblib.load(artifact)
    return Tokenizer(vocab, merges, params["special_tokens"])


# kind -> how to fit it, how to load it back. Loaded encoders need only an
# .encode_iterable for encode() below.
KINDS = {"bpe": {"fit": _fit_bpe, "load": _load_bpe}}


def _kind(definition: dict, what: str):
    kind = definition["kind"]
    if kind not in KINDS:
        raise ValueError(f"cannot {what} encoder kind {kind!r}; registered: {sorted(KINDS)}")
    return KINDS[kind][what]


def fit(definition: dict, sources: list[str], artifact: str) -> None:
    """Fit the encoder `definition` describes on `sources`, writing `artifact`."""
    _kind(definition, "fit")(definition["params"], sources, artifact)


def encode(definition: dict, artifact: str, source: str, target: str) -> None:
    """Write `source` as a token stream at `target`, using the fitted `artifact`.

    `definition` is the encoder's own config.json, so this works for any encoder
    on disk, not just the one the current run asked for.
    """
    encoder = _kind(definition, "load")(definition["params"], artifact)
    # All four paths are absolute here; joining an absolute path onto volume
    # discards volume, which is what makes "/" the right root to pass.
    textfile_to_tokens_as_binary(
        source_text=source,
        binary_target=target,
        tokenizer=encoder,
        volume=Path("/"),
    )
