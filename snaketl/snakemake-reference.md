# Snakemake — What You Need to Know

## 1. The core model

A Snakefile is a set of **rules**. A rule says: *"here is a file pattern I can produce, here are the files I need to do it, here is the code."*

You never tell Snakemake what order to run things. You ask for a **file**, and Snakemake works backwards: which rule produces it, what does that rule need, which rule produces *those*, recursively, until it reaches files that already exist. That recursion is the DAG. Execution then flows forward, in parallel where the graph allows.

**The currency is filenames, not variables.** Everything a job needs to know must be recoverable from the path it was asked to produce.

## 2. Two phases

**Parse time** — the Snakefile *is* a Python file, exec'd top to bottom. Imports, `configfile:`, constants, helper functions, `for` loops all run in written order. Rules are only *registered*: their patterns are recorded, nothing executes. Even `params=lambda w: ...` is stored uncalled.

**Build time** — Snakemake matches your target to a rule, resolves dependencies recursively, then runs jobs in DAG order.

So **file order does not determine execution order.** In the pipeline below, `fit_encoder` runs before `encode_source` because `encode_source` lists `tokenizer.joblib` as an input — not because it appears earlier. Three exceptions where order matters: top-level Python (a name must exist before use), and the first rule being the default target (avoid this dependency with `default_target: True`).

## 3. Anatomy of a rule

```python
rule encode_source:
    input:
        text=".../sources/{source_uid}/content.txt",
        tokenizer=".../tokenizers/{encoder_uid}/tokenizer.joblib",
    output:
        bin=".../bins/{encoder_uid}/{source_uid}.bin",
    params:
        spec=lambda wildcards: config["encoders"][wildcards.encoder_uid],
    threads: 1
    resources: mem_mb=4000
    retries: 3
    run:
        ...  # plain Python
```

| Directive | Meaning |
|---|---|
| `input:` | Files that must exist first. Creates the DAG edges. |
| `output:` | Files produced. **Declares the wildcards.** |
| `params:` | Non-file values. Compared between runs → triggers reruns. |
| `threads:`/`resources:` | Scheduler budget, subtracted from `-j`. |
| `retries:` | Re-attempt on failure. Failed jobs' outputs are deleted. |
| `run:` vs `shell:` | Python body vs shell command. |

## 4. Wildcards

A wildcard is a **named hole in an output path**. One rule text becomes many jobs.

Three distinct moments:

| | when | what happens |
|---|---|---|
| **Declared** | parse time | `output: ".../{source_uid}/content.txt"` names the hole |
| **Bound** | DAG build, per job | request `.../odyssey/...` → `source_uid = "odyssey"` |
| **Accessed** | per job | `wildcards.source_uid` |

Binding is **reverse string interpolation**: `{source_uid}` sits where `odyssey` sits, so matching reads the value *out*. Under the hood it's a regex named group — hence values are **always strings** (`shard_{n}` gives `"003"`, not `3`).

Once bound, the values substitute into `input:` and `params:`, producing new targets, and the recursion continues.

**Rules:**
- Wildcards must appear in `output:`. Inputs *consume* them; outputs *define* them. A wildcard only in `input:` is an error.
- Scope is the **job**, not the rule. Three sources = three independent `wildcards` objects.
- Same name in two rules = unrelated variables.
- Not globs. Nothing scans the disk. A wildcard is filled only with values something *asked for*.
- Default regex is greedy (`.+`) and matches `/`. Always constrain:
  ```python
  wildcard_constraints: source_uid="[^/]+", encoder_uid="[^/]+"
  ```

**The object:** `snakemake.io.Wildcards`, a `list` subclass with a name→index map. Attributes are set for real at construction, so a typo raises `AttributeError`, not `None`.

```python
wildcards.encoder_uid    # "base"
wildcards["source_uid"]  # "odyssey"
dict(wildcards.items())  # {"encoder_uid": "base", "source_uid": "odyssey"}
```

## 5. The three costumes

The same bound value, three syntaxes:

```python
output: ".../{source_uid}/content.txt"                    # path strings — implicit
params: urls=lambda wildcards: cfg[wildcards.source_uid]  # lambdas — argument
run:    Tokenizer.from_files(wildcards.encoder_uid)       # run: block — injected name
```

**`{braces}` work only in directive strings and in `shell:`.** They do **not** work in `run:`, because `run:` is real Python — `{encoder_uid}` there is set-literal syntax and raises `NameError`.

Snakemake injects into every `run:` block, undeclared: `input`, `output`, `params`, `wildcards`, `threads`, `resources`, `log`, `config`, plus the Snakefile's globals.

## 6. Why lambdas

`params:` and `input:` are written **once**, at parse time, when no job exists. You cannot write `config["sources"][source_uid]` — that name has no value yet. The lambda **defers** the lookup: a recipe saying *"when you know which job this is, do this."* Snakemake calls it once per job with that job's wildcards.

**Use a lambda only when the dependency requires a lookup.** Compare:

```python
# encode_source — ZERO lookups. Both coordinates are in the output path itself.
input: text=".../sources/{source_uid}/content.txt"

# fit_encoder — needs a lookup. "base" cannot be textually rewritten
# into its fit_sources; the config must be consulted.
input: lambda w: [f".../sources/{s}/content.txt"
                  for s in config["encoders"][w.encoder_uid]["fit_sources"]]
```

Input lambdas run at **DAG-build time**, so they may read config and the filesystem — never the *contents* of inputs, which may not exist yet.

## 7. `rule all` and targets

```python
rule all:
    input: [f"sources/{uid}/content.txt" for uid in config["sources"]]
```

No `output:`, no action — it exists purely to hold a list of requests. It's the default target because it's first (or via `default_target: True`). Its comprehension evaluates at parse time into literal paths; **each becomes a target and re-enters rule matching.** That fan-out is the demand that binds every wildcard downstream.

It's a *query over the config*: add a source, bare `snakemake` picks it up.

**Targets are files first, rule names second.** `snakemake data/sources/odyssey/content.txt` always works. Named aliases are ergonomics:

```python
for uid in config["sources"]:
    rule:
        name: uid                    # single token — CLI args split on whitespace
        input: f"sources/{uid}/content.txt"
```

CLI args are all targets; there is no subcommand grammar. `snakemake source odyssey` means "build two targets." Mark aliases and `all` as `localrules:`.

## 8. What triggers a rerun

The DAG is built from **existence and structure only** — no timestamps. Staleness then decides which nodes actually run:

1. **Output missing** or incomplete (interrupted mid-write)
2. **mtime** — any input newer than any output. Transitive: refit encoder → every `.bin` stale → every manifest stale.
3. **params** — evaluated params differ from last run's recorded metadata. *This is why config reads belong in `params:`, not hidden inside `run:` where they're invisible.*
4. **input** — the *set* of input filenames changed (not contents)
5. **code** — the rule's own body changed
6. **software-env** — conda/container spec changed

Triggers 3–6 need `.snakemake/` metadata; delete it and only existence + mtime remain.

**Blind spot:** trigger 5 hashes the `run:` block, **not** imported library code. Editing `train_bpe` inside your package triggers nothing. Use `-R <rule>` after library edits.

## 9. Config

`configfile:` loads YAML/JSON into the global `config` dict. Multiple `configfile:` lines **merge**, so splitting by edit-frequency is trivial. Precedence: `configfile:` < `--configfile` < `--config key=value`.

Keep uids as **dict keys**, not list entries — identities belong as keys, and duplicates become YAML errors instead of silent bugs.

**Validate at parse time.** Everything cross-cutting is in `config` before any DAG exists, so a bad config can fail in milliseconds:

```python
def validate(config):
    for uid, s in config["splits"].items():
        fit = set(config["encoders"][s["encoder"]]["fit_sources"])
        assert not set(s["train"]) & set(s["valid"]), f"{uid}: train/valid overlap"
        assert not set(s["valid"]) & fit, f"{uid}: valid in fit set"

validate(config)
```

`snakemake.utils.validate(config, schema)` does this via JSON Schema and also **injects defaults** for missing keys.

## 10. Parallelism

The design is parallel-safe when every job writes files no other job touches; the DAG serializes the real orderings. Then parallelism is just `-j`.

```python
rule write_content:
    resources: net=1      # snakemake --resources net=2 caps concurrent downloads
rule fit_encoder:
    threads: 8            # reserve real cores; scheduler subtracts from -j
rule encode_source:
    threads: 1            # embarrassingly parallel
```

Snakemake writes outputs **in place** — it will not do atomic replacement for you. If a reader may hold a file mapped, write to a temp path and `os.replace` inside your own code.

## 11. Commands

```bash
snakemake -n -r          # dry run + REASON per job — the debugging tool
snakemake -j 8           # 8 cores
snakemake --dag | dot    # the graph Snakemake actually derived
snakemake -R fit_encoder # force a rule and everything downstream
snakemake -F             # force everything
snakemake --touch        # update mtimes without running
snakemake --lint         # catches config-in-run:, missing logs, etc.
```

`-n -r` prints exactly which trigger fired per job. Never guess why something rebuilt.

## 12. Practical shape

- **Thin `run:` blocks.** Unpack wildcards, call your library. One line. The Snakefile stays orchestration; the logic stays unit-testable without Snakemake.
- **Make the package installable** (`pip install -e .`). Then `import mypackage` works identically in the Snakefile, in tests, and in downstream scripts — no `sys.path` juggling.
- **Label outputs** (`output.bin`) rather than `output[0]`; positional indices get fragile the moment there's a second output.
- **`snakemake -n` in CI** — parses, validates config, builds the full DAG. A free integration test of everything except the science.
- **Test input lambdas directly:** `snakemake.io.Wildcards(fromdict={"source_uid": "odyssey"})`.
- **Reorder rules to read like the pipeline.** It's free, and it's for the human.
