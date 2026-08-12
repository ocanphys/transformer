# Logs: what exists, what is missing, and what to build

Two asks drive this:

1. A **launcher log view** — everything `modal app logs training-launcher` shows, captured after
   the launcher starts, one file per session under a `launcher_logs/` folder on the Volume, with
   each line attributed to the function and container it came from.
2. **ETL logs on runs** — an `etl` call against a run folder should leave a log trail there, so the
   run's aggregated log view shows the ETL side beside the training side.

Both land on the same underlying problem: there are seven places that write something log-shaped,
three formats between them, and only one of the seven is durable and viewable. This document is the
inventory, the design that normalizes it, and a staged plan.

---

## 1. Inventory: every log sink today

| # | Producer | Destination | Format | Durable? | Viewable? |
|---|---|---|---|---|---|
| 1 | `logs.py::worker_logger`, via `work` and `Train` | `runs/{run_id}/logs/{call_id}/worker.log` **and** container stdout | `2026-08-12T10:15:00.123Z msg`, UTC, `LEVEL ` prefix when not INFO | yes (Volume) | yes — dashboard `/logs/{run_id}` |
| 2 | `Train.write_progress` | `runs/{run_id}/logs/{call_id}/progress.json` | JSON, last step only | background commits | yes — the ghost bar |
| 3 | `Train.flush_pending` | `runs/{run_id}/train.jsonl` | JSONL metrics | yes | yes — `/ledger/{run_id}` |
| 4 | `app.py::etl` and `gather_run_data` — four `print()` calls | container stdout | none | **no** | `modal app logs` only, until TTL |
| 5 | snakemake subprocess | container stdout/stderr | snakemake's own | **no** | `modal app logs` only |
| 6 | snakemake's own logfile (because `--directory /storage`) | `/storage/.snakemake/log/{ts}.snakemake.log` | snakemake's own | **yes, already** | **no** — orphaned, no `run_id` anywhere in the path |
| 7 | `transformer.util.run_training` | `{rdir}/events.log` | `%(asctime)s %(name)s %(levelname)s %(message)s`, **local time** | yes | no |
| 8 | `transformer.util.configure_logging` | root `basicConfig` → stderr | `%H:%M:%S name LEVEL msg`, **local time** | no | notebook only |
| 9 | `transformer.tokenizer`, `transformer.model` module loggers | nowhere in a container (no root handler); `lastResort` puts WARNING+ on stderr | varies | no | `modal app logs` |
| 10 | Modal itself — container lifecycle, tracebacks, everything above that reached stdout/stderr | Modal's log store, TTL'd | Modal's | **no** | `modal app logs` only |

Sink 6 is the surprise worth calling out: because `etl` passes `--directory /storage`, snakemake has
been writing a complete per-invocation logfile onto the Volume all along, and `volume.commit()` at
the end of `etl` publishes it. The detail exists. Nothing links it to a run and nothing reads it.

### What is wrong with this

- **P1 — three timestamp formats.** Sink 1 is UTC ISO with milliseconds; sinks 7 and 8 are local
  time in two other shapes. The dashboard's merge of a run's logs works *only* because every line
  in that tree is format 1 — it sorts the raw strings and calls that time order
  (`dashboard.read_logs`). Dropping a differently-formatted file into `logs/` silently corrupts
  that sort. The format is a contract and nothing enforces it.
- **P2 — the ETL is invisible.** Everything an ETL call did (which rules ran, what was refit, what
  a failure said) exists only in the container log and in an orphaned `.snakemake` file. A run's
  log page shows training and nothing before it — which is exactly the half that is slowest and
  most likely to be the thing that went wrong.
- **P3 — library records are dropped on purpose, and that now costs something.** `logs.py` captures
  only `worker.{call_id}` and its children, for a good reason (no call id, no honest place to file
  it). Inside the ETL container, `transformer.util.download_and_concat`'s `logger.info` lines
  therefore go nowhere at all: no root handler is configured there.
- **P4 — nothing persists the app-wide stream.** `modal app logs` is the only view of the launcher
  as a system, it is ephemeral, it cannot be diffed across sessions, and it has no grouping.
- **P5 — the naming says "worker" but the mechanism is general.** `worker_logger`,
  `release_worker_logger`, `WorkerFormatter`, logger name `worker.{call_id}` — "worker" is a
  specific role in this codebase (the training container). The ETL needs the identical mechanism
  and is not a worker.
- **P6 — stale docstring.** `logs.py` line 24 says trainer detail "has its own record in the run
  dir (`run_training` writes events.log)". Nothing on the Modal path calls `run_training`;
  `jobs.Train` replaced it. `events.log` is never written by a Modal worker.

---

## 2. Can we tell which container and function a log line came from?

Yes — completely, and it needs no parsing of text. Verified against **modal 1.5.2** by reading
`modal/_logs.py`, `modal/_output/pty.py`, `modal/cli/app.py`.

**Right now, with no code at all**, the CLI already surfaces it:

```
uv run modal app logs training-launcher \
    --timestamps --show-function-id --show-container-id --show-function-call-id
```

and it can be filtered server-side: `--function fu-…`, `--container ta-…`, `--function-call fc-…`,
`--source stdout|stderr|system`, `--search TEXT`, `--since 2h`, `--tail N`.

Underneath, `AppGetLogs` streams `TaskLogsBatch`, and every log line arrives with its provenance
already attached:

| Field | On | Meaning |
|---|---|---|
| `function_id` | batch | `fu-…` — **which function**: `launch`, `work`, `etl`, `dashboard` |
| `task_id` | batch | `ta-…` — **which container** |
| `function_call_id` | item | `fc-…` — **which call**; this is the same id the lease stores and the same id that names `runs/{run_id}/logs/{call_id}/` |
| `container_id` | item | container, again, per line (`container_name` exists but is empty in practice) |
| `file_descriptor` | item | stdout / stderr / system |
| `timestamp`, `timestamp_ns` | item | server-side time |
| `input_id` | item | `in-…`, which input within the call |
| `entry_id` | batch | the **cursor** — resume a stream exactly where it stopped |

`function_id` → readable name comes from `AppGetLayout(app_id).app_layout.function_ids`, which is a
plain `{tag: fu-id}` dict for the deployed app. So the map is `{fu-…: "etl"}` and it is fetched
once per session, not guessed.

`function_call_id` is the load-bearing one: it is **already the join key** between the app-wide
stream and the run tree. A line tagged `fc-XYZ` in a session log is the same attempt whose file is
`runs/{run_id}/logs/fc-XYZ/worker.log`.

Three APIs, two modes:

- **follow** — `client.stub.AppGetLogs.unary_stream(AppGetLogsRequest(app_id, timeout=55, last_entry_id=…))`.
  Long-poll. With an empty cursor it does **not** start at the tail — it replays the app's retained
  stream from the beginning (measured: the first batch of a fresh follow came back at the app's
  deploy time, not "now"). So a session started at noon captures the whole day for free, and
  persisting the cursor is what stops a restart from capturing it twice.
- **range** — `modal._logs.fetch_logs(client, app_id, since, until, filters=…)`, up to 35 days back.
  This is the gap-filler: a collector that died can backfill what it missed.
- **tail** — `modal._logs.tail_logs(client, app_id, n, …)`.

**Caveat, and it must be isolated:** `modal._logs` and `client.stub.*` are private. Same posture as
`app.status()` — one module, one "checked against modal 1.5.2" comment, one place to fix on upgrade.

### Verified, not assumed

Run against the live deployment. First from the laptop, then from inside a throwaway app's
container reading `training-launcher`'s logs — deliberately the *cross-app* case, which is stricter
than what the collector needs.

| | laptop (`CLIENT_TYPE_CLIENT`) | container (`CLIENT_TYPE_CONTAINER`) |
|---|---|---|
| `AppGetByDeploymentName` → `ap-…` | ok | ok |
| `AppGetLayout` → 4 function names | ok | ok |
| `AppGetLogs` (follow stream) | ok | **ok** |
| `AppFetchLogs` (range / backfill) | ok | ok |

**A container's own client can read app logs.** No `modal.Secret`, no
`_Client.from_credentials`, no local-collector fallback. Stage 3 is unblocked as designed.

A real captured line, which is the whole argument in one row:

```
etl  ta=ta-01KZVJYH4PA2EGZ96TN4X1MJGR  fc=fc-01KZVJYG5Q9DG72MDD15WA1G5F  out | $ snakemake all --snakefile /pipeline/Snakefile …
```

That is `etl`'s own `print(f"$ {command}")`, arriving with function, container and call id attached
— and that `fc-` is the same id that names `runs/{run_id}/logs/{call_id}/`.

Four things the probe settled that change the implementation:

- **`modal.App.lookup` cannot be used from async code.** Calling the blocking public wrapper from
  inside the synchronizer's loop raises `Exception: Deadlock detected: calling a sync function from
  the synchronizer loop`. `applog.py` uses the raw `AppGetByDeploymentName` proto call instead,
  exactly as `cli/app.py::resolve_app_identifier` does. (Not a permission problem — the probe's own
  bug, and worth a comment at the call site so nobody "simplifies" it back.)
- **`app_id` is not eternal.** Observed changing mid-spike: the app was stopped and redeployed, and
  the new record came back with `version: 1` and a fresh `created_at`. A follow stream on the old id
  ends with `app_done=True`. See stage 3.
- **`timestamp` is a float epoch** (`1786558500.015219`), so the writer converts with
  `datetime.fromtimestamp(ts, UTC)`.
- **System lines carry no `fc`.** Every `sys` entry (`GET /rows -> 200 OK`, container lifecycle) has
  an empty `function_call_id`; only user-code stdout/stderr carries one. The provenance column has
  to degrade cleanly, which the `{fn}/{ta6}/{fc6}` shape already does.

---

## 3. The target design

Two trees, and one rule that decides which is which.

> **Provenance lives in the path when there is a path, and in the line when there is not.**

A per-run log file already sits at `logs/{call_id}/…`, so its lines carry no ids — they would be the
same three tokens on every line. The app-wide session log has no per-call path, so provenance is
the second column of every line.

### Tree A — per run: "what happened to this run"

```
runs/{run_id}/
    config.json
    train.bin  valid.bin  encoder.json
    train.jsonl                       the metrics ledger
    checkpoints/
    logs/
        {call_id}/worker.log          a training attempt          (exists)
        {call_id}/etl.log             an ETL call for this run    (NEW)
        {call_id}/progress.json       live step                   (exists)
```

One file per call, named for the **actor** that made it. The call id is unique per call, so there is
exactly one `*.log` per directory; the filename is therefore free to carry the actor, and the
dashboard derives it with `path.stem` — no metadata file, no registry.

### Tree B — app-wide: "what happened on the launcher"

```
launcher_logs/
    {session_id}.log                  the captured stream for one session
    {session_id}.json                 its sidecar: app_id, started, ended, lines, bytes, cursor
```

`{session_id}` is `{YYYYMMDDTHHMMSS}Z`. A sidecar per session rather than one shared
`sessions.jsonl`: a session is then the only writer of its own two files, so there is no
read-modify-write of anything shared and no append log of status rows growing by one every ten
seconds. Listing sessions is a glob.

### The two line formats

Per-run — **unchanged**, so `dashboard.read_logs` keeps working:

```
2026-08-12T10:15:00.123Z boot call_id=fc-01JQ…
2026-08-12T10:15:02.884Z WARNING Train: loop exited before finishing …
```

Session — three space-separated columns, then free text; parsed with `line.split(" ", 2)`:

```
2026-08-12T10:15:00.123Z etl/ta-01KZVJYH4PA2EGZ96TN4X1MJGR/fc-01KZVJYG5Q9DG72MDD15WA1G5F Building DAG of jobs...
2026-08-12T10:15:01.515Z dashboard/ta-01KZVJXX7H9TE7QKB3JYSD9BPR/ GET /rows -> 200 OK
2026-08-12T10:15:02.015Z work/ta-01KZVBFC7ZCKR554RKT7THMFER/fc-112233 ! RuntimeError: CUDA out of memory
```

Column 2 is `{function}/{task_id}/{call_id}` — one token, greppable (`grep ' etl/'`), and it
degrades cleanly when a field is absent, which system lines always are (they carry no call id).

**Full ids, not shortened.** The file's job is to be grepped, and
`grep fc-01KZVJYG5Q9DG72MDD15WA1G5F session.log` is exactly the question you want to ask of it —
truncating to six characters would break the one operation the format exists for. The view
abbreviates for display, where space is the constraint instead.

stderr gets a `!` at the head of the message so a failure is visible when scanning; the view turns
it back into colour.

Both formats share column 1 exactly: fixed-width UTC ISO with milliseconds. That is the one
invariant the whole system rests on, and it is what makes any two files mergeable by string sort.

---

## 4. Implementation plan

Six stages. Stages 0–2 are the ETL ask and are independent of Modal internals; stages 3–4 are the
launcher-session ask. Each is separately deployable.

### Stage 0 — generalize `logs.py` from "worker" to "call"

*Files:* `launcher/logs.py`, `launcher/app.py`, `launcher/jobs.py` (docstrings).

The mechanism is per-call, not per-worker, and the ETL is about to become its second user.

| now | after |
|---|---|
| `worker_logger(call_id, logfile)` | `call_logger(call_id, logfile)` |
| `release_worker_logger()` | `release_call_logger()` |
| `WorkerFormatter` | `CallFormatter` |
| logger name `worker.{call_id}` | `call.{call_id}` |
| `_MARK = "_worker_handler"` | `_MARK = "_call_handler"` |

Pure rename; no behaviour changes. Sweep every mention in `app.py` (imports, `work`, module
docstring line 27) and the prose in `jobs.py`. Fix **P6** in the same pass: `logs.py`'s claim about
`run_training`/`events.log` is not true of any Modal path.

### Stage 1 — the ETL writes into the run's log tree

*File:* `launcher/app.py` (`etl`, `gather_run_data`).

```python
@app.function(image=etl_image, volumes={STORAGE: volume}, cpu=4, timeout=3600, max_containers=1)
def etl(run_id: str) -> dict:
    assert re.fullmatch(r"[A-Za-z0-9._-]+", run_id), …

    volume.reload()                       # must precede opening any file on the Volume
    if load_config(run_id) is None:       # NEW: refuse before creating a log dir under a
        raise RuntimeError(...)           # folder that isn't a run — it would show as "unreadable"

    call_id = modal.current_function_call_id()
    logdir = run_dir(run_id) / "logs" / call_id
    logdir.mkdir(parents=True, exist_ok=True)
    log = call_logger(call_id, logdir / "etl.log")
    try:
        ...                               # the body below
    finally:
        release_call_logger()             # load-bearing here — see note
```

Four changes inside the body:

1. **Tee snakemake instead of inheriting stdout.** `subprocess.run(command, shell=True)` sends every
   line straight to the container log and nowhere else. Replace with a streaming read:

   ```python
   process = subprocess.Popen(
       command, shell=True, text=True, bufsize=1,
       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
       env={**os.environ, "PYTHONUNBUFFERED": "1"},
   )
   for line in process.stdout:
       log.info(line.rstrip())
   returncode = process.wait()
   ```

   Each line now lands in `etl.log` **and** on stdout — that is what `call_logger`'s two handlers
   are for, so `modal app logs` loses nothing.

2. **Add `--nocolor`** to the snakemake command. It shells out through a pipe, but the flag makes
   "no ANSI escapes in the file" a guarantee rather than an observation.

3. **`gather_run_data(run_id, log)`** — take the logger, replace its two `print()`s with
   `log.info`. Same for the `$ {command}` print.

4. **Bracket it like a worker.** A `boot`-shaped opening line (`etl: run_id=… call_id=…`) and an
   `EXIT` line on both paths (`EXIT finished` / `EXIT snakemake exited 1`), so the merged run view
   reads with the same punctuation as the training half. The tee also captures snakemake's own
   `Complete log(s): /storage/.snakemake/log/….snakemake.log` line for free — which is how sink 6
   stops being orphaned: the pointer to it is now in the run's own log.

**Why the `finally` is load-bearing.** `etl` has no `single_use_containers`, so containers are
reused, and every call starts with `volume.reload()` — which **fails while any file under the mount
is open**. A leaked file handler from a previous call would break the next call outright. The
call-id-in-the-logger-name design already makes a leaked handler harmless for *correctness*; here
the `finally` is what keeps it from being fatal for *liveness*.

**Visibility during the run.** No extra commits needed: Modal mounts volumes with background
commits, which is what already publishes `progress.json` mid-run. A long encoder fit will stream
into the dashboard as it goes; the final `volume.commit()` makes it durable.

*Optional, cheap:* have the ETL write `logs/{call_id}/progress.json` too (`{"step": rules_done,
"total": rules_total}` parsed from snakemake's `N of M steps (X%) done` lines), which the dashboard's
ghost bar renders with no changes at all.

### Stage 2 — the run's log view shows both actors

*File:* `launcher/dashboard.py` (`read_logs`, `logs_page_html`).

```python
for path in sorted(logs_dir.glob("*/*.log")):
    call_id, actor = path.parent.name, path.stem     # "fc-…", "worker" | "etl"
```

Return `(ts, call_id, actor, msg)`; sort key stays `(ts, call_id)`. In the page: keep colour keyed
on `call_id` (it is what distinguishes overlapping attempts), add the actor as a small badge before
the message, and put actor filter chips in the header (`all · worker · etl`) as a client-side class
toggle — no new route.

Two things to add while in here:

- **Cap the render.** A snakemake run over a fresh encoder produces far more lines than a training
  attempt. Show the last ~5000 lines with a "showing last N of M" note. The current page has no cap
  either; the ETL is what will make that matter.
- The legend already shows `fc` suffixes; prefix each with its actor so `etl a1b2c3` and
  `worker d4e5f6` are distinguishable at a glance.

### Stage 3 — capture the launcher's own stream into sessions

*New file:* `launcher/applog.py`. *Files touched:* `launcher/app.py`, `launcher/jobs.py`
(one path constant).

`jobs.py` owns the Volume layout, so the constant goes there beside `RUNS`:

```python
LAUNCHER_LOGS = VOLUME / "launcher_logs"
```

`applog.py` — the only file that touches Modal's private log APIs:

```python
def resolve_app_id(name: str) -> str          # modal.App.lookup(name).app_id  (public API)
def function_names(client, app_id) -> dict    # AppGetLayout -> {fu-id: "etl", …}
async def follow(client, app_id, cursor="")   # yields Record(ts, function, ta, fc, fd, text, entry_id)
async def backfill(client, app_id, since, until)   # same Record shape, via _logs.fetch_logs
def format_record(r) -> str                   # "{ts} {fn}/{ta6}/{fc6} {text}"
class Session                                 # opens the file, appends, commits, maintains sessions.jsonl
```

and the collector in `app.py`:

```python
@app.function(
    image=base_image.add_local_python_source(*LOCAL_MODULES, "applog"),
    volumes={STORAGE: volume},
    timeout=21600,          # 6h; Modal's ceiling is 24h, so sessions chain rather than run forever
    max_containers=1,       # one session at a time, the same lock launch and etl use
    retries=0,
)
def collect() -> dict:
    """Follow this app's own logs into launcher_logs/{session}.log until the timeout."""
```

Five things this function must get right:

1. **Never echo a captured line to stdout.** The collector's own container is inside the app whose
   logs it is reading — echoing line *N* emits line *N+1*, forever. So it writes with a plain file
   handle, **not** `call_logger` (whose second handler is stdout, and which is correct everywhere
   else). Its own status lines — session opened, cursor committed, session closed — may go to
   stdout: a handful per session, and they *should* appear in the next session's capture.
2. **Cursor.** Persist `last_entry_id` into the session's `sessions.jsonl` row on every commit.
   This is load-bearing in a way the original design underestimated: an empty cursor replays the
   app's whole retained stream, so without a persisted cursor every restart re-captures everything
   it already has. With one, a restart resumes exactly.
3. **Backfill the gap.** On start, read the last session's row; if it has an `ended` time, fetch
   `[ended, now)` with `applog.backfill` and write it into the new file first, marked in the header.
   That closes the hole between two 6-hour sessions.
   *Given (2), most of this is free* — resuming from the cursor already replays what was missed, so
   the range fetch is only needed when the cursor is gone or the app id changed under it.
4. **Commit cadence.** Flush and `volume.commit()` every ~10 s or ~200 lines, whichever first;
   background commits cover the rest. The session log is readable in the dashboard while it is
   being written.
5. **Header.** A few `#`-prefixed lines at the top of each file: session id, app id, start time,
   modal client version, and the `fu-… → name` map as resolved. The reader skips `#`.
6. **Handle `app_done`.** An app id does not live forever — stopping and redeploying mints a new
   record, and the follow stream on the old id then terminates with `app_done=True`. On that
   signal: write a session-closing line, finalize the row, re-resolve the name to the new `ap-…`,
   and open a **new** session against it. A session belongs to one app id, which is why the id is
   in the header and in the index row.

Credentials are a non-issue — see §2. The container's own client is sufficient.

Starting a session: `modal run launcher/app.py::collect` (a local entrypoint that spawns it and
returns the call id), or a cell in `notebooks/launch.ipynb` beside the existing `deployed()` helper.
See the open question in §6 about whether it should instead be always-on.

### Stage 4 — the sessions view

*File:* `launcher/dashboard.py`.

- `GET /sessions` — the session list from `sessions.jsonl`: id, started, ended (or "live"), lines,
  and a link. A nav link to it in the runs page header.
- `GET /sessions/{id}` — the file, rendered like the run log page (reusing `LOG_CSS`), with
  colour keyed on **function** rather than call id, a stderr marker, and filter chips per function
  (`launch · work · etl · dashboard · system`).
- **The cross-link that makes this worth building:** `/sessions/{id}?fc=fc-XYZ` filters to one call.
  A run row's `call_id` is exactly that value, so the run log page gets a "see this attempt in the
  app stream" link, and the app stream shows what the run tree cannot — image pulls, container
  starts, OOM kills, tracebacks from before the logger was attached.

That last point is the real payoff. `work`'s docstring already names the one failure that can never
reach `worker.log`: a `volume.reload()` failure happens *before* the log file can be opened, and is
visible only in the container log. After stage 3 that container log is on the Volume, and after
stage 4 it is one click from the run.

### Stage 5 — retire the divergent formats

*Files:* `transformer/util.py`, `launcher/logs.py`.

- `run_training` / `events.log` (**sink 7**) is legacy: only `modal_lab/run.ipynb` still calls it,
  and the Modal path uses `jobs.Train`. Mark it legacy in its docstring and switch its formatter to
  the `CallFormatter` shape (UTC ISO ms) so any file it does leave sorts by the same rule as
  everything else.
- `configure_logging` (**sink 8**) — notebook-only, but there is no reason for a third format.
  Move it to `%(asctime)s.%(msecs)03dZ %(name)s %(levelname)s %(message)s` with
  `converter = time.gmtime` and `datefmt="%Y-%m-%dT%H:%M:%S"`, so a line copied out of a notebook
  and a line out of a Volume log read identically.

---

## 4b. As built — where the code differs from the plan above

All six stages are implemented. Five things came out differently, each for a reason found while
building it:

- **Sidecars, not `sessions.jsonl`.** Already folded into §3. One shared index would have needed a
  read-modify-write every ten seconds; a sidecar per session has exactly one writer.
- **Full ids in the session line, not six-character suffixes.** Also folded into §3: the file exists
  to be grepped, and truncation breaks the one operation it exists for.
- **`GET /trace/{call_id}` instead of a query-param filter.** The plan had
  `/sessions/{id}?fc=…`, which makes you find the session first. The route searches every session
  for one call id and merges what it finds, so a run's log page can link straight to it without
  knowing which session was running at the time.
- **Bursts render as blocks.** A line sharing both its timestamp *and* its source with the line
  above leaves the timestamp column empty and the line above drops its rule, so a traceback reads
  as one thing. Both keys, not just the timestamp: adjacent lines from two sources in the same
  millisecond are not one event. `dashboard.merged_lines_html` is shared by the run view and the
  session view so that rule cannot drift between them.
- **`transformer.util.utc_formatter`.** Stage 5 needed the timestamp contract in a third place, and
  `launcher/` is not importable from the installed package, so the shape is defined twice on
  purpose with the reasoning written at both sites.

One regression was introduced and fixed in the same pass, worth recording because it is the exact
shape of thing this refactor invites. `dashboard.scan` decided whether a worker had ever booted by
testing `runs/{run_id}/logs/` for existence — sound until stage 1, which made the *ETL* create that
folder first. Every freshly-built run would have shown **Stopped** with a Resume button instead of
**New** with a Start button. It now tests for `logs/*/worker.log`, which is the fact it actually
wanted: a file `work` writes and nothing else does.

## 5. Verification

| Stage | Check |
|---|---|
| 0 | `modal deploy launcher/app.py` succeeds; a run still writes `logs/{call}/worker.log`. |
| 1 | Locally: run the tee against a real `snakemake --nocolor` invocation, confirm lines arrive incrementally (not in one block at exit) and carry no escapes. On Modal: `modal volume ls test-volume runs/{id}/logs/{call}` shows `etl.log`; it contains the DAG output and the `Complete log(s):` pointer. |
| 2 | The run's `/logs/{run_id}` page opens with ETL lines first, worker lines after, one timeline. |
| 3 | Start `collect`, run a launch, confirm the session file has lines attributed to all four functions with correct names, and that the collector's own container appears without runaway repetition. Then kill the collector and restart it: the cursor must resume, not replay. |
| 4 | From a run row, the `fc` cross-link opens the session log filtered to that attempt. |
| 5 | `configure_logging()` in a notebook prints the same timestamp shape as a Volume log line. |

---

## 6. Risks and open questions

**Risks**

- *Private Modal APIs.* `modal._logs`, `client.stub.AppGetLogs`, `AppGetLayout`. Contained to
  `applog.py` with a version note, mirroring how `status()` handles the same exposure. This is now
  the **only** open exposure in stage 3.
- ~~*Container client permissions.*~~ Resolved — see §2. A container can read app logs.
- *Self-capture.* Mitigated by one rule (never echo), which is worth a comment at the write site
  since it is the non-obvious reason the collector cannot use `call_logger`.
- *Session file growth, mostly from the dashboard.* Measured on the live app: an **open dashboard
  tab** emits a `GET /rows -> 200 OK` system line per poll per warm container — with the 3 s
  refresh and several containers warm, that is the dominant source of volume, on the order of a few
  MB/day, and it continues whether or not anything is training. Capture it anyway (the session log
  is the raw record), but the view should default to hiding `fd=system` from `dashboard`, and a size
  roll (new file at ~32 MB) is worth having sooner than "when it first matters".
- *Cost.* The collector is a small always-warm CPU container for as long as a session runs. Real,
  small, and worth deciding deliberately — see the question below.
- *Log TTL.* `TaskLogsBatch.ttl_days` tells us how far back backfill can reach. A collector down
  longer than that loses that window permanently; the session index will show the gap honestly
  rather than pretending it was quiet.

**Decided — what starts a session: explicit start.**

A session is *one run of the collector*, started when you start working:
`modal run launcher/app.py::collect`, or the notebook cell beside `deployed()`. Each call is one
session, capped by the function's 6 h timeout. No background container, so the capture costs
nothing when nobody is working, and a gap between two sessions is a real fact recorded in
`sessions.jsonl` rather than something the design pretends away.

Two alternatives were weighed and rejected for now, both reachable later without changing the file
layout or the reader: an **always-on** `modal.Period` supervisor that chains sessions and backfills
between them (misses nothing, runs a container continuously), and a **per-launch** collector
started by `launch` itself (captures exactly the interesting windows, but puts the collector's
lifetime inside the launcher's single-container lock). The backfill machinery in stage 3 is what
either upgrade would build on, which is why it is in the plan from the start.
