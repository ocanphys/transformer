# launch spec — deployed app + folder-as-source-of-truth

Toy target: `scale-experiment` (the `count_to` job in launch.ipynb). It rehearses
the folder-as-state idea, but it is **deliberately self-contained and different
from web.py**: no `summary.json`, no `events.log`. Only three files exist per run,
and status is reconstructed from two orthogonal signals — folder progress and
Modal call liveness. Because there is no crash-marker file, liveness is not
optional here; it is the only thing that can tell a dead run from a slow one.

Contrast with the real system ([web.py](web.py), [util.py](../transformer/util.py)):
web.py leans on `summary.json` (write-once "done") and an `events.log` crash
string. This toy has neither — so it must be robust without them. That is the
whole point of the exercise.

---

## 1. The one rule: the folder *is* the state. No sidecar DB.

Every durable fact about a run lives in exactly one file under `runs/<run_id>/`,
with exactly one writer and one write-discipline. Status is **derived by
scanning**, never stored.

| Fact                | File                          | Writer                    | Discipline                          |
|---------------------|-------------------------------|---------------------------|-------------------------------------|
| identity            | folder name                   | spawner mints, job mkdirs | immutable (primary key)             |
| **current call id** | `modal_function_call_id.txt`  | the job (startup)         | **overwrite** (1 line, every spawn) |
| progress record     | `count.jsonl`                 | the job                   | **append-only**, one line = one step|
| config              | `config.json`                 | the job (first attempt)   | immutable — contains `max_steps`    |

Two derived facts fall out of these three files and one Modal API call — they are
**never written down**:

- **complete?** — `tail(count.jsonl).i >= config.max_steps`
- **alive?** — `FunctionCall.from_id(<call id>).get(timeout=0)` (see §2)

`run_id` format: `{timestamp}_{name}` (e.g. `20260730T142211_alpha`). The spawner
mints it and passes it to `count_to.spawn(run_id, max_steps)`; the **job** creates
the folder and writes `config.json` + `modal_function_call_id.txt` itself at
startup, then commits so they're visible to the dashboard.

### Why the *job* writes its own config + call id (reverted to web.py's approach)

An earlier draft had the FastAPI launcher write them (it has `call.object_id` the
instant `.spawn()` returns). But launching moved **out** of the endpoint — there's
no launch form; runs are spawned from the notebook (§4). And the volume path
`/storage` only exists *inside a container*, so a notebook client can't write
`config.json` or the call-id there at all. So the job writes both at startup, in
the container, via `modal.current_function_call_id()` — exactly what web.py's real
trainer does. Launching from anywhere is then just `spawn(run_id, max_steps)`.

"Overwrite every time a function is associated (fresh **or** resuming)" still holds
for free: each (re)spawn is a new container → new `current_function_call_id()` →
overwrites the file. The one cost is a brief startup window where a just-spawned
job hasn't written its id yet — harmless here (the row simply reads **starting**).

---

## 2. Status derivation — two orthogonal signals, one matrix

On every dashboard refresh:

1. `volume.reload()` — the dashboard container's mount is a snapshot from its last
   reload; other containers write from elsewhere, so without this you serve stale
   files.
2. For each `runs/<run_id>/`, gather the two signals below.
3. Combine them into a status via the matrix. **Never store the result** — recompute.

### Signal A — progress (folder only, cheap)

```python
latest    = tail_last_json(rdir / "count.jsonl")            # §3, may be None
max_steps = json.loads((rdir / "config.json").read_text())["max_steps"]
complete  = latest is not None and latest["i"] >= max_steps
```

### Signal B — liveness via `status()` (asks Modal; only call it when you must)

A non-blocking poll of the real call. This is exactly `launchpad.status()` (see
[launchpad.py](launchpad.py), pinned by [test_launchpad.py](test_launchpad.py)),
verified against modal 1.5.2 — it maps every outcome of `FunctionCall.get(timeout=0)`
to one of six states:

| `status()`  | what `get(timeout=0)` did                | means                                   |
|-------------|------------------------------------------|-----------------------------------------|
| `running`   | raised builtin `TimeoutError` (no output)| still executing                         |
| `done`      | returned cleanly (value may be `None`)   | finished successfully                   |
| `expired`   | raised `OutputExpiredError`              | result garbage-collected — outcome lost |
| `timed_out` | raised `FunctionTimeoutError`            | exceeded its own `timeout=`             |
| `failed`    | raised other modal `Error` (RemoteError…)| infra fault **or a cancel**             |
| `crashed`   | re-raised the function's own exception   | your code raised                        |

Catch order matters: `OutputExpiredError`/`FunctionTimeoutError` subclass *modal's*
`TimeoutError`, which is **not** the builtin — so `except TimeoutError` isolates
only `running`, and those two must be caught before the generic `Error` branch.

The dashboard recovers the call from the id file and owns the missing-id case
(nothing to ask Modal about):

```python
async def liveness(call_id: str | None) -> str:
    if not call_id:
        return "unknown"                       # no id recorded — can't ask Modal
    _, state, _ = await status(modal.FunctionCall.from_id(call_id))
    return state                               # one of status()'s six states
```

**Optimization / robustness:** if Signal A already says `complete`, *skip Signal B
entirely* — a run with all its steps written is done regardless of what Modal
remembers (call results expire after a while; the folder does not). Only query
liveness for **not-complete** runs. That keeps the scan O(active runs) on the
Modal API, and means a finished run stays "completed" forever even after its call
id is garbage-collected server-side.

### The matrix (this is the robust part)

`complete` short-circuits — every step written means done no matter what Modal
remembers, so **don't even poll**. Only not-complete runs reach `status()`:

| progress        | `status()`                          | dashboard status | meaning                                             |
|-----------------|-------------------------------------|------------------|-----------------------------------------------------|
| complete        | *(skipped)*                         | **completed**    | all `max_steps` written — folder is ground truth    |
| partial (i<max) | `running`                           | **in_progress**  | call alive, steps still landing                     |
| partial         | `done`                              | **failed**       | returned early, steps unfinished ⇒ died / cancelled |
| partial         | `failed` / `crashed` / `timed_out`  | **failed**       | call ended abnormally — includes **cancel**         |
| partial         | `expired`                           | **orphaned**     | result GC'd; outcome unrecoverable — treat as stale |
| partial         | `unknown` (no id)                   | **orphaned**     | not done, nothing to poll                           |
| no rows yet     | `running`                           | **starting**     | container booting, no `count.jsonl` yet             |
| no rows yet     | `done`/`failed`/`crashed`/`timed_out`| **failed**      | ended before writing a single step                  |
| no rows yet     | `expired` / `unknown`               | **orphaned**     | can't tell                                          |

The rows web.py **cannot** produce without a crash-marker file are the abnormal-end
ones (`partial × {failed, crashed, timed_out} → failed`): a container hard-killed
by OOM / preemption / cancel never writes a marker, yet `status()` reports the call
is over while the step count is short — so we flag it for free. That is why liveness
is load-bearing in this toy, not a refinement.

**Verified (2026-07-30):** cancelling a live run drives it to `failed`, not
`running`. [test_launchpad.py](test_launchpad.py)'s integration run observed the
trajectory `running → running → failed` within seconds of `.cancel()`, and Modal
logged "Successfully canceled input" — so the `partial × failed → failed` row *is*
the cancel case, confirmed. Plain `.cancel()` interrupts the loop; no
`terminate_containers=True` needed (settles the §5 caveat below).

---

## 3. Progress = tail the last line, don't parse the whole file

`count.jsonl` grows one line per step; the dashboard only needs the last one.
Seek from the end. Hardened against a **partial final line** (an append or commit
caught mid-write leaves a truncated last line — common when reading a live file):

```python
def tail_last_json(path: Path) -> dict | None:
    """Last complete JSON object in a .jsonl, or None. Tolerates a torn final
    line by falling back to the previous line."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path, "rb") as f:
        f.seek(0, 2)
        end = f.tell()
        f.seek(max(0, end - 8192))              # last 8 KB is plenty for a few lines
        lines = f.read().splitlines()
    for raw in reversed(lines):                 # walk back past any torn last line
        raw = raw.strip()
        if not raw:
            continue
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    return None
```

`latest["i"]` is the live step number. Elapsed = `latest["timestamp"]` minus the
first row's timestamp (read the first line separately if you want it — one small
read from the top).

### ⚠️ The real freshness ceiling isn't the reader — it's commit cadence

The dashboard only sees what's been **committed** to the volume. A job appending
to `count.jsonl` on its own container's disk is invisible to the dashboard until
`volume.commit()`. So the knob that governs how live your progress is:

- **Toy** (1 line/sec): `volume.commit()` every ~10 lines → ≤10 s lag. Cheap.
- **Real GPU run**: commit every ~30–60 s of steps is plenty; committing every
  single step would swamp the volume with writes.

Pair it with `volume.reload()` on the reader (§2). Writer commits + reader reloads
are the two halves of visibility — miss either and the dashboard lies. **This
cadence, not the tail helper, is the single biggest lever on "how fresh is the
latest step."**

---

## 4. App shape — notebook-hosted, deployed with `app.deploy()`

Everything lives in [launch.ipynb](launch.ipynb) (not a separate `launch.py`): the
`count_to` job, the `dashboard` endpoint, a deploy cell, and a launch cell. A
*served* FastAPI endpoint needs the app persistently running, which a notebook
`with app.run()` block can't provide — so a cell calls `app.deploy()`, which
registers the app server-side and keeps the endpoint live after the notebook stops.
`.spawn()`/`.cancel()` then work from anywhere without the
`with app.run(detach=True)` wrapper (launchpad decision 1). Notebook-defined
functions can't be imported by reference, so they're all `serialized=True`.

**The job** (`count_to`) self-writes `config.json` + its own call-id at startup
(§1), then counts:

```python
@app.function(volumes={"/storage": volume}, timeout=3600, scaledown_window=2, serialized=True)
def count_to(run_id, max_steps=20):
    rdir = RUNS / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    if not (rdir / "config.json").exists():                # first attempt fixes max_steps
        (rdir / "config.json").write_text(json.dumps({"max_steps": max_steps}))
    cid = modal.current_function_call_id()                 # overwritten every (re)spawn
    if cid:
        (rdir / "modal_function_call_id.txt").write_text(cid)
    volume.commit()                                        # make config + id visible now
    ...  # resume from tail, append {i, timestamp} rows, commit every COMMIT_EVERY
```

**The endpoint** (`dashboard`) is read-only + cancel + resume — no launch route. Its
image ships `launchpad.py` (`add_local_python_source("launchpad")`) so it can
`from launchpad import status`. It handles cancel/resume, serves a `?partial=1` rows
fragment for the poll (§6), and otherwise renders the page:

```python
@app.function(image=dashboard_image, volumes={"/storage": volume}, serialized=True)
@modal.fastapi_endpoint()
async def dashboard(action="", run_id="", partial=0):
    volume.reload()
    if action == "cancel" and run_id:                      # recover id, cancel, redirect
        cid = RUNS / run_id / "modal_function_call_id.txt"
        if cid.exists():
            modal.FunctionCall.from_id(cid.read_text().strip()).cancel()
        return RedirectResponse("./", 303)
    if action == "resume" and run_id:                      # respawn -> continues from tail
        count_to.spawn(run_id)                             # keeps config, writes a fresh id
        return RedirectResponse("./", 303)
    rows = scan_runs()                                     # §2 matrix over every folder
    return HTMLResponse(_rows_html(rows) if partial else _page(_rows_html(rows), len(rows)))
```

**Resume** is just a respawn: `count_to` reads the existing `config.json`, continues
from the `count.jsonl` tail, and overwrites the call-id. It's offered exactly for
runs that are **neither alive nor complete** — `failed` or `orphaned` in §2 terms.
Never for `starting` (that run is alive/booting; a second spawn would race it into
the same `count.jsonl`) nor `completed`.

`scan_runs()` is the §2 loop: per folder read `config.json`, tail `count.jsonl`,
poll `status()` *only if not complete*, map to the matrix. `?partial=1` returns
just the `<tbody>` — that's what the page polls (§6).

**Launching** happens from a notebook cell against the deployed function — the
caller writes nothing to the volume, since the job self-writes its config/id:

```python
count_to_fn = modal.Function.from_name("scale-experiment", "count_to")
for name, steps in [("alpha", 20), ("beta", 40)]:
    count_to_fn.spawn(f"{datetime.now():%Y%m%dT%H%M%S}_{name}", steps)
```

Deploy: run the `app.deploy()` cell; it prints `dashboard.get_web_url()` (stable
across redeploys).

---

## 5. Does `cancel` interrupt a sync loop? — RESOLVED (2026-07-30): yes

launchpad decision 2's caveat: does `.cancel()` actually **interrupt** a sync job
mid-loop, or just mark the input cancelled while the container runs to completion?
This was load-bearing: the whole "failed" detection in §2 assumes a stopped run
transitions out of `running`. If `.cancel()` left the container happily counting,
liveness would stay `running` (dashboard stuck at **in_progress**) and the run
would actually **complete**, contradicting the user's intent to stop it.

**Answer: `.cancel()` interrupts it.** [test_launchpad.py](test_launchpad.py)'s
integration test spawns a real counter, cancels it, and polls `status()`: the call
went `running → running → failed` within seconds, with Modal logging "Received a
cancellation signal" → "Successfully canceled input". Plain `.cancel()` is enough —
**no `terminate_containers=True`, no cooperative stop-flag file** in the loop. The
cancelled call surfaces as `failed` (a cancelled input is a non-success terminal
output → `get()` raises `RemoteError` → `status()` maps it to `failed`), which is
exactly the `partial × failed → failed` row in §2.

**Scope:** proven for a plain `@app.function()` (serialized). The real
`Trainer.train` is a `@modal.method()` on an `@app.cls()` — also sync, same
signal mechanism, so it should carry over; web.py's separate note is only that the
server rejects `terminate_containers=True` for class methods, which we no longer
need. Confirming the class-method case directly is the one remaining gap.

---

## 6. Dashboard UI — the simple version

One server-rendered HTML page from the `dashboard` endpoint (§4), plus one tiny
poll script — otherwise no framework, no client state. Everything is derived from
the volume on each request (`volume.reload()` → scan `runs/` → §2 matrix + §3 tail),
so the folders stay the only source of truth; the page holds nothing but the rows
it last rendered.

### What it shows — one row per `runs/<run_id>/`

| Column       | Source                        | Notes                                   |
|--------------|-------------------------------|-----------------------------------------|
| **Run**      | folder name                   | plain text (no link — keep it simple)   |
| **Status**   | §2 matrix                     | colored badge                           |
| **Count**    | `tail(count.jsonl).i`          | latest committed step — the live value  |
| **Progress** | `i / config.max_steps`         | `1,234 / 10,000 (12%)` + a thin bar     |
| **Action**   | status (§2)                    | **Cancel** if running, **Resume** if failed/orphaned |

The Action cell is state-driven: `in_progress` (with a call-id) → **Cancel**
(`?action=cancel`, recovers the id and `.cancel()`s — proven to stop the run, §5);
`failed`/`orphaned` → **Resume** (`?action=resume`, respawns `count_to` to continue
from the tail, §4); `completed` and `starting` → `—`. One button at a time, never
both.

### Mock

```
Training runs (4)                                       auto-refreshes every 5s

Run                    Status       Count    Progress                    Action
────────────────────────────────────────────────────────────────────────────────
20260730T142211_run    ● running     3,150   ███░░░░░░░  3,150 / 10,000  [Cancel]
20260730T140902_run    ● completed  10,000   ██████████ 10,000 / 10,000     —
20260730T139001_run    ● failed      4,517   ████░░░░░░  4,517 / 10,000  [Resume]
20260730T138800_run    ● starting      —     ░░░░░░░░░░      — / 10,000     —
```

### Freshness & refresh — smooth in-place poll, not a full reload

- A ~15-line script polls `fetch('?partial=1')` every `refresh_ms` (default 5 s)
  and swaps only the `<tbody>` innerHTML (plus the run count). No full-page reload,
  so no white flash and no scroll jump; the progress bars glide via a CSS
  `transition`. Each poll still re-runs the **whole server scan** — the endpoint
  just returns the rows fragment instead of the page.
- Cancel stays a plain `<a href="?action=cancel&run_id=…">`: a normal navigation →
  303 redirect → the row flips to **failed** on the next poll.
- How live **Count** is stays bounded by commit cadence, not the poll (§3): the
  number only moves as fast as the job commits `count.jsonl`. Keep `refresh_ms` ≥
  the job's commit interval (`COMMIT_EVERY`), or you're just re-fetching the same row.

### Status → badge (toy palette)

Reuse web.py's `STATUS_STYLE` shape, mapped to §2's dashboard statuses:

| status        | label      | tone       |
|---------------|------------|------------|
| `completed`   | Completed  | green      |
| `in_progress` | Running    | amber      |
| `failed`      | Failed     | red        |
| `starting`    | Starting   | grey       |
| `orphaned`    | Orphaned   | muted grey |

Rendering splits in two, both plain string-builds like web.py's: `_rows_html(rows)`
emits just the `<tbody>` rows (returned alone for `?partial=1`), and `_page(...)`
wraps them with the header, CSS, and poll script. Keep the light/dark
`@media (prefers-color-scheme)` CSS. A full request is `volume.reload()` →
`scan_runs()` → `_page(_rows_html(rows), n)`; a poll is the same minus the `_page`
wrapper.
