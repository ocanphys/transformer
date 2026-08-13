"""A page showing every run: how far it got, whether it is alive, and two buttons.

All the hard parts already exist in app.py. Progress comes from the job, liveness
from Modal, ownership from the Dict, and resuming is just a call to the launcher.
This file only reads those and draws them.

Status is worked out fresh on every request and stored nowhere. It has to be: a
run can die -- crash, preemption, cancel -- with nobody left to write down that it
died, so any saved status would be wrong exactly when it matters most.

Imported and served by app.py's `dashboard` function. fastapi is imported inside
build_web_app so this module can be imported locally, where fastapi isn't
installed.
"""

import asyncio
import json
from pathlib import Path

import modal

import applog
from app import JOB, APP_NAME, lease_key, leases, status, volume
from jobs import RUNS, Aborted, load_config, run_dir

REFRESH_MS = 3000  # how often the page asks for fresh rows

# How many log lines one page will draw. A training attempt writes a line per
# checkpoint; an ETL that has to fit an encoder writes snakemake's entire account
# of the DAG, which is what makes a cap worth having. The newest lines are the
# ones kept -- a log is read from the end.
MAX_LOG_LINES = 5000

# What each derived status is called on screen. The colour lives in the stylesheet
# as `.st-<status>`, so light and dark can differ -- plain coloured text has to
# carry itself against the page background, with no chip behind it to sit on.
STATUS_LABEL = {
    "completed": "Completed",
    "in_progress": "Running",
    "starting": "Starting",
    "stopped": "Stopped",
    "new": "New",
    "unreadable": "Unreadable",
}

CANCELLABLE = {"in_progress", "starting"}
RESUMABLE = {"new", "stopped"}

# Drawn in currentColor so they inherit hover states. Deliberately different
# shapes rather than different colours: three lines for a log, bars for a ledger.
LOGS_ICON = (
    "<svg width='13' height='13' viewBox='0 0 16 16' fill='none' stroke='currentColor' "
    "stroke-width='1.7' stroke-linecap='round'><path d='M2.6 4h10.8M2.6 8h10.8M2.6 12h6.6'/></svg>"
)
LEDGER_ICON = (
    "<svg width='13' height='13' viewBox='0 0 16 16' fill='none' stroke='currentColor' "
    "stroke-width='1.7' stroke-linecap='round'><path d='M3 13V7M8 13V3M13 13V9'/></svg>"
)


def derive_status(complete: bool, done: int, live: str | None, started: bool) -> str:
    """Fold progress, liveness and history into one word.

    `complete` wins outright and is decided from committed files alone -- a
    finished run must read finished forever, and Modal eventually forgets old call
    outcomes, so we never ask about them.

    Otherwise `live` says whether anyone is on it, and when nobody is, `started`
    separates the two very different reasons for that:

      `new`     -- no worker has ever booted here. The normal look of a run whose
                   ETL is still building, or one whose data is ready and whose
                   launcher has not spawned yet. Nothing has gone wrong.
      `stopped` -- it ran, and it is not running now.

    `stopped` covers a crash, a cancel, a preemption and an expired lease alike,
    on purpose. The dashboard cannot honestly tell them apart -- Modal forgets old
    call outcomes, and a lease the launcher popped looks exactly like one that was
    never granted -- and all of them want the same thing done about them. The
    worker's own log is where which-one-it-was is written down.
    """
    if complete:
        return "completed"
    if live == "running":
        return "in_progress" if done else "starting"
    return "stopped" if started else "new"


def live_step(rdir: Path) -> int | None:
    """The furthest step any attempt has written to its progress file, or None.

    Read off the volume along with the rest of a run's state, and gated by
    nothing -- not liveness, not the lease. This is display only; `JOB.progress`
    still reads checkpoints, because only those are durable.

    The file carries no commit of its own and rides Modal's background commits,
    so it is exactly as fresh as those happen to be. Nothing here waits on that
    or works around it: when there is nothing to read, the ghost bar simply does
    not draw. Missing is the ordinary case, not an error -- a job that writes no
    progress file, an attempt that has not reached its first step, or a commit
    that has not landed yet.

    Stale is ordinary too, and needs no guarding: `rows_html` only draws a step
    that is *ahead* of the last checkpoint, so a dead attempt's leftover file
    stops showing by itself the moment a later checkpoint passes it. Taking the
    furthest step rather than the current attempt's is what lets this ignore the
    lease entirely; the cost is that a superseded attempt which got further than
    the live one keeps its ghost until the next checkpoint.
    """
    steps = []
    for progress in rdir.glob("logs/*/progress.json"):
        try:
            steps.append(json.loads(progress.read_text())["step"])
        except (OSError, ValueError, KeyError):
            continue  # torn mid-write, or not a progress file we understand
    return max(steps, default=None)


async def scan() -> list[dict]:
    """One row per run folder, newest first.

    Reloads the volume first: this container is long-lived, so its snapshot is
    stale by default -- the same reason `launch` and `work` reload.

    Liveness is asked only about runs that are not finished, and all of those
    questions go out together, so the page costs one round-trip rather than one
    per run.
    """
    await volume.reload.aio()

    runs = []
    for rdir in sorted(RUNS.iterdir(), reverse=True) if RUNS.exists() else []:
        if not rdir.is_dir():
            continue
        run_id = rdir.name
        config = load_config(run_id)
        try:
            if config is None:
                raise Aborted("config.json missing or unparseable")
            done, total = JOB.progress(run_dir(run_id), config)
        except Aborted:
            # Either no readable definition, or one this job does not understand --
            # older folders written by other tools live here too. Listed, but with
            # no progress and no buttons: the launcher refuses these outright, so
            # offering a Resume would promise something that cannot happen.
            runs.append({"run_id": run_id, "status": "unreadable", "done": None, "total": None, "attempt": None})
            continue

        # A worker's own log file, not the `logs/` folder: the ETL writes
        # `logs/{call_id}/etl.log` into that same folder before any worker is
        # spawned, so the folder's existence stopped meaning "a worker booted here"
        # the moment the ETL started logging. `worker.log` is written by `work` and
        # by nothing else, which is the fact this actually wants.
        #
        # Not "config.json is the only file" either, for the same reason: the ETL
        # leaves run.yaml, encoder.json and the bins here too, so a run between a
        # finished build and its first worker would read as started.
        runs.append(
            {
                "run_id": run_id,
                "done": done,
                "total": total,
                "complete": done >= total,
                "started": any(rdir.glob("logs/*/worker.log")),
                "live_step": live_step(rdir),
            }
        )

    pending = [r for r in runs if "complete" in r and not r["complete"]]
    grants = await asyncio.gather(*(leases.get.aio(lease_key(r["run_id"])) for r in pending))
    live = await asyncio.gather(
        *(status(modal.FunctionCall.from_id(g["call_id"])) if g else _no_lease() for g in grants)
    )

    for row, grant, state in zip(pending, grants, live):
        row["attempt"] = grant["attempt"] if grant else None
        row["status"] = derive_status(False, row["done"], state, row["started"])

    for row in runs:
        row.setdefault("status", "completed" if row.get("complete") else "stopped")
        row.setdefault("attempt", None)
    return runs


async def _no_lease():
    """Stand-in for a liveness answer we never asked for, so the gather above
    stays index-aligned with its rows."""
    return None


async def read_logs(run_id: str) -> list[tuple[str, str, str, str]]:
    """Every log in one run's folder, merged into one stream in time order.

    Each line is written as `<iso-timestamp> <message>`, so the timestamp is
    simply the first column. They are all UTC and all the same width, which means
    sorting the strings sorts the times -- no parsing into datetimes needed.

    Two actors write here and the path says which: `logs/{call_id}/worker.log` is a
    training attempt, `logs/{call_id}/etl.log` is the build that prepared its data.
    One Modal call is one directory, so the directory holds exactly one file and
    its *name* is free to say who wrote it -- no manifest, nothing to keep in sync.

    Interleaving is the whole point. Calls overlap: a superseded worker is still
    logging while its replacement boots, and a resume's ETL runs while the previous
    attempt's file sits beside it. Merged and sorted, that reads as one story
    instead of three files to line up by eye.

    Returns (timestamp, call_id, actor, message) per line.
    """
    await volume.reload.aio()

    logs_dir = run_dir(run_id) / "logs"
    if not logs_dir.exists():
        return []

    lines = []
    for log in sorted(logs_dir.glob("*/*.log")):
        call_id, actor = log.parent.name, log.stem
        last_ts = ""
        for line in log.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            ts, _, msg = line.partition(" ")
            if ts[:1].isdigit():
                last_ts = ts
            else:  # no leading timestamp: keep it next to the line it follows
                ts, msg = last_ts, line
            lines.append((ts, call_id, actor, msg))

    lines.sort(key=lambda row: (row[0], row[1]))
    return lines


async def read_ledger(run_id: str) -> list[str]:
    """Every committed line of train.jsonl, in the order it was written.

    Left as raw text on purpose. The ledger is whatever the job chose to append,
    and rendering it as text shows exactly that -- including a torn final line or
    a repeated step, which any attempt to parse it would tidy away.

    Only committed lines exist here. A worker's most recent interval lives in its
    container until the boundary commit, so this always lags the true step count
    by less than one checkpoint.
    """
    await volume.reload.aio()

    ledger = run_dir(run_id) / "train.jsonl"
    if not ledger.is_file():
        return []
    return [line for line in ledger.read_text(errors="replace").splitlines() if line.strip()]


# --------------------------------------------------------------------------------------
# markup
# --------------------------------------------------------------------------------------


def rows_html(runs: list[dict]) -> str:
    """Just the <tbody> contents -- served on its own to the poller, so refreshing
    never reloads the page or moves the scroll position."""
    import html

    if not runs:
        return "<tr><td colspan='5' class='empty'>No runs yet.</td></tr>"

    out = []
    for r in runs:
        status_key = r["status"]
        label = STATUS_LABEL.get(status_key, status_key)
        name = html.escape(r["run_id"])
        done, total = r.get("done"), r.get("total")

        if done is None or not total:
            progress, bar = "—", ""
        else:
            pct = min(100.0, 100 * done / total)
            progress = f"{done:,} / {total:,}"
            # The bar is committed progress; the ghost ahead of it is where the
            # live attempt has got to since its last checkpoint. Two different
            # facts -- one survives a crash, the other does not -- so they are
            # drawn differently rather than added together.
            #
            # The caption is deliberately not gated on the ghost having width.
            # A boundary writes the progress file and the checkpoint at the same
            # step, and the commit that publishes the checkpoint publishes that
            # file with it -- so at every boundary the two numbers coincide and
            # the ghost has nothing left to draw. Gated on the same test, "· at N"
            # blinked out at each boundary and came back a step later. A reading
            # level with the checkpoint is still a reading, and saying so keeps
            # the row the same shape from one poll to the next.
            step = r.get("live_step")
            ghost = ""
            if step is not None and step >= done:
                progress += f" <span class='muted'>· at {step:,}</span>"
                if step > done:  # a zero-width ghost draws the same as no ghost
                    ghost = f"<div class='fill live' style='width:{min(100.0, 100 * step / total):.1f}%'></div>"
            bar = f"<div class='bar'>{ghost}<div class='fill' style='width:{pct:.1f}%'></div></div>"

        # `data-busy` is the label to show while the POST is in flight, carried on
        # the element so the script never has to know the status vocabulary.
        if status_key in CANCELLABLE:
            action = (
                f"<button class='cancel' data-act='cancel' data-busy='Cancelling…' data-run='{name}'>Cancel</button>"
            )
        elif status_key in RESUMABLE:
            # One route and one call: `launch` does not distinguish starting a run
            # from resuming one, and must not -- deciding which it is happens inside
            # its lock, on evidence this page is too far away to have. Only the word
            # differs, because to someone reading the row they are different events.
            verb, busy = ("Start", "Starting…") if status_key == "new" else ("Resume", "Resuming…")
            action = f"<button class='resume' data-act='resume' data-busy='{busy}' data-run='{name}'>{verb}</button>"
        else:
            action = ""

        # Plain links. The browser already knows how to open a tab, and each page
        # wants its own address so it can be reloaded, shared, and left open beside
        # the dashboard.
        logs = f"<a class='icon' href='logs/{name}' target='_blank' rel='noopener' title='Worker logs'>{LOGS_ICON}</a>"
        ledger = (
            f"<a class='icon' href='ledger/{name}' target='_blank' rel='noopener' title='train.jsonl'>{LEDGER_ICON}</a>"
        )

        out.append(f"""
      <tr>
        <td class='mono'>{name}</td>
        <td><span class='st st-{status_key}'>{label}</span></td>
        <td>{bar}<span class='mono small'>{progress}</span></td>
        <td class='mono small'>{r["attempt"] or "—"}</td>
        <td class='actions'>{action}{logs}{ledger}</td>
      </tr>""")
    return "".join(out)


# Shared by both pages, so the logs view cannot drift away from the run list. A
# plain string, not an f-string, so the braces stay braces.
THEME_CSS = """
  :root { color-scheme: light dark; --fg:#1f2328; --bg:#fff; --muted:#57606a; --line:#d8dee4; --accent:#0969da; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e6edf3; --bg:#0d1117; --muted:#8b949e; --line:#30363d; --accent:#4493f8; }
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:2.5rem 1.5rem; background:var(--bg); color:var(--fg);
         font:13px/1.35 -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
  main { max-width: 60rem; margin: 0 auto; }
  header { display:flex; align-items:baseline; gap:.75rem; margin-bottom:1.25rem; }
  h1 { font-size:1.05rem; font-weight:600; margin:0; letter-spacing:-.01em; }
  a { color:var(--accent); text-decoration:none; }
  a:hover { text-decoration:underline; }
  .muted { color:var(--muted); }
  .small { font-size:.75rem; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
"""


def page_html(runs: list[dict]) -> str:
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Runs</title>
<style>{THEME_CSS}
  @media (prefers-color-scheme: dark) {{
    .st-completed {{ color:#3fb950; }}
    .st-in_progress {{ color:#d29922; }}
    .st-stopped {{ color:#f85149; }}
    .st-unreadable {{ color:#bc8cff; }}
  }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ text-align:left; font-size:.68rem; text-transform:uppercase; letter-spacing:.04em;
        color:var(--muted); font-weight:600; padding:0 .75rem .5rem; border-bottom:1px solid var(--line); }}
  td {{ padding:.3rem .75rem; border-bottom:1px solid var(--line); vertical-align:middle; white-space:nowrap; }}
  td:first-child, th:first-child {{ padding-left:0; }}
  td:last-child, th:last-child {{ padding-right:0; text-align:right; }}
  .empty {{ color:var(--muted); padding:2rem 0; text-align:center; }}
  .st {{ font-weight:600; font-size:.78rem; }}
  .st-completed {{ color:#1a7f37; }}
  .st-in_progress {{ color:#9a6700; }}
  .st-stopped {{ color:#cf222e; }}
  .st-unreadable {{ color:#8250df; }}
  /* `new` is not an outcome and nothing has gone wrong in it -- it reads like
     `starting`, quiet, rather than competing with the states that did happen. */
  .st-starting, .st-new {{ color:var(--muted); font-weight:500; }}
  /* inline, not stacked above the numbers -- stacking is what made rows tall */
  .bar {{ display:inline-block; vertical-align:middle; margin-right:.5rem; width:110px; height:4px;
          border-radius:2px; background:var(--line); overflow:hidden; position:relative; }}
  /* both fills sit at left:0 and overlap; the solid one is later in the DOM, so it
     paints over the ghost and what shows past it is the uncommitted stretch */
  .fill {{ height:100%; background:#2da44e; transition:width .4s ease; position:absolute; left:0; top:0; }}
  .fill.live {{ opacity:.3; }}
  button {{ font:inherit; font-size:.75rem; padding:.1rem .5rem; border-radius:5px; cursor:pointer;
            border:1px solid var(--line); background:transparent; color:var(--accent); }}
  button:hover {{ border-color:var(--accent); }}
  button:disabled {{ opacity:.5; cursor:default; border-color:var(--line); }}
  button.cancel {{ color:#cf222e; }}
  button.cancel:hover {{ border-color:#cf222e; }}
  .actions {{ display:flex; gap:.4rem; align-items:center; justify-content:flex-end; }}
  .icon {{ display:inline-flex; align-items:center; padding:.2rem .3rem; border-radius:5px;
           color:var(--muted); border:1px solid transparent; }}
  .icon:hover {{ color:var(--accent); border-color:var(--line); text-decoration:none; }}
</style></head>
<body><main>
  <header>
    <h1>Runs</h1>
    <span class='muted small' id='meta'>{len(runs)} runs</span>
    <span class='muted small'>· <a href='modal'>modal logs</a></span>
  </header>
  <table>
    <thead><tr><th>Run</th><th>Status</th><th>Progress</th><th>Attempt</th><th></th></tr></thead>
    <tbody id='rows'>{rows_html(runs)}</tbody>
  </table>
</main>
<script>
const rows = document.getElementById('rows'), meta = document.getElementById('meta');

async function refresh() {{
  try {{
    const r = await fetch('rows', {{cache: 'no-store'}});
    if (!r.ok) return;
    rows.innerHTML = await r.text();
    const n = rows.querySelectorAll('tr').length;
    meta.textContent = `${{n}} runs · updated ${{new Date().toLocaleTimeString()}}`;
  }} catch (e) {{}}
}}

// One listener for the whole table, so it keeps working after rows are replaced.
// The logs link is left alone -- it is an ordinary link and the browser handles it.
rows.addEventListener('click', async (e) => {{
  const b = e.target.closest('button[data-act]');
  if (!b) return;
  b.disabled = true;
  b.textContent = b.dataset.busy;
  try {{ await fetch(`${{b.dataset.act}}/${{b.dataset.run}}`, {{method: 'POST'}}); }} catch (e) {{}}
  refresh();
}});

setInterval(refresh, {REFRESH_MS});
</script>
</body></html>"""


# One colour per call, assigned in order of appearance. The point is not
# decoration: when two calls overlap, colour is what lets you see the handover
# at a glance instead of comparing call ids character by character.
CALL_COLOURS = ["#0969da", "#8250df", "#bf8700", "#1a7f37", "#cf222e", "#0f766e"]


# Every column is a fixed width. Not cosmetic: with `flex:0 0 auto` the badge grew
# with its text, so a `dashboard` line and an `etl` line started their messages at
# different columns and a scrolling log read as a zigzag. Widths are in `ch`, which
# in a monospace face is exactly one character.
LOG_CSS = """
  .log { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px;
         border-top:1px solid var(--line); }
  .line { display:flex; gap:.75rem; padding:.08rem 0; border-bottom:1px solid var(--line); }
  .line:hover { background:color-mix(in srgb, var(--fg) 5%, transparent); }
  .line.hi .msg { font-weight:600; }
  .line.off { display:none; }
  /* the last line of a same-instant burst keeps the rule; the ones above it drop
     it, so a traceback reads as one block instead of a stack of framed rows */
  .line.nb { border-bottom:none; }
  /* column headings, kept in view while a long log scrolls under them */
  .line.head { position:sticky; top:0; background:var(--bg); color:var(--muted);
               font-size:.66rem; text-transform:uppercase; letter-spacing:.05em; font-weight:600;
               padding:.2rem 0; border-bottom:1px solid var(--line); }
  .line.head:hover { background:var(--bg); }
  /* fixed, because a continuation line leaves it empty and the columns still have
     to line up under the timestamp they belong to */
  .ts { color:var(--muted); flex:0 0 19ch; }
  .num { color:var(--muted); flex:0 0 3rem; text-align:right; }
  .att { flex:0 0 10ch; font-weight:600; font-size:.72rem; letter-spacing:.02em;
         overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .actor { flex:0 0 7ch; color:var(--muted); font-size:.72rem;
           overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .msg { white-space:pre-wrap; word-break:break-word; }
  .msg.err { color:#cf222e; }
  @media (prefers-color-scheme: dark) { .msg.err { color:#f85149; } }
  .empty { color:var(--muted); padding:2rem 0; }
  .legend { display:flex; gap:.9rem; margin-bottom:.8rem; font-size:.72rem; }
  .chips { display:flex; gap:.4rem; margin-bottom:.8rem; align-items:center; flex-wrap:wrap; }
  .chip { font:inherit; font-size:.72rem; padding:.1rem .55rem; border-radius:999px; cursor:pointer;
          border:1px solid var(--line); background:transparent; color:var(--muted); }
  .chip:hover { border-color:var(--accent); }
  .chip.on { color:var(--accent); border-color:var(--accent); }
  label.chip { display:inline-flex; gap:.35rem; align-items:center; }
  label.chip input { margin:0; }
"""


def merged_lines_html(rows: list[dict]) -> str:
    """The `.line` divs for any merged log, with same-instant bursts drawn as blocks.

    Shared by the run view and the session view, so the one rule they both depend
    on cannot drift between them: a line that shares its timestamp *and* its
    `group` with the line above leaves the timestamp column empty, and the line
    above drops its rule. A traceback or a snakemake job table then reads as one
    block instead of a stack of framed rows repeating the same millisecond.

    Both keys, not just the timestamp. Lines are sorted by time, so two different
    sources logging in the same millisecond land adjacent -- and drawing those as
    one block would claim a relationship between them that isn't there.

    Each row: `ts`, `group`, `colour`, `badge`, `tag`, `message`, and the optional
    flags `hi` (bold), `err` (stderr) and `filter` (which chip shows it).
    """
    import html

    def ident(row: dict) -> tuple:
        """What makes two lines the same speaker.

        `filter` and `poll` are in here even though they are not drawn, because
        both can be hidden independently: if a poll line and an ordinary one could
        continue each other, hiding the polls would leave a visible line whose
        identity was blanked against a predecessor that is no longer on screen.
        """
        return (row["badge"], row.get("tag", ""), row.get("filter", ""), bool(row.get("poll")))

    out = []
    for i, row in enumerate(rows):
        key = (row["ts"], row["group"])
        continues = i > 0 and (rows[i - 1]["ts"], rows[i - 1]["group"]) == key
        continued = i + 1 < len(rows) and (rows[i + 1]["ts"], rows[i + 1]["group"]) == key
        # Said once per run of lines, not once per line: a hundred snakemake lines
        # from one ETL call do not need the call id a hundred times, and the
        # repetition is what buries the messages.
        same_speaker = i > 0 and ident(rows[i - 1]) == ident(row)

        classes = "line" + (" hi" if row.get("hi") else "") + (" nb" if continued else "")
        # An optional link on the tag, which is how a line gets a way out of the
        # page it is on -- from Modal's view into one call's trace, say. It sits on
        # the first line of a run, which is the one that still draws the tag.
        tag = "" if same_speaker else html.escape(row.get("tag", ""))
        if row.get("href") and tag:
            tag = f"<a href='{html.escape(row['href'])}'>{tag}</a>"
        badge = "" if same_speaker else html.escape(row["badge"])

        # Rendered UTC, converted to local by the browser -- see LOG_JS. The short
        # form is written server-side so the column is already the right width
        # before any script runs, and `data-ts` carries the full instant for the
        # conversion. Continuation lines get neither: their cell stays empty.
        stamp = "" if continues else f"<span class='ts' data-ts='{html.escape(row['ts'])}'>{short_ts(row['ts'])}</span>"

        out.append(
            f"<div class='{classes}' data-filter='{html.escape(row.get('filter', ''))}'"
            f"{' data-poll=1' if row.get('poll') else ''}>"
            f"{stamp or "<span class='ts'></span>"}"
            f"<span class='att' style='color:{row['colour']}'>{badge}</span>"
            f"<span class='actor'>{tag}</span>"
            f"<span class='msg{' err' if row.get('err') else ''}'>{html.escape(row['message'])}</span>"
            "</div>"
        )
    return "".join(out)


def short_ts(ts: str) -> str:
    """`2026-08-13T03:47:48.868Z` -> `08/13 03:47:48.868`.

    The year is never in question in a log you are reading now, and dropping it
    buys five characters of message width on every line. Still UTC at this point;
    LOG_JS rewrites it to local, in this same shape.
    """
    return f"{ts[5:7]}/{ts[8:10]} {ts[11:23]}" if len(ts) >= 23 else ts


def log_header_html(badge: str, tag: str) -> str:
    """A heading row using the same columns as the lines below it."""
    return (
        "<div class='line head'>"
        "<span class='ts'>Time</span>"
        f"<span class='att'>{badge}</span>"
        f"<span class='actor'>{tag}</span>"
        "<span class='msg'>Message</span></div>"
    )


def chips_html(values: list[str]) -> str:
    """Filter chips, or nothing when there is only one thing to filter for."""
    import html

    if len(values) < 2:
        return ""
    buttons = "".join(
        f"<button class='chip{' on' if v == 'all' else ''}' data-filter='{html.escape(v)}'>{html.escape(v)}</button>"
        for v in ["all", *values]
    )
    return f"<div class='chips'>{buttons}</div>"


# Everything the log pages do client-side. Filtering and hiding are class toggles
# rather than fetches: the lines are already on the page, and re-reading the volume
# to show fewer of them would be the slow way to do nothing.
#
# A plain string, not an f-string, so the braces stay braces. Each behaviour is
# guarded on its own elements, so pages that lack a control simply skip it.
LOG_JS = """
<script>
// Timestamps are stored and rendered in UTC -- one fixed format is what lets logs
// from different files be merged by sorting strings. Only the reader wants local
// time, and only the reader knows what local is, so the conversion happens here.
document.querySelectorAll('.ts[data-ts]').forEach(el => {
  const d = new Date(el.dataset.ts);
  if (isNaN(d)) return;
  const p = (n, w = 2) => String(n).padStart(w, '0');
  el.textContent = `${p(d.getMonth() + 1)}/${p(d.getDate())} ` +
                   `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}.${p(d.getMilliseconds(), 3)}`;
});

document.querySelectorAll('.chip[data-filter]').forEach(chip => chip.addEventListener('click', () => {
  const want = chip.dataset.filter;
  document.querySelectorAll('.chip[data-filter]').forEach(c => c.classList.toggle('on', c === chip));
  document.querySelectorAll('.line').forEach(line => {
    if (line.classList.contains('head')) return;
    line.classList.toggle('off', want !== 'all' && line.dataset.filter !== want);
  });
}));

// The dashboard polls itself every few seconds, so on a quiet app its own request
// log is most of what there is to read -- hidden by default, and one click away
// when the question is about the dashboard itself. The visible count is kept
// honest, or the header would claim thousands of lines you cannot see.
const polls = document.getElementById('hide-polls');
if (polls) {
  const apply = () => {
    document.querySelectorAll('.line[data-poll]').forEach(
      line => line.classList.toggle('off', polls.checked));
    const count = document.getElementById('linecount');
    if (count) {
      const shown = document.querySelectorAll('.line:not(.head):not(.off)').length;
      count.textContent = shown.toLocaleString();
    }
  };
  polls.addEventListener('change', apply);
  apply();
}
</script>
"""


def logs_page_html(run_id: str, lines: list[tuple[str, str, str, str]]) -> str:
    """One run's merged log -- every call, both actors -- as its own page.

    The ETL that built this run's data and the workers that trained on it are one
    timeline here, which is the point: the two halves fail into each other, and a
    worker aborting on a missing train.bin is only legible beside the build that
    failed to produce it.
    """
    import html

    shown = lines[-MAX_LOG_LINES:]

    # A call writes one file, so it has exactly one actor.
    actor_of = {call_id: actor for _, call_id, actor, _ in shown}
    calls = list(dict.fromkeys(call_id for _, call_id, _, _ in shown))
    colour = {c: CALL_COLOURS[i % len(CALL_COLOURS)] for i, c in enumerate(calls)}
    actors = sorted({actor for _, _, actor, _ in shown})

    legend = " ".join(
        f"<span class='att' style='color:{colour[c]}'>&#9679; {html.escape(actor_of[c])} {html.escape(c[-6:])}</span>"
        for c in calls
    )

    rows = [
        {
            "ts": ts,
            "group": call_id,
            "colour": colour[call_id],
            "badge": call_id[-6:],
            "tag": actor,
            "message": msg,
            "hi": "EXIT" in msg,
            "filter": actor,
        }
        for ts, call_id, actor, msg in shown
    ]
    body = (
        log_header_html("Call", "Actor") + merged_lines_html(rows)
        if rows
        else "<div class='empty'>Nothing logged yet.</div>"
    )
    truncated = f" · showing last {len(shown):,}" if len(shown) < len(lines) else ""

    # Straight to what Modal saw about this run's calls -- the container starts, the
    # kills, and anything that happened before a logger existed to write it down.
    traces = " ".join(
        f"<a class='mono small' href='../trace/{html.escape(c)}'>{html.escape(actor_of[c])} {html.escape(c[-6:])}</a>"
        for c in calls
    )

    return f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{html.escape(run_id)} · logs</title>
<style>{THEME_CSS}{LOG_CSS}
  main {{ max-width: 76rem; }}
</style></head>
<body><main>
  <header>
    <h1 class='mono'>{html.escape(run_id)}</h1>
    <span class='muted small'>{len(lines):,} lines · {len(calls)} call(s){truncated}</span>
    <span class='muted small'>· <a href='../ledger/{html.escape(run_id)}'>ledger</a>
                              · <a href='../modal'>modal logs</a>
                              · <a href='../'>all runs</a></span>
  </header>
  {chips_html(actors)}
  <div class='legend'>{legend}</div>
  <div class='log'>{body}</div>
  <p class='muted small'>in Modal's own log: {traces or "—"}</p>
</main>{LOG_JS}
</body></html>"""


def ledger_page_html(run_id: str, lines: list[str]) -> str:
    """One run's train.jsonl, one row per line, as written.

    Numbered by file position rather than by step: the two usually agree, but a
    resumed run can legitimately repeat a step, and the line number is the thing
    that stays true when it does.
    """
    import html

    if lines:
        body = "".join(
            f"<div class='line'><span class='num'>{i}</span><span class='msg'>{html.escape(line)}</span></div>"
            for i, line in enumerate(lines, 1)
        )
    else:
        body = "<div class='empty'>Nothing committed yet.</div>"

    return f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{html.escape(run_id)} · ledger</title>
<style>{THEME_CSS}{LOG_CSS}
  main {{ max-width: 76rem; }}
</style></head>
<body><main>
  <header>
    <h1 class='mono'>{html.escape(run_id)}</h1>
    <span class='muted small'>{len(lines)} rows · train.jsonl</span>
    <span class='muted small'>· <a href='../logs/{html.escape(run_id)}'>logs</a>
                              · <a href='../'>all runs</a></span>
  </header>
  <div class='log'>{body}</div>
</main></body></html>"""


# --------------------------------------------------------------------------------------
# Modal's own view of the app, asked for when someone opens the page
# --------------------------------------------------------------------------------------
#
# A different question from everything above. The run pages answer "what happened
# to this run", out of files our own code wrote into the run's folder. These answer
# "what happened on this app", out of what Modal saw -- including the containers
# that died before they could write anything down.
#
# Nothing here is stored. Modal keeps these logs already, `applog.fetch` asks for
# them, and the answer is rendered and thrown away. There is no collector and no
# second container: this one serves the page and makes the call.

MODAL_HOURS = 2.0  # default range for the app-wide view
TRACE_HOURS = 24 * 7  # a call is looked up long after it ran, so reach back further


def modal_rows(rows: list[dict], link: bool = True) -> list[dict]:
    """`applog.fetch` rows in the shape `merged_lines_html` draws.

    Coloured by function rather than by call: the question on this page is which
    part of the app spoke, and a busy window holds far too many calls for one
    colour each to mean anything. Grouped by container, so a traceback out of one
    stays a single block even while another logs into the same millisecond.

    `link=False` for the trace page: its links are relative to `/trace/{call_id}`,
    where another `trace/…` would nest, and every line there is already the call
    being looked at.
    """
    functions = list(dict.fromkeys(row["function"] for row in rows))
    colour = {f: CALL_COLOURS[i % len(CALL_COLOURS)] for i, f in enumerate(functions)}
    return [
        {
            "ts": row["ts"],
            "group": row["task_id"],
            "colour": colour[row["function"]],
            "badge": row["function"],
            # The call when there is one, falling back to the container. System
            # lines -- container lifecycle, the web server's own request log --
            # carry no call id, and only the call is worth a link: it is the id
            # that also names a folder under `runs/`.
            "tag": (row["call_id"] or row["task_id"])[-6:],
            "href": f"trace/{row['call_id']}" if (link and row["call_id"]) else "",
            "message": row["message"],
            "hi": "EXIT" in row["message"],
            "err": row["stderr"],
            "filter": row["function"],
            # The dashboard polling itself. Marked rather than dropped, so the page
            # can hide it without the server deciding what is worth keeping.
            "poll": row["function"] == "dashboard" and "GET /rows" in row["message"],
        }
        for row in rows
    ]


# The long end reaches past what Modal will serve: a fetch is clamped to 35 days,
# but `applog.load` is not, so the wider ranges answer from the archive alone.
RANGES = [("30m", 0.5), ("2h", 2.0), ("12h", 12.0), ("2d", 48.0), ("7d", 168.0), ("30d", 720.0)]


def modal_page_html(rows: list[dict], hours: float, search: str, app: str = "") -> str:
    """The app-wide view: what Modal saw, over a window, fetched just now.

    The range buttons are plain links rather than a filter, because the range is
    the one thing the browser cannot narrow on its own -- widening it is another
    request to Modal, and a cheap one. The function chips are client-side, since
    those lines are already on the page.
    """
    import html

    shown = rows[-MAX_LOG_LINES:]
    functions = sorted({row["function"] for row in shown})
    body = (
        log_header_html("Function", "Call") + merged_lines_html(modal_rows(shown))
        if shown
        else "<div class='empty'>Nothing in this window.</div>"
    )
    truncated = f" · showing last {len(shown):,}" if len(shown) < len(rows) else ""
    polls = sum(1 for row in shown if row["function"] == "dashboard" and "GET /rows" in row["message"])

    # Carried through every range link, or changing the window would silently drop
    # the search and the app you are looking at.
    carried = ("&q=" + html.escape(search, quote=True) if search else "") + (
        "&app=" + html.escape(app, quote=True) if app else ""
    )
    picker = " ".join(
        f"<a class='chip{' on' if abs(value - hours) < 1e-6 else ''}' href='?hours={value:g}{carried}'>{label}</a>"
        for label, value in RANGES
    )
    q = html.escape(search, quote=True)
    pinned = (
        f" · pinned to <span class='mono'>{html.escape(app)}</span> · <a href='?hours={hours:g}'>all records</a>"
        if app
        else ""
    )

    return f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Modal · logs</title>
<style>{THEME_CSS}{LOG_CSS}
  main {{ max-width: 88rem; }}
  a.chip {{ text-decoration:none; }}
  form.q {{ display:inline-flex; gap:.4rem; margin-left:.5rem; }}
  form.q input {{ font:inherit; font-size:.72rem; padding:.1rem .5rem; border-radius:999px;
                  border:1px solid var(--line); background:transparent; color:var(--fg); }}
</style></head>
<body><main>
  <header>
    <h1>Modal logs</h1>
    <span class='muted small'><span id='linecount'>{len(rows):,}</span> lines
      · last {hours:g}h{truncated}{pinned}</span>
    <span class='muted small'>· <a href='./'>all runs</a></span>
  </header>
  <p class='muted small'>Everything <span class='mono'>modal app logs {html.escape(APP_NAME)}</span>
     would show, asked for when you opened this page and archived to
     <span class='mono'>modal_logs/</span> on the way past — so the window can reach back further
     than Modal will still answer for. Add <span class='mono'>?app=ap-…</span> to pull in a record
     it has forgotten.</p>
  <div class='chips'>{picker}
    <form class='q' method='get'>
      <input type='hidden' name='hours' value='{hours:g}'>
      <input type='hidden' name='app' value='{html.escape(app, quote=True)}'>
      <input type='search' name='q' value='{q}' placeholder='search…'>
    </form>
    <label class='chip' title='The dashboard polling itself every {REFRESH_MS // 1000}s'>
      <input type='checkbox' id='hide-polls' checked> hide {polls:,} dashboard polls</label>
  </div>
  {chips_html(functions)}
  <div class='log'>{body}</div>
</main>{LOG_JS}
</body></html>"""


def trace_page_html(call_id: str, rows: list[dict]) -> str:
    """Everything Modal saw about one call.

    The join, and the reason any of this is worth having: a run's own log stops
    wherever its container stopped being able to write, and this is the rest of
    that sentence -- the OOM, the preemption, the traceback out of an import that
    happened before a logger existed.

    Filtered by Modal, not here. `fc-` is a first-class filter on the fetch, so
    this asks a narrow question and gets a small answer, however busy the app was.
    """
    import html

    shown = rows[-MAX_LOG_LINES:]
    body = (
        log_header_html("Function", "Container") + merged_lines_html(modal_rows(shown, link=False))
        if shown
        else f"<div class='empty'>Nothing for this call in the last {TRACE_HOURS / 24:g} days.</div>"
    )
    containers = sorted({row["task_id"] for row in shown})

    return f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{html.escape(call_id)} · trace</title>
<style>{THEME_CSS}{LOG_CSS}
  main {{ max-width: 88rem; }}
</style></head>
<body><main>
  <header>
    <h1 class='mono'>{html.escape(call_id)}</h1>
    <span class='muted small'>{len(rows):,} lines · {len(containers)} container(s)</span>
    <span class='muted small'>· <a href='../modal'>all modal logs</a>
                              · <a href='../'>all runs</a></span>
  </header>
  <p class='muted small'>What Modal saw of this one call, over the last
     {TRACE_HOURS / 24:g} days.</p>
  <div class='log'>{body}</div>
</main>{LOG_JS}
</body></html>"""


# --------------------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------------------


def build_web_app():
    """The ASGI app app.py serves. fastapi is imported here, not at module level,
    so this file can be imported locally without it installed."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    web = FastAPI()

    # Polls must never be served from a cache, or the page silently freezes while
    # looking perfectly healthy.
    no_store = {"Cache-Control": "no-store"}

    @web.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(page_html(await scan()), headers=no_store)

    @web.get("/rows", response_class=HTMLResponse)
    async def rows():
        return HTMLResponse(rows_html(await scan()), headers=no_store)

    @web.get("/logs/{run_id}", response_class=HTMLResponse)
    async def logs(run_id: str):
        """Every worker's log for one run, merged in time order. Its own page,
        opened in its own tab."""
        return HTMLResponse(logs_page_html(run_id, await read_logs(run_id)), headers=no_store)

    @web.get("/ledger/{run_id}", response_class=HTMLResponse)
    async def ledger(run_id: str):
        """One run's train.jsonl, as written."""
        return HTMLResponse(ledger_page_html(run_id, await read_ledger(run_id)), headers=no_store)

    # The two Modal-log routes are `def`, not `async def`, and that is load-bearing.
    # FastAPI runs a sync handler in its threadpool, which is exactly where
    # `applog.fetch` has to be called from: it drives synchronicity's own event
    # loop, and doing that from inside this one would deadlock. They touch no
    # volume, so they skip the reload every other route pays for.

    @web.get("/modal", response_class=HTMLResponse)
    def modal_logs(hours: float = MODAL_HOURS, q: str = "", app: str = ""):
        """What Modal saw of the whole app, over a window, asked for right now.

        Also what archives it: `applog.fetch` merges the answer into `modal_logs/`
        on the way past, so opening this page is how history accumulates.

        `?app=ap-…` reads one specific app record instead of resolving the name.
        That is the rescue path for history stranded under an id `AppList` has
        forgotten -- and since the read archives, visiting it once is enough.
        """
        rows = applog.fetch(volume, APP_NAME, hours=hours, search=q, app_id=app)
        return HTMLResponse(modal_page_html(rows, hours, q, app), headers=no_store)

    @web.get("/trace/{call_id}", response_class=HTMLResponse)
    def trace(call_id: str):
        """Everything Modal saw about one call.

        The join between the two halves: `call_id` names a folder under a run and
        tags every line Modal kept, so one id reaches both records. Modal does the
        filtering, so this stays cheap however noisy the app was.
        """
        rows = applog.fetch(volume, APP_NAME, hours=TRACE_HOURS, call_id=call_id)
        return HTMLResponse(trace_page_html(call_id, rows), headers=no_store)

    @web.post("/cancel/{run_id}")
    async def cancel(run_id: str):
        """Cancel whoever currently holds the run. The lease is what tells us which
        call that is -- there is nowhere else it is written down."""
        grant = await leases.get.aio(lease_key(run_id))
        if not grant:
            return {"ok": False, "reason": "no lease -- nothing to cancel"}
        await modal.FunctionCall.from_id(grant["call_id"]).cancel.aio()
        return {"ok": True, "reason": f"cancelled {grant['call_id']}"}

    @web.post("/resume/{run_id}")
    async def resume(run_id: str):
        """Hand it back to the launcher, which decides what resuming means here --
        including deciding not to, if the run is already finished or still alive.

        Reached by name rather than by calling the imported function, because the
        one-at-a-time lock only holds for the deployed launcher.
        """
        launcher = modal.Function.from_name(APP_NAME, "launch")
        result = await launcher.remote.aio(run_id=run_id)
        return {"ok": True, "reason": result["reason"]}

    return web
