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

import modal

from app import JOB, RUNS, APP_NAME, Aborted, lease_key, leases, read_config, run_dir, status, volume

REFRESH_MS = 3000  # how often the page asks for fresh rows

# What each derived status is called on screen. The colour lives in the stylesheet
# as `.st-<status>`, so light and dark can differ -- plain coloured text has to
# carry itself against the page background, with no chip behind it to sit on.
STATUS_LABEL = {
    "completed": "Completed",
    "in_progress": "Running",
    "starting": "Starting",
    "failed": "Failed",
    "orphaned": "Orphaned",
    "unreadable": "Unreadable",
}

CANCELLABLE = {"in_progress", "starting"}
RESUMABLE = {"failed", "orphaned"}

# Three stacked lines, drawn in currentColor so it inherits hover states.
LOGS_ICON = (
    "<svg width='13' height='13' viewBox='0 0 16 16' fill='none' stroke='currentColor' "
    "stroke-width='1.7' stroke-linecap='round'><path d='M2.6 4h10.8M2.6 8h10.8M2.6 12h6.6'/></svg>"
)


def derive_status(complete: bool, done: int, live: str | None) -> str:
    """Fold progress and liveness into one word.

    `complete` wins outright and is decided from committed files alone -- a
    finished run must read finished forever, and Modal eventually forgets old call
    outcomes, so we never ask about them.

    Otherwise `live` decides. `orphaned` is the honest "cannot know" case: the
    lease expired, or there is no key at all. It is not an error state -- it is
    the normal look of a run nobody has touched in a while, and it is resumable.
    """
    if complete:
        return "completed"
    if live == "running":
        return "in_progress" if done else "starting"
    if live in ("done", "failed", "crashed", "timed_out"):
        return "failed"
    return "orphaned"  # expired, or no lease key


async def scan() -> list[dict]:
    """One row per run folder, newest first.

    Reloads the volume first: this container is long-lived, so its snapshot is
    stale by default -- the same reason `launch` and `Attempt` reload.

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
        config = read_config(run_id)
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

        runs.append({"run_id": run_id, "done": done, "total": total, "complete": done >= total})

    pending = [r for r in runs if "complete" in r and not r["complete"]]
    grants = await asyncio.gather(*(leases.get.aio(lease_key(r["run_id"])) for r in pending))
    live = await asyncio.gather(
        *(status(modal.FunctionCall.from_id(g["call_id"])) if g else _no_lease() for g in grants)
    )

    for row, grant, state in zip(pending, grants, live):
        row["attempt"] = grant["attempt"] if grant else None
        row["call_id"] = grant["call_id"] if grant else None
        row["status"] = derive_status(False, row["done"], state)

    for row in runs:
        row.setdefault("status", "completed" if row.get("complete") else "orphaned")
        row.setdefault("attempt", None)
        row.setdefault("call_id", None)
    return runs


async def _no_lease():
    """Stand-in for a liveness answer we never asked for, so the gather above
    stays index-aligned with its rows."""
    return None


async def read_logs(run_id: str) -> list[tuple[str, str, str]]:
    """Every attempt's log for one run, merged into one stream in time order.

    Each line is written as `<iso-timestamp> <message>`, so the timestamp is
    simply the first column. They are all UTC and all the same width, which means
    sorting the strings sorts the times -- no parsing into datetimes needed.

    Interleaving matters here. Attempts overlap: a superseded worker is still
    logging while its replacement boots, and reading two files side by side hides
    that. Merged and sorted, a takeover reads as one story.

    Returns (timestamp, call_id, message) per line.
    """
    await volume.reload.aio()

    logs_dir = run_dir(run_id) / "logs"
    if not logs_dir.exists():
        return []

    lines = []
    for attempt_dir in sorted(logs_dir.iterdir()):
        log = attempt_dir / "attempt.log"
        if not log.is_file():
            continue
        last_ts = ""
        for line in log.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            ts, _, msg = line.partition(" ")
            if ts[:1].isdigit():
                last_ts = ts
            else:  # no leading timestamp: keep it next to the line it follows
                ts, msg = last_ts, line
            lines.append((ts, attempt_dir.name, msg))

    lines.sort(key=lambda row: (row[0], row[1]))
    return lines


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
            bar = f"<div class='bar'><div class='fill' style='width:{pct:.1f}%'></div></div>"

        if status_key in CANCELLABLE:
            action = f"<button class='cancel' data-act='cancel' data-run='{name}'>Cancel</button>"
        elif status_key in RESUMABLE:
            action = f"<button class='resume' data-act='resume' data-run='{name}'>Resume</button>"
        else:
            action = ""

        # A plain link. The browser already knows how to open a tab, and the logs
        # page wants its own address so it can be reloaded, shared, and left open
        # beside the dashboard.
        logs = f"<a class='icon' href='logs/{name}' target='_blank' rel='noopener' title='Attempt logs'>{LOGS_ICON}</a>"

        out.append(f"""
      <tr>
        <td class='mono'>{name}</td>
        <td><span class='st st-{status_key}'>{label}</span></td>
        <td>{bar}<span class='mono small'>{progress}</span></td>
        <td class='mono small'>{r["attempt"] or "—"}</td>
        <td class='actions'>{action}{logs}</td>
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
    .st-failed {{ color:#f85149; }}
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
  .st-failed {{ color:#cf222e; }}
  .st-unreadable {{ color:#8250df; }}
  .st-starting, .st-orphaned {{ color:var(--muted); font-weight:500; }}
  /* inline, not stacked above the numbers -- stacking is what made rows tall */
  .bar {{ display:inline-block; vertical-align:middle; margin-right:.5rem; width:110px; height:4px;
          border-radius:2px; background:var(--line); overflow:hidden; }}
  .fill {{ height:100%; background:#2da44e; transition:width .4s ease; }}
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
  b.textContent = b.dataset.act === 'cancel' ? 'Cancelling…' : 'Resuming…';
  try {{ await fetch(`${{b.dataset.act}}/${{b.dataset.run}}`, {{method: 'POST'}}); }} catch (e) {{}}
  refresh();
}});

setInterval(refresh, {REFRESH_MS});
</script>
</body></html>"""


# One colour per attempt, assigned in order of appearance. The point is not
# decoration: when two attempts overlap, colour is what lets you see the handover
# at a glance instead of comparing call ids character by character.
ATTEMPT_COLOURS = ["#0969da", "#8250df", "#bf8700", "#1a7f37", "#cf222e", "#0f766e"]


LOG_CSS = """
  .log { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px;
         border-top:1px solid var(--line); }
  .line { display:flex; gap:1rem; padding:.18rem 0; border-bottom:1px solid var(--line); }
  .line:hover { background:color-mix(in srgb, var(--fg) 5%, transparent); }
  .line.hi .msg { font-weight:600; }
  .ts { color:var(--muted); flex:0 0 auto; }
  .att { flex:0 0 auto; font-weight:600; font-size:.72rem; letter-spacing:.02em; }
  .msg { white-space:pre-wrap; word-break:break-word; }
  .empty { color:var(--muted); padding:2rem 0; }
  .legend { display:flex; gap:.9rem; margin-bottom:.8rem; font-size:.72rem; }
"""


def logs_page_html(run_id: str, lines: list[tuple[str, str, str]]) -> str:
    """One run's merged log, as its own page in its own tab."""
    import html

    attempts = list(dict.fromkeys(call_id for _, call_id, _ in lines))
    colour = {c: ATTEMPT_COLOURS[i % len(ATTEMPT_COLOURS)] for i, c in enumerate(attempts)}

    legend = " ".join(
        f"<span class='att' style='color:{colour[c]}'>&#9679; {html.escape(c[-6:])}</span>" for c in attempts
    )

    if lines:
        body = "".join(
            f"<div class='line{' hi' if 'EXIT' in msg else ''}'>"
            f"<span class='ts'>{html.escape(ts)}</span>"
            f"<span class='att' style='color:{colour[call_id]}'>{html.escape(call_id[-6:])}</span>"
            f"<span class='msg'>{html.escape(msg)}</span></div>"
            for ts, call_id, msg in lines
        )
    else:
        body = "<div class='empty'>No attempt logs yet.</div>"

    return f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{html.escape(run_id)} · logs</title>
<style>{THEME_CSS}{LOG_CSS}
  main {{ max-width: 76rem; }}
</style></head>
<body><main>
  <header>
    <h1 class='mono'>{html.escape(run_id)}</h1>
    <span class='muted small'>{len(lines)} lines · {len(attempts)} attempt(s)</span>
    <span class='muted small'>· <a href='./'>all runs</a></span>
  </header>
  <div class='legend'>{legend}</div>
  <div class='log'>{body}</div>
</main></body></html>"""


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
        """Every attempt's log for one run, merged in time order. Its own page,
        opened in its own tab."""
        return HTMLResponse(logs_page_html(run_id, await read_logs(run_id)), headers=no_store)

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
