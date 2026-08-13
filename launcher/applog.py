"""Modal's own view of this app, fetched on demand.

THE OTHER HALF. `logs.py` is what our code writes about itself: a stdlib logger
per Modal call, filed under the run it belongs to, on the Volume. This module is
what *Modal* saw -- every container's stdout and stderr, plus the things no
container is alive to record. The two share no code and this one never imports
that one; they answer different questions and fail in different directions.

READ THROUGH, ARCHIVE BEHIND. Modal already keeps these logs -- they are what
`modal app logs` reads -- so nothing here runs in the background waiting for them.
The dashboard asks when someone opens a page, and what comes back is merged into
`modal_logs/{YYYYMMDD}_{app_id}` on the Volume before being rendered. Opening the
page is what collects the logs.

The archive is not a cache; it is the only thing that outlives Modal's own
reachability. Modal's store is keyed by *app record*, a record's id changes every
time the app is stopped and redeployed, and `AppList` returns only the deployed
and recently-stopped ones -- so a week-old id cannot be found by name any more and
its logs, still sitting there, become unaskable-for. Two weeks of this project's
history went that way before this file existed. Fetching reaches what Modal can
still serve; the archive holds everything ever fetched, across every app id, for
as long as the Volume does.

An earlier version instead ran a collector container that streamed the same logs
onto the Volume, one file per session. It is gone: it cost a long-lived container,
a resume cursor, session boundaries, a size roll, and a standing rule against
echoing captured lines to stdout (a collector living inside the app it reads will
otherwise feed on itself, for ever) -- to buy the same durability this gets from a
dict and a sorted write.

CONCURRENCY. Several dashboard containers serve at once and all of them write
here. Two handling overlapping requests can rewrite the same day file and one will
lose the other's newest lines. That is why `store` merges by line rather than
appending, and why `fetch` reloads first: the loser's lines come back on the next
request that covers them, so the archive converges instead of tearing.

WHAT IS ONLY EVER HERE, and cannot be in the run tree:

  - anything before a call's logger exists. `work` must reload the Volume before
    it can open a log file on it, so a reload failure is invisible to the run tree
    by construction -- see the comment in `app.work`.
  - anything after it is gone: an OOM kill, a preemption, a SIGKILL. No `finally`
    runs for those.
  - `print()`, and every library that logs without a call id -- `transformer.*`,
    `wandb`, torch warnings. `logs.py` drops those on purpose.
  - `launch` and `dashboard`, which write no run-tree log at all. Every lease
    decision -- takeover, "already running", the attempt counter -- happens in
    `launch` and is otherwise a return value nobody keeps.
  - image builds, container starts, task states.

THE JOIN. Every line comes back tagged with the `fc-` call id that produced it,
and that is the same id naming `runs/{run_id}/logs/{call_id}/`. One id reaches
both records, which is what `trace()` and the dashboard's `/trace` route are for.

CALLING CONVENTION. The public functions here are blocking, and they must be
called off the event loop -- from a FastAPI handler declared `def`, not `async
def`, so it runs in the threadpool. Everything inside runs on synchronicity's own
loop via `create_blocking`, where objects are translated to their implementation
types (`_Client`, not `Client`) and the public blocking wrappers are unusable:
`modal.App.lookup` raises `Deadlock detected` from in there, and anything spelled
`.aio()` raises AttributeError because `.aio` lives on the wrapper you do not
have. Hence the raw protos below, which is also how `modal/cli/app.py` does it.

PRIVATE APIS. `modal._logs`, `client.stub.*`. Checked against modal 1.5.2, and
contained here for the same reason `app.status` is contained there: one place to
fix when it moves.
"""

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

from modal._logs import LogsFilters, fetch_logs
from modal._utils.async_utils import synchronizer
from modal.client import _Client
from modal_proto import api_pb2

from jobs import MODAL_LOGS

# How far back a view reaches when nobody says. Two hours covers "what just
# happened", which is what the page is for; the caller can ask for more.
DEFAULT_HOURS = 2.0

# Modal enforces 35 days on a fetch range, so asking for more is an error rather
# than a bigger answer.
MAX_HOURS = 34 * 24

# The dashboard serves many requests per container (`modal.concurrent`), and the
# `def` handlers that call in here run in FastAPI's threadpool -- so two archive
# updates can land at once. `volume.reload()` swaps the mount underneath whatever
# else is reading it, which is the part that would actually break, so the whole
# reload/write/commit section is serialized. Per container, not per app: the
# cross-container race is handled by overwriting whole days.
_ARCHIVE_LOCK = threading.Lock()

# How far back an ordinary view refreshes the archive. Days older than this are
# read and never rewritten -- they belong to app records that have stopped
# producing lines, so there is nothing to refresh them with. Bounds the fetch too:
# a 30-day view costs a file read plus three days, not thirty.
#
# `?app=` overrides it, because that is the deliberate "go and get history" path.
ARCHIVE_DAYS = 3

STDERR = api_pb2.FILE_DESCRIPTOR_STDERR


def utc_iso(epoch: float) -> str:
    """A float epoch as the 24-character UTC stamp every log line here starts with.

    The same shape `logs.CallFormatter` produces, so these rows and a run's own log
    sort against each other. That is a convention the two halves agree on, not a
    dependency between them.
    """
    return datetime.fromtimestamp(epoch, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


async def resolve_app(client, name: str) -> str:
    """The `ap-` id of the app deployed under this name.

    `previous_app_id` is the fallback: a stopped app still has readable logs, and
    reading them after the fact is often the whole point. Note that a stop and
    redeploy mints a *new* id, so history from before that lives under the old one
    and is not reachable through this name any more.
    """
    resp = await client.stub.AppGetByDeploymentName(
        api_pb2.AppGetByDeploymentNameRequest(name=name, environment_name="")
    )
    app_id = resp.app_id or resp.previous_app_id
    if not app_id:
        raise RuntimeError(f"no app named {name!r} is deployed or recently stopped")
    return app_id


async def app_records(client, name: str, since: datetime, until: datetime, limit: int = 6) -> list[str]:
    """Every `ap-` ever deployed under this name whose life overlaps the window.

    One name is not one app. Stopping and redeploying mints a new record, and every
    `modal run` of the module mints an ephemeral one carrying the same name -- so a
    call from this morning routinely belongs to an id that is no longer the current
    one, and a lookup by name alone silently returns nothing for it. That is not a
    hypothetical: tracing an eight-hour-old call by name gave 0 lines, because four
    deploys had happened since.

    Newest first, capped: a busy day of `modal run` leaves a long tail of ephemeral
    records, and the recent ones are the ones anybody asks about.

    What this cannot do is outlive Modal's own bookkeeping. Once a record ages out
    of the list its logs are unreachable by name, which is the argument for copying
    a dead call's lines into its run folder while they can still be had.
    """
    resp = await client.stub.AppList(api_pb2.AppListRequest(environment_name=""))
    start, end = since.timestamp(), until.timestamp()

    overlapping = [
        (app.created_at, app.app_id)
        for app in resp.apps
        if name in (app.description, app.name)
        # stopped_at is 0 for one still running, which is "no end", not "ended in 1970"
        if app.created_at <= end and (app.stopped_at or float("inf")) >= start
    ]
    return [app_id for _, app_id in sorted(overlapping, reverse=True)[:limit]]


async def function_names(client, app_id: str) -> dict[str, str]:
    """`{fu-id: "etl"}` for one app -- what turns an id into a name.

    A function deployed since the range being asked about is unknown to this map,
    and falls back to its raw id, which is still exact.
    """
    layout = await client.stub.AppGetLayout(api_pb2.AppGetLayoutRequest(app_id=app_id))
    return {fu: tag for tag, fu in layout.app_layout.function_ids.items()}


def record(batch, item, text: str, names: dict[str, str]) -> dict:
    """One log line, with the provenance Modal already attached to it.

    Split across two levels of the response, which is why both halves are needed:
    the batch knows which function and container, the item knows which call, when,
    and on which stream.
    """
    # A function that has since been removed from the app has no entry in the name
    # map, so it falls back to its own id -- shortened, because the full 26
    # characters make a useless chip and the line already carries the exact
    # container and call.
    function = names.get(batch.function_id) or (f"{batch.function_id[:7]}…" if batch.function_id else "-")

    return {
        "ts": utc_iso(item.timestamp),
        "function": function,
        "task_id": batch.task_id,
        "call_id": item.function_call_id,
        "message": text,
        "stderr": item.file_descriptor == STDERR,
    }


# --------------------------------------------------------------------------------------
# the archive: what has been read, kept
# --------------------------------------------------------------------------------------
#
# Modal keeps these logs, but only reachably so. Its store is keyed by app record,
# a record's id changes every time the app is stopped and redeployed, and `AppList`
# only returns the deployed and recently-stopped ones -- so an id a week old cannot
# be found by name any more, and its logs, which are still there, become
# unaskable-for. Two weeks of history went that way before this existed.
#
# So every read is also a write. What comes back from Modal is merged into
# `modal_logs/{YYYYMMDD}_{app_id}`, and the view renders from those files. The
# archive only ever grows, it survives the app id changing under it, and nothing
# has to be running for it to be collected -- opening the page is what collects it.
#
# One line per record, in the format at the top of this file: column one is the
# timestamp contract, column two is provenance. The app id is in the filename
# rather than on every line, and the date is there so a range query can skip whole
# files without opening them.


def log_path(date: str, app_id: str) -> Path:
    """Where one UTC day of one app record's logs live. `{YYYYMMDD}_{app_id}`."""
    return MODAL_LOGS / f"{date}_{app_id}"


def format_line(row: dict) -> str:
    """One row as its stored line. `!` marks stderr, as it does in a run's log."""
    mark = "! " if row["stderr"] else ""
    return f"{row['ts']} {row['function']}/{row['task_id']}/{row['call_id']} {mark}{row['message']}"


def parse_line(line: str) -> dict | None:
    """A stored line back into a row, or None if it is not one.

    Tolerant on purpose: a torn last line comes back with whatever fields it has
    rather than raising. This is a log; refusing to show its final line is worse
    than showing it ragged.
    """
    if not line.strip():
        return None
    ts, _, rest = line.partition(" ")
    provenance, _, message = rest.partition(" ")
    function, task_id, call_id = (provenance.split("/") + ["", ""])[:3]
    stderr = message.startswith("! ")
    return {
        "ts": ts,
        "function": function,
        "task_id": task_id,
        "call_id": call_id,
        "message": message[2:] if stderr else message,
        "stderr": stderr,
    }


def store(rows: list[dict]) -> list[Path]:
    """Write each day-and-app file as exactly what Modal just returned for it.

    Overwrite, not merge, and that is safe because of how the caller picks its
    window: it always starts at a UTC midnight, so every day it fetches is covered
    end to end and what came back *is* that day. There is no partial day to
    preserve, so there is nothing to merge -- which also means no dedupe, no
    ordering rules, and no way for the file to drift out of agreement with Modal.

    Identical content is not rewritten. The commit that follows costs several
    seconds no matter how little changed, so the cheapest write is the one that
    does not happen.
    """
    by_file: dict[Path, set[str]] = {}
    for row in rows:
        # The row's own timestamp decides its file: one fetch straddles midnight
        # whenever the window does.
        by_file.setdefault(log_path(row["ts"][:10].replace("-", ""), row["app_id"]), set()).add(format_line(row))

    MODAL_LOGS.mkdir(parents=True, exist_ok=True)
    changed = []
    for path, lines in by_file.items():
        text = "\n".join(sorted(lines)) + "\n"
        if path.exists() and path.read_text(errors="replace") == text:
            continue
        path.write_text(text)
        changed.append(path)
    return changed


def load(since: datetime, until: datetime, app_id: str = "") -> list[dict]:
    """Every archived row in this window, from every app record that has one.

    Both filters come off the filename, so a file outside the range is never
    opened -- which is the point of putting the date and the app id there. Rows are
    then filtered to the exact window, since a day file holds a whole day.
    """
    if not MODAL_LOGS.exists():
        return []

    first, last = since.strftime("%Y%m%d"), until.strftime("%Y%m%d")
    lo, hi = utc_iso(since.timestamp()), utc_iso(until.timestamp())

    rows = []
    for path in sorted(MODAL_LOGS.iterdir()):
        date, _, record_id = path.name.partition("_")
        if not (first <= date <= last):
            continue  # a day outside the range: never opened
        if app_id and record_id != app_id:
            continue  # a different app record: likewise
        for line in path.read_text(errors="replace").splitlines():
            if (row := parse_line(line)) and lo <= row["ts"] <= hi:
                rows.append(row)

    rows.sort(key=lambda row: row["ts"])
    return rows


async def _fetch(app_name: str, hours: float, call_id: str, search: str, app_id: str = "") -> list[dict]:
    client = await _Client.from_env()

    until = datetime.now(UTC)
    since = until - timedelta(hours=min(hours, MAX_HOURS))
    filters = LogsFilters(function_call_id=call_id, search_text=search)

    # An explicit id is the rescue path: `AppList` forgets old records within days,
    # and once it has, an app's logs are still served but no longer findable by
    # name. Given the id -- from Modal's web UI, or from a run that recorded it --
    # this reaches them and, because every read archives, rescues them permanently.
    records = [app_id] if app_id else await app_records(client, app_name, since, until)

    rows = []
    for app_id in records:
        # Per app record: the same function can have a different `fu-` in each, so
        # the name map has to come from the record its lines belong to.
        names = await function_names(client, app_id)
        async for batch in fetch_logs(client, app_id, since, until, filters=filters):
            for item in batch.items:
                if not item.data:
                    continue  # a task-state or progress item: nothing to show
                # One row per output line. Modal batches whatever the container
                # wrote, so a single item can carry an entire traceback.
                for text in item.data.splitlines():
                    # The app id rides along because it decides which file the row
                    # is archived in, and it is not recoverable from the row itself.
                    rows.append(record(batch, item, text, names) | {"app_id": app_id})

    return rows


def fetch(
    volume,
    app_name: str,
    hours: float = DEFAULT_HOURS,
    call_id: str = "",
    search: str = "",
    app_id: str = "",
) -> list[dict]:
    """Modal's log for one app over the last `hours`, archived on the way past.

    Every read is also a write: what Modal returns is merged into `modal_logs/`,
    and the answer is then read back out of the archive. So the view shows what
    Modal can still reach *plus* everything it has ever been asked for -- which is
    the only way history survives an app id changing, since Modal's own store
    stops being reachable by name the moment `AppList` forgets the record.

    Blocking, and it must be called from a threadpool rather than an event loop:
    see CALLING CONVENTION above. That is also what makes the writes safe. The
    network half runs under `create_blocking`; the file half and the commit happen
    out here, in ordinary sync code, where `volume.commit()` means what it says.

    `call_id` and `search` narrow the fetch server-side. They are applied again to
    the archive on the way out, because the files hold everything ever fetched,
    not just what this request asked for.
    """
    until = datetime.now(UTC)
    # The archive is read over the full window asked for; only the *fetch* is
    # clamped, because Modal refuses a range beyond 35 days. Asking for 90 days is
    # therefore a legitimate question -- it just answers from files alone past the
    # clamp, which is exactly what the archive is for.
    since = until - timedelta(hours=hours)

    # What to refresh, as opposed to what to show. Two rules, and between them they
    # are the reason `store` can simply overwrite:
    #
    #   snapped to midnight  every day fetched is covered end to end, so what comes
    #                        back is the whole of that day and not a slice of it
    #   capped               an ordinary view refreshes the last few days; older
    #                        days are read from the archive and never rewritten,
    #                        which is also what stops a 30-day view re-downloading
    #                        a month every time somebody opens it
    #
    # `?app=` lifts the cap: pulling a forgotten record's history in is the one
    # time fetching far back is the whole point.
    horizon = since if app_id else max(since, until - timedelta(days=ARCHIVE_DAYS))
    refresh_from = horizon.replace(hour=0, minute=0, second=0, microsecond=0)

    fresh = synchronizer.create_blocking(_fetch)(
        app_name, (until - refresh_from).total_seconds() / 3600, call_id, search, app_id
    )

    # The network call above is deliberately outside the lock -- it is the slow part
    # and it touches nothing shared. Everything that reads or writes the mount is
    # inside it; see _ARCHIVE_LOCK.
    with _ARCHIVE_LOCK:
        volume.reload()  # other containers write here too
        if fresh and store(fresh):
            volume.commit()
        # Read back out of the archive, which is now current for everything
        # refreshed and is the only source for anything older.
        rows = load(since, until, app_id)
    if call_id:
        rows = [row for row in rows if row["call_id"] == call_id]
    if search:
        rows = [row for row in rows if search in row["message"]]
    return rows
