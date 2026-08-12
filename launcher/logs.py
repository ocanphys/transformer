"""One call, one logger, one file.

A Modal container is not a process per call. It imports this module once and then
serves many inputs, so anything installed in interpreter globals -- root handlers
above all -- outlives the call that installed it. Teardown alone cannot be the
answer, because teardown is exactly what a crash skips.

So the call id is baked into the logger's *name*, and the handlers only accept
records from that name:

    log = call_logger(call_id, logfile)     # logger "call.fc-AAA"
    log.info("boot ...")                    # -> logfile
    log.getChild("count").info("step 3")    # -> logfile, same call
    release_call_logger()                   # close the file

A handler left behind by an earlier call is then harmless. It sits on the root
logger and sees everything, including the next call's records -- and refuses them,
because they come from `call.fc-BBB` and it only answers to `call.fc-AAA`.
Nothing has to be remembered for that to hold.

A call, not a worker: `work` and `etl` both take one of these, and neither is the
other's kind of thing. What they share is the shape -- one Modal call, one folder
under `runs/{run_id}/logs/{call_id}/`, one file in it named for the actor
(`worker.log`, `etl.log`). The call id is unique per call, so the directory holds
exactly one log and its *name* is free to say who wrote it.

The flip side is that only this logger is captured. Records from `transformer.*`,
`modal`, `wandb` and any other library are not written to the call's file: they
carry no call id, so there is no honest way to file them under one. Python's
lastResort handler still puts library warnings on stderr, where Modal's container
log picks them up -- and `launcher/applog.py` is what makes that durable, in a
tree of its own that this module knows nothing about.

`print()` is not captured either. Nothing here touches `sys.stdout`, which is what
makes it impossible for a handler to write back into the stream it writes from.
"""

import logging
import sys
import time
from pathlib import Path

# Marks the handlers this module installs. Found by attribute rather than by
# identity, because the call that installed them is exactly the thing that is
# gone by the time anyone needs to clean them up.
_MARK = "_call_handler"


def logger_name(call_id: str | None) -> str:
    """The logger one call owns.

    `local` stands in when there is no call id -- a `.local()` run or a unit test
    -- so the name is always well-formed and two such runs in one process still
    share a single, obviously-named logger.
    """
    return f"call.{call_id or 'local'}"


class CallFormatter(logging.Formatter):
    """`<iso-timestamp> <message>`, in UTC.

    The first column is a contract, not a style choice: the dashboard merges every
    log in a run's folder by sorting on it, and that only works because the
    timestamps are one fixed-width UTC format. `converter = time.gmtime` is what
    keeps them UTC even on a machine that thinks otherwise.

    `applog.py` writes its own files to the same column-1 shape without going
    through this class or the logging module at all. That is a convention the two
    halves agree on, deliberately not a dependency between them.

    INFO lines stay bare, since by construction everything in this file is the
    call's own voice. Anything louder is worth marking, so it carries its level.
    """

    converter = time.gmtime

    def __init__(self):
        super().__init__(fmt="%(asctime)s.%(msecs)03dZ %(prefix)s%(message)s", datefmt="%Y-%m-%dT%H:%M:%S")

    def format(self, record):
        record.prefix = "" if record.levelno == logging.INFO else f"{record.levelname} "
        return super().format(record)


def call_logger(call_id: str | None, logfile: Path) -> logging.Logger:
    """The logger for this call, writing to this file and nothing else to it.

    Hand the returned logger to anything that should be part of the call's story --
    the lease, the job, a subprocess being teed -- and take `getChild` for a
    sub-voice. Anything under `call.{call_id}.` is family and lands in the same
    file.

    On a Volume, call this only after `reload()`: reloading fails while a file on
    the volume is open, and this opens one.
    """
    release_call_logger()  # close whatever file an earlier call left open

    name = logger_name(call_id)

    # The stdlib filter is exactly this rule already: pass `name`, pass anything
    # under `name.`, reject everything else. It is the whole call-id guard, so
    # there is nothing here worth writing a second time.
    own_call = logging.Filter(name)

    root = logging.getLogger()
    for handler in (logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)):
        # Two handlers, on purpose: the file is the durable record that ships with
        # the run folder, the console keeps `modal app logs` working -- and is what
        # the session collector picks up, so a line written here reaches both trees.
        handler.setFormatter(CallFormatter())
        handler.addFilter(own_call)
        setattr(handler, _MARK, True)
        root.addHandler(handler)

    logger = logging.getLogger(name)
    # The only level that matters. A record's fate is decided by the level of the
    # logger it came from, never by the root logger's -- so nothing global is
    # touched here, and the handlers stay at NOTSET to take whatever arrives.
    logger.setLevel(logging.INFO)
    return logger


def release_call_logger() -> None:
    """Detach and close this module's handlers.

    Housekeeping for `work`, correctness for `etl`. The filter is what stops a
    stale handler from writing, and it does that whether or not anyone remembers
    to call this. What is left to do is close the file -- and an open file on a
    Volume blocks `reload()`. `work` gets a fresh container per attempt so it
    never meets its own leftovers; `etl` reuses containers and reloads at the top
    of every call, so for that one a leaked descriptor is not untidiness, it is
    the next call failing to start.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _MARK, False):
            root.removeHandler(handler)
            handler.close()
