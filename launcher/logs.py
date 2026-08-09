"""One worker, one logger, one file.

A Modal container is not a process per call. It imports this module once and then
serves many inputs, so anything installed in interpreter globals -- root handlers
above all -- outlives the call that installed it. Teardown alone cannot be the
answer, because teardown is exactly what a crash skips.

So the call id is baked into the logger's *name*, and the handlers only accept
records from that name:

    log = worker_logger(call_id, logfile)   # logger "worker.fc-AAA"
    log.info("boot ...")                    # -> logfile
    log.getChild("count").info("step 3")    # -> logfile, same call
    release_worker_logger()                 # close the file

A handler left behind by an earlier call is then harmless. It sits on the root
logger and sees everything, including the next call's records -- and refuses them,
because they come from `worker.fc-BBB` and it only answers to `worker.fc-AAA`.
Nothing has to be remembered for that to hold.

The flip side is that only this logger is captured. Records from `transformer.*`,
`modal`, `wandb` and any other library are not written to the run's file: they
carry no call id, so there is no honest way to file them under one. Trainer detail
has its own record in the run dir (`run_training` writes events.log), and Python's
lastResort handler still puts library warnings on stderr, where Modal's container
log picks them up.

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
_MARK = "_worker_handler"


def logger_name(call_id: str | None) -> str:
    """The logger one call owns.

    `local` stands in when there is no call id -- a `.local()` run or a unit test
    -- so the name is always well-formed and two such runs in one process still
    share a single, obviously-named logger.
    """
    return f"worker.{call_id or 'local'}"


class WorkerFormatter(logging.Formatter):
    """`<iso-timestamp> <message>`, in UTC.

    The first column is a contract, not a style choice: the dashboard merges every
    worker's log for a run by sorting on it, and that only works because the
    timestamps are one fixed-width UTC format. `converter = time.gmtime` is what
    keeps them UTC even on a machine that thinks otherwise.

    INFO lines stay bare, since by construction everything in this file is the
    worker's own voice. Anything louder is worth marking, so it carries its level.
    """

    converter = time.gmtime

    def __init__(self):
        super().__init__(fmt="%(asctime)s.%(msecs)03dZ %(prefix)s%(message)s", datefmt="%Y-%m-%dT%H:%M:%S")

    def format(self, record):
        record.prefix = "" if record.levelno == logging.INFO else f"{record.levelname} "
        return super().format(record)


def worker_logger(call_id: str | None, logfile: Path) -> logging.Logger:
    """The logger for this call, writing to this file and nothing else to it.

    Hand the returned logger to anything that should be part of the run's story --
    the lease, the job -- and take `getChild` for a sub-voice. Anything under
    `worker.{call_id}.` is family and lands in the same file.

    On a Volume, call this only after `reload()`: reloading fails while a file on
    the volume is open, and this opens one.
    """
    release_worker_logger()  # close whatever file an earlier call left open

    name = logger_name(call_id)

    # The stdlib filter is exactly this rule already: pass `name`, pass anything
    # under `name.`, reject everything else. It is the whole call-id guard, so
    # there is nothing here worth writing a second time.
    own_call = logging.Filter(name)

    root = logging.getLogger()
    for handler in (logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)):
        # Two handlers, on purpose: the file is the durable record that ships with
        # the run folder, the console keeps `modal app logs` working.
        handler.setFormatter(WorkerFormatter())
        handler.addFilter(own_call)
        setattr(handler, _MARK, True)
        root.addHandler(handler)

    logger = logging.getLogger(name)
    # The only level that matters. A record's fate is decided by the level of the
    # logger it came from, never by the root logger's -- so nothing global is
    # touched here, and the handlers stay at NOTSET to take whatever arrives.
    logger.setLevel(logging.INFO)
    return logger


def release_worker_logger() -> None:
    """Detach and close this module's handlers.

    Housekeeping, not correctness: the filter is what stops a stale handler from
    writing, and it does that whether or not anyone remembers to call this. What
    is left to do is close the file, because an open file on a Volume blocks
    `reload()` and a leaked handler is a leaked descriptor.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _MARK, False):
            root.removeHandler(handler)
            handler.close()
