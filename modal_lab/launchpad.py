"""Liveness of a Modal FunctionCall, derived purely from a non-blocking poll.

Single source of truth for launch.ipynb's dashboard and its tests. `status`
maps every outcome of `FunctionCall.get(timeout=0)` to one state string --
verified against modal 1.5.2 (`_functions.py:poll_function` +
`_utils/function_utils.py:_process_result`).
"""

from modal.exception import Error, FunctionTimeoutError, OutputExpiredError


async def status(call):
    """Map every outcome of `call.get(timeout=0)` to `(call, state, detail)`.

      - still running        -> builtin TimeoutError   (poll returned no output)
      - success              -> clean return            (value may be None)
      - result GC'd (too old)-> OutputExpiredError
      - hit its own timeout=  -> FunctionTimeoutError
      - infra fault          -> other modal Error       (RemoteError/InternalFailure)
      - your code raised      -> that exception, re-raised on get()

    Ordering matters: OutputExpiredError/FunctionTimeoutError subclass *modal's*
    TimeoutError, which is NOT the builtin -- so `except TimeoutError` catches
    only "running", and those two must be caught before the generic Error branch.

    `detail` carries the return value on "done" and the exception on
    "failed"/"crashed"; it is None otherwise.
    """
    try:
        result = await call.get.aio(timeout=0)
        return call, "done", result
    except TimeoutError:  # builtin -- still executing
        return call, "running", None
    except OutputExpiredError:  # result garbage-collected; outcome unknowable now
        return call, "expired", None
    except FunctionTimeoutError:  # exceeded its own timeout=
        return call, "timed_out", None
    except Error as e:  # any other modal-side failure (RemoteError, ...)
        return call, "failed", e
    except Exception as e:  # your function raised -> re-raised on get()
        return call, "crashed", e


# Every state `status` can return -- tests and the dashboard both assert against this.
STATES = frozenset({"done", "running", "expired", "timed_out", "failed", "crashed"})
