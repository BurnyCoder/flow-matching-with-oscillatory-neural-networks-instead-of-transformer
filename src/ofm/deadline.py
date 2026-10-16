"""Local: independent process deadline. Global: a blocked metric call cannot spend past a reserved GPU allocation.

Sources: https://docs.python.org/3/library/select.html; https://docs.python.org/3/library/signal.html.
This helper imports no numerical packages and consumes no GPU resources.
"""

import os  # Terminates only the specific worker process supplied by its parent.
import select  # Waits for parent-pipe closure with an operating-system timeout.
import signal  # SIGKILL stops a worker even while it is blocked inside native numerical code.
import sys  # Reads the worker PID, timeout, and cancellation pipe.


def main() -> None:
    """Local: wait for cancellation or expiry. Global: hard stops leave the full reservation charged for crash recovery."""
    pid, seconds = int(sys.argv[1]), float(sys.argv[2])  # Inputs are generated internally, never shell-interpolated.
    ready, _, _ = select.select([sys.stdin], [], [], max(0, seconds))  # Closing the parent's pipe cancels the deadline.
    if not ready:  # No cancellation arrived before the reserved duration expired.
        try:  # The parent may have exited in the narrow interval after timeout.
            os.kill(pid, signal.SIGKILL)  # Releases the worker's CUDA context without allowing further model computation.
        except ProcessLookupError:  # Already-exited workers require no action.
            pass  # Their pipe closure or previous failure already ended GPU work.


if __name__ == "__main__":  # Invoked only as a subprocess by BudgetSession.
    main()  # Keeps deadline enforcement independent of the worker's Python interpreter lock.
