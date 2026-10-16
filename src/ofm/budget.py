"""Local: crash-conservative GPU reservations. Global: all commands share a persistent 24-hour ceiling.

Source: https://docs.python.org/3/library/fcntl.html. This ledger targets Linux/macOS/WSL.
Reservations are charged before execution and unused time is refunded only on an orderly exit.
"""

import fcntl  # Prevents concurrent commands from spending the same remaining allocation.
import json  # Keeps the cumulative ledger human-reviewable.
import os  # Records the process owning a reservation.
import time  # Monotonic elapsed time is immune to wall-clock adjustments.
from pathlib import Path  # Stores the lock alongside the experiment ledger.

from ofm.artifacts import timestamp, write_json  # Reuses atomic metadata handling.


class BudgetExhausted(RuntimeError):
    """Local: stop a compute phase. Global: exhaustion is a recorded outcome, not permission to renew the budget."""


class BudgetSession:
    """Local: reserve one bounded operation. Global: crashes keep their full reservation charged."""

    def __init__(self, path: Path, limit_seconds: float, requested_seconds: float, label: str, enabled: bool = True):
        """Local: store reservation inputs. Global: CPU operations retain deadlines without consuming GPU allocation."""
        self.path, self.limit, self.requested = Path(path), float(limit_seconds), float(requested_seconds)  # Normalize inputs.
        self.label, self.enabled = label, enabled  # Labels attribute consumption to an experimental phase.
        self.handle = None  # The lock is acquired only when work enters the context.
        self.start = 0.0  # Actual elapsed time starts after a successful reservation.
        self.reserved = self.requested  # CPU-only sessions retain their requested wall-time deadline.

    def __enter__(self):
        """Local: atomically reserve available time. Global: multiple runs cannot overbook the cumulative cap."""
        if self.enabled:  # GPU work participates in the persistent ledger.
            self.path.parent.mkdir(parents=True, exist_ok=True)  # Creates the local experiment root.
            self.handle = self.path.with_suffix(".lock").open("a+")  # A stable inode survives atomic JSON replacement.
            try:  # Failed entry must not retain a lock or open file.
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # Rejects overlapping GPU commands.
                self.ledger = json.loads(self.path.read_text()) if self.path.exists() else {  # Restores previous spending.
                    "limit_seconds": self.limit, "charged_seconds": 0.0, "active": None, "history": []
                }
                if self.ledger["limit_seconds"] != self.limit:  # Prevents changing the cap through a later CLI argument.
                    raise ValueError("Budget limit differs from the existing ledger; record an explicit protocol revision")
                if self.ledger["active"] is not None:  # A prior process exited without refunding its reservation.
                    self.ledger["history"].append({**self.ledger["active"], "status": "interrupted_full_charge"})  # Audit crash.
                available = self.limit - self.ledger["charged_seconds"]  # Full outstanding reservations remain charged.
                self.reserved = min(self.requested, available)  # Never reserve beyond remaining authorized time.
                if self.reserved <= 0:  # No operation may start after exhaustion.
                    raise BudgetExhausted("The cumulative GPU budget is exhausted")  # Preserve a recognizable stop reason.
                self.ledger["charged_seconds"] += self.reserved  # Charge before the first GPU operation.
                self.ledger["active"] = {"label": self.label, "pid": os.getpid(), "started": timestamp(),
                                         "reserved_seconds": self.reserved}  # Crash recovery needs no live process state.
                write_json(self.path, self.ledger)  # Commits the reservation while holding the exclusive lock.
            except BaseException:  # Covers ordinary validation failures and user interruption during entry.
                self.handle.close()  # Closing also releases any acquired flock.
                self.handle = None  # Prevents later double-close attempts.
                raise  # Let the command record the actual failure.
        self.start = time.monotonic()  # Measures occupied time, including GPU stalls and checkpoint work.
        return self  # Callers check remaining time at bounded computation boundaries.

    @property
    def remaining(self) -> float:
        """Local: compute remaining reservation. Global: loops stop before their allotted compute expires."""
        return max(0.0, self.reserved - (time.monotonic() - self.start))  # Clamps only the displayed remaining duration.

    def check(self, margin: float = 0.0) -> None:
        """Local: guard the next bounded operation. Global: stop with a checkpoint before consuming the last seconds."""
        if self.remaining <= margin:  # Training/sampling choose a margin larger than their measured batch latency.
            raise BudgetExhausted(f"Reservation exhausted for {self.label}")  # Caller preserves partial outputs.

    def __exit__(self, exc_type, exc_value, traceback):
        """Local: refund unused reserved time. Global: completed work remains charged across every future restart."""
        if self.enabled and self.handle is not None:  # CPU-only operations leave the GPU ledger untouched.
            elapsed = time.monotonic() - self.start  # Includes cleanup performed inside the context.
            self.ledger["charged_seconds"] += elapsed - self.reserved  # Refunds unused time; records overruns honestly.
            record = {**self.ledger["active"], "elapsed_seconds": elapsed,
                      "status": "completed" if exc_type is None else exc_type.__name__}  # Retains failed attempts as evidence.
            self.ledger["history"].append(record)  # Appends rather than overwriting historical consumption.
            self.ledger["active"] = None  # Allows the next serialized GPU operation to reserve time.
            write_json(self.path, self.ledger)  # Persists usage before releasing the lock.
            self.handle.close()  # Releases the exclusive reservation lock.
        return False  # Exceptions are never hidden by accounting cleanup.
