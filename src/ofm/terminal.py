"""Local: full stdout/stderr capture. Global: library progress, tracebacks, and command results survive terminal truncation.

Source: https://docs.python.org/3/library/contextlib.html#contextlib.redirect_stdout.
Runtime commands contain no LLM calls; future prompt/response logging must use this untruncated path without credentials.
"""

import contextlib  # Redirects Python streams while preserving their original terminal destinations.
import sys  # Supplies original output/error streams and command arguments.
import threading  # Serializes writes from stdout and stderr into one complete command log.
import traceback  # Captures exception details before the redirected streams close.
from pathlib import Path  # Keeps command logs inside the configured run root.

from ofm.artifacts import timestamp  # Shares UTC timestamps with model artifacts.


class Tee:
    """Local: mirror a text stream. Global: never shorten prompts, results, warnings, or tracebacks."""

    def __init__(self, terminal, log, lock, name):
        """Local: retain output destinations. Global: terminal and file receive every write immediately."""
        self.terminal, self.log, self.lock, self.name = terminal, log, lock, name  # Share one lock across both streams.

    def write(self, value):
        """Local: duplicate complete text. Global: timestamp every emitted chunk without imposing a length limit."""
        with self.lock:  # Prevent interleaved library messages from corrupting the persistent transcript.
            self.terminal.write(value)  # Preserve ordinary CLI rendering, including progress carriage returns.
            self.terminal.flush()  # Make long operations visible in real time.
            if value:  # Empty writes contain no additional evidence.
                self.log.write(f"{timestamp()} {self.name} {value}")  # Prefix the full chunk; never slice its content.
                self.log.flush()  # A crash retains every completed write.
        return len(value)  # Match the standard text-stream write contract.

    def flush(self):
        """Local: flush both destinations. Global: logging handlers can use the tee as an ordinary stream."""
        self.terminal.flush()  # Preserve interactive progress behavior.
        self.log.flush()  # Persist buffered command output immediately.

    def isatty(self):
        """Local: identify a captured stream. Global: discourage enormous animated terminal redraw sequences."""
        return False  # Libraries choose ordinary textual progress for this persistent output stream.


@contextlib.contextmanager
def capture_terminal(root: Path):
    """Local: capture one entire CLI invocation. Global: all emitted text is recoverable from a timestamped local file."""
    directory = Path(root) / "commands"  # Separates raw command transcripts from structured per-run progress.
    directory.mkdir(parents=True, exist_ok=True)  # A fresh checkout needs no manual logging setup.
    with (directory / f"{timestamp()}.log").open("w", encoding="utf-8") as log:  # Unique command attempt, never overwrite.
        lock = threading.RLock()  # Allows nested logging flushes while retaining cross-stream serialization.
        with contextlib.redirect_stdout(Tee(sys.stdout, log, lock, "stdout")), \
             contextlib.redirect_stderr(Tee(sys.stderr, log, lock, "stderr")):  # Captures third-party Python output too.
            try:  # Ensure unexpected failures remain in the same command transcript.
                yield  # The caller runs parsing, phase dispatch, and complete JSON printing inside the capture.
            except BaseException:  # Include interruptions and argument parsing failures without hiding them.
                traceback.print_exc()  # Full stack trace is written to terminal and file before restoration.
                raise  # Preserve the actual nonzero command exit for study orchestration.
