"""Local: atomic artifacts and readable logs. Global: preserve complete, attributable experimental evidence.

Sources: https://docs.python.org/3/library/logging.html; https://docs.python.org/3/library/os.html#os.replace.
"""

import hashlib  # Produces stable seeds and content digests across Python processes.
import json  # Keeps experiment metadata readable without custom serialization.
import logging  # Shares timestamped output between the terminal and a persistent file.
import os  # Atomically replaces finalized files.
import subprocess  # Records version-control provenance without invoking a shell.
import sys  # Routes live logging to the terminal.
import time  # Supplies UTC formatting for log timestamps.
from datetime import UTC, datetime  # Names artifacts independently of machine-local timezone.
from pathlib import Path  # Resolves artifact destinations within the workspace.

import torch  # Serializes tensor and optimizer state with the supported PyTorch format.


def timestamp() -> str:
    """Local: create a UTC run identifier. Global: prevent accidental overwrites of previous experiments."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")  # Microseconds distinguish consecutive invocations.


def seed_for(seed: int, stream: str) -> int:
    """Local: derive independent RNG streams. Global: architecture initialization cannot disturb paired minibatches."""
    digest = hashlib.sha256(f"{seed}:{stream}".encode()).digest()  # Avoids Python's randomized built-in hash.
    return int.from_bytes(digest[:8], "little") % (2**31)  # Fits NumPy, torch, and sklearn seed ranges.


def write_json(path: Path, value) -> None:
    """Local: replace a JSON file atomically. Global: interruption cannot leave a half-written manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)  # Creates only the requested artifact tree.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")  # Keeps the replacement on the same filesystem.
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")  # Rejects invalid metrics.
    os.replace(temporary, path)  # Readers observe either the previous complete file or the new one.


def save_checkpoint(path: Path, value: dict) -> None:
    """Local: atomically persist tensors. Global: long experiments can resume from complete optimizer boundaries."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")  # Avoids overwriting a valid checkpoint in place.
    torch.save(value, temporary)  # Uses tensors and primitive containers compatible with weights_only loading.
    os.replace(temporary, path)  # Commits the saved state as one filesystem operation.


def file_hash(path: Path) -> str:
    """Local: hash a file incrementally. Global: exported weights and evidence can be verified after download."""
    with path.open("rb") as handle:  # Streams large tensor artifacts without loading them into RAM.
        return hashlib.file_digest(handle, "sha256").hexdigest()  # Uses the standard-library digest implementation.


def git_revision() -> str:
    """Local: read the checked-out commit. Global: connect every result to source code that was pushed."""
    result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)  # Read-only query.
    return result.stdout.strip() if result.returncode == 0 else "unversioned"  # Labels missing provenance honestly.


def setup_logging(directory: Path) -> logging.Logger:
    """Local: configure non-truncating handlers. Global: inspect live progress and the identical persistent record."""
    directory.mkdir(parents=True, exist_ok=True)  # Each command has a durable artifact directory.
    logger = logging.getLogger("ofm")  # One named logger is shared across implementation phases.
    logger.setLevel(logging.INFO)  # Emits all research progress and diagnostics.
    logger.propagate = False  # Prevents duplicate lines through global library logging configuration.
    for handler in list(logger.handlers):  # Reconfiguration happens between sequential study runs.
        handler.close()  # Releases the previous run's file handle.
        logger.removeHandler(handler)  # Prevents accidentally writing into an earlier run's log.
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")  # ISO-like UTC prefix.
    formatter.converter = time.gmtime  # Makes timestamps independent of the execution host timezone.
    for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(directory / "run.log", encoding="utf-8")]:
        handler.setFormatter(formatter)  # File and terminal use identical complete messages.
        logger.addHandler(handler)  # Logging handlers flush each emitted record in real time.
    return logger  # Callers log JSON messages or full exception tracebacks, never truncated string slices.


def event(name: str, **fields) -> None:
    """Local: emit structured evidence. Global: the log remains readable while preserving complete measurements."""
    logging.getLogger("ofm").info(json.dumps({"event": name, **fields}, allow_nan=False))  # No text-length limit.
