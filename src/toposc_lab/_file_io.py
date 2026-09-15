"""Filesystem mechanics only: bounded Windows namespace contention."""

import os
import stat
import sys
import time
from collections.abc import Callable
from pathlib import Path


def _retry_windows(operation: Callable[[], None], description: str) -> None:
    # Eight attempts, at most 1.13 seconds of sleeping. Never rerun serialization,
    # numerical work or a transaction; only retry a failed OS namespace operation.
    delays = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.5)
    for attempt in range(len(delays) + 1):
        try:
            operation()
            return
        except OSError as error:
            if sys.platform != "win32" or getattr(error, "winerror", None) not in (5, 32, 33):
                raise
            if attempt == len(delays):
                error.add_note(
                    f"{description}: failed after 8 attempts / 1.13 s backoff; "
                    "persistent access/sharing failure. No non-atomic fallback used."
                )
                raise
            time.sleep(delays[attempt])


def replace_atomic(source: str | Path, destination: Path) -> None:
    """Retry only failed replacement, retaining the original OS exception on failure.

    Windows error 5 can mean a transient reader OR a permanent ACL denial. It is
    not possible to distinguish these from that code alone: known invalid targets
    fail immediately, other denials receive bounded retries and then propagate.
    Never chmod, unlink the destination, copy over it, or change caller policy.
    """

    def replace() -> None:
        try:
            os.replace(source, destination)
        except OSError as error:
            if sys.platform == "win32" and getattr(error, "winerror", None) == 5:
                try:
                    target = destination.stat()
                except OSError:
                    pass
                else:
                    if not stat.S_ISREG(target.st_mode) or (
                        getattr(target, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_READONLY
                    ):
                        # Exclude a known permanent configuration/attribute failure.
                        error.add_note("Replacement target is not a writable regular file.")
                        raise _PermanentReplaceError(error) from error
            raise

    try:
        _retry_windows(replace, f"Atomic replacement of {destination}")
    except _PermanentReplaceError as failure:
        raise failure.original


class _PermanentReplaceError(Exception):
    def __init__(self, original: OSError) -> None:
        self.original = original


def remove_temporary(path: Path) -> None:
    """Delete only the caller-owned temporary; do not mask an in-flight failure."""
    original = sys.exception()
    try:
        _retry_windows(lambda: path.unlink(missing_ok=True), f"Temporary cleanup of {path}")
    except OSError as cleanup:
        if original is None:
            raise
        original.add_note(f"Temporary cleanup also failed; retained {path}: {cleanup}")
