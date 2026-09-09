"""Environment checks and observational heartbeat for the frozen research CLI."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
import threading
import time
from contextlib import redirect_stdout
from datetime import UTC, datetime
from importlib.metadata import version
from io import StringIO
from pathlib import Path
from typing import Any, Self, TextIO

import numpy as np

from toposc_lab.phase_9_8_cli import _process_memory_bytes
from toposc_lab.search._research_storage import ResearchAbort, json_bytes
from toposc_lab.search.phase_10_research import (
    RESEARCH_PROTOCOL_COMMIT,
    RESEARCH_PROTOCOL_PATH,
    THREAD_VARIABLES,
)


def research_environment() -> dict[str, Any]:
    """Require committed implementation and the frozen numerical runtime before writing."""
    root = Path(__file__).resolve().parents[3]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()

    if sys.version_info[:2] != (3, 14) or not sys.dont_write_bytecode:
        raise ValueError("Python 3.14 und PYTHONDONTWRITEBYTECODE=1 sind erforderlich")
    if any(os.environ.get(name) != "1" for name in THREAD_VARIABLES):
        raise ValueError(
            "Vor Pythonstart OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS "
            "und BLIS_NUM_THREADS auf 1 setzen (siehe Bedienungsanleitung)"
        )
    code = git("rev-parse", "HEAD")
    if code == RESEARCH_PROTOCOL_COMMIT:
        raise ValueError("Die Implementierung muss nach dem Protokoll separat committed sein")
    git("merge-base", "--is-ancestor", RESEARCH_PROTOCOL_COMMIT, code)
    allowed = {
        "src/toposc_lab/observables/__pycache__/__init__.cpython-314.pyc",
        "src/toposc_lab/observables/__pycache__/spectrum.cpython-314.pyc",
        "geometry_demo.npz",
    }
    # Do not strip leading status spaces: a worktree-only modification is ' M'.
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    unexpected = [
        line for line in status.splitlines() if line[3:].replace("\\", "/") not in allowed
    ]
    if unexpected:
        raise ValueError("Worktree vor Forschungslauf committen: " + "; ".join(unexpected))
    frozen = subprocess.run(
        ["git", "-C", str(root), "show", f"{RESEARCH_PROTOCOL_COMMIT}:{RESEARCH_PROTOCOL_PATH}"],
        check=True,
        capture_output=True,
    ).stdout.replace(b"\r\n", b"\n")
    current = (root / RESEARCH_PROTOCOL_PATH).read_bytes().replace(b"\r\n", b"\n")
    if current != frozen:
        raise ValueError("Protokoll weicht vom eingefrorenen Commit ab")
    output = StringIO()
    with redirect_stdout(output):
        np.show_config()
    return {
        "code_commit": code,
        "protocol_commit": RESEARCH_PROTOCOL_COMMIT,
        "protocol_sha256": hashlib.sha256(frozen).hexdigest(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "matplotlib", "pydantic")},
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "numpy_backend": output.getvalue(),
        "threads": {k: os.environ[k] for k in THREAD_VARIABLES},
    }


class ResearchMonitor:
    """One JSONL writer and heartbeat; instrumentation never makes scientific decisions."""

    def __init__(self, path: Path, *, stream: TextIO | None = None, interval: float = 10.0) -> None:
        self.path = path
        self.stream = stream if stream is not None else sys.stdout
        self.interval = interval
        self.started = time.perf_counter()
        self.cpu_started = time.process_time()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.failure: BaseException | None = None
        self.measurements: dict[str, dict[str, float]] = {}
        self.evaluations_observed = 0
        self.state: dict[str, Any] = {"completed": 0, "total": 0, "last_sealed": None}
        self.thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        # Preserve a crash-truncated diagnostic line and explicitly separate the next event.
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as stream:
                stream.seek(-1, os.SEEK_END)
                complete = stream.read(1) == b"\n"
            if not complete:
                with self.path.open("ab") as stream:
                    stream.write(b"\n")
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=self.interval + 1)

    def _heartbeat(self) -> None:
        while not self.stop.wait(self.interval):
            try:
                self.emit("heartbeat")
            except (Exception, ResearchAbort) as error:  # noqa: BLE001 - propagate monitor failure to main thread
                self.failure = error
                return

    def emit(self, event: str, **changes: Any) -> None:
        if self.failure is not None:
            raise ResearchAbort(f"monitor failed: {self.failure}")
        with self.lock:
            self.state.update(changes)
            elapsed = time.perf_counter() - self.started
            if "operation_seconds" in changes and "category" in changes:
                totals = self.measurements.setdefault(
                    changes["category"], {"count": 0.0, "seconds": 0.0}
                )
                totals["count"] += 1
                totals["seconds"] += changes["operation_seconds"]
                if changes["category"] == "physics":
                    self.evaluations_observed += 1
                    self.state["estimated_seconds_per_evaluation"] = (
                        elapsed / self.evaluations_observed
                    )
            completed, total = self.state["completed"], self.state["total"]
            # completed includes only this execution's logical progress; after resume
            # ETA is supplied separately from the saved preflight or remains unknown.
            eta = self.state.get("estimated_seconds_per_evaluation")
            remaining = max(0, total - completed)
            eta = None if eta is None else eta * remaining
            memory, peak = _process_memory_bytes()
            entry = dict(
                self.state,
                event=event,
                timestamp=datetime.now(UTC).isoformat(),
                elapsed_seconds=elapsed,
                remaining=remaining,
                eta_seconds=eta,
                cpu_cores=(time.process_time() - self.cpu_started) / max(elapsed, 1e-12),
                memory_bytes=memory,
                peak_memory_bytes=peak,
            )
            try:
                with self.path.open("ab") as stream:
                    stream.write(json_bytes(entry))
                    stream.flush()
                label = " ".join(
                    f"{k}={entry[k]}"
                    for k in (
                        "stage", "block", "trial", "arm", "model_role", "size", "method",
                        "amplitude", "edge_step", "generation", "slot", "hits",
                        "classification", "clean_eligible", "screening_pass",
                    )
                    if k in entry
                )
                eta_text = "offen" if eta is None else f"{eta / 60:.1f} min"
                ram = "?" if memory is None else f"{memory / 1024**2:.0f} MiB"
                print(
                    f"[{event}] {label} | {completed}/{total} | Zeit {elapsed:.1f}s | ETA {eta_text} "
                    f"| CPU {entry['cpu_cores']:.2f} Kerne | RAM {ram} | "
                    f"versiegelt={entry['last_sealed']}",
                    file=self.stream,
                    flush=True,
                )
            except Exception as error:
                raise ResearchAbort(f"cannot write progress: {error}") from error
