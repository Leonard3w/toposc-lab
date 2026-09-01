"""Simple console runner and process monitor for the frozen Phase-9.8 experiment."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import ctypes
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import TextIO, TypeAlias

from toposc_lab.search import (
    Phase98ProgressEvent,
    Phase98ProgressStage,
    run_phase_9_8_dry_run,
    run_phase_9_8_random_search,
)


_MemoryReader: TypeAlias = Callable[[], tuple[int | None, int | None]]
_Clock: TypeAlias = Callable[[], float]

_STAGE_LABELS = {
    Phase98ProgressStage.STARTING: "Start",
    Phase98ProgressStage.DRY_RUN: "Dry-Run",
    Phase98ProgressStage.REFERENCES: "Referenzen",
    Phase98ProgressStage.SEARCH: "Suche",
    Phase98ProgressStage.SELECTION: "Auswahl",
    Phase98ProgressStage.VALIDATION: "Validierung",
    Phase98ProgressStage.CONFIRMATION: "Bestätigung",
    Phase98ProgressStage.COMPLETE: "Fertig",
}


class _WindowsProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


def _process_memory_bytes() -> tuple[int | None, int | None]:
    """Return current and peak resident process memory when Windows exposes it."""
    if os.name != "nt":
        return None, None
    counters = _WindowsProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_WindowsProcessMemoryCounters),
            ctypes.c_ulong,
        )
        get_process_memory_info.restype = ctypes.c_int
        succeeded = get_process_memory_info(
            get_current_process(),
            ctypes.byref(counters),
            counters.cb,
        )
    except (AttributeError, OSError):
        return None, None
    if not succeeded:
        return None, None
    return int(counters.working_set_size), int(counters.peak_working_set_size)


class Phase98ConsoleMonitor:
    """Render progress plus observational wall-time, CPU, and memory metrics."""

    def __init__(
        self,
        *,
        output: TextIO = sys.stdout,
        wall_clock: _Clock = time.perf_counter,
        cpu_clock: _Clock = time.process_time,
        memory_reader: _MemoryReader = _process_memory_bytes,
        logical_cpu_count: int | None = None,
    ) -> None:
        self._output = output
        self._wall_clock = wall_clock
        self._cpu_clock = cpu_clock
        self._memory_reader = memory_reader
        self._logical_cpu_count = logical_cpu_count or os.cpu_count() or 1
        self._started_at = wall_clock()
        self._cpu_started_at = cpu_clock()
        self._stage: Phase98ProgressStage | None = None
        self._stage_started_at = self._started_at

    @property
    def logical_cpu_count(self) -> int:
        """Number of logical CPUs used to normalize total machine capacity."""
        return self._logical_cpu_count

    def __call__(self, event: Phase98ProgressEvent) -> None:
        now = self._wall_clock()
        if event.stage is not self._stage:
            self._stage = event.stage
            self._stage_started_at = now
        elapsed = max(0.0, now - self._started_at)
        stage_elapsed = max(0.0, now - self._stage_started_at)
        cpu_seconds = max(0.0, self._cpu_clock() - self._cpu_started_at)
        average_cores = cpu_seconds / elapsed if elapsed > 0.0 else 0.0
        total_capacity = 100.0 * average_cores / self._logical_cpu_count
        current_memory, peak_memory = self._memory_reader()
        progress = _format_progress(event)
        eta = _estimate_remaining(stage_elapsed, event.completed, event.total)
        metrics = (
            f"Zeit {_format_duration(elapsed)} | ETA {_format_duration(eta)} | "
            f"CPU {average_cores:.2f} Kerne ({total_capacity:.1f}% von "
            f"{self._logical_cpu_count}) | RAM {_format_memory(current_memory)} | "
            f"Peak {_format_memory(peak_memory)}"
        )
        print(
            f"[{_STAGE_LABELS[event.stage]}] {progress} - {event.message}\n"
            f"    {metrics}",
            file=self._output,
            flush=True,
        )


def run_phase_9_8_command(args: argparse.Namespace) -> int:
    """Execute the user-facing dry or full Phase-9.8 command."""
    monitor = Phase98ConsoleMonitor()
    print("TopoSC Lab - Phase 9.8", flush=True)
    print(f"Logische CPUs: {monitor.logical_cpu_count}", flush=True)
    if args.dry_run:
        if args.output is not None:
            raise ValueError("--output ist nur zusammen mit --full erlaubt")
        print("Modus: kurzer Geometrie-Dry-Run ohne wissenschaftliche Suchseeds", flush=True)
        records = run_phase_9_8_dry_run(progress=monitor)
        print(
            "Dry-Run erfolgreich: "
            f"{len(records)} Seeds, maximal "
            f"{max(item.complete_attempt_count for item in records)} Versuche und "
            f"{max(item.proposal_count for item in records)} Punktvorschläge.",
            flush=True,
        )
        return 0

    code_commit = _git_head_commit()
    destination = args.output or _default_output_directory()
    print("Modus: vollständiger eingefrorener Lauf/Reproduktionslauf", flush=True)
    print(f"Code-Commit: {code_commit}", flush=True)
    print(f"Ausgabe: {destination.resolve()}", flush=True)
    result = run_phase_9_8_random_search(
        destination,
        code_commit=code_commit,
        progress=monitor,
    )
    print("\nErgebnis", flush=True)
    print(f"  Trials: {len(result.trials)}", flush=True)
    print(f"  Ausgewählte Kandidaten: {len(result.selected_candidates)}", flush=True)
    print(f"  Validierungsberichte: {len(result.validation)}", flush=True)
    print(f"  Confirmation ausgelöst: {result.confirmation_triggered}", flush=True)
    print(f"  Zusammenfassung: {result.summary_manifest}", flush=True)
    return 0


def _git_head_commit() -> str:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            (
                "git",
                "-c",
                f"safe.directory={repository_root.as_posix()}",
                "-C",
                str(repository_root),
                "rev-parse",
                "HEAD",
            ),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Git HEAD konnte nicht gelesen werden") from error
    commit = completed.stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("Git HEAD konnte nicht als vollständiger Commit gelesen werden")
    return commit


def _default_output_directory() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"phase_9_8_random_search_v1_{timestamp}"


def _estimate_remaining(elapsed: float, completed: int, total: int) -> float | None:
    if completed <= 0 or total <= 0:
        return None
    return max(0.0, elapsed / completed * (total - completed))


def _format_progress(event: Phase98ProgressEvent) -> str:
    if event.total == 0:
        return "übersprungen"
    percentage = 100.0 * event.completed / event.total
    return f"{event.completed}/{event.total} ({percentage:5.1f}%)"


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    rounded = max(0, round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"


def _format_memory(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "n/v"
    return f"{size_bytes / (1024.0 * 1024.0):.1f} MiB"
