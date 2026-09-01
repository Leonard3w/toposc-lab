from __future__ import annotations

from io import StringIO

from toposc_lab.phase_9_8_cli import Phase98ConsoleMonitor
from toposc_lab.search import Phase98ProgressEvent, Phase98ProgressStage


def test_console_monitor_formats_progress_eta_cpu_and_memory() -> None:
    wall_values = iter((0.0, 1.0, 5.0))
    cpu_values = iter((0.0, 0.5, 3.0))
    output = StringIO()
    monitor = Phase98ConsoleMonitor(
        output=output,
        wall_clock=lambda: next(wall_values),
        cpu_clock=lambda: next(cpu_values),
        memory_reader=lambda: (128 * 1024**2, 256 * 1024**2),
        logical_cpu_count=8,
    )

    monitor(
        Phase98ProgressEvent(
            stage=Phase98ProgressStage.SEARCH,
            completed=0,
            total=4,
            message="started",
        )
    )
    monitor(
        Phase98ProgressEvent(
            stage=Phase98ProgressStage.SEARCH,
            completed=1,
            total=4,
            message="trial sealed",
        )
    )

    rendered = output.getvalue()
    assert "[Suche] 1/4 ( 25.0%)" in rendered
    assert "Zeit 00:00:05" in rendered
    assert "ETA 00:00:12" in rendered
    assert "CPU 0.60 Kerne (7.5% von 8)" in rendered
    assert "RAM 128.0 MiB" in rendered
    assert "Peak 256.0 MiB" in rendered
