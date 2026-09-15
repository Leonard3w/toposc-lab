import json
import subprocess
import sys
import time

import pytest

from toposc_lab.discovery.config import DiscoveryConfig
from toposc_live.configuration import LaunchPlan
from toposc_live.processes import (
    ProcessService,
    detached_options,
    engine_command,
    process_alive,
    read_metadata,
    write_metadata,
)


def test_command_is_official_cli_without_shell(tmp_path):
    plan = LaunchPlan("name", tmp_path / "with spaces & symbols", DiscoveryConfig(), iterations=1)
    command = engine_command(plan, tmp_path / "config.json", "python")
    assert command == [
        "python",
        "-u",
        "-m",
        "toposc_lab.discovery",
        str(plan.directory),
        "--config",
        str(tmp_path / "config.json"),
        "--iterations",
        "1",
    ]


def test_process_construction_and_reproducibility(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: calls.append((args, kwargs)))
    service = ProcessService(tmp_path / "ui-state")
    plan = LaunchPlan("fixture", tmp_path / "science", DiscoveryConfig())
    request_path = service.launch(plan)
    args, options = calls[0]
    assert args[0][3] == "toposc_live.worker"
    assert options["stdin"] == subprocess.DEVNULL
    assert options["close_fds"] is True
    assert not options.get("shell", False)
    assert "env" not in options  # Original runtime environment is inherited.
    assert all(options[k] == v for k, v in detached_options().items())
    assert list(plan.directory.iterdir()) == []
    request = read_metadata(request_path)
    assert request["config_sha256"] == plan.config.fingerprint
    stored = json.loads(request_path.with_name("config.json").read_text())
    assert DiscoveryConfig(**stored) == plan.config
    with pytest.raises(ValueError, match="active"):
        service.launch(plan)


def test_spawn_failure_visible(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("fixture failed")

    monkeypatch.setattr(subprocess, "Popen", fail)
    service = ProcessService(tmp_path / "state")
    plan = LaunchPlan("test", tmp_path / "science", DiscoveryConfig())
    with pytest.raises(OSError):
        service.launch(plan)
    assert service.inspect(plan.directory)["status"] == "Failed"


def test_pid_identity_reconnect(tmp_path):
    psutil = pytest.importorskip("psutil")
    current = psutil.Process()
    assert process_alive({"pid": current.pid, "create_time": current.create_time()}) is True
    assert process_alive({"pid": current.pid, "create_time": current.create_time() - 100}) is False
    assert process_alive({}) is None


def test_detached_supervisor_survives_launcher_exit(tmp_path):
    """Launch a harmless sleeping Python process, never the discovery engine."""
    pytest.importorskip("psutil")
    job = tmp_path / "jobs/fixture"
    request_path = job / "request.json"
    marker = tmp_path / "finished.txt"
    child_code = (
        "import time; from pathlib import Path; time.sleep(1.5); Path("
        + repr(str(marker))
        + ").write_text('complete')"
    )
    write_metadata(
        request_path,
        {
            "directory": str(tmp_path / "campaign"),
            "requested_at": time.time(),
            "command": [sys.executable, "-c", child_code],
        },
    )
    launcher_code = """
import subprocess, sys
from pathlib import Path
from toposc_live.processes import detached_options
subprocess.Popen([sys.executable, '-m', 'toposc_live.worker', sys.argv[1]],
 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
 close_fds=True, **detached_options())
"""
    subprocess.run([sys.executable, "-c", launcher_code, str(request_path)], check=True, timeout=10)
    service = ProcessService(tmp_path)
    deadline = time.monotonic() + 15
    statuses = []
    while time.monotonic() < deadline:
        status = service.inspect(tmp_path / "campaign").get("status")
        statuses.append(status)
        if status in ("Exited", "Failed"):
            break
        time.sleep(0.05)
    assert "Running" in statuses
    assert statuses[-1] == "Exited"
    assert marker.read_text() == "complete"
    assert read_metadata(job / "state.json")["exit_code"] == 0


def test_worker_exit_failure_and_log(tmp_path):
    pytest.importorskip("psutil")
    from toposc_live.worker import run

    request = tmp_path / "request.json"
    write_metadata(
        request, {"command": [sys.executable, "-c", "print('fixture stderr'); raise SystemExit(3)"]}
    )
    assert run(request) == 3
    assert read_metadata(tmp_path / "state.json")["status"] == "Failed"
    assert "fixture stderr" in ProcessService(tmp_path).log_tail({"request_path": str(request)})


def test_official_cli_zero_cycle_launch_and_reconnect(tmp_path):
    """Exercise the production launch path with zero numerical evaluations."""
    pytest.importorskip("psutil")
    from toposc_lab.discovery.storage import read_json

    service = ProcessService(tmp_path / "application-state")
    directory = tmp_path / "zero-cycle-test-fixture"
    plan = LaunchPlan(
        "ZERO CYCLE TEST FIXTURE", directory, DiscoveryConfig(plots=False), iterations=0
    )
    service.launch(plan)
    # A fresh service represents a reopened GUI; no retained Popen handle is needed.
    reconnected = ProcessService(service.state_directory)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        job = reconnected.inspect(directory)
        if job.get("status") in ("Exited", "Failed"):
            break
        time.sleep(0.05)
    assert job["status"] == "Exited", reconnected.log_tail(job)
    assert read_json(directory / "manifest.json")["config_sha256"] == plan.config.fingerprint
    assert read_json(directory / "checkpoint.json")["completed_cycles"] == 0
    assert read_json(directory / "summary.json")["exact_attempts"] == 0
    assert not (directory / "attempts").exists()
    assert "exact_attempts" in reconnected.log_tail(job)


def test_malformed_optional_process_metadata(tmp_path):
    request = tmp_path / "jobs/test/request.json"
    write_metadata(request, {"directory": str(tmp_path / "campaign"), "requested_at": "broken"})
    write_metadata(request.with_name("state.json"), {"status": "Running", "engine": "broken"})
    assert ProcessService(tmp_path).inspect(tmp_path / "campaign")["status"] == "Unknown"
