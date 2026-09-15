"""Filesystem-only regression fixtures; no production campaigns are executed."""

import errno
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from toposc_lab import _file_io as io
from toposc_lab.discovery.storage import atomic_json, read_json
from toposc_live.processes import ProcessService, read_metadata, write_metadata
from toposc_live.reader import ArtifactCache, CampaignReader


def windows_error(code):
    error = OSError(errno.EACCES, "fixture contention")
    error.winerror = code
    return error


def test_ordinary_write_existing_replacement_and_encoding(tmp_path):
    path = tmp_path / "attempts/000150.json"
    atomic_json(path, {"status": "started", "unicode": "Ω"})
    assert read_json(path)["status"] == "started"
    atomic_json(path, {"status": "complete"})
    assert read_json(path) == {"status": "complete"}
    assert not list(path.parent.glob("*.tmp"))


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_retry_same_synced_temp_no_partial_json(tmp_path, monkeypatch, code):
    path = tmp_path / "artifact.json"
    atomic_json(path, {"old": True})
    real_replace = os.replace
    calls, sleeps, sources = [], [], []
    monkeypatch.setattr(io.sys, "platform", "win32")
    monkeypatch.setattr(io.time, "sleep", sleeps.append)

    def replace(source, destination):
        calls.append(destination)
        sources.append(Path(source))
        assert Path(source).parent == destination.parent
        assert read_json(Path(source)) == {"new": "x" * 4096}
        assert read_json(destination) == {"old": True}
        if len(calls) <= 3:
            raise windows_error(code)
        real_replace(source, destination)

    monkeypatch.setattr(io.os, "replace", replace)
    atomic_json(path, {"new": "x" * 4096})
    assert sleeps == [0.01, 0.02, 0.04]
    assert len(set(sources)) == 1
    assert len(calls) == 4
    assert read_json(path) == {"new": "x" * 4096}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("code", [5, 32, 33])
def test_exhaustion_keeps_original_exception_old_file_and_cleans(tmp_path, monkeypatch, code):
    path = tmp_path / "artifact.json"
    atomic_json(path, {"old": True})
    original = path.read_bytes()
    calls, sleeps = [], []
    error = windows_error(code)

    def fail(*args):
        calls.append(args)
        raise error

    monkeypatch.setattr(io.sys, "platform", "win32")
    monkeypatch.setattr(io.os, "replace", fail)
    monkeypatch.setattr(io.time, "sleep", sleeps.append)
    with pytest.raises(OSError) as caught:
        atomic_json(path, {"new": True})
    assert caught.value is error
    assert len(calls) == 8
    assert sum(sleeps) == pytest.approx(1.13)
    assert any("failed after 8 attempts" in n for n in error.__notes__)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("code", [2, 3, 17, 87, 112, None])
def test_other_errors_do_not_retry(tmp_path, monkeypatch, code):
    path = tmp_path / "artifact.json"
    error = windows_error(code)
    calls = []

    def fail(*args):
        calls.append(args)
        raise error

    monkeypatch.setattr(io.sys, "platform", "win32")
    monkeypatch.setattr(io.os, "replace", fail)
    monkeypatch.setattr(io.time, "sleep", lambda seconds: pytest.fail("unexpected retry"))
    with pytest.raises(OSError):
        atomic_json(path, {})
    assert len(calls) == 1
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_non_windows_eacces_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(io.sys, "platform", "linux")
    monkeypatch.setattr(io.os, "replace", lambda *args: (_ for _ in ()).throw(windows_error(5)))
    monkeypatch.setattr(io.time, "sleep", lambda seconds: pytest.fail("unexpected retry"))
    with pytest.raises(OSError):
        atomic_json(tmp_path / "artifact.json", {})


def test_directory_destination_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(io.sys, "platform", "win32")
    monkeypatch.setattr(io.os, "replace", lambda *args: (_ for _ in ()).throw(windows_error(5)))
    monkeypatch.setattr(io.time, "sleep", lambda seconds: pytest.fail("unexpected retry"))
    with pytest.raises(OSError):
        atomic_json(tmp_path, {})


def test_serialization_failure_does_not_touch_existing_file(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_bytes(b"corrupt existing evidence")
    with pytest.raises(ValueError):
        atomic_json(path, {"invalid": float("nan")})
    assert path.read_bytes() == b"corrupt existing evidence"
    with pytest.raises(ValueError):
        read_json(path)
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_cleanup_preserves_primary_exception(tmp_path, monkeypatch):
    primary = OSError(errno.ENOSPC, "primary disk failure")
    monkeypatch.setattr(io.os, "replace", lambda *args: (_ for _ in ()).throw(primary))
    real_unlink = Path.unlink

    def unlink(path, **kwargs):
        if path.suffix == ".tmp":
            raise OSError(errno.EPERM, "cleanup denied")
        return real_unlink(path, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(OSError) as caught:
        atomic_json(tmp_path / "artifact.json", {})
    assert caught.value is primary
    assert "Temporary cleanup also failed" in primary.__notes__[0]
    assert not (tmp_path / "artifact.json").exists()
    # Test cleanup only; the production function reported the retained temp.
    for path in tmp_path.glob("*.tmp"):
        real_unlink(path)


@pytest.mark.skipif(sys.platform != "win32", reason="Real Windows sharing semantics")
def test_real_ordinary_reader_reproduces_winerror5_then_retry_succeeds(tmp_path, monkeypatch):
    path = tmp_path / "000150.json"
    atomic_json(path, {"status": "started"})
    temporary = tmp_path / "probe.tmp"
    temporary.write_text("replacement")
    reader = path.open("rb")
    try:
        with pytest.raises(PermissionError) as caught:
            os.replace(temporary, path)
        assert caught.value.winerror in (5, 32)
        sleeps = []

        def release(seconds):
            sleeps.append(seconds)
            reader.close()

        monkeypatch.setattr(io.time, "sleep", release)
        atomic_json(path, {"status": "complete"})
        assert sleeps == [0.01]
        assert read_json(path)["status"] == "complete"
    finally:
        reader.close()
        temporary.unlink(missing_ok=True)


@pytest.mark.skipif(sys.platform != "win32", reason="Real Windows readonly attribute")
def test_readonly_target_fails_immediately(tmp_path, monkeypatch):
    import stat

    path = tmp_path / "artifact.json"
    atomic_json(path, {"old": True})
    path.chmod(stat.S_IREAD)
    monkeypatch.setattr(io.time, "sleep", lambda seconds: pytest.fail("readonly target retried"))
    try:
        with pytest.raises(PermissionError):
            atomic_json(path, {"new": True})
        assert read_json(path) == {"old": True}
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        path.chmod(stat.S_IWRITE | stat.S_IREAD)


def test_monitor_refresh_concurrent_representative_writes(tmp_path):
    from dataclasses import asdict

    from toposc_lab.discovery.config import DiscoveryConfig

    root = tmp_path / "fixture"
    config = DiscoveryConfig()
    atomic_json(
        root / "manifest.json", {"config": asdict(config), "config_sha256": config.fingerprint}
    )
    path = root / "attempts/000150.json"
    atomic_json(path, {"number": 150, "stage": "fixture", "status": "started"})
    reader = CampaignReader(ProcessService(tmp_path / "ui-state"))
    ready, done = threading.Event(), threading.Event()

    def monitor():
        reads = 0
        ready.set()
        while not done.is_set() or reads < 2:
            snapshot = reader.load(root)
            assert not any("checksum mismatch" in w for w in snapshot.warnings)
            reads += 1
        return reads

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(monitor)
        assert ready.wait(5)
        try:
            for i in range(30):
                atomic_json(
                    path, {"number": 150, "stage": "fixture", "status": "complete", "revision": i}
                )
                atomic_json(
                    root / "checkpoint.json",
                    {"completed_cycles": 0, "config_sha256": config.fingerprint},
                )
        finally:
            done.set()
        assert future.result(timeout=10) >= 2
    assert read_json(path)["revision"] == 29
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    reader.load(root)
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert not list(root.rglob("*.tmp"))


def test_monitor_closes_reader_before_json_decode(tmp_path, monkeypatch):
    import toposc_live.reader as reader_module

    path = tmp_path / "artifact.json"
    atomic_json(path, {"old": True})
    original = reader_module.json.loads
    replaced = False

    def decode(raw, *args, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            # Must succeed while the monitor is parsing; read_bytes has closed
            # its OS handle. Cache may return the complete older snapshot.
            atomic_json(path, {"new revision": True})
        return original(raw, *args, **kwargs)

    monkeypatch.setattr(reader_module.json, "loads", decode)
    cache = ArtifactCache()
    assert cache.read(path, []) == {"old": True}
    assert cache.read(path, []) == {"new revision": True}


@pytest.mark.skipif(sys.platform != "win32", reason="Real Windows sharing semantics")
def test_metadata_replacement_retries_after_reader_release(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    write_metadata(path, {"status": "Running"})
    with path.open("rb") as reader:
        monkeypatch.setattr(io.time, "sleep", lambda seconds: reader.close())
        write_metadata(path, {"status": "Exited"})
    assert read_metadata(path) == {"status": "Exited"}


def test_failed_ledger_finalization_reuses_saved_stage_without_recomputation(tmp_path, monkeypatch):
    from toposc_lab.discovery import DiscoveryConfig, DiscoveryEngine

    engine = DiscoveryEngine(tmp_path, DiscoveryConfig())
    stage = tmp_path / "cycle-0006/base-0001.json"
    ledger = tmp_path / "attempts/000000.json"
    real_replace = os.replace
    calculations = []

    def replace(source, destination):
        if destination == ledger and read_json(Path(source)).get("status") == "complete":
            raise windows_error(5)
        real_replace(source, destination)

    monkeypatch.setattr(io.sys, "platform", "win32")
    monkeypatch.setattr(io.os, "replace", replace)
    monkeypatch.setattr(io.time, "sleep", lambda seconds: None)

    def operation():
        calculations.append("called")
        return {"fixture": "stored output, no numerical calculation"}

    with pytest.raises(OSError):
        engine._exact_stage(stage, 123, operation)
    assert read_json(ledger)["status"] == "started"
    saved = read_json(stage)
    assert saved["status"] == "complete"
    # This is the precise production crash boundary; no recovery policy changed.
    assert engine._exact_stage(stage, 123, operation) == saved
    assert calculations == ["called"]
    assert len(list((tmp_path / "attempts").glob("*.json"))) == 1
    assert read_json(ledger)["status"] == "started"
    stage.write_text("corrupt fixture")
    with pytest.raises(ValueError):
        engine._exact_stage(stage, 123, operation)
    assert calculations == ["called"]
    assert stage.read_text() == "corrupt fixture"


def test_fsync_failure_does_not_retry_or_publish(tmp_path, monkeypatch):
    path = tmp_path / "artifact.json"
    atomic_json(path, {"old": True})
    monkeypatch.setattr(io.os, "fsync", lambda fd: (_ for _ in ()).throw(windows_error(5)))
    monkeypatch.setattr(io.time, "sleep", lambda seconds: pytest.fail("fsync must not be retried"))
    with pytest.raises(OSError):
        atomic_json(path, {"new": True})
    assert read_json(path) == {"old": True}
    assert not list(tmp_path.glob("*.tmp"))
