"""Detached process orchestration. UI metadata lives outside scientific campaigns."""

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from toposc_lab._file_io import remove_temporary, replace_atomic
from toposc_live.configuration import LaunchPlan
from toposc_live.models import mapping, number


def default_state_directory() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "state"))
    return base / "toposc-live"


def write_metadata(path: Path, value: dict[str, Any]) -> None:
    """Atomic application metadata, never used to write scientific artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
        replace_atomic(temporary, path)
    finally:
        remove_temporary(Path(temporary))


def read_metadata(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def detached_options() -> dict[str, Any]:
    if sys.platform == "win32":
        return {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def engine_command(
    plan: LaunchPlan, config_path: Path, executable: str = sys.executable
) -> list[str]:
    command = [
        executable,
        "-u",
        "-m",
        "toposc_lab.discovery",
        str(plan.directory),
        "--config",
        str(config_path),
    ]
    if plan.iterations is not None:
        command.extend(["--iterations", str(plan.iterations)])
    return command


def process_alive(state: dict[str, Any]) -> bool | None:
    """PID plus creation time prevents reconnecting to a reused PID."""
    try:
        import psutil

        pid, created = state.get("pid"), state.get("create_time")
        if type(pid) is not int or not isinstance(created, (int, float)):
            return None
        process = psutil.Process(pid)
        return bool(process.is_running() and abs(process.create_time() - created) < 0.01)
    except ImportError:
        return None
    except psutil.NoSuchProcess:
        return False
    except (psutil.AccessDenied, OSError, ValueError):
        return None


@contextmanager
def application_lease(state_directory: Path) -> Iterator[None]:
    """Serialize launch requests from multiple windows without touching writer.lock."""
    state_directory.mkdir(parents=True, exist_ok=True)
    with (state_directory / "launcher.lock").open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class ProcessService:
    def __init__(self, state_directory: Path | None = None) -> None:
        self.state_directory = (state_directory or default_state_directory()).resolve()

    def jobs(self) -> list[dict[str, Any]]:
        result = []
        for path in self.state_directory.glob("jobs/*/request.json"):
            request = read_metadata(path)
            if isinstance(request.get("directory"), str):
                state = read_metadata(path.with_name("state.json"))
                result.append(
                    {
                        **request,
                        "requested_at": number(request.get("requested_at")) or 0,
                        "state": state,
                        "request_path": str(path),
                    }
                )
        return sorted(result, key=lambda item: item.get("requested_at", 0), reverse=True)

    def inspect(self, directory: Path) -> dict[str, Any]:
        for job in self.jobs():
            if Path(job["directory"]).resolve() != directory.resolve():
                continue
            state = job["state"]
            alive = process_alive(state)
            engine_alive = process_alive(mapping(state.get("engine")))
            status = state.get("status", "Starting")
            if status in ("Running", "Starting"):
                if alive is False and engine_alive is not True:
                    status = "Interrupted" if engine_alive is False else "Unknown"
                elif alive is None and time.time() - job.get("requested_at", 0) > 30:
                    status = "Unknown"
            return {**job, "status": status, "alive": alive, "engine_alive": engine_alive}
        return {}

    def launch(self, plan: LaunchPlan) -> Path:
        # The engine remains final authority, including its own single-writer lease.
        from toposc_live.configuration import ConfigurationAdapter

        with application_lease(self.state_directory):
            current = self.inspect(plan.directory)
            if current.get("status") in ("Starting", "Running", "Unknown"):
                raise ValueError("A process may already be active here; inspect its state first.")
            if plan.resume:
                checked = ConfigurationAdapter().resume_plan(
                    plan.directory, one_cycle=plan.iterations == 1
                )
                if checked.config != plan.config:
                    raise ValueError("Configuration changed since validation; review again.")
            else:
                # Exclusive reservation prevents launch races and overwriting old campaigns.
                plan.directory.mkdir(parents=True, exist_ok=False)
            job = self.state_directory / "jobs" / uuid.uuid4().hex
            job.mkdir(parents=True)
            config_path = job / "config.json"
            write_metadata(config_path, asdict(plan.config))
            request_path = job / "request.json"
            write_metadata(
                request_path,
                {
                    "schema_version": 1,
                    "name": plan.name,
                    "directory": str(plan.directory),
                    "requested_at": time.time(),
                    "resume": plan.resume,
                    "config_sha256": plan.config.fingerprint,
                    "command": engine_command(plan, config_path),
                },
            )
            write_metadata(job / "state.json", {"status": "Starting"})
            try:
                with (job / "supervisor.log").open("ab") as log:
                    subprocess.Popen(
                        [sys.executable, "-u", "-m", "toposc_live.worker", str(request_path)],
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        close_fds=True,
                        **detached_options(),
                    )
            except OSError as error:
                write_metadata(job / "state.json", {"status": "Failed", "error": str(error)})
                raise
            return request_path

    def log_tail(self, job: dict[str, Any], limit: int = 16000) -> str:
        if not job.get("request_path"):
            return "Logs unavailable for campaigns not launched by this application."
        path = Path(job["request_path"]).with_name("engine.log")
        try:
            with path.open("rb") as stream:
                stream.seek(0, 2)
                stream.seek(max(0, stream.tell() - limit))
                return stream.read(limit).decode("utf-8", errors="replace")
        except OSError:
            return "Log not yet available."
