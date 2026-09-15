"""Detached supervisor of the official CLI. No Qt imports or scientific logic."""

import subprocess
import sys
import time
from pathlib import Path

from toposc_live.processes import read_metadata, write_metadata


def run(request_path: Path) -> int:
    import psutil

    request = read_metadata(request_path)
    state_path = request_path.with_name("state.json")
    self_process = psutil.Process()
    state = {
        "status": "Running",
        "pid": self_process.pid,
        "create_time": self_process.create_time(),
        "started_at": time.time(),
    }
    write_metadata(state_path, state)
    try:
        with request_path.with_name("engine.log").open("ab") as log:
            process = subprocess.Popen(
                request["command"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
            try:
                state["engine"] = {
                    "pid": process.pid,
                    "create_time": psutil.Process(process.pid).create_time(),
                }
                write_metadata(state_path, state)
            except psutil.NoSuchProcess:
                pass  # A failed CLI can exit before its process metadata is queried.
            code = process.wait()
        state.update(status="Exited" if code == 0 else "Failed", exit_code=code)
    except Exception as error:  # noqa: BLE001 -- preserve supervisor failure for reconnect
        code = 1
        state.update(status="Failed", error=f"{type(error).__name__}: {error}")
    state["ended_at"] = time.time()
    write_metadata(state_path, state)
    return code


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1])))
