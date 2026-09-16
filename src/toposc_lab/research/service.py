"""Shared application/CLI API; GUI lifetime is independent of exact workers."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

from toposc_lab.discovery.storage import writer_lease
from toposc_lab.research.config import ExperimentConfig
from toposc_lab.research.reporting import diagnostics
from toposc_lab.research.storage import ResearchStore, atomic_text


def _worker_active(directory: Path) -> bool:
    try:
        with writer_lease(directory):
            return False
    except OSError:
        return True


class ResearchService:
    @staticmethod
    def create(config: dict[str, Any] | ExperimentConfig, directory: str | Path) -> Path:
        from toposc_lab.research.engine import ResearchEngine
        return ResearchEngine.create(directory, config)

    @staticmethod
    def snapshot(directory: str | Path, *, detect_interrupted: bool = True) -> dict[str, Any]:
        store = ResearchStore(directory)
        # One read transaction: the UI cannot combine different cycle commits.
        with store.connect(readonly=True) as db:
            db.execute("BEGIN")
            objects: dict[str, dict[str, Any]] = {}
            for row in db.execute("SELECT * FROM objects"):
                objects.setdefault(row["kind"], {})[row["id"]] = store.decode(row)
            attempts = [dict(row) for row in db.execute("SELECT * FROM attempts ORDER BY number")]
            events = [{**dict(row), "payload": json.loads(row["payload"])}
                      for row in db.execute("SELECT * FROM events ORDER BY number")]
        state = objects["state"]["current"]
        config = objects["config"]["current"]
        state["exact_evaluations"] = len(attempts)
        state["remaining_exact_budget"] = max(0, config["exact_budget"] - len(attempts))
        state["exact_seconds"] = sum(a["seconds"] or 0 for a in attempts)
        if detect_interrupted and state["status"] in ("RUNNING", "FINALIZING") and not _worker_active(Path(directory)):
            state["status"] = "INTERRUPTED"
        strategy = objects.get("strategy", {}).get("current", {})
        candidates = list(objects.get("candidate", {}).values())
        # Completed candidate records carry their own stage results. During an
        # evaluation expose committed stages too, always labeled partial exact.
        stages = objects.get("exact_result", {})
        for candidate in candidates:
            if not candidate.get("observed"):
                prefix = candidate["id"] + ":"
                partial = {key[len(prefix):]: value for key, value in stages.items()
                           if key.startswith(prefix)}
                if partial:
                    candidate["exact_results"] = partial
                    candidate["origin"] = "partial_exact"
                    candidate["score"] = None
        snapshot = {"config": config, "state": state,
                    "manifest": objects["manifest"]["current"], "candidates": candidates,
                    "archive": list(strategy.get("archive", {}).values()),
                    "history": list(objects.get("history", {}).values()), "events": events,
                    "checkpoints": [dict(value, id=key) for key, value in
                                    objects.get("checkpoint", {}).items()],
                    "report": objects.get("report", {}).get("current", {}).get("markdown", ""),
                    "surrogate": objects.get("model", {}).get("current", {}),
                    "attempts": attempts, "model_versions": list(objects.get("model_version", {}).values()),
                    "baselines": [c for c in candidates if c.get("baseline")],
                    "allocation": config["search"].get("allocation", {}),
                    "selection_counts": strategy.get("selection_counts", {})}
        report = diagnostics(snapshot)
        state.update({k: report[k] for k in ("best_score", "best_validated_candidate", "archive_coverage",
                      "duplicate_rate", "near_duplicate_rate", "invalid_rate", "surrogate_error", "diversity")})
        elapsed = max(1e-9, state.get("elapsed_seconds", 0))
        state["candidate_throughput"] = state.get("generated", 0) / elapsed
        state["simulation_throughput"] = len(attempts) / elapsed
        state["warnings"] = list(dict.fromkeys(state.get("warnings", []) + report["anomalies"]))
        snapshot["diagnostics"] = report
        return snapshot

    @staticmethod
    def control(directory: str | Path, action: str) -> None:
        store = ResearchStore(directory)
        if action == "resume":
            ResearchService.launch(directory)
            return
        if action == "archive":
            with writer_lease(Path(directory)):
                state = store.get("state")
                if state["status"] == "RUNNING":
                    state["status"] = "INTERRUPTED"
                state.update(previous_status=state["status"], status="ARCHIVED")
                with store.connect() as db:
                    store.put(db, "state", "current", state)
                    store.event(db, "archived", {})
            return
        if action == "checkpoint" and not _worker_active(Path(directory)):
            from toposc_lab.research.engine import ResearchEngine
            with writer_lease(Path(directory)):
                engine = ResearchEngine(directory)
                engine.repair_exports()
                engine.checkpoint("manual_idle")
            return
        if action in ("pause", "stop") and not _worker_active(Path(directory)):
            from toposc_lab.research.engine import ResearchEngine
            with writer_lease(Path(directory)):
                engine = ResearchEngine(directory)
                if store.get("state")["status"] in ("COMPLETED", "STOPPED", "ARCHIVED"):
                    return
                if action == "stop":
                    engine._finish("STOPPED", "manual_idle_stop")
                else:
                    engine._save_state(status="PAUSED")
                    engine.checkpoint("manual_idle_pause")
            return
        store.request(action)

    @staticmethod
    def launch(directory: str | Path) -> int:
        from toposc_lab.research.provenance import ROOT, verify
        directory = Path(directory).resolve()
        store = ResearchStore(directory)
        config = ExperimentConfig(**store.get("config"))
        verify(directory, store.get("manifest"))
        if store.get("state")["status"] in ("COMPLETED", "STOPPED", "ARCHIVED"):
            raise ValueError("Run has finished; duplicate its configuration to start a new experiment")
        with writer_lease(directory):
            options: dict[str, Any] = {}
            if os.name == "nt":
                options["creationflags"] = (subprocess.DETACHED_PROCESS |
                    subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
            else:
                options["start_new_session"] = True
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
                env[name] = str(config.blas_threads)
            with (directory / "logs" / "worker.log").open("ab") as log:
                worker = subprocess.Popen([sys.executable, "-B", "-m", "toposc_lab.research", "run",
                                           str(directory)], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT, close_fds=True, **options)
        return worker.pid

    @staticmethod
    def list_experiments(root: str | Path) -> list[dict[str, Any]]:
        root = Path(root)
        found: list[dict[str, Any]] = []
        if not root.exists():
            return found
        for path in sorted(root.rglob("research.sqlite3")):
            try:
                store = ResearchStore(path.parent)
                state, config = store.get("state"), store.get("config")
                if state["status"] == "RUNNING" and not _worker_active(path.parent):
                    state["status"] = "INTERRUPTED"
                found.append({"directory": str(path.parent), "config": config,
                              "state": state, **state})
            except (ValueError, KeyError, OSError, sqlite3.DatabaseError) as error:
                found.append({"directory": str(path.parent), "status": "CORRUPT", "error": str(error)})
        return found

    @staticmethod
    def duplicate(directory: str | Path, destination: str | Path,
                  changes: dict[str, Any] | None = None) -> Path:
        original = ResearchStore(directory)
        config = original.get("config")
        config.update(changes or {})
        new = ResearchService.create(config, destination)
        store = ResearchStore(new)
        with store.connect() as db:
            store.event(db, "cloned_from", {"experiment_id": original.get("manifest")["experiment_id"],
                                           "configuration_changes": changes or {}})
        return new

    @staticmethod
    def save_config(config: dict[str, Any], path: str | Path) -> None:
        validated = ExperimentConfig(**config)
        validated.validate_plugins()
        atomic_text(Path(path), json.dumps(validated.to_dict(), indent=2))
