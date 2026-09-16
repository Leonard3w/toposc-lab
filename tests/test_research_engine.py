"""Lifecycle, budget, transactional recovery, and real scientific integration gates."""

import json
from dataclasses import replace

import pytest

from toposc_lab.discovery.storage import read_json, writer_lease
from toposc_lab.research.config import ExperimentConfig
from toposc_lab.research.engine import ResearchEngine
from toposc_lab.research.service import ResearchService
from toposc_lab.research.storage import ResearchStore


class CheapExact:
    """Deterministic test oracle exercises orchestration without physics claims."""
    calls = 0

    def plan(self):
        return [{"key": "clean"}, {"key": "confirm"}]

    def evaluate(self, geometry, stage):
        self.calls += 1
        return {"score": sum((e.source + 1) * (e.target + 1) for e in geometry.edges) / 10000,
                "origin": "exact"}

    def summarize(self, results):
        return {"score": results["clean"]["score"], "origin": "exact",
                "validation_state": "EXACT_EVALUATED", "validation_results": {},
                "raw_metrics": {"quality": results["clean"]["score"]}}


@pytest.fixture
def config():
    return ExperimentConfig(cycles=2, exact_budget=36, candidate_budget=160, pool_size=6,
        batch_size=4, checkpoint_every=8, space={"side": 4, "max_degree": 6, "min_degree": 2,
        "bond_tolerance": 0.0}, physics={"disorder_widths": [.2], "disorder_seeds": [17001, 17002]},
        surrogate={"ensemble_size": 2, "n_estimators": 3})


def completed(directory):
    return [c for c in ResearchStore(directory).all("candidate") if c.get("observed")]


def scientific_sequence(directory):
    return [(c["id"], c["score"], c["geometry"]["edges"], c.get("acquisition"))
            for c in completed(directory)]


def test_pause_resume_dataset_and_model_persist(tmp_path, config):
    path = ResearchEngine.create(tmp_path / "resume", config)
    assert ResearchEngine(path, evaluator=CheapExact()).run(max_cycles=1)["status"] == "PAUSED"
    assert ResearchEngine(path, evaluator=CheapExact()).run()["status"] == "COMPLETED"
    snapshot = ResearchService.snapshot(path)
    assert len(completed(path)) == 10
    assert snapshot["state"]["exact_evaluations"] == 20
    assert snapshot["archive"] and snapshot["model_versions"]
    assert snapshot["surrogate"]["calibration"]
    assert "Inconclusive" in snapshot["report"]
    assert (path / "final_report.md").read_text(encoding="utf-8") == snapshot["report"]
    calls = CheapExact()
    ResearchEngine(path, evaluator=calls).run()
    assert calls.calls == 0
    reference = ResearchEngine.create(tmp_path / "reference", config)
    ResearchEngine(reference, evaluator=CheapExact()).run()
    assert scientific_sequence(path) == scientific_sequence(reference)


@pytest.mark.parametrize("boundary", ["plan_saved", "attempt_started", "after_simulation",
    "stage_saved", "candidate_saved", "checkpoint_saved", "cycle_committed"])
def test_crash_recovery_and_charged_retry(tmp_path, config, boundary):
    config = replace(config, algorithm="random", cycles=1)
    path = ResearchEngine.create(tmp_path / "crash", config)
    def crash(name, payload):
        if name == boundary:
            raise KeyboardInterrupt("injected interruption")
    with pytest.raises(KeyboardInterrupt):
        ResearchEngine(path, evaluator=CheapExact(), hook=crash).run()
    final = ResearchEngine(path, evaluator=CheapExact()).run()
    assert final["status"] == "COMPLETED"
    assert final["exact_evaluations"] == 12 + (boundary in ("attempt_started", "after_simulation"))
    reference = ResearchEngine.create(tmp_path / "reference", config)
    ResearchEngine(reference, evaluator=CheapExact()).run()
    assert scientific_sequence(path) == scientific_sequence(reference)
    assert read_json(path / "state.json")["status"] == "COMPLETED"


def test_safe_control_at_exact_boundary(tmp_path, config):
    path = ResearchEngine.create(tmp_path, config)
    fired = False
    def pause(name, payload):
        nonlocal fired
        if name == "stage_saved" and not fired:
            fired = True
            ResearchStore(path).request("pause")
    result = ResearchEngine(path, evaluator=CheapExact(), hook=pause).run()
    assert result["status"] == "PAUSED" and result["exact_evaluations"] == 1
    assert len(ResearchStore(path).all("exact_result")) == 1
    result = ResearchEngine(path, evaluator=CheapExact()).run()
    assert result["status"] == "COMPLETED" and result["exact_evaluations"] == 20


@pytest.mark.parametrize("algorithm", ["random", "evolution", "map_elites", "surrogate_map_elites"])
def test_all_strategies_same_charged_budget(tmp_path, config, algorithm):
    path = ResearchEngine.create(tmp_path, replace(config, algorithm=algorithm, exact_budget=16))
    state = ResearchEngine(path, evaluator=CheapExact()).run()
    assert state["exact_evaluations"] == 16
    assert len(completed(path)) == 8
    assert len([c for c in completed(path) if c["baseline"]]) == 2
    assert state["remaining_exact_budget"] == 0


def test_solver_failure_and_nan_excluded_from_archive(tmp_path, config):
    class Failed(CheapExact):
        def evaluate(self, geometry, stage):
            return {"score": float("nan")}
    path = ResearchEngine.create(tmp_path, replace(config, cycles=1))
    state = ResearchEngine(path, evaluator=Failed()).run()
    assert state["status"] == "COMPLETED"
    snapshot = ResearchService.snapshot(path)
    assert not snapshot["archive"]
    assert all(c["score"] is None and c["validation_state"] == "FAILED" for c in completed(path))
    assert ResearchStore(path).all("error")


def test_corrupt_checkpoint_export_rebuilt_database_corruption_rejected(tmp_path, config):
    path = ResearchEngine.create(tmp_path, config)
    ResearchEngine(path, evaluator=CheapExact()).run(max_cycles=1)
    checkpoint = next((path / "checkpoints").glob("*.json"))
    checkpoint.write_text("not json", encoding="utf-8")
    ResearchEngine(path, evaluator=CheapExact()).run()
    assert read_json(checkpoint)["config_sha256"] == config.fingerprint
    store = ResearchStore(path)
    with store.connect() as db:
        db.execute("UPDATE objects SET checksum='bad' WHERE kind='candidate'")
    with pytest.raises(ValueError, match="Corrupt database"):
        ResearchService.snapshot(path)


def test_source_and_config_guard_and_single_writer(tmp_path, config):
    path = ResearchEngine.create(tmp_path, config)
    store = ResearchStore(path)
    with writer_lease(path), pytest.raises(OSError):
        ResearchEngine(path, evaluator=CheapExact()).run()
    manifest = store.get("manifest")
    original = manifest["source_sha256"]
    manifest["source_sha256"] = "wrong"
    store.save("manifest", manifest)
    with pytest.raises(ValueError, match="source mismatch"):
        ResearchEngine(path, evaluator=CheapExact()).run()
    manifest["source_sha256"] = original
    store.save("manifest", manifest)
    payload = config.to_dict()
    payload["seed"] += 1
    (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="config.json differs"):
        ResearchEngine(path, evaluator=CheapExact()).run()


def test_idle_stop_clone_archive_and_interrupted_discovery(tmp_path, config):
    path = ResearchEngine.create(tmp_path / "run", config)
    store = ResearchStore(path)
    state = store.get("state")
    state["status"] = "RUNNING"
    store.save("state", state)
    assert ResearchService.snapshot(path)["state"]["status"] == "INTERRUPTED"
    assert ResearchService.list_experiments(tmp_path)[0]["status"] == "INTERRUPTED"
    ResearchService.control(path, "stop")
    assert store.get("state")["status"] == "STOPPED"
    clone = ResearchService.duplicate(path, tmp_path / "clone", {"seed": 19})
    assert ResearchStore(clone).get("config")["seed"] == 19
    assert ResearchStore(clone).get("state")["experiment_id"] != store.get("state")["experiment_id"]
    ResearchService.control(path, "archive")
    assert store.get("state")["status"] == "ARCHIVED"


def test_actual_physics_surrogate_and_exact_budget(tmp_path, config):
    path = ResearchEngine.create(tmp_path, replace(config, exact_budget=40))
    ResearchEngine(path).run(max_cycles=1)
    state = ResearchEngine(path).run()
    assert state["status"] == "COMPLETED" and state["exact_evaluations"] == 40
    snapshot = ResearchService.snapshot(path)
    assert len(snapshot["model_versions"]) == 1
    assert len(snapshot["surrogate"]["calibration"]) == 4
    assert snapshot["selection_counts"].get("uncertainty", 0) > 0
    assert snapshot["selection_counts"].get("novelty", 0) > 0
    for candidate in completed(path):
        assert candidate["origin"] == "exact"
        assert candidate["validation_state"] not in ("MAJORANA_VALIDATED", "FINITE_SIZE_VALIDATED")
        assert len(candidate["exact_results"]) == 4
        assert candidate["exact_results"]["clean"]["dataset_record"]["provenance"]["git_commit"] != "unrecorded"


def test_stale_engine_reloads_under_lease(tmp_path, config):
    path = ResearchEngine.create(tmp_path / "run", config)
    stale = ResearchEngine(path, evaluator=CheapExact())
    ResearchEngine(path, evaluator=CheapExact()).run(max_cycles=1)
    stale.run()
    reference = ResearchEngine.create(tmp_path / "reference", config)
    ResearchEngine(reference, evaluator=CheapExact()).run()
    assert scientific_sequence(path) == scientific_sequence(reference)


@pytest.mark.parametrize("boundary", ["control_pending", "control_committed", "report_saved"])
def test_durable_stop_request_and_terminal_repair(tmp_path, config, boundary):
    path = ResearchEngine.create(tmp_path, config)
    store = ResearchStore(path)
    store.request("stop")
    def crash(name, payload):
        if name == boundary:
            raise KeyboardInterrupt("injected control/report crash")
    with pytest.raises(KeyboardInterrupt):
        ResearchEngine(path, evaluator=CheapExact(), hook=crash).run()
    final = ResearchEngine(path, evaluator=CheapExact()).run()
    assert final["status"] == "STOPPED"
    assert final["exact_evaluations"] == 0
    (path / "final_report.md").unlink()
    ResearchService.control(path, "checkpoint")
    assert (path / "final_report.md").is_file()
    assert "STOPPED" in (path / "final_report.md").read_text(encoding="utf-8")


def test_wall_budget_and_candidate_budget_are_hard_limits(tmp_path, config):
    path = ResearchEngine.create(tmp_path / "wall", replace(config, wall_seconds=.00001))
    final = ResearchEngine(path, evaluator=CheapExact()).run()
    assert final["status"] == "PAUSED" and final["exact_evaluations"] == 0
    with pytest.raises(ValueError, match="Wall-clock budget exhausted"):
        ResearchEngine(path, evaluator=CheapExact()).run()
    path = ResearchEngine.create(tmp_path / "candidate", replace(config, candidate_budget=1))
    final = ResearchEngine(path, evaluator=CheapExact()).run()
    assert final["generated"] == 1
    assert final["exact_evaluations"] <= 6


def test_retry_reserve_is_charged_and_bounded(tmp_path, config):
    path = ResearchEngine.create(tmp_path, replace(config, cycles=1, retry_limit=0))
    def crash(name, payload):
        if name == "attempt_started":
            raise KeyboardInterrupt("lost worker")
    with pytest.raises(KeyboardInterrupt):
        ResearchEngine(path, evaluator=CheapExact(), hook=crash).run()
    state = ResearchEngine(path, evaluator=CheapExact()).run()
    assert state["status"] == "COMPLETED"
    first = next(c for c in completed(path) if c["id"] == "baseline-regular")
    assert first["score"] is None and first["validation_state"] == "FAILED"
    assert len([a for a in ResearchStore(path).attempts()
                if a["candidate"] == "baseline-regular"]) == 1


def test_thread_runtime_resume_guard(tmp_path, config, monkeypatch):
    path = ResearchEngine.create(tmp_path, config)
    ResearchEngine(path, evaluator=CheapExact()).run(max_cycles=0)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "999")
    with pytest.raises(ValueError, match="BLAS execution environment mismatch"):
        ResearchEngine(path, evaluator=CheapExact()).run()
