from dataclasses import replace

import pytest

from toposc_lab.data import load_dataset
from toposc_lab.discovery import DiscoveryConfig, DiscoveryEngine
from toposc_lab.discovery.engine import scientific_record
from toposc_lab.discovery.storage import read_json


@pytest.fixture
def config():
    return DiscoveryConfig(cycles=2, pool_size=4, disorder_samples=2, plots=False)


def science(path):
    return [scientific_record(r) for r in load_dataset(path / "dataset.json").records]


def test_cycle_resume_exact_dataset_and_diversity(tmp_path, config):
    first = tmp_path / "first"
    engine = DiscoveryEngine(first, config)
    assert engine.run(iterations=1)["completed_cycles"] == 1
    result = DiscoveryEngine(first, config).run()
    assert result["validated_dataset_records"] == 8
    assert result["exact_attempts"] == 32
    assert result["minimum_pairwise_distance"] > 0.06
    assert result["selected_ood"] >= 4
    assert result["successes"] == 0
    second = tmp_path / "second"
    DiscoveryEngine(second, config).run()
    assert science(first) == science(second)
    assert DiscoveryEngine(first, config).run()["exact_attempts"] == 32
    with pytest.raises(ValueError, match="mismatch"):
        DiscoveryEngine(first, replace(config, seed=4)).run()
    with pytest.raises(ValueError, match="cap"):
        engine.run(iterations=1)


@pytest.mark.parametrize(
    "boundary",
    [
        "plan_saved",
        "attempt_started",
        "after_simulation",
        "stage_saved",
        "candidate_saved",
        "dataset_saved",
        "cycle_committed",
        "checkpoint_saved",
    ],
)
def test_crash_boundaries(tmp_path, config, boundary):
    config = replace(config, cycles=1)
    root = tmp_path / "crash"

    def crash(stage, payload):
        if stage == boundary:
            raise KeyboardInterrupt("injected process interruption")

    with pytest.raises(KeyboardInterrupt):
        DiscoveryEngine(root, config, hook=crash).run()
    summary = DiscoveryEngine(root, config).run()
    assert summary["validated_dataset_records"] == 4
    assert summary["exact_attempts"] == 16 + (boundary in ("attempt_started", "after_simulation"))
    assert read_json(root / "checkpoint.json")["completed_cycles"] == 1
    reference = tmp_path / "reference"
    DiscoveryEngine(reference, config).run()
    assert science(root) == science(reference)


@pytest.mark.parametrize("method", ["random", "evolution", "coverage"])
def test_supported_baselines(tmp_path, config, method):
    summary = DiscoveryEngine(tmp_path, replace(config, generator=method)).run()
    assert summary["validated_dataset_records"] == 8
    assert summary["candidate_failures"] == []


def test_optional_surrogate_no_future_labels_and_cadence(tmp_path, config):
    config = replace(config, surrogate=True, cycles=3, pool_size=8)
    summary = DiscoveryEngine(tmp_path, config).run()
    assert summary["predictions"] == 16
    first, second = summary["training_events"]
    assert first == second
    assert len(first["record_ids"]) == 4
    for cycle in (1, 2):
        plan = read_json(tmp_path / f"cycle-{cycle:04d}" / "plan.json")
        assert set(plan["training"]["record_ids"]) <= set(plan["history_record_ids"])
        assert all(p["kind"] == "predicted" for p in plan["predictions"])
        assert not any(p["is_ood"] and p["strategy"] == "exploitation" for p in plan["predictions"])


def test_external_dataset_blocks_repeated_geometries(tmp_path, config):
    config = replace(config, cycles=1)
    first = tmp_path / "first"
    DiscoveryEngine(first, config).run()
    dataset = load_dataset(first / "dataset.json")
    summary = DiscoveryEngine(tmp_path / "excluded", config, seed_dataset=dataset).run()
    assert summary["duplicate"] >= 4
    assert summary["validated_dataset_records"] == 4
    assert {r.geometry.exact_id for r in dataset.records}.isdisjoint(
        r.geometry.exact_id for r in load_dataset(tmp_path / "excluded" / "dataset.json").records
    )


def test_numerical_failure_isolated_and_not_a_label(tmp_path, config, monkeypatch):
    from toposc_lab.generative.physics import ExactGeometryEvaluator

    original = ExactGeometryEvaluator.evaluate
    calls = 0

    def fail_one(self, proposal, seed):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ArithmeticError("numerical failure fixture")
        return original(self, proposal, seed)

    monkeypatch.setattr(ExactGeometryEvaluator, "evaluate", fail_one)
    summary = DiscoveryEngine(tmp_path, replace(config, cycles=1)).run()
    assert summary["failed_attempts"] == 1
    assert summary["validated_dataset_records"] == 3
    assert len(summary["candidate_failures"]) == 1


def test_retry_budget_is_charged(tmp_path, config):
    config = replace(config, cycles=1, retry_reserve=0)

    def crash(stage, payload):
        if stage == "after_simulation":
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        DiscoveryEngine(tmp_path, config, hook=crash).run()
    with pytest.raises(RuntimeError, match="attempt cap"):
        DiscoveryEngine(tmp_path, config).run()
    assert len(list((tmp_path / "attempts").glob("*.json"))) == 16


def test_committed_artifacts_fail_closed(tmp_path, config):
    root = tmp_path / "corrupt"
    config = replace(config, cycles=1)
    DiscoveryEngine(root, config).run()
    path = root / "cycle-0000" / "plan.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="corruption"):
        DiscoveryEngine(root, config).run()
