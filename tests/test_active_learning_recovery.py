from __future__ import annotations

import pytest
from test_active_learning import campaign_config, cycle_fixture, exact_fixture, training_config

from toposc_lab.active_learning.campaign import run_campaign
from toposc_lab.active_learning.cycle import run_cycle
from toposc_lab.active_learning.training import FittedSurrogate, retrain
from toposc_lab.data import ExactPhysicsDataset, dataset_to_bytes


def test_prediction_failure_falls_back_to_seeded_exact_exploration(monkeypatch):
    space, dataset, evaluate = cycle_fixture()

    def invalid(self, pool):
        raise ValueError("NaN prediction")

    monkeypatch.setattr(FittedSurrogate, "predict", invalid)
    result = run_cycle(
        space, dataset, config=campaign_config().cycle, seed=13, cycle_index=0, evaluator=evaluate
    )
    assert "NaN" in result.prediction_failure
    assert all(s.strategy == "fallback_exploration" for s in result.selected)
    assert len(result.dataset.records) == 6
    assert result.exact_attempts == 2


def test_completed_journal_survives_crash_before_cycle_commit(tmp_path, monkeypatch):
    import toposc_lab.active_learning.campaign as module

    space, dataset, evaluate = cycle_fixture()
    config = campaign_config()
    original = module.atomic_json
    calls = []

    def counted(c, s):
        calls.append((c.candidate_id, s))
        return evaluate(c, s)

    def crash(path, state):
        if state["completed_cycles"] == 1:
            raise KeyboardInterrupt("simulated process death before cycle commit")
        original(path, state)

    path = tmp_path / "interrupted.json"
    monkeypatch.setattr(module, "atomic_json", crash)
    with pytest.raises(KeyboardInterrupt):
        run_campaign(space, dataset, config=config, evaluator=counted, checkpoint_path=path)
    assert len(calls) == 2
    monkeypatch.setattr(module, "atomic_json", original)
    resumed = run_campaign(space, dataset, config=config, evaluator=counted, checkpoint_path=path)
    assert len(calls) == resumed.exact_attempts == 4
    reference = run_campaign(
        space,
        dataset,
        config=config,
        evaluator=evaluate,
        checkpoint_path=tmp_path / "reference.json",
    )
    assert dataset_to_bytes(resumed.dataset) == dataset_to_bytes(reference.dataset)
    assert resumed.history == reference.history


def test_interrupted_solver_is_charged_and_never_repeated(tmp_path):
    space, dataset, evaluate = cycle_fixture()
    path = tmp_path / "interrupted.json"
    interrupted = []

    def crash(c, s):
        interrupted.append(c.candidate_id)
        raise KeyboardInterrupt("solver terminated")

    with pytest.raises(KeyboardInterrupt):
        run_campaign(
            space, dataset, config=campaign_config(), evaluator=crash, checkpoint_path=path
        )
    calls = []

    def counted(c, s):
        calls.append(c.candidate_id)
        return evaluate(c, s)

    resumed = run_campaign(
        space, dataset, config=campaign_config(), evaluator=counted, checkpoint_path=path
    )
    assert resumed.exact_attempts == 4
    assert len(calls) == 3
    assert interrupted[0] not in calls
    assert len(resumed.dataset.records) == 7
    assert "interrupted exact attempt" in resumed.history[0]["verifications"][0]["failure"]


def test_corrupt_partial_file_ignored_but_corrupt_committed_snapshot_rejected(tmp_path):
    space, dataset, evaluate = cycle_fixture()
    path = tmp_path / "state.json"
    run_campaign(
        space,
        dataset,
        config=campaign_config(),
        evaluator=evaluate,
        checkpoint_path=path,
        stop_after=1,
    )
    (tmp_path / ".state.json.incomplete").write_text("{broken", encoding="utf-8")
    result = run_campaign(
        space, dataset, config=campaign_config(), evaluator=evaluate, checkpoint_path=path
    )
    assert result.exact_attempts == 4
    path.write_text(path.read_text().replace('"exact_attempts": 4', '"exact_attempts": 3'))
    with pytest.raises(ValueError, match="checksum"):
        run_campaign(
            space, dataset, config=campaign_config(), evaluator=evaluate, checkpoint_path=path
        )


def test_duplicate_training_labels_rejected_even_with_different_seeds():
    records = tuple(exact_fixture(n=n) for n in (6, 8, 10, 12))
    with pytest.raises(ValueError, match="duplicate exact"):
        retrain(
            ExactPhysicsDataset((*records, exact_fixture(n=6, seed=2))),
            config=training_config(),
            seed=1,
        )


def test_failed_simulations_preserved_without_dataset_poison(tmp_path):
    space, dataset, _ = cycle_fixture()

    def fail(c, s):
        raise ArithmeticError("solver failure")

    result = run_campaign(
        space,
        dataset,
        config=campaign_config(),
        evaluator=fail,
        checkpoint_path=tmp_path / "failed.json",
    )
    assert dataset_to_bytes(result.dataset) == dataset_to_bytes(dataset)
    assert result.exact_attempts == 4
    assert len({v["candidate_id"] for h in result.history for v in h["verifications"]}) == 4
    assert all(v["failure"] for h in result.history for v in h["verifications"])
