"""Synthetic engineering checks for the paired frozen-policy control."""

import json

import numpy as np
import pytest

from toposc_lab.design.benchmark import (
    FrozenNeuralPolicy,
    run_benchmark,
    summarize_pairs,
    summarize_trial,
)
from toposc_lab.design.environment import FEATURE_COUNT, BuildRules
from toposc_lab.design.policy import NeuralPolicy, restore_policy
from toposc_lab.design.training import load_sealed


def test_frozen_actions_match_initial_actor_without_updates():
    learner = NeuralPolicy(seed=9)
    frozen = FrozenNeuralPolicy(NeuralPolicy(seed=9))
    first_rng, second_rng = np.random.default_rng(2), np.random.default_rng(2)
    features = np.random.default_rng(3).normal(size=(7, FEATURE_COUNT))
    original = frozen.snapshot()
    for _ in range(10):
        assert learner.choose(features, first_rng) == frozen.choose(features, second_rng)
        frozen.finish(1.0)
    assert frozen.snapshot() == original
    learner.finish(1.0)
    assert learner.snapshot()["weights"] != frozen.snapshot()["weights"]
    assert restore_policy(original).snapshot() == original


def test_metric_keeps_operational_failures_unavailable():
    result = {
        "history": [
            {"reward": 0.5, "geometry_valid": True},
            {"reward": None, "geometry_valid": True},
            {"reward": -1.0, "geometry_valid": False},
        ],
        "evaluator_calls_in_sealed_episodes": 2,
        "cache_hits": 0,
        "unavailable_rewards": 1,
        "additional_executions": 0,
    }
    summary = summarize_trial(result)
    assert summary["mean_reward"] is None
    assert summary["valid_fraction"] == pytest.approx(2 / 3)
    assert summary["mean_valid_reward"] == 0.5
    pair = summarize_pairs(
        [{"seed": 1, "arms": {"neural": summary, "frozen": summary, "random": summary}}]
    )
    assert pair["comparisons"]["neural_minus_frozen"]["available_pairs"] == 0


def test_benchmark_paired_initialization_budget_and_resume(tmp_path, monkeypatch):
    from toposc_lab.design import benchmark

    rules = BuildRules(
        site_count=4,
        edge_count=4,
        width=1,
        height=1,
        maximum_degree=3,
        minimum_degree=2,
        maximum_edge_length=1,
        fixed_points=((0, 0), (0, 1), (1, 0), (1, 1)),
    )
    monkeypatch.setattr(benchmark, "BuildRules", lambda: rules)
    output = tmp_path / "paired"
    result = run_benchmark(output, seeds=(31, 32), episodes=2)
    assert len(result["trials"]) == 2
    for seed in (31, 32):
        base = output / f"seed_{seed}"
        neural = load_sealed(base / "neural" / "manifest.json")["initial_policy"]
        frozen = load_sealed(base / "frozen" / "manifest.json")["initial_policy"]
        assert neural["weights"] == frozen["weights"]
        assert neural["output"] == frozen["output"]
        assert load_sealed(base / "frozen" / "episode_000001.json")["policy"] == frozen
        assert load_sealed(base / "neural" / "episode_000001.json")["policy"]["updates"] == 2
    before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert run_benchmark(output, seeds=(31, 32), episodes=2, resume=True) == result
    assert before == {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="protocol"):
        run_benchmark(output, seeds=(31, 32), episodes=3, resume=True)
    event = json.loads((output / "live.json").read_text())
    assert event["benchmark_seed"] == 32
    assert event["benchmark_arm"] in ("neural", "frozen", "random")


@pytest.mark.parametrize("seeds", [(), (1, 1), (-1,)])
def test_invalid_seed_schedule_never_creates_archive(tmp_path, seeds):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        run_benchmark(output, seeds=seeds)
    assert not output.exists()
