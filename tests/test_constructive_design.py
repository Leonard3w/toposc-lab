"""Geometric, gradient, persistence and observer regression tests; no physics runs."""

import json
from dataclasses import replace
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pytest

from toposc_lab.design.environment import (
    FEATURE_COUNT,
    BuildRules,
    BuildState,
    GraphBuilder,
    components,
)
from toposc_lab.design.policy import NeuralPolicy, RandomPolicy, restore_policy
from toposc_lab.design.training import (
    CompactWiringDemo,
    Evaluation,
    TrainingConfig,
    load_sealed,
    run_training,
)
from toposc_lab.design.viewer import make_server


def test_live_permission_error_does_not_abort_training(tmp_path, monkeypatch):
    from toposc_lab.design import training

    def locked(*args):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(training.os, "replace", locked)
    output = tmp_path / "locked"
    result = run_training(
        square_rules(), TrainingConfig(2), RandomPolicy(), CompactWiringDemo(), output
    )
    assert result["episodes"] == 2
    assert load_sealed(output / "complete.json")["episodes"] == 2


def test_exact_display_predecessor_resumes_without_changing_manifest(tmp_path):
    from toposc_lab.design.training import save_sealed

    output = tmp_path / "predecessor"
    rules, config = square_rules(), TrainingConfig(1)

    def interrupt(event):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_training(rules, config, RandomPolicy(), CompactWiringDemo(), output, observer=interrupt)
    manifest_path = output / "manifest.json"
    manifest = load_sealed(manifest_path)
    manifest["code_sha256"] = "2b4a4fdccd514989be5562534199fb4381909ac5d609946c73cb3c88df161aae"
    manifest_path.unlink()  # Synthetic fixture only, never a user archive.
    save_sealed(manifest_path, manifest)
    original = manifest_path.read_bytes()
    result = run_training(rules, config, RandomPolicy(), CompactWiringDemo(), output, resume=True)
    assert result["episodes"] == 1
    assert manifest_path.read_bytes() == original
    assert (
        load_sealed(output / "episode_000000.json")["execution_code_sha256"]
        != manifest["code_sha256"]
    )


def test_live_recovers_after_lock_and_other_io_errors_propagate(tmp_path, monkeypatch):
    from toposc_lab.design import training

    original = training.os.replace
    with monkeypatch.context() as patch:

        def locked(*args):
            raise PermissionError(5, "reader lock")

        patch.setattr(training.os, "replace", locked)
        assert not training.publish_live(tmp_path, {"step": 1})
    assert training.os.replace is original
    assert training.publish_live(tmp_path, {"step": 2})
    assert json.loads((tmp_path / "live.json").read_text()) == {"step": 2}
    with monkeypatch.context() as patch:

        def full(*args):
            raise OSError(28, "disk full")

        patch.setattr(training.os, "replace", full)
        with pytest.raises(OSError, match="disk full"):
            training.publish_live(tmp_path, {"step": 3})


def square_rules() -> BuildRules:
    return BuildRules(
        site_count=4,
        edge_count=4,
        width=1,
        height=1,
        minimum_degree=2,
        maximum_degree=3,
        fixed_points=((0, 0), (0, 1), (1, 0), (1, 1)),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("site_count", True),
        ("site_count", 1),
        ("edge_count", 0),
        ("width", 0),
        ("height", float("nan")),
        ("maximum_degree", 12),
        ("point_proposals", 0),
        ("minimum_degree", 5),
        ("minimum_separation", -1),
        ("tolerance", 3),
    ],
)
def test_rules_reject_bad_contract(field, value):
    with pytest.raises(ValueError):
        replace(BuildRules(), **{field: value})


def test_fixed_points_validate_and_freeze():
    rules = square_rules()
    assert rules.fixed_points == ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))
    with pytest.raises(ValueError, match="separation"):
        replace(rules, fixed_points=((0, 0),) * 4)
    with pytest.raises(ValueError, match="outside"):
        replace(rules, fixed_points=((0, 0), (0, 1), (1, 0), (2, 1)))


def test_square_is_reachable_without_reward_or_target_pattern():
    env = GraphBuilder(square_rules())
    state = env.initial()
    assert not state.edges
    for edge in ((0, 1), (0, 2), (1, 3), (2, 3)):
        assert edge in env.choices(state, np.random.default_rng(0))
        state = env.step(state, edge)
    assert env.validate_final(state) == ()
    geometry = state.geometry()
    assert geometry.n_sites == 4 and geometry.n_edges == 4
    assert not geometry.boundary_sites  # No made-up physical boundary from degree.
    for edge in geometry.edges:
        assert edge.displacement == tuple(
            geometry.coordinates[edge.target] - geometry.coordinates[edge.source]
        )


def test_diagonal_crossings_and_third_site_are_masked():
    env = GraphBuilder(replace(square_rules(), minimum_degree=1, edge_count=3))
    state = env.step(env.initial(), (0, 3))
    assert (1, 2) not in env.choices(state, np.random.default_rng(0))
    with pytest.raises(ValueError):
        env.step(state, (1, 2))
    line = BuildRules(
        site_count=3, edge_count=2, maximum_degree=2, fixed_points=((0, 0), (1, 0), (2, 0))
    )
    assert (0, 2) not in GraphBuilder(line).choices(
        GraphBuilder(line).initial(), np.random.default_rng(0)
    )


def test_insufficient_connectivity_budget_masks_cycle():
    rules = replace(square_rules(), edge_count=3, minimum_degree=1)
    env = GraphBuilder(rules)
    state = env.step(env.step(env.initial(), (0, 1)), (0, 2))
    assert (1, 2) not in env.choices(state, np.random.default_rng(0))


def test_final_validation_rejects_forged_snapshots():
    env = GraphBuilder(square_rules())
    assert env.validate_final(BuildState(env.rules.fixed_points, ((0, 1),) * 4))
    assert env.validate_final(BuildState(((4, 4),) * 4, ((0, 1),) * 4))
    assert env.validate_final(env.initial()) == ("incomplete",)


def test_continuous_free_proposals_are_seeded_and_no_lattice():
    env = GraphBuilder(BuildRules())
    first = env.choices(env.initial(), np.random.default_rng(7))
    assert first == env.choices(env.initial(), np.random.default_rng(7))
    assert first != env.choices(env.initial(), np.random.default_rng(8))
    assert any(x != round(x) for x, y in first)
    state = env.step(env.initial(), first[0])
    next_choices = env.choices(state, np.random.default_rng(9))
    assert all(np.linalg.norm(np.array(p) - np.array(first[0])) >= 0.4 for p in next_choices)
    assert env.features(state, next_choices).shape == (len(next_choices), FEATURE_COUNT)


@pytest.mark.parametrize("parameter,gradient_index", [("weights", 0), ("bias", 1), ("output", 2)])
def test_policy_gradient_matches_finite_differences(parameter, gradient_index):
    policy = NeuralPolicy(seed=2, hidden=3)
    features = np.random.default_rng(4).normal(size=(5, FEATURE_COUNT))
    expected = policy.log_gradient(features, 2)[gradient_index]
    array = getattr(policy, parameter)
    for index in np.ndindex(array.shape):
        original = array[index]
        array[index] = original + 1e-6
        high = np.log(policy.probabilities(features)[2])
        array[index] = original - 1e-6
        low = np.log(policy.probabilities(features)[2])
        array[index] = original
        assert (high - low) / 2e-6 == pytest.approx(expected[index], abs=1e-8)


def test_neural_update_learns_and_none_discards_gradient():
    policy = NeuralPolicy(seed=4)
    features = np.zeros((2, FEATURE_COUNT))
    features[1, 0] = 1
    before = policy.probabilities(features)[1]
    # Deterministic synthetic bandit, not a research/training campaign.
    rng = np.random.default_rng(1)
    for _ in range(250):
        action = policy.choose(features, rng)
        policy.finish(float(action))
    assert policy.probabilities(features)[1] > before + 0.1
    snapshot = policy.snapshot()
    policy.choose(features, rng)
    policy.finish(None)
    assert policy.snapshot() == snapshot
    assert restore_policy(snapshot).snapshot() == snapshot


def test_action_features_and_probabilities_are_permutation_equivariant():
    policy = NeuralPolicy(seed=3)
    features = np.random.default_rng(1).normal(size=(7, FEATURE_COUNT))
    order = [4, 3, 1, 5, 2, 6, 0]
    assert np.allclose(policy.probabilities(features[order]), policy.probabilities(features)[order])


def test_episode_resume_matches_uninterrupted_and_observer_cannot_mutate(tmp_path):
    rules, config = square_rules(), TrainingConfig(4, 18)
    uninterrupted, interrupted = tmp_path / "a", tmp_path / "b"
    result = run_training(rules, config, NeuralPolicy(seed=18), CompactWiringDemo(), uninterrupted)

    def stop(event):
        if event["episode"] == 1 and event["step"] == 2:
            raise KeyboardInterrupt
        event["state"]["points"].clear()

    with pytest.raises(KeyboardInterrupt):
        run_training(
            rules, config, NeuralPolicy(seed=18), CompactWiringDemo(), interrupted, observer=stop
        )
    assert not (interrupted / "writer.lock").exists()
    resumed = run_training(
        rules, config, NeuralPolicy(seed=18), CompactWiringDemo(), interrupted, resume=True
    )
    assert len(resumed["history"]) == len(result["history"]) == 4
    for i in range(4):
        a = load_sealed(uninterrupted / f"episode_{i:06d}.json")
        b = load_sealed(interrupted / f"episode_{i:06d}.json")
        assert a["policy"] == b["policy"]
        assert a["state"] == b["state"]
        assert a["evaluation"] == b["evaluation"]
    assert len(list((interrupted / "attempts_000001").iterdir())) == 2
    before = {p: p.read_bytes() for p in interrupted.rglob("*") if p.is_file()}
    assert (
        run_training(
            rules, config, NeuralPolicy(seed=18), CompactWiringDemo(), interrupted, resume=True
        )
        == resumed
    )
    assert before == {p: p.read_bytes() for p in interrupted.rglob("*") if p.is_file()}


def test_archive_exclusive_and_resume_contract_and_corruption(tmp_path):
    rules, config = square_rules(), TrainingConfig(1, 3)
    output = tmp_path / "run"
    run_training(rules, config, RandomPolicy(), CompactWiringDemo(), output)
    with pytest.raises(FileExistsError):
        run_training(rules, config, RandomPolicy(), CompactWiringDemo(), output)
    with pytest.raises(ValueError, match="resume"):
        run_training(
            rules,
            replace(config, episodes=2),
            RandomPolicy(),
            CompactWiringDemo(),
            output,
            resume=True,
        )
    path = output / "episode_000000.json"
    content = json.loads(path.read_text())
    content["payload"]["policy"]["kind"] = "tampered"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="checksum"):
        run_training(rules, config, RandomPolicy(), CompactWiringDemo(), output, resume=True)


def test_evaluator_failure_not_used_as_reward(tmp_path):
    class Broken:
        identifier = "broken_test_v1"

        def __call__(self, *args):
            raise ArithmeticError("test failure")

    policy = NeuralPolicy(seed=1)
    before = policy.snapshot()
    rules = replace(square_rules(), maximum_edge_length=1.0)
    result = run_training(rules, TrainingConfig(1), policy, Broken(), tmp_path / "broken")
    assert result["history"][0]["reward"] is None
    assert "ArithmeticError" in result["history"][0]["status"]
    assert policy.snapshot() == before


def test_dead_end_skips_evaluator_and_keeps_denominator(tmp_path):
    rules = replace(square_rules(), maximum_edge_length=0.1)
    result = run_training(
        rules, TrainingConfig(2), RandomPolicy(), CompactWiringDemo(), tmp_path / "dead"
    )
    assert len(result["history"]) == 2
    assert all(row["reward"] == -1 and not row["evaluator_called"] for row in result["history"])


def test_free_placement_episodes_remain_geometric(tmp_path):
    rules = BuildRules(
        site_count=5, edge_count=4, maximum_degree=4, maximum_edge_length=6, minimum_separation=0.1
    )
    output = tmp_path / "free"
    run_training(rules, TrainingConfig(3, 12), NeuralPolicy(seed=2), CompactWiringDemo(), output)
    for i in range(3):
        data = load_sealed(output / f"episode_{i:06d}.json")
        points = tuple(tuple(p) for p in data["state"]["points"])
        edges = tuple(tuple(e) for e in data["state"]["edges"])
        assert len(points) == 5
        assert data["evaluation"]["status"] == "engineering_only"
        assert len(set(components(5, edges))) == 1


def test_evaluation_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        Evaluation(float("nan"), "bad", {})
    with pytest.raises(ValueError):
        Evaluation(None, "bad", {"x": float("inf")})


def test_deterministic_cache_counts_duplicates_without_new_evaluations(tmp_path):
    rules = replace(square_rules(), maximum_edge_length=1.0)
    result = run_training(
        rules, TrainingConfig(4), RandomPolicy(), CompactWiringDemo(), tmp_path / "cache"
    )
    assert result["unique_valid_geometries"] == 1
    assert result["evaluator_calls_in_sealed_episodes"] == 1
    assert result["cache_hits"] == 3
    assert result["episodes"] == 4


def test_stochastic_evaluator_seeds_prevent_cross_seed_cache(tmp_path):
    class Seeded:
        identifier = "test_seeded_v1"

        def __call__(self, state, rules, seed):
            return Evaluation(float(np.random.default_rng(seed).random()), "test", {})

    result = run_training(
        replace(square_rules(), maximum_edge_length=1.0),
        TrainingConfig(4),
        RandomPolicy(),
        Seeded(),
        tmp_path / "seeded",
    )
    assert result["evaluator_calls_in_sealed_episodes"] == 4
    assert result["cache_hits"] == 0
    assert len({row["reward"] for row in result["history"]}) == 4


def test_completed_archive_missing_episode_is_never_repaired(tmp_path):
    output = tmp_path / "complete"
    rules, config = square_rules(), TrainingConfig(2)
    run_training(rules, config, RandomPolicy(), CompactWiringDemo(), output)
    (output / "episode_000001.json").unlink()
    before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="refusing to modify"):
        run_training(rules, config, RandomPolicy(), CompactWiringDemo(), output, resume=True)
    assert before == {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}


@pytest.mark.parametrize("n", [2, 3])
def test_cli_config_emits_usable_rules_and_cli_import_does_not_train(capsys, monkeypatch, n):
    from toposc_lab.design.__main__ import main

    monkeypatch.setattr("sys.argv", ["design", "config", "--square", str(n)])
    main()
    rules = BuildRules(**json.loads(capsys.readouterr().out))
    assert rules.site_count == n * n and rules.edge_count == 2 * n * (n - 1)
    assert len(rules.fixed_points) == n * n


def test_loopback_viewer_serves_only_approved_resources(tmp_path):
    output = tmp_path / "view"
    run_training(square_rules(), TrainingConfig(1), RandomPolicy(), CompactWiringDemo(), output)
    server = make_server(output, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base) as response:
            assert "Graphbauer" in response.read().decode()
        with urlopen(base + "/state") as response:
            assert json.load(response)["stage"] == "episode_complete"
        with urlopen(base + "/episode/0") as response:
            assert len(json.load(response)) >= 6
        for route in ("/manifest.json", "/../manifest.json", "/episode/../manifest.json"):
            with pytest.raises(HTTPError) as error:
                urlopen(base + route)
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
