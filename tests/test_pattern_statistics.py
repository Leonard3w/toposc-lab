import numpy as np
import pytest

from toposc_lab.patterns.statistics import (
    family_labels,
    grouped_mean_interval,
    holm,
    importance_stability,
    purged_group_splits,
    rank_associations,
    residualize,
    sign_flip_pvalue,
)


def test_adjustment_removes_exact_confound_and_flags_unidentifiable_feature():
    x = np.arange(20.0)[:, None]
    assert np.linalg.norm(residualize(x, x)) < 1e-10
    rows = rank_associations(
        x, x[:, 0], ["confound"], x, np.zeros(20), np.arange(20), 5, permutations=19
    )
    assert rows[0]["spearman"] == pytest.approx(1)
    assert rows[0]["partial_spearman"] is None
    assert rows[0]["p"] is None


def test_exact_confound_is_unidentifiable_even_with_large_outcome_residual():
    x = np.arange(120.0)[:, None]
    y = np.random.default_rng(8).normal(size=120)
    rows = rank_associations(
        x,
        y,
        ["confound"],
        x,
        np.zeros(120),
        np.zeros(120),
        5,
        permutations=19,
        bootstrap_samples=19,
    )
    assert rows[0]["partial_spearman"] is None
    assert rows[0]["status"] == "unidentifiable_after_adjustment"


def test_multiple_testing_family_retains_unidentifiable_variable_features():
    rng = np.random.default_rng(42)
    y = rng.normal(size=40)
    control = rng.normal(size=(40, 1))
    x = np.column_stack((y, control))
    rows = rank_associations(
        x,
        y,
        ["signal", "confound"],
        control,
        np.zeros(40),
        np.zeros(40),
        5,
        permutations=19,
        bootstrap_samples=19,
    )
    assert rows[0]["p"] == 0.05
    assert rows[0]["bh_q"] == 0.1
    assert rows[0]["by_q"] == pytest.approx(0.15)
    assert rows[0]["multiplicity_family_size"] == 2
    assert rows[1]["p"] is None
    assert rows[1]["bh_q"] is None


def test_cluster_bootstrap_weights_parents_not_replicates():
    result = grouped_mean_interval(np.array([1.0, 1.0, 1.0, 3.0]), np.array([0, 0, 0, 1]), 4)
    assert result["mean"] == 2
    assert result["groups"] == 2
    assert grouped_mean_interval(np.ones(3), np.zeros(3), 4)["ci95"] is None
    assert sign_flip_pvalue(np.ones(6)) == 2 / 64
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_family_chaining_and_split_purge():
    d = np.array([[0, 0.1, 0.3, 0.8], [0.1, 0, 0.1, 0.8], [0.3, 0.1, 0, 0.8], [0.8, 0.8, 0.8, 0]])
    assert len(set(family_labels(d, 0.15))) == 2
    groups = np.array([0, 1, 1, 1])
    train, test = purged_group_splits(groups, d)[0]
    assert test.tolist() == [0]
    assert train.tolist() == [2, 3]


@pytest.mark.parametrize(
    "distances",
    [
        np.ones((2, 3)),
        np.array([[0.0, -1], [-1, 0]]),
        np.array([[0.0, np.nan], [np.nan, 0]]),
        np.array([[0.0, 0.1], [0.2, 0]]),
    ],
)
def test_family_and_purge_reject_invalid_distance_matrices(distances):
    with pytest.raises(ValueError, match="distances"):
        family_labels(distances, 0.15)
    with pytest.raises(ValueError, match="distances"):
        purged_group_splits(np.zeros(2), distances)


def test_importance_predictions_do_not_depend_on_heldout_labels():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(24, 2))
    y = x[:, 0] + 0.05 * rng.normal(size=24)
    groups = np.repeat(np.arange(3), 8)
    d = np.ones((24, 24)) - np.eye(24)
    first = importance_stability(x, y, ["signal", "noise"], groups, d)
    changed = y.copy()
    changed[:8] += 100
    second = importance_stability(x, changed, ["signal", "noise"], groups, d)
    for a, b in zip(first["folds"][:2], second["folds"][:2], strict=True):
        assert a["predictions"] == b["predictions"]
        assert a["test_indices"] == list(range(8))


def test_importance_training_scaling_and_group_names_are_auditable():
    rng = np.random.default_rng(93)
    x = rng.normal(size=(24, 2))
    x[:8] += 100
    groups = np.repeat(["a", "b", "c"], 8)
    d = np.ones((24, 24)) - np.eye(24)
    result = importance_stability(x, x[:, 0], ["signal", "noise"], groups, d)
    for fold in result["folds"][:2]:
        assert fold["training_feature_mean"] == pytest.approx(x[8:].mean(axis=0))
        assert fold["training_feature_scale"] == pytest.approx(x[8:].std(axis=0))
    assert result["causal_interpretation"] is False
    assert result["summaries"][0]["features"][0]["seed_bootstrap"]["groups"] == 3
