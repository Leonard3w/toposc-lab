"""Long-run regressions: exact distances, recovery, policy and honest diagnostics."""
import copy
import json

import numpy as np
import pytest

from toposc_lab.research.reporting import search_assessment
from toposc_lab.research.space import FixedConnectivitySpace, geometry_from_payload
from toposc_lab.research.strategies import create_strategy


@pytest.mark.parametrize("side,tolerance", [(3, 0), (4, .2), (10, 0)])
def test_compact_masks_preserve_set_jaccard_for_all_symmetries(side, tolerance):
    space = FixedConnectivitySpace(side=side, bond_tolerance=tolerance,
                                   max_bond_length=2, site_crossings="unconnected")
    strategy = create_strategy("random", space, 17301)
    rng = np.random.default_rng(17301)
    geometries = [space.reference(), space.sample(rng), space.sample(rng)]
    for first in geometries:
        edges = frozenset(space.edges(first))
        mask = strategy._edge_mask(edges)
        for second in geometries:
            for other, encoded in zip(space.orbit(second), strategy._masks(second), strict=True):
                expected = len(edges ^ other) / max(1, len(edges | other))
                actual = (mask ^ encoded).bit_count() / max(1, (mask | encoded).bit_count())
                assert actual == expected


def test_legacy_checkpoint_migrates_and_replays_with_compact_seen():
    space = FixedConnectivitySpace(side=4, minimum_distance=.1)
    strategy = create_strategy("map_elites", space, 17302)
    proposals = strategy.propose(8)
    old_seen = {c["id"]: copy.deepcopy(c) for c in proposals}
    for i, candidate in enumerate(proposals):
        strategy.observe(candidate, {"origin": "exact", "score": i / 8,
                                    "validation_state": "EXACT_EVALUATED"})
    compact = json.loads(json.dumps(strategy.checkpoint(), sort_keys=True))
    legacy = {**copy.deepcopy(compact), "version": 1, "seen": old_seen}
    resumed = create_strategy("map_elites", space, 17302).resume(legacy)
    assert resumed.checkpoint() == compact
    expected, actual = strategy.propose(8), resumed.propose(8)
    assert [(c["id"], c["seed"]) for c in expected] == [(c["id"], c["seed"]) for c in actual]
    assert strategy.summarize() == resumed.summarize()
    assert len(json.dumps(compact["seen"])) < len(json.dumps(old_seen)) / 3


def test_quality_parent_mixture_keeps_uniform_exploration_and_resume():
    space = FixedConnectivitySpace(side=4)
    options = {"quality_parent_probability": .7, "quality_parent_fraction": .25}
    strategy = create_strategy("surrogate_map_elites", space, 22, **options)
    proposals = strategy.propose(8)
    for i, candidate in enumerate(proposals):
        strategy.observe(candidate, {"origin": "exact", "score": i,
                                    "validation_state": "EXACT_EVALUATED"})
    # Isolate archive-parent sampling from descriptor bin collisions.
    strategy.archive = {str(i): c for i, c in enumerate(strategy.history)}
    state = json.loads(json.dumps(strategy.checkpoint(), sort_keys=True))
    resumed = create_strategy("surrogate_map_elites", space, 22, **options).resume(state)
    expected = [strategy._parent()["id"] for _ in range(400)]
    assert expected == [resumed._parent()["id"] for _ in range(400)]
    top = {c["id"] for c in sorted(strategy.history, key=lambda c: -c["score"])[:2]}
    assert 250 < sum(identity in top for identity in expected) < 370
    assert set(expected) == {c["id"] for c in strategy.history}


def test_batch_novelty_spreads_selection_when_alternatives_exist():
    class Model:
        fitted = True

        def predict(self, proposals):
            return [{"score": 0, "uncertainty": 0, "ood": False} for _ in proposals]

    options = {"behavior_descriptors": ["regular_edge_distance"], "descriptor_bounds": [[0, 1]],
                   "archive_bins": [10], "allocation": {"novelty": 1}, "family_fraction": 1}
    proposals = [{"id": name, "descriptors": {"regular_edge_distance": value}}
                 for name, value in [("z", .1), ("y", .11), ("a", .9)]]
    old = create_strategy("surrogate_map_elites", FixedConnectivitySpace(side=3), 1, **options)
    new = create_strategy("surrogate_map_elites", FixedConnectivitySpace(side=3), 1,
                          batch_novelty=True, **options)
    assert [c["id"] for c in old.select(copy.deepcopy(proposals), 2, Model())] == ["z", "y"]
    assert [c["id"] for c in new.select(copy.deepcopy(proposals), 2, Model())] == ["z", "a"]


@pytest.mark.parametrize("options", [{"quality_parent_probability": -.1},
    {"quality_parent_probability": float("nan")}, {"quality_parent_fraction": 0},
    {"quality_parent_fraction": 1.1}, {"batch_novelty": 1}])
def test_search_policy_rejects_invalid_options(options):
    with pytest.raises(ValueError):
        create_strategy("map_elites", FixedConnectivitySpace(side=3), **options)


def test_tradeoff_report_excludes_failed_partial_and_surrogate_evidence():
    def candidate(identity, score, weight, **changes):
        return {"id": identity, "score": score, "origin": "exact",
                "validation_state": "ROBUSTNESS_VALIDATED", "baseline": False,
                "raw_metrics": {"boundary_weight": weight, "robustness_success_fraction": 1},
                **changes}
    rows = [candidate("regular", .37, .99, family="regular", baseline=True),
            candidate("best", .39, .8), candidate("dominated", .30, .7),
            candidate("failed", 10, 1, validation_state="FAILED"),
            candidate("partial", None, 1, origin="partial_exact"),
            candidate("prediction", 10, 1, origin="surrogate")]
    report = search_assessment({"candidates": rows})
    assert report["complete_searched"] == 2
    assert report["all_sampled_successes"] == 2
    assert len(report["warnings"]) == 2
    assert {c["id"] for c in report["objective_boundary_frontier"]} == {"best", "regular"}


def test_observe_keeps_detailed_validation_only_in_original_candidate():
    strategy = create_strategy("map_elites", FixedConnectivitySpace(side=3), 173)
    candidate = strategy.propose(1)[0]
    result = {"score": .3, "origin": "exact", "validation_state": "EXACT_EVALUATED",
              "validation_results": {"majorana": {"passed": False, "status": "diagnostics_only",
                                                    "site_probability": [1] * 10000}}}
    strategy.observe(candidate, result)
    assert "site_probability" not in strategy.history[0]["validation_results"]["majorana"]
    assert len(result["validation_results"]["majorana"]["site_probability"]) == 10000
    assert strategy.space.hash(geometry_from_payload(strategy.history[0]["geometry"])) == candidate["id"]
