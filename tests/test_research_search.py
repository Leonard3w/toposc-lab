"""Research search contracts: fixed science inputs, exact-only archive and reproducible policy."""
from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.research.descriptors import compute_descriptors, register_descriptor
from toposc_lab.research.space import (
    MUTATION_REGISTRY,
    FixedConnectivitySpace,
    geometry_from_payload,
    geometry_to_payload,
)
from toposc_lab.research.strategies import STRATEGY_REGISTRY, create_strategy
from toposc_lab.research.surrogate import GraphSurrogate


def exact(candidate, score):
    return {**candidate, "origin": "exact", "score": score,
            "validation_state": "EXACT_EVALUATED"}


def records(count=8):
    strategy = create_strategy("random", FixedConnectivitySpace(side=4), seed=123)
    return [exact(p, float(i) / count) for i, p in enumerate(strategy.propose(count))]


def test_default_fixed_positions_and_approximate_bond_budget():
    space = FixedConnectivitySpace()
    reference = space.reference()
    assert reference.n_sites == 100
    assert reference.n_edges == 180
    assert not space.validate(reference)
    moved = replace(reference, coordinates=reference.coordinates + 0.01)
    assert "fixed_sites_required" in space.validate(moved)
    with pytest.raises(ValueError, match="fixed sites"):
        space.hash(moved)


@pytest.mark.parametrize("operator", list(MUTATION_REGISTRY))
def test_mutation_registry_preserves_fixed_sites_and_records_edits(operator):
    space = FixedConnectivitySpace(side=4, bond_tolerance=0.1)
    parent = space.reference()
    # Seed a diagonal so the geometrically valid flip can operate.
    if operator == "diagonal_flip":
        parent = space.build(set(space.edges(parent)) | {(0, 5)})
    changed = False
    for seed in range(200):
        geometry, mutation = space.mutate(parent, np.random.default_rng(seed), operator)
        np.testing.assert_array_equal(geometry.coordinates, parent.coordinates)
        assert geometry.n_sites == parent.n_sites
        assert mutation["operator"] == operator
        if mutation["added"] or mutation["removed"]:
            changed = True
            if operator not in ("add_local_bond", "remove_local_bond"):
                assert geometry.n_edges == parent.n_edges
            if operator == "edge_swap":
                assert [len(geometry.neighbors(i)) for i in range(16)] == [
                    len(parent.neighbors(i)) for i in range(16)]
            break
    assert changed, operator


def test_constraints_reject_disconnection_crossings_long_bonds_and_extra_budget():
    space = FixedConnectivitySpace(side=3, bond_tolerance=0, forbid_crossings=True)
    edges = set(space.edges(space.reference()))
    disconnected = space.build({e for e in edges if 0 not in e})
    assert space.validate(disconnected)
    crossed = space.build((edges - {(4, 5), (5, 8)}) | {(0, 4), (1, 3)})
    assert any("cross" in reason for reason in space.validate(crossed))
    long = space.build((edges - {(0, 1)}) | {(0, 8)})
    assert space.validate(long)
    assert space.validate(space.build(edges | {(0, 4)}))


def test_serialization_roundtrip_and_symmetry_duplicate_identity():
    space = FixedConnectivitySpace(side=4)
    geometry = space.sample(np.random.default_rng(1))
    loaded = geometry_from_payload(json.loads(json.dumps(geometry_to_payload(geometry))))
    assert space.hash(geometry) == space.hash(loaded)
    assert space.distance(geometry, loaded) == 0
    reflected = space.build(set(space.orbit(geometry)[-1]))
    assert space.hash(geometry) == space.hash(reflected)
    assert space.distance(geometry, reflected) == 0
    labels = np.arange(16)[::-1]
    relabeled = Geometry(n_sites=16, coordinates=geometry.coordinates[labels],
        edges=tuple(GeometryEdge(int(labels[e.source]), int(labels[e.target])) for e in geometry.edges))
    assert space.hash(relabeled) == space.hash(geometry)


def test_descriptor_definitions_square_and_extensible_registry():
    space = FixedConnectivitySpace(side=3)
    descriptor = compute_descriptors(space.reference())
    assert descriptor["triangle_count"] == 0
    assert descriptor["four_cycle_count"] == 4
    assert descriptor["graph_diameter"] == 4
    assert descriptor["clustering_coefficient"] == 0
    assert descriptor["short_loop_mean_size"] == 4
    assert descriptor["bond_length_mean"] == 1
    assert descriptor["anisotropy"] == 0
    name = "test_research_custom_descriptor"
    register_descriptor(name, lambda geometry: float(geometry.n_sites**2))
    assert compute_descriptors(space.reference(), [name]) == {name: 81.0}
    from toposc_lab.research.descriptors import DESCRIPTOR_REGISTRY
    DESCRIPTOR_REGISTRY.pop(name)


@pytest.mark.parametrize("name", list(STRATEGY_REGISTRY))
def test_all_search_strategies_seed_and_checkpoint_resume(name):
    space = FixedConnectivitySpace(side=4)
    first = create_strategy(name, space, 123)
    for i, proposal in enumerate(first.propose(5)):
        first.observe(proposal, exact(proposal, i / 5))
    state = json.loads(json.dumps(first.checkpoint(), allow_nan=False, sort_keys=True))
    second = create_strategy(name, space, 123).resume(state)
    expected, actual = first.propose(5), second.propose(5)
    assert [p["id"] for p in expected] == [p["id"] for p in actual]
    assert [p["seed"] for p in expected] == [p["seed"] for p in actual]
    assert first.summarize() == second.summarize()
    assert first.archive
    if name != "random":
        assert all(p["parent"] is not None for p in expected)


@pytest.mark.parametrize("result", [
    {"score": 10.0, "origin": "surrogate", "validation_state": "EXACT_EVALUATED"},
    {"score": float("nan"), "origin": "exact", "validation_state": "EXACT_EVALUATED"},
    {"score": float("inf"), "origin": "exact", "validation_state": "EXACT_EVALUATED"},
    {"score": 100.0, "origin": "exact", "validation_state": "FAILED"},
    {"score": None, "origin": "exact", "validation_state": "EXACT_EVALUATED"},
])
def test_archive_never_accepts_surrogates_failed_or_nonfinite_scores(result):
    strategy = create_strategy("map_elites", FixedConnectivitySpace(side=4), 10)
    proposal = strategy.propose(1)[0]
    strategy.observe(proposal, result)
    assert not strategy.archive
    assert strategy.summarize()["best_score"] is None


def test_out_of_bounds_descriptor_does_not_silently_clip_into_archive():
    strategy = create_strategy("map_elites", FixedConnectivitySpace(side=4), 10,
                               descriptor_bounds=((0, 0.001), (0, 1)))
    proposal = strategy.propose(1)[0]
    strategy.observe(proposal, exact(proposal, 0.1))
    assert not strategy.archive


def test_duplicate_near_duplicate_rejections_traceable_and_proposal_budget():
    # Fixed reference sampler makes duplicate accounting independent of random luck.
    class RepeatingSpace(FixedConnectivitySpace):
        def sample(self, rng):
            return self.reference()
    strategy = create_strategy("random", RepeatingSpace(side=3), 10)
    strategy.proposal_limit = 3
    assert len(strategy.propose(5)) == 1
    rejected = strategy.drain_rejections()
    assert len(rejected) == 2
    assert all(p["rejection_reason"] == "duplicate" for p in rejected)
    assert len({p["id"] for p in rejected}) == 2
    assert strategy.generated == 3
    assert strategy.drain_rejections() == []
    near = create_strategy("random", FixedConnectivitySpace(side=4, minimum_distance=0.9), 2)
    assert len(near.propose(3, max_attempts=5)) == 1
    assert any(r["rejection_reason"] == "near_duplicate" for r in near.drain_rejections())


def test_default_planar_hundred_site_space_produces_valid_candidates():
    strategy = create_strategy("random", FixedConnectivitySpace(
        side=10, min_degree=2, max_degree=6, forbid_crossings=True, bond_tolerance=0), 17100)
    proposals = strategy.propose(2, max_attempts=120)
    assert len(proposals) == 2
    assert all(not strategy.space.validate(geometry_from_payload(p["geometry"])) for p in proposals)


def test_surrogate_resume_batched_predictions_ood_and_prequential_calibration():
    data = records(8)
    model = GraphSurrogate(ensemble_size=3, n_estimators=4, seed=5).fit(data[:5])
    prediction = model.predict(data[5:])
    assert all(p["origin"] == "surrogate" and np.isfinite(p["uncertainty"]) for p in prediction)
    assert model.summarize()["calibration_count"] == 0
    model.observe(prediction[0], data[5]["score"])
    model.observe(prediction[0], data[5]["score"])
    assert model.summarize()["calibration_count"] == 1
    model.observe(model.predict(data[:1])[0], data[0]["score"])
    assert model.summarize()["calibration_count"] == 1
    restored = GraphSurrogate().resume(json.loads(json.dumps(
        model.checkpoint(), allow_nan=False, sort_keys=True)))
    assert restored.predict(data[5:]) == prediction
    assert restored.summarize() == model.summarize()
    far = copy.deepcopy(data[5])
    far["descriptors"] = {k: v + 10000 for k, v in far["descriptors"].items()}
    assert model.predict([far])[0]["ood"] is True


def test_surrogate_rejects_duplicates_and_never_trains_on_predicted_labels():
    data = records(5)
    with pytest.raises(ValueError, match="duplicate"):
        GraphSurrogate().fit([*data, data[0]])
    for record in data:
        record["origin"] = "surrogate"
    with pytest.raises(ValueError, match="four valid exact"):
        GraphSurrogate().fit(data)


def test_active_selection_uses_configured_allocation_across_small_batches():
    strategy = create_strategy("surrogate_map_elites", FixedConnectivitySpace(side=4), 10,
                               allocation={"exploitation": 0.6, "uncertainty": 0.2, "novelty": 0.2})
    model = GraphSurrogate(ensemble_size=3, n_estimators=3).fit(records(5))
    proposals = strategy.propose(15)
    assert len(proposals) == 15
    for _ in range(5):
        selected = strategy.select(proposals, 1, model)
        proposals = [p for p in proposals if p["id"] != selected[0]["id"]]
        assert selected[0]["score"] is None
        assert selected[0]["validation_state"] == "SURROGATE_SCREENED"
    assert dict(strategy.selection_counts) == {"exploitation": 3, "uncertainty": 1, "novelty": 1}
    assert not strategy.archive


def test_failed_candidate_does_not_become_evolution_parent():
    strategy = create_strategy("evolution", FixedConnectivitySpace(side=4), 3)
    proposal = strategy.propose(1)[0]
    strategy.observe(proposal, {"origin": "exact", "score": 100, "validation_state": "FAILED"})
    assert strategy.propose(1)[0]["parent"] is None


@pytest.mark.parametrize("kwargs", [
    {"behavior_descriptors": ["unregistered", "clustering_coefficient"]},
    {"behavior_descriptors": ["coordination_variance", "coordination_variance"]},
    {"archive_bins": [3.5, 4]}, {"archive_bins": [True, 3]},
    {"mutation_rates": {}}, {"mutation_rates": {"edge_swap": -1}},
    {"mutation_rates": {"edge_swap": float("nan")}},
    {"mutation_rates": {"unknown": 1}},
])
def test_search_rejects_invalid_plugin_configuration(kwargs):
    with pytest.raises(ValueError):
        create_strategy("random", FixedConnectivitySpace(side=3), **kwargs)


def test_archive_checkpoint_retains_metadata_without_exact_eigensystems():
    strategy = create_strategy("map_elites", FixedConnectivitySpace(side=4), 10)
    candidate = strategy.propose(1)[0]
    result = exact(candidate, 0.2)
    result["exact_results"] = {"eigenvectors": [[1, 0], [0, 1]]}
    result["raw_metrics"] = {"quality": 0.2, "bulk_gap": 0.1}
    strategy.observe(candidate, result)
    checkpoint = json.loads(json.dumps(strategy.checkpoint(), allow_nan=False))
    assert "eigenvectors" not in json.dumps(checkpoint)
    elite = next(iter(checkpoint["archive"].values()))
    assert elite["raw_metrics"] == result["raw_metrics"]
    assert elite["seed"] == candidate["seed"]
    assert elite["code_version"] == candidate["code_version"]
    assert elite["geometry"] == candidate["geometry"]
    invalid = strategy.propose(1)[0]
    strategy.observe(invalid, exact(invalid, float("nan")))
    json.dumps(strategy.checkpoint(), allow_nan=False)


def test_active_selection_limits_actual_family_when_alternatives_exist():
    strategy = create_strategy("surrogate_map_elites", FixedConnectivitySpace(side=4), 10,
                               allocation={"exploitation": 1, "uncertainty": 0, "novelty": 0})
    model = GraphSurrogate(ensemble_size=3, n_estimators=3).fit(records(5))
    proposals = strategy.propose(10)
    for index, proposal in enumerate(proposals):
        proposal["family"] = "family_a" if index < 5 else "family_b"
    selected = strategy.select(proposals, 4, model)
    assert sum(p["family"] == "family_a" for p in selected) == 2
    assert sum(p["family"] == "family_b" for p in selected) == 2
