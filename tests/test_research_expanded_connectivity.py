"""Phase 17.1 structural definitions, independent of physical success labels."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from toposc_lab.geometry import GeometryEdge
from toposc_lab.research.descriptors import compute_descriptors, regular_edge_distance
from toposc_lab.research.space import (
    FixedConnectivitySpace,
    geometry_from_payload,
    geometry_to_payload,
)


def expanded_space(side=4, **overrides):
    from toposc_lab.research.config import ExperimentConfig
    return FixedConnectivitySpace(**{**ExperimentConfig().space, "side": side,
                                     "max_bond_length": 2.0, "site_crossings": "unconnected",
                                     "initialization_rewires": (5, 10, 20, 30, 40), **overrides})


def test_regular_distance_counts_replaced_edges_and_is_order_invariant():
    space = FixedConnectivitySpace(side=4)
    reference = space.reference()
    edges = set(space.edges(reference))
    once = space.build((edges - {(0, 1)}) | {(0, 5)})
    twice = space.build((edges - {(0, 1), (10, 11)}) | {(0, 5), (10, 15)})
    assert regular_edge_distance(reference) == 0
    assert regular_edge_distance(once) == pytest.approx(1 / 24)
    assert regular_edge_distance(twice) == pytest.approx(2 / 24)
    assert regular_edge_distance(replace(twice, edges=tuple(reversed(twice.edges)))) == regular_edge_distance(twice)
    labels = np.arange(reference.n_sites)[::-1]
    relabeled = replace(twice, coordinates=twice.coordinates[labels],
                        edges=tuple(GeometryEdge(int(labels[e.source]), int(labels[e.target]))
                                    for e in twice.edges))
    assert regular_edge_distance(relabeled) == regular_edge_distance(twice)


def test_long_bond_descriptors_measure_geometry_and_survive_serialization():
    # Descriptor input deliberately bypasses search validity: measures are not
    # a certificate that the embedding satisfies any chosen crossing policy.
    space = FixedConnectivitySpace(side=4)
    geometry = space.build({(0, 1), (0, 5), (0, 2)})
    descriptors = compute_descriptors(geometry)
    lengths = np.array([1, np.sqrt(2), 2])
    assert descriptors["long_bond_fraction"] == pytest.approx(1 / 3)
    assert descriptors["mean_bond_length"] == pytest.approx(lengths.mean())
    assert descriptors["bond_length_variance"] == pytest.approx(lengths.var())
    assert descriptors["max_bond_length_observed"] == 2
    assert descriptors["mean_bond_length"] == descriptors["bond_length_mean"]
    restored = geometry_from_payload(json.loads(json.dumps(geometry_to_payload(geometry))))
    assert compute_descriptors(restored) == descriptors
    regular = compute_descriptors(space.reference())
    assert regular["long_bond_fraction"] == 0


def test_regular_distance_rejects_undefined_reference():
    geometry = FixedConnectivitySpace(side=4).reference()
    with pytest.raises(ValueError, match="unit-square"):
        regular_edge_distance(replace(geometry, coordinates=geometry.coordinates + .1))


@pytest.mark.parametrize("side", [4, 10])
def test_seeded_expansion_reaches_valid_long_edges_and_distances(side):
    space = expanded_space(side)
    samples = [space.sample(np.random.default_rng(seed)) for seed in range(10)]
    for geometry in samples:
        assert not space.validate(geometry)
        assert space.valid_edges(set(space.edges(geometry)))
        assert geometry.n_edges == 2 * side * (side - 1)
    descriptors = [compute_descriptors(geometry) for geometry in samples]
    assert max(d["regular_edge_distance"] for d in descriptors) > .15
    assert sum(d["long_bond_fraction"] > 0 for d in descriptors) >= 8


def test_crossing_convention_allows_only_unconnected_site_contacts():
    space = expanded_space()
    conflicts = space.crossing_conflicts
    assert (1, 5) not in conflicts[(0, 2)]  # midpoint incident bond
    assert (1, 9) not in conflicts[(4, 6)]  # two long bonds cross at site 5
    assert (0, 1) in conflicts[(0, 2)]  # positive-length overlap, even adjacent
    assert (1, 2) in conflicts[(0, 2)]
    assert (1, 4) in conflicts[(0, 5)]  # diagonals cross away from a site
    assert (2, 6) not in conflicts[(0, 2)]  # normal shared endpoint
    old = expanded_space(site_crossings="forbid")
    assert (0, 2) not in old.edge_pool
    assert (1, 4) in old.crossing_conflicts[(0, 5)]


@pytest.mark.parametrize("change,reason", [
    ("too_long", "maximum_edge_length"),
    ("overlap", "straight_edge_crossing_or_overlap"),
    ("crossing", "straight_edge_crossing_or_overlap"),
    ("extra_edge", "edge_count_above_maximum"),
    ("disconnect", "disconnected"),
    ("low_degree", "minimum_degree"),
    ("high_degree", "maximum_degree"),
])
def test_expanded_common_validation_still_rejects_constraint_violations(change, reason):
    space = expanded_space()
    edges = set(space.edges(space.reference()))
    if change == "too_long":
        edges.add((0, 3))
    elif change == "overlap":
        edges.add((0, 2))
    elif change == "crossing":
        edges.update({(0, 5), (1, 4)})
    elif change == "extra_edge":
        edges.add((0, 5))
    elif change == "disconnect":
        edges = {e for e in edges if 0 not in e}
    elif change == "low_degree":
        edges.remove((0, 1))
    elif change == "high_degree":
        edges.update({(0, 5), (2, 5), (5, 8), (5, 10)})
    assert any(reason in code for code in space.validate(space.build(edges)))


@pytest.mark.parametrize("operator", ["multi_rewire_local", "multi_rewire_explore"])
def test_composed_mutations_are_valid_and_repeatable(operator):
    space = expanded_space()
    parent = space.reference()
    first, metadata = space.mutate(parent, np.random.default_rng(17), operator)
    repeated, repeated_metadata = space.mutate(parent, np.random.default_rng(17), operator)
    assert not space.validate(first)
    assert space.edges(first) == space.edges(repeated)
    assert metadata == repeated_metadata
    assert metadata["added"] and len(metadata["added"]) == len(metadata["removed"])


def test_new_search_descriptors_and_reseeding_resume_through_json():
    from toposc_lab.research.strategies import create_strategy
    from toposc_lab.research.surrogate import GraphSurrogate
    kwargs = {"behavior_descriptors": ["regular_edge_distance", "long_bond_fraction"],
              "descriptor_bounds": [[0, 1], [0, 1]], "initialization_probability": .2,
              "mutation_rates": {"multi_rewire_local": .5, "multi_rewire_explore": .5}}
    space = expanded_space()
    strategy = create_strategy("surrogate_map_elites", space, 17002, **kwargs)
    records = strategy.propose(8)
    for i, record in enumerate(records):
        strategy.observe(record, {"origin": "exact", "score": i / 10,
                                  "validation_state": "EXACT_EVALUATED"})
    assert strategy.archive
    model = GraphSurrogate(ensemble_size=2, n_estimators=3).fit(strategy.history)
    assert {"regular_edge_distance", "long_bond_fraction", "mean_bond_length",
            "bond_length_variance", "max_bond_length_observed"} <= set(model.feature_names)
    restored = create_strategy("surrogate_map_elites", space, 17002, **kwargs).resume(
        json.loads(json.dumps(strategy.checkpoint(), sort_keys=True)))
    first, second = strategy.propose(12), restored.propose(12)
    for records in (first, second):
        for record in records:
            record.pop("created_at")
    assert first == second


def test_old_config_still_loads_without_new_geometry_semantics():
    from toposc_lab.research.config import ExperimentConfig
    old = ExperimentConfig.from_file("examples/research_experiment001.json")
    assert "site_crossings" not in old.space
    assert "initialization_probability" not in old.search
    assert old.space["max_bond_length"] == np.sqrt(2)
    assert old.objective == "robustness_success_fraction"
    assert FixedConnectivitySpace(**old.space).site_crossings == "forbid"


def test_experiment002_config_and_frozen_bond_amplitudes():
    from toposc_lab.models.chiral_p_wave import ChiralPWaveModel, ChiralPWaveParameters
    from toposc_lab.research.config import ExperimentConfig
    from toposc_lab.research.physics import FiniteSystemEvaluator
    config = ExperimentConfig.from_file("examples/research_experiment002_long_connectivity.json")
    config.validate_plugins()
    assert config.exact_budget == 6000
    assert config.objective == "robustness_quality_mean"
    assert len(FiniteSystemEvaluator(config.physics_protocol()).plan()) == 18
    geometry = expanded_space().sample(np.random.default_rng(0))
    long = [e for e in geometry.edges if geometry.distance(e.source, e.target) > np.sqrt(2)]
    assert long
    model = ChiralPWaveModel(geometry, ChiralPWaveParameters(
        hopping=1, pairing=1, chemical_potential=2, chirality=1))
    for edge in long:
        assert model.normal_hamiltonian()[edge.source, edge.target] == -1
        assert abs(model.pairing_matrix()[edge.source, edge.target]) == pytest.approx(1)
    evaluator = FiniteSystemEvaluator(config.physics_protocol())
    result = evaluator.evaluate(geometry, evaluator.plan()[0])
    assert result["kind"] == "exact"
    assert len(result["spectrum"]) == 32


def test_comparison_retains_width_evidence_and_excludes_predictions():
    from toposc_lab.research.reporting import structural_comparison
    baseline = {"id": "regular", "baseline": True, "origin": "exact", "score": .3,
                "raw_metrics": {"robustness_quality_mean": .3,
                                "robustness_success_fraction": 1, "quality": .4},
                "robustness": [{"width": .4, "quality_mean": .3}],
                "descriptors": {"regular_edge_distance": 0}}
    searched = {**baseline, "id": "searched", "baseline": False, "score": .4,
                "raw_metrics": {**baseline["raw_metrics"], "robustness_quality_mean": .4}}
    prediction = {**searched, "id": "predicted", "origin": "proposed", "score": 100}
    report = structural_comparison({"candidates": [baseline, searched, prediction]})
    assert report["best_searched"]["id"] == "searched"
    row = report["baselines"][0]
    assert row["observed_comparison"] == "higher observed finite-system quality"
    assert row["disorder_width_statistics"] == baseline["robustness"]
    assert row["descriptors"]["regular_edge_distance"] == 0
