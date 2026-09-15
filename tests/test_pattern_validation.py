import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from toposc_lab.generative.space import GeometrySearchSpace, embedded_edges, structural_distance
from toposc_lab.patterns.interventions import valid_swaps
from toposc_lab.patterns.validation import (
    HYPOTHESES,
    ExclusionIndex,
    edge_mask,
    orbit_masks,
    pair_eligible,
    select_pair,
    summarize,
    swap_description,
)


def descriptor(**delta):
    return {
        "delta": {
            k: delta.get(k, 0)
            for k in (
                "triangle_count",
                "chordless_square_count",
                "boundary_bulk_edges",
                "central_diagonal_count",
                "diagonal_count",
            )
        },
        "lost_boundary_triangles": 0,
        "lost_bulk_triangles": 0,
        "gained_triangles": 0,
        "centroid": [2.0, 2.0],
        "radius": 1.0,
    }


def test_bitset_distance_equals_existing_d4_distance_including_deletions():
    space = GeometrySearchSpace()
    rng = np.random.default_rng(991)
    geometries = [space.sample(rng, patch=bool(i % 2)) for i in range(8)]
    edges = [[list(e) for e in embedded_edges(g)] for g in geometries]
    for i in range(1, len(edges)):
        index = ExclusionIndex(edges[:i])
        expected = min(structural_distance(geometries[i], g) for g in geometries[:i])
        assert index.distance(edges[i]) == expected
        deletion = space.build(frozenset(tuple(e) for e in edges[i][1:]))
        assert index.distance(edges[i][1:]) == min(
            structural_distance(deletion, g) for g in geometries[:i]
        )
        index.add(edges[i])
        assert index.distance(edges[i]) == 0
        assert len(orbit_masks(edges[i])) <= 8


@pytest.mark.parametrize(
    "edges", [[[1, 1]], [[-1, 2]], [[True, 2]], [[0, 36]], [[1, 2], [2, 1]], [[1]]]
)
def test_invalid_bitset_inputs(edges):
    with pytest.raises(ValueError):
        edge_mask(edges)


@pytest.mark.parametrize(
    "h,key",
    [
        ("h1", "boundary_bulk_edges"),
        ("h4", "chordless_square_count"),
        ("h5", "central_diagonal_count"),
    ],
)
def test_strict_reciprocal_matching_rejects_nuisance_drift(h, key):
    a, b = descriptor(**{key: -1}), descriptor(**{key: 1})
    assert pair_eligible(h, a, b)
    assert not pair_eligible(h, b, a)
    b["delta"]["triangle_count"] = 1
    assert not pair_eligible(h, a, b)


def test_h2_demands_actual_boundary_and_bulk_removal_not_net_counts():
    a, b = descriptor(triangle_count=-1), descriptor(triangle_count=-1)
    a["lost_boundary_triangles"] = b["lost_bulk_triangles"] = 1
    assert pair_eligible("h2", a, b)
    b["gained_triangles"] = 1
    assert not pair_eligible("h2", a, b)


def test_matching_spatial_cost_and_seed_are_reproducible():
    a, b, c = (
        descriptor(boundary_bulk_edges=-1),
        descriptor(boundary_bulk_edges=1),
        descriptor(boundary_bulk_edges=1),
    )
    c["centroid"] = [9, 9]
    pair = select_pair("h1", [a, b, c], 4)
    assert pair == select_pair("h1", [a, b, c], 4)
    assert pair["b"] == b
    assert pair["spatial_cost"] == 0
    assert select_pair("h1", [a], 4) is None


def effects(value, count=16):
    result = {h: [] for h in HYPOTHESES}
    result["h1"] = [
        {
            "seed": i,
            "effect": value,
            "sensitivity": {"1.9": value, "2.1": value},
            "a_minus_parent": value / 2,
            "b_minus_parent": -value / 2,
        }
        for i in range(count)
    ]
    return result


def test_support_requires_direction_sensitivity_reciprocal_effects_and_independent_units():
    data = effects(0.1)
    assert summarize(data)["h1"]["supported"]
    assert not summarize(effects(-0.1))["h1"]["supported"]
    assert not summarize(effects(0.1, 7))["h1"]["supported"]
    drift = deepcopy(data)
    for row in drift["h1"]:
        row["sensitivity"]["1.9"] = -0.1
    assert not summarize(drift)["h1"]["supported"]
    drift = deepcopy(data)
    for row in drift["h1"]:
        row["a_minus_parent"] = -0.1
    assert not summarize(drift)["h1"]["supported"]
    data["h1"].append(data["h1"][0])
    with pytest.raises(ValueError, match="independent seed"):
        summarize(data)


def test_h3_zero_parent_failures_and_no_equivalence_from_h4_null():
    data = {h: [] for h in HYPOTHESES}
    data["h3"] = [{"seed": i, "effect": 0.5} for i in range(16)]
    data["h4"] = [{"seed": i, "effect": 0.0} for i in range(16)]
    report = summarize(data)
    assert not report["h3"]["supported"]
    assert report["h3"]["holm_p"] < 0.05
    assert not report["h4"]["supported"]
    assert report["h4"]["p"] == 1
    assert report["h2"]["status"] == "insufficient_matched_parents"


def test_real_swap_descriptor_preserves_degree_and_triangle_instance_accounting():
    space = GeometrySearchSpace()
    grid = space.build(
        frozenset(
            e
            for e in space.edge_pool
            if np.linalg.norm(space.coordinates[e[0]] - space.coordinates[e[1]]) == 1
        )
    )
    swaps = valid_swaps(grid)
    assert swaps
    row = swap_description(grid, swaps[0])
    assert row["delta"]["triangle_count"] == (
        row["gained_triangles"] - row["lost_boundary_triangles"] - row["lost_bulk_triangles"]
    )
    assert len(row["edges"]) == 60


def test_driver_exact_references_resume_and_audit_without_new_solves(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "phase16b_driver", root / "scripts/phase_16b_validation.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    for name in driver.THREADS:
        monkeypatch.setenv(name, "1")
    driver.freeze(tmp_path / "exclusions.json", {"edges": []})
    provenance = driver.initialize(root, tmp_path)
    space = GeometrySearchSpace()
    grid = space.build(
        frozenset(
            e
            for e in space.edge_pool
            if np.linalg.norm(space.coordinates[e[0]] - space.coordinates[e[1]]) == 1
        )
    )
    plan = {
        "parents": [
            {
                "seed": 16201,
                "generator": "reference_fixture",
                "edges": [list(e) for e in embedded_edges(grid)],
                "pairs": {},
                "deletions": [],
            }
        ]
    }
    driver.freeze(tmp_path / "plan.json", plan)
    driver.freeze(
        tmp_path / "plan-identity.json", {"sha256": driver.digest(tmp_path / "plan.json")}
    )
    result = driver.evaluate_plan(tmp_path, plan, provenance)
    driver.freeze(tmp_path / "results.json", result)
    driver.freeze(tmp_path / "analysis.json", driver.analyze(result))
    assert result["references"]["2.0"]["indices"] == [1, 1, 1]
    assert result["references"]["8.0"]["indices"] == [0, 0, 0]

    def no_solve(*args, **kwargs):
        pytest.fail("resume/audit must not solve again")

    monkeypatch.setattr(np.linalg, "eigh", no_solve)
    monkeypatch.setattr(np.linalg, "eigvalsh", no_solve)
    assert driver.evaluate_plan(tmp_path, plan, provenance) == result
    assert driver.audit(tmp_path)["attempts"] == 8
    with pytest.raises(ValueError, match="frozen artifact mismatch"):
        driver.freeze(tmp_path / "plan.json", {"parents": []})
