"""Validity, physical references, honest budgets and deterministic geometry generation."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from toposc_lab.active_learning.benchmark import source_provenance
from toposc_lab.generative.generators import METHODS, Evidence, GeometryGenerator
from toposc_lab.generative.physics import ExactGeometryEvaluator, candidate
from toposc_lab.generative.space import GeometrySearchSpace, embedded_edges, structural_distance
from toposc_lab.geometry import Geometry, GeometryEdge


@pytest.fixture(scope="module")
def warm():
    return GeometryGenerator("random").propose((), seed=14001)


@pytest.fixture(scope="module")
def evaluator():
    return ExactGeometryEvaluator(source_provenance(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="module")
def history(warm, evaluator):
    return tuple(Evidence(*evaluator.evaluate(candidate(g), 14001 + i)) for i, g in enumerate(warm))


def test_validity_and_determinism(warm):
    space = GeometrySearchSpace()
    again = GeometryGenerator("random").propose((), seed=14001)
    assert [embedded_edges(g) for g in warm] == [embedded_edges(g) for g in again]
    assert all(not space.reasons(g) for g in warm)
    assert all(structural_distance(a, b) > 0.06 for i, a in enumerate(warm) for b in warm[i + 1 :])


def test_physical_compatibility(warm, evaluator):
    p = candidate(warm[0])
    with pytest.raises(ValueError, match="unsupported physical"):
        evaluator.evaluate(replace(p, model=replace(p.model, model_name="kitaev_chain")), 0)
    with pytest.raises(ValueError, match="chemical potential"):
        evaluator.evaluate(candidate(warm[0], 3.0), 0)
    shifted = replace(warm[0], coordinates=warm[0].coordinates + 0.1)
    assert "fixed_lattice_required" in GeometrySearchSpace().reasons(shifted)
    with pytest.raises(ValueError, match="inadmissible"):
        evaluator.evaluate(candidate(shifted), 0)


def test_symmetries_and_relabeling_are_duplicates(warm):
    original = warm[0]
    rotation = np.column_stack((5 - original.coordinates[:, 1], original.coordinates[:, 0]))
    assert structural_distance(original, replace(original, coordinates=rotation)) == 0
    permutation = np.random.default_rng(5).permutation(36)
    inverse = np.argsort(permutation)
    relabeled = Geometry(
        n_sites=36,
        coordinates=original.coordinates[permutation],
        embedding_dimension=2,
        edges=tuple(
            GeometryEdge(int(inverse[e.source]), int(inverse[e.target])) for e in original.edges
        ),
    )
    assert structural_distance(original, relabeled) == 0


def test_crossings_and_disconnection_rejected(warm):
    space = GeometrySearchSpace()
    crossed = space.build(frozenset(embedded_edges(warm[0])) | {(0, 7), (1, 6)})
    assert any("cross" in r for r in space.reasons(crossed))
    disconnected = space.build(frozenset(e for e in embedded_edges(warm[0]) if 14 not in e))
    assert any("connect" in r for r in space.reasons(disconnected))


@pytest.mark.parametrize("method", METHODS)
def test_generators_reuse_exact_history_and_exclude_duplicates(method, history):
    generator = GeometryGenerator(method)
    first = generator.propose(history, seed=14002)
    second = GeometryGenerator(method).propose(history, seed=14002)
    assert [embedded_edges(g) for g in first] == [embedded_edges(g) for g in second]
    assert len(first) == 4
    for i, g in enumerate(first):
        assert not generator.space.reasons(g)
        for old in [e.record.geometry.to_geometry() for e in history] + list(first[:i]):
            assert structural_distance(g, old) > 0.06
    assert generator.audit.attempted == len(generator.audit.events)
    assert all(p["kind"] == "predicted" for p in generator.audit.predictions)
    assert not any(
        p["is_ood"] and p["strategy"] == "exploitation" for p in generator.audit.predictions
    )


def test_attempt_cap_is_hard():
    with pytest.raises(RuntimeError, match="attempt cap"):
        GeometryGenerator("random", maximum_attempts=1).propose((), seed=0, count=4)


def test_positive_and_negative_exact_references(evaluator):
    space = GeometrySearchSpace()
    xy = space.coordinates
    grid = space.build(
        frozenset(e for e in space.edge_pool if np.linalg.norm(xy[e[0]] - xy[e[1]]) == 1)
    )
    positive, _ = evaluator.evaluate(candidate(grid, 2.0), 14001)
    negative, _ = evaluator.evaluate(candidate(grid, 8.0), 14001)
    assert positive.observables[0].values["eligible"] == 1
    assert negative.observables[0].values["eligible"] == 0
    assert negative.observables[0].values["quality"] == 0
    assert all(t.invariant_value == 0 for t in negative.topology)
    assert positive.observables[0].values["phs_mismatch"] < 1e-10


def test_near_duplicate_filter_without_relaxation(warm):
    excluded = tuple(warm)
    generator = GeometryGenerator("random")
    generator.propose((), seed=14001, excluded=excluded)
    assert generator.audit.duplicate >= 4
