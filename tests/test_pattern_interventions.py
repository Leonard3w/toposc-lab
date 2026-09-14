import json
from dataclasses import replace

import numpy as np
import pytest

from toposc_lab.generative.space import GeometrySearchSpace, embedded_edges
from toposc_lab.patterns.features import adjacency
from toposc_lab.patterns.interventions import (
    MOTIFS,
    deletion_candidates,
    deletion_reasons,
    matched_swaps,
    motif_counts,
    valid_swaps,
)


@pytest.fixture(scope="module")
def parent_and_swaps():
    space = GeometrySearchSpace()
    rng = np.random.default_rng(16301)
    while True:
        geometry = space.sample(rng)
        if not space.reasons(geometry):
            return geometry, valid_swaps(geometry)


def test_all_swaps_preserve_per_vertex_degrees_and_validity(parent_and_swaps):
    parent, swaps = parent_and_swaps
    assert swaps
    for swap in swaps:
        result = swap.apply(parent)
        np.testing.assert_array_equal(adjacency(result).sum(axis=1), adjacency(parent).sum(axis=1))
        assert not GeometrySearchSpace().reasons(result)
        assert len(set(embedded_edges(parent)) ^ set(embedded_edges(result))) == 4


def test_motif_counts_are_plain_json_integers_for_frozen_plans(parent_and_swaps):
    parent, _ = parent_and_swaps
    counts = motif_counts(parent)
    assert counts["central_diagonal_count"] > 0
    assert all(type(value) is int for value in counts.values())
    assert json.loads(json.dumps(counts, allow_nan=False)) == counts


def test_matching_is_deterministic_and_controls_preserve_target(parent_and_swaps):
    parent, swaps = parent_and_swaps
    base = motif_counts(parent)
    total = 0
    for motif in MOTIFS:
        pairs = matched_swaps(parent, motif, swaps, 16401)
        assert pairs == matched_swaps(parent, motif, swaps, 16401)
        for target, control in pairs:
            t, c = motif_counts(target.apply(parent)), motif_counts(control.apply(parent))
            assert t[motif] < base[motif] == c[motif]
            assert t["diagonal_count"] == c["diagonal_count"]
            if motif != "boundary_bulk_edges":
                assert t["boundary_bulk_edges"] == c["boundary_bulk_edges"]
        total += len(pairs)
    assert total > 0


def test_explicit_perturbation_seeds_preserve_first_replicate_and_do_not_reuse_edits(
    parent_and_swaps,
):
    parent, swaps = parent_and_swaps
    for motif in MOTIFS:
        first = matched_swaps(parent, motif, swaps, 0, replicate_seeds=(16411, 16412))
        second = matched_swaps(parent, motif, swaps, 0, replicate_seeds=(16411, 16413))
        assert first[:1] == second[:1]
        assert len({s for pair in first for s in pair}) == 2 * len(first)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"replicates": 0},
        {"replicates": True},
        {"replicate_seeds": (1,)},
        {"replicate_seeds": (1, 1)},
        {"replicate_seeds": (1, True)},
    ],
)
def test_invalid_perturbation_protocol_is_rejected(parent_and_swaps, kwargs):
    parent, swaps = parent_and_swaps
    with pytest.raises(ValueError):
        matched_swaps(parent, MOTIFS[0], swaps, 16401, **kwargs)


def test_swap_rejects_stale_counts_and_incompatible_parent(parent_and_swaps):
    parent, swaps = parent_and_swaps
    swap = swaps[0]
    altered = swap.apply(parent)
    with pytest.raises(ValueError, match="compatible"):
        swap.apply(altered)
    with pytest.raises(ValueError, match="counts"):
        replace(swap, counts=(0, 0, 0, 0, 0)).apply(parent)


def test_deletion_scope_is_separate_and_strict(parent_and_swaps):
    parent, _ = parent_and_swaps
    assert deletion_reasons(parent)
    candidates = deletion_candidates(parent)
    assert candidates
    g = GeometrySearchSpace().build(frozenset(embedded_edges(parent)) - {candidates[0]})
    assert not deletion_reasons(g)
    assert GeometrySearchSpace().reasons(g)


def test_deletion_rejects_moved_or_missing_coordinates_and_changed_boundary(parent_and_swaps):
    parent, _ = parent_and_swaps
    space = GeometrySearchSpace()
    edges = frozenset(embedded_edges(parent))
    deleted = space.build(edges - {deletion_candidates(parent)[0]})
    assert "fixed_coordinates_required" in deletion_reasons(
        replace(deleted, coordinates=deleted.coordinates + (0.1, 0.0))
    )
    assert "fixed_coordinates_required" in deletion_reasons(replace(deleted, coordinates=None))
    assert "fixed_boundary_required" in deletion_reasons(
        replace(deleted, boundary_sites=space.boundary - {0})
    )
    assert "fixed_edge_pool_and_perimeter_required" in deletion_reasons(
        space.build(edges - {min(space.perimeter)})
    )
