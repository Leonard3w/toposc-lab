from __future__ import annotations

from dataclasses import replace

from dataset_fixtures import representative_dataset_record

from toposc_lab.data.dataset_duplicates import (
    DuplicatePolicy,
    DuplicateRelation,
    assess_geometry_duplicate,
    find_duplicate_groups,
)
from toposc_lab.data.dataset_schema import GeometryRecord
from toposc_lab.geometry import Geometry, GeometryEdge, chain


def _relabel_chain() -> Geometry:
    original = chain(4)
    order = [2, 1, 3, 0]
    return Geometry(
        n_sites=4,
        edges=(
            GeometryEdge(3, 1, edge_type="nearest_neighbor"),
            GeometryEdge(1, 0, edge_type="nearest_neighbor"),
            GeometryEdge(0, 2, edge_type="nearest_neighbor"),
        ),
        coordinates=original.coordinates[order],
        embedding_dimension=1,
        boundary_sites=frozenset({2, 3}),
    )


def test_exact_and_relabeling_isomorphic_relations_are_distinguished() -> None:
    original = GeometryRecord.from_geometry(chain(4))
    exact = assess_geometry_duplicate(original, original)
    isomorphic = assess_geometry_duplicate(original, GeometryRecord.from_geometry(_relabel_chain()))

    assert exact.relation is DuplicateRelation.EXACT_SNAPSHOT
    assert isomorphic.relation is DuplicateRelation.GRAPH_ISOMORPHIC
    assert isomorphic.equivalent is True


def test_pairwise_coordinate_policy_is_rigid_motion_invariant() -> None:
    translated = _relabel_chain()
    translated = replace(translated, coordinates=translated.coordinates + 10.0)
    assessment = assess_geometry_duplicate(
        chain(4),
        translated,
        policy=DuplicatePolicy(coordinate_relation="pairwise_distance", coordinate_tolerance=1e-12),
    )

    assert assessment.relation is DuplicateRelation.COORDINATE_NEAR_DUPLICATE
    assert assessment.equivalent is True


def test_wl_collision_is_not_mistaken_for_isomorphism() -> None:
    complete_bipartite = Geometry(
        n_sites=6,
        edges=tuple(GeometryEdge(left, right) for left in range(3) for right in range(3, 6)),
    )
    triangular_prism = Geometry(
        n_sites=6,
        edges=tuple(
            GeometryEdge(*edge)
            for edge in ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5))
        ),
    )
    assessment = assess_geometry_duplicate(complete_bipartite, triangular_prism)

    assert assessment.relation is DuplicateRelation.DISTINCT
    assert assessment.equivalent is False
    assert "isomorphism was disproved" in assessment.reason


def test_search_limit_preserves_uncertainty() -> None:
    first = Geometry(n_sites=6, edges=tuple(GeometryEdge(index, (index + 1) % 6) for index in range(6)))
    second = Geometry(
        n_sites=6,
        edges=tuple(GeometryEdge((index + 1) % 6, index) for index in range(6)),
    )
    assessment = assess_geometry_duplicate(
        first,
        second,
        policy=DuplicatePolicy(max_backtracking_states=1),
    )

    assert assessment.relation is DuplicateRelation.POSSIBLE_FINGERPRINT_COLLISION
    assert assessment.equivalent is None


def test_duplicate_groups_are_deterministic_and_keep_different_records() -> None:
    first = representative_dataset_record(seed=1, family_label=None)
    second = representative_dataset_record(geometry=_relabel_chain(), seed=2, family_label=None)
    third = representative_dataset_record(geometry=chain(5), seed=3, family_label=None)

    result = find_duplicate_groups((third, second, first))

    assert len(result.groups) == 2
    assert tuple(sorted((first.record_id, second.record_id))) in result.groups
    assert result.uncertain_pairs == ()
