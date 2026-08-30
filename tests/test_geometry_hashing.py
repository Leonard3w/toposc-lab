from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from toposc_lab.geometry import (
    CANONICAL_GRAPH_HASH_ALGORITHM,
    Geometry,
    GeometryEdge,
    GeometryValidationError,
    canonical_graph_hash,
    geometry_from_bytes,
    geometry_to_bytes,
)


def _graph(n_sites: int, edge_pairs: Sequence[tuple[int, int]]) -> Geometry:
    return Geometry(
        n_sites=n_sites,
        edges=tuple(GeometryEdge(source, target) for source, target in edge_pairs),
    )


def _relabel_graph(geometry: Geometry, mapping: tuple[int, ...]) -> Geometry:
    assert sorted(mapping) == list(geometry.site_indices)
    site_types: list[str | None] | None = None
    if geometry.site_types is not None:
        site_types = [None] * geometry.n_sites
        for old_site, new_site in enumerate(mapping):
            site_types[new_site] = geometry.site_types[old_site]
    return Geometry(
        n_sites=geometry.n_sites,
        edges=tuple(
            GeometryEdge(
                mapping[edge.target],
                mapping[edge.source],
                edge_type=edge.edge_type,
                boundary_crossing=edge.boundary_crossing,
            )
            for edge in reversed(geometry.edges)
        ),
        boundary_sites=frozenset(mapping[site] for site in geometry.boundary_sites),
        site_types=None if site_types is None else tuple(site_types),
    )


def test_hash_is_deterministic_versioned_and_has_fixed_digest_width() -> None:
    geometry = _graph(4, ((0, 1), (1, 2), (2, 3)))
    first = canonical_graph_hash(geometry)
    second = canonical_graph_hash(geometry)
    algorithm, digest = first.split(":", maxsplit=1)

    assert first == second
    assert algorithm == CANONICAL_GRAPH_HASH_ALGORITHM
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")
    assert first == (
        "wl1-v1-sha256:"
        "628321d47cf8765349001292dae3ec086c4b4277e65f63e8a2971ce4facfe273"
    )


def test_topology_hash_is_invariant_to_site_relabeling_edge_order_and_orientation() -> None:
    geometry = _graph(
        7,
        ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6)),
    )
    relabeled = _relabel_graph(geometry, (4, 1, 6, 0, 5, 2, 3))

    assert canonical_graph_hash(relabeled) == canonical_graph_hash(geometry)


def test_attributed_hash_is_invariant_to_matching_relabeling() -> None:
    geometry = Geometry(
        n_sites=5,
        edges=(
            GeometryEdge(0, 1, edge_type="short"),
            GeometryEdge(1, 2, edge_type="long", boundary_crossing=True),
            GeometryEdge(2, 3, edge_type="short"),
            GeometryEdge(2, 4, edge_type="branch"),
        ),
        boundary_sites=frozenset({0, 3, 4}),
        site_types=("A", "B", "A", None, "C"),
    )
    relabeled = _relabel_graph(geometry, (3, 0, 4, 2, 1))

    options = {
        "include_site_types": True,
        "include_edge_types": True,
        "include_boundary": True,
    }
    assert canonical_graph_hash(relabeled, **options) == canonical_graph_hash(
        geometry,
        **options,
    )


def test_default_hash_ignores_coordinates_faces_roots_and_metadata() -> None:
    edges = (
        GeometryEdge(0, 1),
        GeometryEdge(1, 2),
        GeometryEdge(2, 3),
        GeometryEdge(3, 0),
    )
    plain = Geometry(n_sites=4, edges=edges, metadata={"name": "plain"})
    embedded = Geometry(
        n_sites=4,
        edges=edges,
        coordinates=np.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))),
        metadata={"name": "different", "array": np.arange(3)},
    )

    assert canonical_graph_hash(embedded) == canonical_graph_hash(plain)


def test_site_types_only_affect_explicit_attributed_scope() -> None:
    first = Geometry(
        n_sites=3,
        edges=(GeometryEdge(0, 1), GeometryEdge(1, 2)),
        site_types=("A", "A", "A"),
    )
    second = Geometry(
        n_sites=3,
        edges=(GeometryEdge(0, 1), GeometryEdge(1, 2)),
        site_types=("A", "B", "A"),
    )

    assert canonical_graph_hash(first) == canonical_graph_hash(second)
    assert canonical_graph_hash(
        first,
        include_site_types=True,
    ) != canonical_graph_hash(second, include_site_types=True)


def test_edge_types_only_affect_explicit_attributed_scope() -> None:
    first = Geometry(
        n_sites=3,
        edges=(
            GeometryEdge(0, 1, edge_type="short"),
            GeometryEdge(1, 2, edge_type="short"),
        ),
    )
    second = Geometry(
        n_sites=3,
        edges=(
            GeometryEdge(0, 1, edge_type="short"),
            GeometryEdge(1, 2, edge_type="long"),
        ),
    )

    assert canonical_graph_hash(first) == canonical_graph_hash(second)
    assert canonical_graph_hash(
        first,
        include_edge_types=True,
    ) != canonical_graph_hash(second, include_edge_types=True)


def test_boundary_membership_and_crossing_only_affect_explicit_scope() -> None:
    first = Geometry(
        n_sites=3,
        edges=(GeometryEdge(0, 1), GeometryEdge(1, 2)),
        boundary_sites=frozenset({0, 2}),
    )
    second = Geometry(
        n_sites=3,
        edges=(GeometryEdge(0, 1, boundary_crossing=True), GeometryEdge(1, 2)),
        boundary_sites=frozenset({1}),
    )

    assert canonical_graph_hash(first) == canonical_graph_hash(second)
    assert canonical_graph_hash(
        first,
        include_boundary=True,
    ) != canonical_graph_hash(second, include_boundary=True)


def test_option_scope_is_part_of_the_hash_domain() -> None:
    geometry = _graph(2, ((0, 1),))

    assert canonical_graph_hash(geometry) != canonical_graph_hash(
        geometry,
        include_site_types=True,
    )
    assert canonical_graph_hash(geometry) != canonical_graph_hash(
        geometry,
        include_edge_types=True,
    )
    assert canonical_graph_hash(geometry) != canonical_graph_hash(
        geometry,
        include_boundary=True,
    )


def test_simple_nonisomorphic_graphs_have_different_hashes() -> None:
    path = _graph(4, ((0, 1), (1, 2), (2, 3)))
    star = _graph(4, ((0, 1), (0, 2), (0, 3)))

    assert canonical_graph_hash(path) != canonical_graph_hash(star)


def test_component_size_profile_strengthens_disconnected_fingerprint() -> None:
    cycle = _graph(6, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)))
    two_triangles = _graph(
        6,
        ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)),
    )

    assert cycle.n_edges == two_triangles.n_edges
    assert canonical_graph_hash(cycle) != canonical_graph_hash(two_triangles)


def test_known_one_wl_collision_is_not_misrepresented_as_isomorphism_proof() -> None:
    complete_bipartite = _graph(
        6,
        tuple((left, right) for left in range(3) for right in range(3, 6)),
    )
    triangular_prism = _graph(
        6,
        (
            (0, 1),
            (1, 2),
            (2, 0),
            (3, 4),
            (4, 5),
            (5, 3),
            (0, 3),
            (1, 4),
            (2, 5),
        ),
    )

    assert all(complete_bipartite.degree(site) == 3 for site in range(6))
    assert all(triangular_prism.degree(site) == 3 for site in range(6))
    assert canonical_graph_hash(complete_bipartite) == canonical_graph_hash(
        triangular_prism
    )


def test_refinement_budget_is_deterministic_and_part_of_truncated_result() -> None:
    path = _graph(20, tuple((site, site + 1) for site in range(19)))

    one_round = canonical_graph_hash(path, max_refinement_rounds=1)
    full_default = canonical_graph_hash(path)
    assert one_round == canonical_graph_hash(path, max_refinement_rounds=1)
    assert one_round != full_default


def test_round_trip_does_not_change_graph_hash() -> None:
    geometry = Geometry(
        n_sites=4,
        edges=(
            GeometryEdge(0, 1, edge_type="a"),
            GeometryEdge(1, 2, edge_type="b"),
            GeometryEdge(1, 3, edge_type="c"),
        ),
        site_types=("root", "hub", "leaf", "leaf"),
        boundary_sites=frozenset({2, 3}),
    )
    loaded = geometry_from_bytes(geometry_to_bytes(geometry))
    options = {
        "include_site_types": True,
        "include_edge_types": True,
        "include_boundary": True,
    }

    assert canonical_graph_hash(loaded, **options) == canonical_graph_hash(
        geometry,
        **options,
    )


def test_invalid_geometry_is_rejected_through_validation_pipeline() -> None:
    geometry = Geometry(n_sites=1, metadata={"bad": object()})

    with pytest.raises(GeometryValidationError, match="invalid_metadata_type"):
        canonical_graph_hash(geometry)


@pytest.mark.parametrize("geometry", (None, object(), "graph"))
def test_hash_requires_geometry_instance(geometry: object) -> None:
    with pytest.raises(TypeError, match="Geometry instance"):
        canonical_graph_hash(geometry)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "options",
    (
        {"include_site_types": 1},
        {"include_edge_types": None},
        {"include_boundary": "yes"},
    ),
)
def test_hash_requires_boolean_scope_options(options: dict[str, object]) -> None:
    with pytest.raises(TypeError, match="must be a boolean"):
        canonical_graph_hash(Geometry(n_sites=1), **options)  # type: ignore[arg-type]


@pytest.mark.parametrize("rounds", (0, -1, -10))
def test_hash_rejects_nonpositive_refinement_budget(rounds: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        canonical_graph_hash(Geometry(n_sites=1), max_refinement_rounds=rounds)


@pytest.mark.parametrize("rounds", (True, 1.5, "3"))
def test_hash_rejects_noninteger_refinement_budget(rounds: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        canonical_graph_hash(
            Geometry(n_sites=1),
            max_refinement_rounds=rounds,  # type: ignore[arg-type]
        )
