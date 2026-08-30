from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    Geometry,
    k_nearest_neighbor_graph,
)


def _edge_records(
    geometry: Geometry,
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    return tuple(
        (
            edge.source,
            edge.target,
            tuple(edge.metadata["selected_by"]),
        )
        for edge in geometry.edges
    )


def test_union_symmetrization_has_exact_reference_edges_and_provenance() -> None:
    geometry = k_nearest_neighbor_graph([(0.0,), (1.0,), (3.0,)], 1)

    assert _edge_records(geometry) == (
        (0, 1, (0, 1)),
        (1, 2, (2,)),
    )
    assert all(edge.edge_type == "k_nearest_neighbor" for edge in geometry.edges)
    assert geometry.metadata["symmetrization"] == "union"
    assert geometry.metadata["n_mutual_edges"] == 1
    assert geometry.metadata["n_one_sided_edges"] == 1


def test_mutual_symmetrization_retains_only_reciprocal_selection() -> None:
    geometry = k_nearest_neighbor_graph(
        [(0.0,), (1.0,), (3.0,)],
        1,
        symmetrization="mutual",
    )

    assert _edge_records(geometry) == ((0, 1, (0, 1)),)
    assert geometry.metadata["symmetrization"] == "mutual"
    assert geometry.metadata["n_mutual_edges"] == 1
    assert geometry.metadata["n_one_sided_edges"] == 0


def test_equal_distance_ties_use_ascending_site_index() -> None:
    geometry = k_nearest_neighbor_graph(
        [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0)],
        2,
    )
    selected_by_zero = {
        edge.target
        for edge in geometry.edges
        if edge.source == 0 and 0 in edge.metadata["selected_by"]
    }

    assert selected_by_zero == {1, 2}
    edge_to_third_tie = geometry.edge_between(0, 3)
    assert edge_to_third_tie is not None
    assert edge_to_third_tie.metadata["selected_by"] == (3,)
    assert geometry.metadata["tie_break"] == "ascending_site_index"


def test_tie_radius_does_not_admit_the_next_larger_float() -> None:
    just_outside = float(np.nextafter(1.0, np.inf))
    geometry = k_nearest_neighbor_graph(
        [(0.0,), (1.0,), (just_outside,)],
        1,
    )
    edge = geometry.edge_between(0, 1)

    assert edge is not None
    assert edge.metadata["selected_by"] == (0,)
    edge_to_outside = geometry.edge_between(0, 2)
    assert edge_to_outside is None
    close_pair = geometry.edge_between(1, 2)
    assert close_pair is not None
    assert close_pair.metadata["selected_by"] == (1, 2)


def test_coincident_sites_follow_zero_distance_index_ties() -> None:
    geometry = k_nearest_neighbor_graph(np.zeros((4, 2)), 1)

    assert _edge_records(geometry) == (
        (0, 1, (0, 1)),
        (0, 2, (2,)),
        (0, 3, (3,)),
    )
    assert all(geometry.distance(edge.source, edge.target) == 0.0 for edge in geometry.edges)
    assert geometry.metadata["coincident_site_policy"] == (
        "rank_distinct_indices_at_zero_distance"
    )


@pytest.mark.parametrize("dimension", (1, 2, 3, 7))
def test_builder_supports_arbitrary_embedding_dimension(dimension: int) -> None:
    coordinates = np.vstack(
        (
            np.zeros(dimension),
            np.ones(dimension),
            3.0 * np.ones(dimension),
        )
    )
    geometry = k_nearest_neighbor_graph(coordinates, 1)

    assert geometry.embedding_dimension == dimension
    assert geometry.coordinates is not None
    assert geometry.coordinates.shape == (3, dimension)
    assert geometry.edges[0].displacement == pytest.approx(tuple(np.ones(dimension)))


def test_k_equal_to_all_other_sites_builds_complete_mutual_graph() -> None:
    geometry = k_nearest_neighbor_graph(
        np.eye(5),
        4,
        symmetrization="mutual",
    )

    assert geometry.n_edges == 10
    assert tuple((edge.source, edge.target) for edge in geometry.edges) == tuple(
        (source, target) for source in range(4) for target in range(source + 1, 5)
    )
    assert all(edge.metadata["mutual"] is True for edge in geometry.edges)


def test_edge_displacement_uses_canonical_lower_to_higher_site_order() -> None:
    geometry = k_nearest_neighbor_graph(
        [(2.0, 3.0), (-1.0, 7.0), (20.0, 20.0)],
        1,
    )
    edge = geometry.edge_between(0, 1)

    assert edge is not None
    assert (edge.source, edge.target) == (0, 1)
    assert edge.displacement == (-3.0, 4.0)
    assert geometry.distance(0, 1) == pytest.approx(5.0)


def test_generic_point_cloud_does_not_invent_geometry_semantics() -> None:
    geometry = k_nearest_neighbor_graph([(0.0, 0.0), (1.0, 0.0)], 1)

    assert geometry.boundary_sites == frozenset()
    assert geometry.boundary_components == ()
    assert geometry.site_types is None
    assert geometry.dimension_records == ()
    assert geometry.faces == ()
    assert geometry.rooted_tree is None
    assert geometry.metadata["boundary_inference"] == "not_performed"


def test_builder_is_deterministic_and_defensively_copies_coordinates() -> None:
    coordinates = np.asarray(
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (3.0, 3.0)]
    )
    first = k_nearest_neighbor_graph(coordinates, 2)
    second = k_nearest_neighbor_graph(coordinates, 2)
    coordinates[0] = (99.0, 99.0)

    assert first.edges == second.edges
    assert first.coordinates is not None
    assert second.coordinates is not None
    assert np.array_equal(first.coordinates, second.coordinates)
    assert np.array_equal(first.coordinates[0], (0.0, 0.0))
    assert first.metadata == second.metadata


def test_builder_is_available_through_common_registry() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "k_nearest_neighbor_graph",
        parameters={
            "coordinates": [[0.0], [1.0], [3.0]],
            "k": 1,
            "symmetrization": "mutual",
        },
    )

    assert geometry.n_edges == 1
    assert geometry.metadata["generation"] == {
        "generator_key": "k_nearest_neighbor_graph",
        "generator_version": 1,
        "parameters": {
            "coordinates": ((0.0,), (1.0,), (3.0,)),
            "k": 1,
            "symmetrization": "mutual",
        },
        "seed": None,
    }


def test_metadata_records_rule_and_no_model_parameters() -> None:
    geometry = k_nearest_neighbor_graph([(0.0,), (1.0,), (3.0,)], 1)

    assert geometry.metadata["generator"] == "k_nearest_neighbor_graph"
    assert geometry.metadata["construction"] == "euclidean_k_nearest_neighbors"
    assert geometry.metadata["distance_metric"] == "euclidean_l2"
    assert geometry.metadata["k"] == 1
    assert geometry.metadata["n_directed_neighbor_selections"] == 3
    assert geometry.metadata["edge_selection_provenance"] == (
        "edge_metadata_selected_by"
    )
    assert not {"hopping", "onsite", "pairing"} & geometry.metadata.keys()


def test_neighbor_selection_budget_accepts_exact_work_and_rejects_next() -> None:
    coordinates = np.arange(5.0).reshape(-1, 1)

    geometry = k_nearest_neighbor_graph(
        coordinates,
        2,
        max_neighbor_selections=10,
    )
    assert geometry.metadata["n_directed_neighbor_selections"] == 10
    with pytest.raises(
        ValueError,
        match=(
            r"requires 10 directed neighbor selections, exceeding "
            r"max_neighbor_selections=9"
        ),
    ):
        k_nearest_neighbor_graph(
            coordinates,
            2,
            max_neighbor_selections=9,
        )


def test_edge_budget_accepts_exact_graph_and_rejects_next() -> None:
    coordinates = [(0.0,), (1.0,), (3.0,)]

    assert k_nearest_neighbor_graph(coordinates, 1, max_edges=2).n_edges == 2
    with pytest.raises(ValueError, match=r"requires 2 edges, exceeding max_edges=1"):
        k_nearest_neighbor_graph(coordinates, 1, max_edges=1)


def test_resource_budgets_can_be_explicitly_disabled() -> None:
    geometry = k_nearest_neighbor_graph(
        np.eye(5),
        4,
        max_neighbor_selections=None,
        max_edges=None,
    )

    assert geometry.n_edges == 10
    assert geometry.metadata["max_neighbor_selections"] is None
    assert geometry.metadata["max_edges"] is None


@pytest.mark.parametrize(
    "coordinates",
    (
        [0.0, 1.0],
        np.zeros((0, 2)),
        [[0.0], [np.nan]],
        [[0.0], [1.0, 2.0]],
    ),
)
def test_builder_reuses_strict_point_cloud_validation(coordinates: object) -> None:
    with pytest.raises(ValueError):
        k_nearest_neighbor_graph(coordinates, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("k", (0, -1, -10))
def test_builder_rejects_nonpositive_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        k_nearest_neighbor_graph([(0.0,), (1.0,)], k)


@pytest.mark.parametrize("k", (True, 1.5, "1"))
def test_builder_rejects_noninteger_k(k: object) -> None:
    with pytest.raises(TypeError, match="k must be an integer"):
        k_nearest_neighbor_graph([(0.0,), (1.0,)], k)  # type: ignore[arg-type]


@pytest.mark.parametrize("n_sites", (1, 2, 5))
def test_builder_rejects_k_not_smaller_than_site_count(n_sites: int) -> None:
    with pytest.raises(ValueError, match="smaller than the number of sites"):
        k_nearest_neighbor_graph(np.zeros((n_sites, 1)), n_sites)


def test_builder_rejects_unknown_symmetrization() -> None:
    with pytest.raises(ValueError, match="either 'union' or 'mutual'"):
        k_nearest_neighbor_graph(
            [(0.0,), (1.0,)],
            1,
            symmetrization="directed",  # type: ignore[arg-type]
        )


def test_builder_rejects_nonstring_symmetrization() -> None:
    with pytest.raises(TypeError, match="symmetrization must be a string"):
        k_nearest_neighbor_graph(
            [(0.0,), (1.0,)],
            1,
            symmetrization=True,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("name", ("max_neighbor_selections", "max_edges"))
@pytest.mark.parametrize("value", (0, -1, -10))
def test_builder_rejects_nonpositive_budgets(name: str, value: int) -> None:
    keyword_arguments: dict[str, Any] = {name: value}
    with pytest.raises(ValueError, match=rf"{name} must be positive"):
        k_nearest_neighbor_graph(
            [(0.0,), (1.0,)],
            1,
            **keyword_arguments,
        )


@pytest.mark.parametrize("name", ("max_neighbor_selections", "max_edges"))
@pytest.mark.parametrize("value", (True, 1.5, "10"))
def test_builder_rejects_noninteger_budgets(name: str, value: object) -> None:
    keyword_arguments: dict[str, Any] = {name: value}
    with pytest.raises(TypeError, match=rf"{name} must be an integer or None"):
        k_nearest_neighbor_graph(
            [(0.0,), (1.0,)],
            1,
            **keyword_arguments,
        )
