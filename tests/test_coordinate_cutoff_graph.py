from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    Geometry,
    coordinate_cutoff_graph,
)


def test_cutoff_graph_has_exact_reference_edges_in_lexicographic_order() -> None:
    geometry = coordinate_cutoff_graph(
        [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, 2.0)],
        1.0,
    )

    assert isinstance(geometry, Geometry)
    assert geometry.n_sites == 4
    assert tuple((edge.source, edge.target) for edge in geometry.edges) == (
        (0, 1),
        (0, 2),
    )
    assert all(edge.edge_type == "distance_cutoff" for edge in geometry.edges)
    assert [edge.displacement for edge in geometry.edges] == [
        (1.0, 0.0),
        (0.0, 1.0),
    ]


def test_distance_exactly_at_cutoff_is_included_and_next_float_is_not() -> None:
    just_outside = float(np.nextafter(1.0, np.inf))
    geometry = coordinate_cutoff_graph(
        [(0.0,), (1.0,), (just_outside,)],
        1.0,
    )

    assert tuple((edge.source, edge.target) for edge in geometry.edges) == (
        (0, 1),
        (1, 2),
    )
    assert not geometry.has_edge(0, 2)


@pytest.mark.parametrize("dimension", (1, 2, 3, 7))
def test_builder_supports_arbitrary_positive_embedding_dimension(
    dimension: int,
) -> None:
    coordinates = np.vstack(
        (
            np.zeros(dimension),
            np.ones(dimension),
            3.0 * np.ones(dimension),
        )
    )
    geometry = coordinate_cutoff_graph(coordinates, sqrt_dimension := np.sqrt(dimension))

    assert geometry.embedding_dimension == dimension
    assert geometry.coordinates is not None
    assert geometry.coordinates.shape == (3, dimension)
    assert tuple((edge.source, edge.target) for edge in geometry.edges) == ((0, 1),)
    assert geometry.distance(0, 1) == pytest.approx(sqrt_dimension)


def test_distinct_coincident_sites_receive_zero_length_edge() -> None:
    geometry = coordinate_cutoff_graph([(0.0, 0.0), (0.0, 0.0)], 0.5)

    assert geometry.n_edges == 1
    assert geometry.edges[0].displacement == (0.0, 0.0)
    assert geometry.distance(0, 1) == 0.0
    assert geometry.metadata["coincident_site_policy"] == (
        "connect_distinct_indices"
    )


def test_single_site_and_disconnected_clouds_are_valid() -> None:
    singleton = coordinate_cutoff_graph([(2.0, -1.0, 4.0)], 1.0)
    disconnected = coordinate_cutoff_graph([(0.0,), (2.0,), (4.0,)], 1.0)

    assert singleton.n_sites == 1
    assert singleton.n_edges == 0
    assert singleton.embedding_dimension == 3
    assert disconnected.n_sites == 3
    assert disconnected.n_edges == 0


def test_large_cutoff_builds_the_complete_simple_graph() -> None:
    geometry = coordinate_cutoff_graph(np.eye(5), 2.0)

    assert geometry.n_edges == 10
    assert tuple((edge.source, edge.target) for edge in geometry.edges) == tuple(
        (source, target) for source in range(4) for target in range(source + 1, 5)
    )


def test_generic_point_cloud_does_not_invent_geometry_semantics() -> None:
    geometry = coordinate_cutoff_graph([(0.0, 0.0), (1.0, 0.0)], 1.0)

    assert geometry.boundary_sites == frozenset()
    assert geometry.boundary_components == ()
    assert geometry.site_types is None
    assert geometry.dimension_records == ()
    assert geometry.faces == ()
    assert geometry.rooted_tree is None
    assert geometry.metadata["boundary_inference"] == "not_performed"


def test_builder_is_deterministic_and_defensively_copies_coordinates() -> None:
    coordinates = np.asarray(
        [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (2.0, 2.0)]
    )
    first = coordinate_cutoff_graph(coordinates, 0.75)
    second = coordinate_cutoff_graph(coordinates, 0.75)
    coordinates[0] = (99.0, 99.0)

    assert first.edges == second.edges
    assert first.coordinates is not None
    assert second.coordinates is not None
    assert np.array_equal(first.coordinates, second.coordinates)
    assert np.array_equal(first.coordinates[0], (0.0, 0.0))
    assert first.metadata == second.metadata


def test_builder_is_available_through_common_registry() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "coordinate_cutoff_graph",
        parameters={
            "coordinates": [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]],
            "cutoff": 1.0,
        },
    )

    assert geometry.n_edges == 1
    assert geometry.metadata["generation"] == {
        "generator_key": "coordinate_cutoff_graph",
        "generator_version": 1,
        "parameters": {
            "coordinates": ((0.0, 0.0), (1.0, 0.0), (3.0, 0.0)),
            "cutoff": 1.0,
        },
        "seed": None,
    }


def test_metadata_records_rule_and_no_model_parameters() -> None:
    geometry = coordinate_cutoff_graph([(0.0,), (1.0,)], 1.0)

    assert geometry.metadata["generator"] == "coordinate_cutoff_graph"
    assert geometry.metadata["construction"] == "euclidean_coordinate_cutoff"
    assert geometry.metadata["distance_metric"] == "euclidean_l2"
    assert geometry.metadata["distance_relation"] == "less_than_or_equal"
    assert geometry.metadata["cutoff"] == 1.0
    assert geometry.metadata["embedding_dimension"] == 1
    assert not {"hopping", "onsite", "pairing"} & geometry.metadata.keys()


def test_edge_budget_accepts_exact_count_and_rejects_next_edge() -> None:
    coordinates = np.zeros((4, 2))

    assert coordinate_cutoff_graph(coordinates, 1.0, max_edges=6).n_edges == 6
    with pytest.raises(ValueError, match=r"requires more than max_edges=5"):
        coordinate_cutoff_graph(coordinates, 1.0, max_edges=5)


def test_edge_budget_can_be_explicitly_disabled() -> None:
    geometry = coordinate_cutoff_graph(np.zeros((5, 2)), 1.0, max_edges=None)

    assert geometry.n_edges == 10
    assert geometry.metadata["max_edges"] is None


@pytest.mark.parametrize(
    "coordinates",
    (
        1.0,
        [0.0, 1.0],
        np.zeros((1, 1, 1)),
    ),
)
def test_builder_rejects_nonmatrix_coordinates(coordinates: object) -> None:
    with pytest.raises(ValueError, match="two-dimensional array"):
        coordinate_cutoff_graph(coordinates, 1.0)  # type: ignore[arg-type]


def test_builder_rejects_empty_site_axis() -> None:
    with pytest.raises(ValueError, match="at least one site"):
        coordinate_cutoff_graph(np.zeros((0, 2)), 1.0)


def test_builder_rejects_empty_embedding_axis() -> None:
    with pytest.raises(ValueError, match="at least one dimension"):
        coordinate_cutoff_graph(np.zeros((2, 0)), 1.0)


@pytest.mark.parametrize(
    "coordinates",
    (
        [[0.0], [np.nan]],
        [[0.0], [np.inf]],
        [[0.0], [-np.inf]],
    ),
)
def test_builder_rejects_nonfinite_coordinates(
    coordinates: list[list[float]],
) -> None:
    with pytest.raises(ValueError, match="only finite values"):
        coordinate_cutoff_graph(coordinates, 1.0)


@pytest.mark.parametrize(
    "coordinates",
    (
        [[0.0], [1.0, 2.0]],
        [["not-a-number"], [1.0]],
        [[0.0 + 1.0j], [1.0 + 0.0j]],
    ),
)
def test_builder_rejects_nonreal_or_ragged_coordinates(
    coordinates: object,
) -> None:
    with pytest.raises(ValueError, match="rectangular array of real numbers"):
        coordinate_cutoff_graph(coordinates, 1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("cutoff", (0.0, -1.0, np.inf, -np.inf, np.nan))
def test_builder_rejects_invalid_cutoff(cutoff: float) -> None:
    with pytest.raises(ValueError, match="cutoff must be finite and positive"):
        coordinate_cutoff_graph([(0.0,), (1.0,)], cutoff)


@pytest.mark.parametrize("cutoff", (True, "1.0"))
def test_builder_rejects_nonreal_cutoff(cutoff: object) -> None:
    with pytest.raises(TypeError, match="cutoff must be a real number"):
        coordinate_cutoff_graph([(0.0,), (1.0,)], cutoff)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_edges", (0, -1, -10))
def test_builder_rejects_nonpositive_edge_budget(max_edges: int) -> None:
    with pytest.raises(ValueError, match="max_edges must be positive"):
        coordinate_cutoff_graph([(0.0,)], 1.0, max_edges=max_edges)


@pytest.mark.parametrize("max_edges", (True, 1.5, "10"))
def test_builder_rejects_noninteger_edge_budget(max_edges: object) -> None:
    with pytest.raises(TypeError, match="max_edges must be an integer or None"):
        coordinate_cutoff_graph(
            [(0.0,)],
            1.0,
            max_edges=max_edges,  # type: ignore[arg-type]
        )
