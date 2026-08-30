from __future__ import annotations

from collections.abc import Callable
from math import isfinite, sqrt

import numpy as np
import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    Geometry,
    body_centered_cubic,
    canonical_graph_hash,
    cubic,
    geometry_from_bytes,
    geometry_to_bytes,
    menger_sponge,
    validate_geometry,
)


GeometryFactory = Callable[[], Geometry]


def _assert_cross_layer_round_trip(geometry: Geometry) -> Geometry:
    report = validate_geometry(geometry, require_connected=True)
    assert report.is_valid, report.errors
    assert report.is_connected

    topology_hash = canonical_graph_hash(geometry)
    attributed_hash = canonical_graph_hash(
        geometry,
        include_site_types=True,
        include_edge_types=True,
        include_boundary=True,
    )
    loaded = geometry_from_bytes(geometry_to_bytes(geometry))

    assert loaded.n_sites == geometry.n_sites
    assert loaded.edges == geometry.edges
    assert loaded.embedding_dimension == geometry.embedding_dimension
    assert loaded.boundary_sites == geometry.boundary_sites
    assert loaded.boundary_components == geometry.boundary_components
    assert loaded.site_types == geometry.site_types
    assert loaded.dimension_records == geometry.dimension_records
    assert loaded.faces == geometry.faces
    assert loaded.rooted_tree == geometry.rooted_tree
    assert loaded.metadata == geometry.metadata
    assert loaded.coordinates is not None
    assert geometry.coordinates is not None
    assert loaded.coordinates.dtype == geometry.coordinates.dtype
    assert loaded.coordinates.shape == geometry.coordinates.shape
    assert loaded.coordinates.tobytes() == geometry.coordinates.tobytes()

    loaded_report = validate_geometry(loaded, require_connected=True)
    assert loaded_report.is_valid, loaded_report.errors
    assert loaded_report.connected_components == report.connected_components
    assert canonical_graph_hash(loaded) == topology_hash
    assert (
        canonical_graph_hash(
            loaded,
            include_site_types=True,
            include_edge_types=True,
            include_boundary=True,
        )
        == attributed_hash
    )
    return loaded


@pytest.mark.parametrize(
    ("generator_name", "factory", "dimension_kinds"),
    (
        (
            "cubic",
            lambda: cubic(
                3,
                4,
                5,
                boundary_x="periodic",
                boundary_y="periodic",
                boundary_z="periodic",
            ),
            ("lattice",),
        ),
        (
            "body_centered_cubic",
            lambda: body_centered_cubic(
                2,
                3,
                4,
                boundary_x="periodic",
                boundary_y="periodic",
                boundary_z="periodic",
            ),
            ("lattice",),
        ),
        (
            "menger_sponge",
            lambda: menger_sponge(1),
            ("topological", "hausdorff"),
        ),
    ),
)
def test_three_dimensional_generators_survive_all_geometry_layers(
    generator_name: str,
    factory: GeometryFactory,
    dimension_kinds: tuple[str, ...],
) -> None:
    geometry = factory()

    assert geometry.metadata["generator"] == generator_name
    assert geometry.embedding_dimension == 3
    assert geometry.coordinates is not None
    assert geometry.coordinates.shape == (geometry.n_sites, 3)
    assert all(len(edge.displacement or ()) == 3 for edge in geometry.edges)
    assert tuple(record.kind for record in geometry.dimension_records) == dimension_kinds

    _assert_cross_layer_round_trip(geometry)


def test_menger_embedding_dimension_is_not_its_fractal_dimension() -> None:
    geometry = menger_sponge(2)
    dimensions = {record.kind: record.value for record in geometry.dimension_records}

    assert geometry.embedding_dimension == 3
    assert dimensions == {
        "topological": 1.0,
        "hausdorff": pytest.approx(np.log(20.0) / np.log(3.0)),
    }
    assert all(value != geometry.embedding_dimension for value in dimensions.values())


def _axis_rules(dimension: int) -> list[list[int]]:
    return [
        [sign if component == axis else 0 for component in range(dimension)]
        for axis in range(dimension)
        for sign in (-1, 1)
    ]


@pytest.mark.parametrize(
    ("generator_key", "dimension", "parameters", "expected_counts"),
    (
        (
            "artificial_rule_graph",
            4,
            {
                "iterations": 2,
                "displacement_rules": _axis_rules(4),
            },
            (41, 64),
        ),
        (
            "k_nearest_neighbor_graph",
            7,
            {
                "coordinates": [
                    [0.0] * 7,
                    [1.0] * 7,
                    [3.0] * 7,
                ],
                "k": 1,
            },
            (3, 2),
        ),
        (
            "coordinate_cutoff_graph",
            11,
            {
                "coordinates": [
                    [0.0] * 11,
                    [1.0] * 11,
                    [2.0] * 11,
                ],
                "cutoff": sqrt(11.0),
            },
            (3, 2),
        ),
    ),
)
def test_public_generators_support_cross_layer_geometries_above_3d(
    generator_key: str,
    dimension: int,
    parameters: dict[str, object],
    expected_counts: tuple[int, int],
) -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        generator_key,
        parameters=parameters,
    )

    assert (geometry.n_sites, geometry.n_edges) == expected_counts
    assert geometry.embedding_dimension == dimension
    assert geometry.coordinates is not None
    assert geometry.coordinates.shape == (geometry.n_sites, dimension)
    assert all(len(edge.displacement or ()) == dimension for edge in geometry.edges)
    assert all(
        isfinite(geometry.distance(edge.source, edge.target))
        for edge in geometry.edges
    )
    assert geometry.metadata["embedding_dimension"] == dimension
    assert geometry.metadata["generation"]["generator_key"] == generator_key

    # An embedding dimension alone is not an intrinsic or physical dimension.
    assert geometry.dimension_records == ()

    loaded = _assert_cross_layer_round_trip(geometry)
    assert loaded.embedding_dimension == dimension
    assert loaded.dimension_records == ()
