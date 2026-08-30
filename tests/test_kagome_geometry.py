from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    Geometry,
    honeycomb,
    kagome,
)


def test_open_kagome_has_three_sublattices_and_expected_edges() -> None:
    geometry = kagome(3, 4)

    assert isinstance(geometry, Geometry)
    assert geometry.n_sites == 36
    assert geometry.n_edges == 59
    assert geometry.embedding_dimension == 2
    assert geometry.site_types == tuple(
        ("A", "B", "C")[site % 3] for site in range(36)
    )
    assert geometry.neighbors(15) == (16, 17, 20, 28)
    assert geometry.neighbors(16) == (3, 8, 15, 17)
    assert geometry.neighbors(17) == (12, 15, 16, 25)
    assert not {15, 16, 17} & geometry.boundary_sites
    assert all(geometry.degree(site) < 4 for site in geometry.boundary_sites)
    assert all(
        geometry.degree(site) == 4
        for site in geometry.site_indices
        if site not in geometry.boundary_sites
    )


def test_kagome_coordinates_have_requested_nearest_neighbor_distance() -> None:
    geometry = kagome(2, 3, spacing=0.4)

    assert geometry.coordinates is not None
    assert np.allclose(
        geometry.coordinates[:3],
        [
            (0.4, np.sqrt(3.0) * 0.2),
            (0.0, np.sqrt(3.0) * 0.2),
            (0.2, 0.0),
        ],
    )
    assert all(
        geometry.distance(edge.source, edge.target) == pytest.approx(0.4)
        for edge in geometry.edges
    )


@pytest.mark.parametrize(("n_x", "n_y"), ((2, 2), (3, 4)))
def test_fully_periodic_kagome_is_four_regular(n_x: int, n_y: int) -> None:
    geometry = kagome(
        n_x,
        n_y,
        boundary_x="periodic",
        boundary_y="periodic",
    )

    assert geometry.n_sites == 3 * n_x * n_y
    assert geometry.n_edges == 6 * n_x * n_y
    assert geometry.boundary_sites == frozenset()
    assert all(geometry.degree(site) == 4 for site in geometry.site_indices)


def test_periodic_kagome_has_expected_boundary_crossings_and_local_edges() -> None:
    geometry = kagome(
        3,
        4,
        spacing=0.5,
        boundary_x="periodic",
        boundary_y="periodic",
    )

    assert sum(edge.boundary_crossing for edge in geometry.edges) == 13
    assert all(
        geometry.distance(edge.source, edge.target) == pytest.approx(0.5)
        for edge in geometry.edges
    )
    assert np.allclose(geometry.displacement_between(24, 1), [0.5, 0.0])
    assert np.allclose(
        geometry.displacement_between(9, 2),
        [0.25, np.sqrt(3.0) * 0.25],
    )


def test_mixed_kagome_boundaries_form_a_cylinder() -> None:
    geometry = kagome(3, 4, boundary_x="periodic", boundary_y="open")

    assert geometry.n_edges == 66
    assert geometry.boundary_sites
    assert all(
        not edge.boundary_crossing
        or edge.metadata["lattice_direction"] in ((1, 0), (-1, 1))
        for edge in geometry.edges
    )


def test_periodic_kagome_is_honeycomb_line_graph() -> None:
    spacing = 0.75
    kagome_geometry = kagome(
        3,
        4,
        spacing=spacing,
        boundary_x="periodic",
        boundary_y="periodic",
    )
    honeycomb_geometry = honeycomb(
        3,
        4,
        spacing=2.0 * spacing / np.sqrt(3.0),
        boundary_x="periodic",
        boundary_y="periodic",
    )

    incidence: dict[int, list[int]] = {
        site: [] for site in honeycomb_geometry.site_indices
    }
    for edge_index, edge in enumerate(honeycomb_geometry.edges):
        incidence[edge.source].append(edge_index)
        incidence[edge.target].append(edge_index)
    line_graph_edges = {
        tuple(sorted(pair))
        for incident_edges in incidence.values()
        for pair in combinations(incident_edges, 2)
    }
    kagome_edges = {
        tuple(sorted((edge.source, edge.target)))
        for edge in kagome_geometry.edges
    }

    assert kagome_geometry.n_sites == honeycomb_geometry.n_edges
    assert kagome_edges == line_graph_edges

    assert honeycomb_geometry.coordinates is not None
    assert kagome_geometry.coordinates is not None
    translation = np.asarray((spacing / 2.0, spacing / np.sqrt(3.0)))
    honeycomb_bond_midpoints = np.asarray(
        [
            honeycomb_geometry.coordinates[edge.source]
            + 0.5
            * honeycomb_geometry.displacement_between(edge.source, edge.target)
            + translation
            for edge in honeycomb_geometry.edges
        ]
    )
    assert np.allclose(kagome_geometry.coordinates, honeycomb_bond_midpoints)


def test_periodic_kagome_has_two_corner_sharing_triangles_per_cell() -> None:
    geometry = kagome(
        3,
        4,
        boundary_x="periodic",
        boundary_y="periodic",
    )
    triangles = {
        tuple(sorted((first, second, third)))
        for first in geometry.site_indices
        for second, third in combinations(geometry.neighbors(first), 2)
        if geometry.has_edge(second, third)
    }

    assert len(triangles) == 24


def test_kagome_metadata_describes_canonical_unit_cell() -> None:
    geometry = kagome(2, 3, spacing=2.0)

    assert geometry.metadata["generator"] == "kagome"
    assert geometry.metadata["shape"] == (2, 3)
    assert geometry.metadata["sites_per_cell"] == 3
    assert geometry.dimension_records[0].kind == "lattice"
    assert geometry.dimension_records[0].value == 2.0
    assert geometry.metadata["construction"] == "honeycomb_line_graph"
    assert geometry.metadata["primitive_vectors"] == (
        (4.0, 0.0),
        (2.0, 2.0 * np.sqrt(3.0)),
    )


def test_kagome_generator_is_available_through_common_registry() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "kagome",
        parameters={
            "n_x": 2,
            "n_y": 3,
            "spacing": 0.5,
            "boundary_y": "periodic",
        },
    )

    assert geometry.n_sites == 18
    assert geometry.metadata["generation"] == {
        "generator_key": "kagome",
        "generator_version": 1,
        "parameters": {
            "boundary_y": "periodic",
            "n_x": 2,
            "n_y": 3,
            "spacing": 0.5,
        },
        "seed": None,
    }


@pytest.mark.parametrize(("n_x", "n_y"), ((1, 3), (3, 1), (0, 3), (3, -1)))
def test_kagome_rejects_too_small_axes(n_x: int, n_y: int) -> None:
    with pytest.raises(ValueError, match="at least two"):
        kagome(n_x, n_y)


@pytest.mark.parametrize(("n_x", "n_y"), ((2.5, 3), (3, True)))
def test_kagome_rejects_non_integer_axes(n_x: object, n_y: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        kagome(n_x, n_y)  # type: ignore[arg-type]


@pytest.mark.parametrize("spacing", (0.0, -1.0, np.inf, np.nan))
def test_kagome_rejects_invalid_spacing(spacing: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        kagome(2, 3, spacing=spacing)


@pytest.mark.parametrize("boundary", ("closed", "", "OPEN"))
def test_kagome_rejects_invalid_boundary(boundary: str) -> None:
    with pytest.raises(ValueError, match="either open or periodic"):
        kagome(3, 3, boundary_x=boundary)
    with pytest.raises(ValueError, match="either open or periodic"):
        kagome(3, 3, boundary_y=boundary)
