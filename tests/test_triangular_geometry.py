from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    Geometry,
    triangular,
)


def test_open_triangular_lattice_has_expected_coordinates_and_edges() -> None:
    geometry = triangular(3, 4)

    assert isinstance(geometry, Geometry)
    assert geometry.n_sites == 12
    assert geometry.n_edges == 23
    assert geometry.embedding_dimension == 2
    assert geometry.coordinates is not None
    assert np.allclose(geometry.coordinates[0], [0.0, 0.0])
    assert np.allclose(geometry.coordinates[1], [0.5, np.sqrt(3.0) / 2.0])
    assert np.allclose(geometry.coordinates[4], [1.0, 0.0])
    assert geometry.neighbors(0) == (1, 4)
    assert geometry.neighbors(5) == (1, 2, 4, 6, 8, 9)
    assert all(edge.edge_type == "nearest_neighbor" for edge in geometry.edges)
    assert not any(edge.boundary_crossing for edge in geometry.edges)


def test_open_triangular_lattice_marks_its_perimeter() -> None:
    geometry = triangular(4, 4)

    assert geometry.boundary_sites == frozenset(
        {0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15}
    )
    assert not geometry.is_boundary_site(5)
    assert not geometry.is_boundary_site(10)


def test_fully_periodic_triangular_lattice_is_six_regular() -> None:
    geometry = triangular(
        3,
        4,
        boundary_x="periodic",
        boundary_y="periodic",
    )

    assert geometry.n_edges == 36
    assert geometry.boundary_sites == frozenset()
    assert all(geometry.degree(site) == 6 for site in geometry.site_indices)
    assert geometry.neighbors(0) == (1, 3, 4, 7, 8, 9)
    assert sum(edge.boundary_crossing for edge in geometry.edges) == 13


def test_periodic_edges_keep_local_triangular_displacements() -> None:
    geometry = triangular(
        3,
        4,
        spacing=0.5,
        boundary_x="periodic",
        boundary_y="periodic",
    )

    assert geometry.distance(8, 0) == pytest.approx(0.5)
    assert np.allclose(geometry.direction(8, 0), [1.0, 0.0])
    assert geometry.distance(3, 0) == pytest.approx(0.5)
    assert np.allclose(
        geometry.direction(3, 0),
        [0.5, np.sqrt(3.0) / 2.0],
    )
    assert geometry.distance(8, 3) == pytest.approx(0.5)
    assert np.allclose(
        geometry.direction(8, 3),
        [0.5, -np.sqrt(3.0) / 2.0],
    )


def test_mixed_boundary_conditions_create_a_triangular_cylinder() -> None:
    geometry = triangular(3, 4, boundary_x="periodic", boundary_y="open")

    assert geometry.boundary_sites == frozenset({0, 3, 4, 7, 8, 11})
    assert geometry.metadata["boundary_x"] == "periodic"
    assert geometry.metadata["boundary_y"] == "open"
    assert all(
        not edge.boundary_crossing
        or edge.metadata["lattice_direction"] in ((1, 0), (1, -1))
        for edge in geometry.edges
    )


def test_triangular_metadata_records_lattice_basis() -> None:
    geometry = triangular(2, 3, spacing=2.0)

    assert geometry.metadata["generator"] == "triangular"
    assert geometry.metadata["shape"] == (2, 3)
    assert geometry.dimension_records[0].kind == "lattice"
    assert geometry.dimension_records[0].value == 2.0
    assert geometry.metadata["primitive_vectors"] == (
        (2.0, 0.0),
        (1.0, np.sqrt(3.0)),
    )
    assert {
        edge.metadata["lattice_direction"] for edge in geometry.edges
    } == {(1, 0), (0, 1), (1, -1)}


def test_triangular_generator_is_available_through_common_registry() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "triangular",
        parameters={
            "n_x": 3,
            "n_y": 4,
            "spacing": 0.75,
            "boundary_x": "periodic",
        },
    )

    assert geometry.n_sites == 12
    assert geometry.metadata["generation"] == {
        "generator_key": "triangular",
        "generator_version": 1,
        "parameters": {
            "boundary_x": "periodic",
            "n_x": 3,
            "n_y": 4,
            "spacing": 0.75,
        },
        "seed": None,
    }


@pytest.mark.parametrize(("n_x", "n_y"), ((1, 3), (3, 1), (0, 3), (3, -1)))
def test_triangular_rejects_too_small_axes(n_x: int, n_y: int) -> None:
    with pytest.raises(ValueError, match="at least two"):
        triangular(n_x, n_y)


@pytest.mark.parametrize(("n_x", "n_y"), ((2.5, 3), (3, True)))
def test_triangular_rejects_non_integer_axes(n_x: object, n_y: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        triangular(n_x, n_y)  # type: ignore[arg-type]


@pytest.mark.parametrize("spacing", (0.0, -1.0, np.inf, np.nan))
def test_triangular_rejects_invalid_spacing(spacing: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        triangular(2, 3, spacing=spacing)


@pytest.mark.parametrize("spacing", (True, "1.0"))
def test_triangular_rejects_non_real_spacing(spacing: object) -> None:
    with pytest.raises(TypeError, match="must be a real number"):
        triangular(2, 3, spacing=spacing)  # type: ignore[arg-type]


@pytest.mark.parametrize("boundary", ("closed", "", "OPEN"))
def test_triangular_rejects_invalid_boundary(boundary: str) -> None:
    with pytest.raises(ValueError, match="either open or periodic"):
        triangular(3, 3, boundary_x=boundary)
    with pytest.raises(ValueError, match="either open or periodic"):
        triangular(3, 3, boundary_y=boundary)


@pytest.mark.parametrize(
    ("n_x", "n_y", "boundary_x", "boundary_y", "axis_name"),
    (
        (2, 3, "periodic", "open", "n_x"),
        (3, 2, "open", "periodic", "n_y"),
    ),
)
def test_triangular_rejects_two_site_periodic_circumference(
    n_x: int,
    n_y: int,
    boundary_x: str,
    boundary_y: str,
    axis_name: str,
) -> None:
    with pytest.raises(ValueError, match=rf"{axis_name} must be at least three"):
        triangular(
            n_x,
            n_y,
            boundary_x=boundary_x,
            boundary_y=boundary_y,
        )
