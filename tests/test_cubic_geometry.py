from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    Geometry,
    cubic,
)


def test_open_cubic_lattice_has_expected_sites_edges_and_coordinates() -> None:
    geometry = cubic(3, 4, 5)

    assert isinstance(geometry, Geometry)
    assert geometry.n_sites == 60
    assert geometry.n_edges == 133
    assert geometry.embedding_dimension == 3
    assert geometry.coordinates is not None
    assert geometry.coordinates.shape == (60, 3)
    assert np.array_equal(geometry.coordinates[0], [0.0, 0.0, 0.0])
    assert np.array_equal(geometry.coordinates[59], [2.0, 3.0, 4.0])
    assert geometry.neighbors(0) == (1, 5, 20)
    assert geometry.neighbors(26) == (6, 21, 25, 27, 31, 46)
    assert all(len(edge.displacement or ()) == 3 for edge in geometry.edges)


def test_open_cubic_lattice_marks_surface_sites() -> None:
    geometry = cubic(3, 4, 5)

    assert len(geometry.boundary_sites) == 54
    assert not geometry.is_boundary_site(26)
    assert geometry.is_boundary_site(0)
    assert geometry.is_boundary_site(59)
    assert all(geometry.degree(site) < 6 for site in geometry.boundary_sites)


def test_fully_periodic_cubic_lattice_is_six_regular() -> None:
    geometry = cubic(
        3,
        4,
        5,
        boundary_x="periodic",
        boundary_y="periodic",
        boundary_z="periodic",
    )

    assert geometry.n_edges == 180
    assert geometry.boundary_sites == frozenset()
    assert all(geometry.degree(site) == 6 for site in geometry.site_indices)
    assert geometry.neighbors(0) == (1, 4, 5, 15, 20, 40)
    assert sum(edge.boundary_crossing for edge in geometry.edges) == 47


def test_periodic_cubic_edges_retain_local_3d_displacements() -> None:
    geometry = cubic(
        3,
        4,
        5,
        spacing=0.5,
        boundary_x="periodic",
        boundary_y="periodic",
        boundary_z="periodic",
    )

    assert np.array_equal(geometry.displacement_between(40, 0), [0.5, 0.0, 0.0])
    assert np.array_equal(geometry.displacement_between(15, 0), [0.0, 0.5, 0.0])
    assert np.array_equal(geometry.displacement_between(4, 0), [0.0, 0.0, 0.5])
    assert all(
        geometry.distance(edge.source, edge.target) == pytest.approx(0.5)
        for edge in geometry.edges
    )


def test_mixed_cubic_boundaries_create_a_periodic_xy_slab() -> None:
    geometry = cubic(
        3,
        4,
        5,
        boundary_x="periodic",
        boundary_y="periodic",
        boundary_z="open",
    )

    assert geometry.n_edges == 168
    assert len(geometry.boundary_sites) == 24
    assert geometry.boundary_sites == frozenset(
        site
        for x in range(3)
        for y in range(4)
        for z in (0, 4)
        for site in ((x * 4 + y) * 5 + z,)
    )
    assert all(
        not edge.boundary_crossing
        or edge.metadata["lattice_direction"] in ((1, 0, 0), (0, 1, 0))
        for edge in geometry.edges
    )


def test_cubic_metadata_records_three_dimensional_basis() -> None:
    geometry = cubic(2, 3, 4, spacing=2.0)

    assert geometry.metadata["generator"] == "cubic"
    assert geometry.metadata["shape"] == (2, 3, 4)
    assert geometry.metadata["intrinsic_dimension"] == 3
    assert geometry.metadata["primitive_vectors"] == (
        (2.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (0.0, 0.0, 2.0),
    )


def test_cubic_generator_is_available_through_common_registry() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "cubic",
        parameters={
            "n_x": 3,
            "n_y": 3,
            "n_z": 4,
            "spacing": 0.75,
            "boundary_x": "periodic",
        },
    )

    assert geometry.n_sites == 36
    assert geometry.metadata["generation"] == {
        "generator_key": "cubic",
        "generator_version": 1,
        "parameters": {
            "boundary_x": "periodic",
            "n_x": 3,
            "n_y": 3,
            "n_z": 4,
            "spacing": 0.75,
        },
        "seed": None,
    }


@pytest.mark.parametrize(
    ("n_x", "n_y", "n_z"),
    ((1, 3, 4), (3, 1, 4), (3, 4, 1), (0, 3, 4), (3, -1, 4)),
)
def test_cubic_rejects_too_small_axes(n_x: int, n_y: int, n_z: int) -> None:
    with pytest.raises(ValueError, match="at least two"):
        cubic(n_x, n_y, n_z)


@pytest.mark.parametrize(
    ("n_x", "n_y", "n_z"),
    ((2.5, 3, 4), (3, True, 4), (3, 4, "5")),
)
def test_cubic_rejects_non_integer_axes(
    n_x: object,
    n_y: object,
    n_z: object,
) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        cubic(n_x, n_y, n_z)  # type: ignore[arg-type]


@pytest.mark.parametrize("spacing", (0.0, -1.0, np.inf, np.nan))
def test_cubic_rejects_invalid_spacing(spacing: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        cubic(2, 3, 4, spacing=spacing)


@pytest.mark.parametrize("boundary", ("closed", "", "OPEN"))
def test_cubic_rejects_invalid_boundary(boundary: str) -> None:
    with pytest.raises(ValueError, match="either open or periodic"):
        cubic(3, 3, 3, boundary_x=boundary)
    with pytest.raises(ValueError, match="either open or periodic"):
        cubic(3, 3, 3, boundary_y=boundary)
    with pytest.raises(ValueError, match="either open or periodic"):
        cubic(3, 3, 3, boundary_z=boundary)


@pytest.mark.parametrize(
    ("n_x", "n_y", "n_z", "boundary_name", "axis_name"),
    (
        (2, 3, 3, "boundary_x", "n_x"),
        (3, 2, 3, "boundary_y", "n_y"),
        (3, 3, 2, "boundary_z", "n_z"),
    ),
)
def test_cubic_rejects_two_site_periodic_circumference(
    n_x: int,
    n_y: int,
    n_z: int,
    boundary_name: str,
    axis_name: str,
) -> None:
    boundary_x = "periodic" if boundary_name == "boundary_x" else "open"
    boundary_y = "periodic" if boundary_name == "boundary_y" else "open"
    boundary_z = "periodic" if boundary_name == "boundary_z" else "open"
    with pytest.raises(ValueError, match=rf"{axis_name} must be at least three"):
        cubic(
            n_x,
            n_y,
            n_z,
            boundary_x=boundary_x,
            boundary_y=boundary_y,
            boundary_z=boundary_z,
        )
