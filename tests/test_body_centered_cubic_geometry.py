from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    Geometry,
    body_centered_cubic,
)


def test_open_bcc_has_expected_sites_edges_and_sublattices() -> None:
    geometry = body_centered_cubic(3, 4, 5)

    assert isinstance(geometry, Geometry)
    assert geometry.n_sites == 120
    assert geometry.n_edges == 315
    assert geometry.embedding_dimension == 3
    assert geometry.coordinates is not None
    assert geometry.coordinates.shape == (120, 3)
    assert geometry.site_types == tuple(
        "A" if site % 2 == 0 else "B" for site in range(120)
    )
    assert all(edge.source % 2 == 0 for edge in geometry.edges)
    assert all(edge.target % 2 == 1 for edge in geometry.edges)
    assert geometry.neighbors(52) == (1, 3, 11, 13, 41, 43, 51, 53)


def test_open_bcc_marks_undercoordinated_surface_sites() -> None:
    geometry = body_centered_cubic(3, 4, 5)

    assert len(geometry.boundary_sites) == 72
    assert not geometry.is_boundary_site(52)
    assert not geometry.is_boundary_site(53)
    assert all(geometry.degree(site) < 8 for site in geometry.boundary_sites)
    assert all(
        geometry.degree(site) == 8
        for site in geometry.site_indices
        if site not in geometry.boundary_sites
    )


@pytest.mark.parametrize(("n_x", "n_y", "n_z"), ((2, 2, 2), (3, 4, 5)))
def test_fully_periodic_bcc_is_eight_regular(
    n_x: int,
    n_y: int,
    n_z: int,
) -> None:
    geometry = body_centered_cubic(
        n_x,
        n_y,
        n_z,
        boundary_x="periodic",
        boundary_y="periodic",
        boundary_z="periodic",
    )

    n_cells = n_x * n_y * n_z
    assert geometry.n_sites == 2 * n_cells
    assert geometry.n_edges == 8 * n_cells
    assert geometry.boundary_sites == frozenset()
    assert all(geometry.degree(site) == 8 for site in geometry.site_indices)


def test_periodic_bcc_has_expected_neighbors_and_boundary_crossings() -> None:
    geometry = body_centered_cubic(
        3,
        4,
        5,
        boundary_x="periodic",
        boundary_y="periodic",
        boundary_z="periodic",
    )

    assert geometry.neighbors(0) == (1, 9, 31, 39, 81, 89, 111, 119)
    assert sum(edge.boundary_crossing for edge in geometry.edges) == 165


def test_bcc_spacing_is_body_diagonal_nearest_neighbor_distance() -> None:
    spacing = 0.6
    geometry = body_centered_cubic(
        3,
        3,
        3,
        spacing=spacing,
        boundary_x="periodic",
        boundary_y="periodic",
        boundary_z="periodic",
    )

    expected_component = spacing / np.sqrt(3.0)
    assert all(
        geometry.distance(edge.source, edge.target) == pytest.approx(spacing)
        for edge in geometry.edges
    )
    assert all(
        np.allclose(
            np.abs(geometry.displacement_between(edge.source, edge.target)),
            expected_component,
        )
        for edge in geometry.edges
    )
    assert np.allclose(
        geometry.displacement_between(0, 53),
        [-expected_component, -expected_component, -expected_component],
    )


def test_mixed_bcc_boundaries_create_a_periodic_xy_slab() -> None:
    geometry = body_centered_cubic(
        3,
        4,
        5,
        boundary_x="periodic",
        boundary_y="periodic",
        boundary_z="open",
    )
    expected_boundary = frozenset(
        site
        for x in range(3)
        for y in range(4)
        for site in (
            2 * ((x * 4 + y) * 5),
            2 * ((x * 4 + y) * 5 + 4) + 1,
        )
    )

    assert geometry.n_edges == 432
    assert geometry.boundary_sites == expected_boundary
    assert len(geometry.boundary_sites) == 24
    assert sum(edge.boundary_crossing for edge in geometry.edges) == 117
    assert all(
        not (
            (edge.source // 2) % 5 == 0
            and edge.metadata["lattice_direction"][2] == -1
        )
        for edge in geometry.edges
    )


def test_bcc_metadata_records_conventional_cell_and_roles() -> None:
    geometry = body_centered_cubic(2, 3, 4, spacing=1.5)
    expected_cell_length = 3.0 / np.sqrt(3.0)

    assert geometry.metadata["generator"] == "body_centered_cubic"
    assert geometry.metadata["lattice_type"] == "bcc"
    assert geometry.metadata["shape"] == (2, 3, 4)
    assert geometry.metadata["sites_per_cell"] == 2
    assert geometry.metadata["intrinsic_dimension"] == 3
    assert geometry.metadata["conventional_cell_length"] == pytest.approx(
        expected_cell_length
    )
    assert geometry.metadata["sublattice_roles"] == {
        "A": "corner",
        "B": "body_center",
    }


def test_bcc_generator_is_available_through_common_registry() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "body_centered_cubic",
        parameters={
            "n_x": 2,
            "n_y": 3,
            "n_z": 4,
            "spacing": 0.75,
            "boundary_x": "periodic",
        },
    )

    assert geometry.n_sites == 48
    assert geometry.metadata["generation"] == {
        "generator_key": "body_centered_cubic",
        "generator_version": 1,
        "parameters": {
            "boundary_x": "periodic",
            "n_x": 2,
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
def test_bcc_rejects_too_small_axes(n_x: int, n_y: int, n_z: int) -> None:
    with pytest.raises(ValueError, match="at least two"):
        body_centered_cubic(n_x, n_y, n_z)


@pytest.mark.parametrize(
    ("n_x", "n_y", "n_z"),
    ((2.5, 3, 4), (3, True, 4), (3, 4, "5")),
)
def test_bcc_rejects_non_integer_axes(
    n_x: object,
    n_y: object,
    n_z: object,
) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        body_centered_cubic(n_x, n_y, n_z)  # type: ignore[arg-type]


@pytest.mark.parametrize("spacing", (0.0, -1.0, np.inf, np.nan))
def test_bcc_rejects_invalid_spacing(spacing: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        body_centered_cubic(2, 3, 4, spacing=spacing)


@pytest.mark.parametrize("boundary", ("closed", "", "OPEN"))
def test_bcc_rejects_invalid_boundary(boundary: str) -> None:
    with pytest.raises(ValueError, match="either open or periodic"):
        body_centered_cubic(2, 3, 4, boundary_x=boundary)
    with pytest.raises(ValueError, match="either open or periodic"):
        body_centered_cubic(2, 3, 4, boundary_y=boundary)
    with pytest.raises(ValueError, match="either open or periodic"):
        body_centered_cubic(2, 3, 4, boundary_z=boundary)
