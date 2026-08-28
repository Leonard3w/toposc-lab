import numpy as np
import pytest

from toposc_lab.geometry import Geometry, square


def test_open_square_has_expected_sites_coordinates_and_edges() -> None:
    geometry = square(3, 4)

    assert isinstance(geometry, Geometry)
    assert geometry.n_sites == 12
    assert geometry.n_edges == 17
    assert geometry.dimension == 2
    assert np.array_equal(geometry.coordinates[0], [0.0, 0.0])
    assert np.array_equal(geometry.coordinates[11], [2.0, 3.0])
    assert geometry.neighbors(0) == (1, 4)
    assert geometry.neighbors(5) == (1, 4, 6, 9)
    assert geometry.metadata["shape"] == (3, 4)
    assert geometry.metadata["intrinsic_dimension"] == 2


def test_open_square_marks_its_perimeter_as_boundary() -> None:
    geometry = square(3, 4)

    assert geometry.boundary_sites == frozenset({0, 1, 2, 3, 4, 7, 8, 9, 10, 11})
    assert not geometry.is_boundary_site(5)
    assert not geometry.is_boundary_site(6)
    assert not any(edge.boundary_crossing for edge in geometry.edges)


def test_fully_periodic_square_has_four_neighbors_and_no_boundary() -> None:
    geometry = square(
        3,
        4,
        boundary_x="periodic",
        boundary_y="periodic",
    )

    assert geometry.n_edges == 24
    assert geometry.boundary_sites == frozenset()
    assert all(geometry.degree(site) == 4 for site in geometry.site_indices)
    assert geometry.neighbors(0) == (1, 3, 4, 8)
    assert sum(edge.boundary_crossing for edge in geometry.edges) == 7


def test_periodic_edges_retain_local_distance_and_direction() -> None:
    geometry = square(
        3,
        4,
        spacing=0.5,
        boundary_x="periodic",
        boundary_y="periodic",
    )

    assert geometry.distance(8, 0) == pytest.approx(0.5)
    assert np.array_equal(geometry.direction(8, 0), [1.0, 0.0])
    assert geometry.distance(3, 0) == pytest.approx(0.5)
    assert np.array_equal(geometry.direction(3, 0), [0.0, 1.0])


def test_mixed_boundary_conditions_create_a_cylinder() -> None:
    geometry = square(3, 4, boundary_x="periodic", boundary_y="open")

    assert geometry.boundary_sites == frozenset({0, 3, 4, 7, 8, 11})
    assert geometry.metadata["boundary_x"] == "periodic"
    assert geometry.metadata["boundary_y"] == "open"


@pytest.mark.parametrize(("n_x", "n_y"), [(1, 3), (3, 1), (0, 3), (3, -1)])
def test_square_rejects_too_small_axes(n_x: int, n_y: int) -> None:
    with pytest.raises(ValueError, match="at least two"):
        square(n_x, n_y)


@pytest.mark.parametrize(("n_x", "n_y"), [(2.5, 3), (3, True)])
def test_square_rejects_non_integer_axes(n_x: object, n_y: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        square(n_x, n_y)  # type: ignore[arg-type]


@pytest.mark.parametrize("spacing", [0.0, -1.0, np.inf, np.nan])
def test_square_rejects_invalid_spacing(spacing: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        square(2, 2, spacing=spacing)


@pytest.mark.parametrize("boundary", ["closed", "", "OPEN"])
def test_square_rejects_invalid_boundary(boundary: str) -> None:
    with pytest.raises(ValueError, match="either open or periodic"):
        square(3, 3, boundary_x=boundary)

    with pytest.raises(ValueError, match="either open or periodic"):
        square(3, 3, boundary_y=boundary)
