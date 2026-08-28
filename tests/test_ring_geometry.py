import numpy as np
import pytest

from toposc_lab.geometry import Geometry, ring


def test_ring_has_expected_cycle_graph() -> None:
    geometry = ring(5)

    assert isinstance(geometry, Geometry)
    assert geometry.n_sites == 5
    assert geometry.n_edges == 5
    assert geometry.dimension == 2
    assert [(edge.source, edge.target) for edge in geometry.edges] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
    ]
    assert geometry.neighbors(0) == (1, 4)
    assert all(geometry.degree(site) == 2 for site in geometry.site_indices)
    assert all(edge.edge_type == "nearest_neighbor" for edge in geometry.edges)


def test_ring_has_no_boundary_sites_and_marks_periodic_closing_edge() -> None:
    geometry = ring(5)

    assert geometry.boundary_sites == frozenset()
    assert [edge.boundary_crossing for edge in geometry.edges] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert geometry.metadata["boundary_condition"] == "periodic"
    assert geometry.metadata["intrinsic_dimension"] == 1


def test_ring_coordinates_lie_on_requested_radius() -> None:
    geometry = ring(4, radius=2.0)

    expected_coordinates = np.asarray(
        [
            (2.0, 0.0),
            (0.0, 2.0),
            (-2.0, 0.0),
            (0.0, -2.0),
        ]
    )
    assert np.allclose(geometry.coordinates, expected_coordinates)
    assert np.allclose(np.linalg.norm(geometry.coordinates, axis=1), 2.0)
    assert geometry.distance(0, 1) == pytest.approx(np.sqrt(8.0))
    assert geometry.distance(3, 0) == pytest.approx(np.sqrt(8.0))
    assert geometry.metadata["radius"] == 2.0


@pytest.mark.parametrize("n_sites", [2, 1, 0, -3])
def test_ring_rejects_too_few_sites(n_sites: int) -> None:
    with pytest.raises(ValueError, match="at least three"):
        ring(n_sites)


@pytest.mark.parametrize("n_sites", [3.5, True])
def test_ring_rejects_non_integer_site_counts(n_sites: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        ring(n_sites)  # type: ignore[arg-type]


@pytest.mark.parametrize("radius", [0.0, -1.0, np.inf, np.nan])
def test_ring_rejects_invalid_radius(radius: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        ring(3, radius=radius)


@pytest.mark.parametrize("radius", [True, "1.0"])
def test_ring_rejects_non_real_radius(radius: object) -> None:
    with pytest.raises(TypeError, match="must be a real number"):
        ring(3, radius=radius)  # type: ignore[arg-type]
