import numpy as np
import pytest

from toposc_lab.geometry import Geometry, chain


def test_chain_has_expected_sites_edges_and_coordinates() -> None:
    geometry = chain(5)

    assert isinstance(geometry, Geometry)
    assert geometry.n_sites == 5
    assert geometry.n_edges == 4
    assert geometry.dimension == 1
    assert np.array_equal(geometry.coordinates[:, 0], [0.0, 1.0, 2.0, 3.0, 4.0])
    assert [(edge.source, edge.target) for edge in geometry.edges] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    assert all(edge.edge_type == "nearest_neighbor" for edge in geometry.edges)


def test_chain_is_open_and_marks_only_its_endpoints_as_boundary() -> None:
    geometry = chain(5)

    assert geometry.boundary_sites == frozenset({0, 4})
    assert geometry.neighbors(0) == (1,)
    assert geometry.neighbors(2) == (1, 3)
    assert geometry.neighbors(4) == (3,)
    assert not geometry.has_edge(0, 4)
    assert not any(edge.boundary_crossing for edge in geometry.edges)


def test_chain_spacing_controls_coordinates_and_distances() -> None:
    geometry = chain(3, spacing=0.25)

    assert np.allclose(geometry.coordinates[:, 0], [0.0, 0.25, 0.5])
    assert geometry.distance(0, 1) == pytest.approx(0.25)
    assert np.array_equal(geometry.direction(1, 2), [1.0])
    assert np.array_equal(geometry.direction(2, 1), [-1.0])
    assert geometry.metadata == {
        "generator": "chain",
        "boundary_condition": "open",
        "spacing": 0.25,
    }


@pytest.mark.parametrize("n_sites", [1, 0, -3])
def test_chain_rejects_too_few_sites(n_sites: int) -> None:
    with pytest.raises(ValueError, match="at least two"):
        chain(n_sites)


@pytest.mark.parametrize("n_sites", [2.5, True])
def test_chain_rejects_non_integer_site_counts(n_sites: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        chain(n_sites)  # type: ignore[arg-type]


@pytest.mark.parametrize("spacing", [0.0, -1.0, np.inf, np.nan])
def test_chain_rejects_invalid_spacing(spacing: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        chain(3, spacing=spacing)


@pytest.mark.parametrize("spacing", [True, "1.0"])
def test_chain_rejects_non_real_spacing(spacing: object) -> None:
    with pytest.raises(TypeError, match="must be a real number"):
        chain(3, spacing=spacing)  # type: ignore[arg-type]
