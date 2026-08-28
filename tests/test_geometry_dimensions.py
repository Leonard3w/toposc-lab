from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.geometry import Geometry, GeometryEdge


@pytest.mark.parametrize("embedding_dimension", [1, 2, 3, 7])
def test_geometry_accepts_arbitrary_embedding_dimensions(
    embedding_dimension: int,
) -> None:
    source = np.zeros(embedding_dimension, dtype=float)
    target = np.arange(1, embedding_dimension + 1, dtype=float)
    coordinates = np.vstack((source, target))
    geometry = Geometry(
        n_sites=2,
        edges=(GeometryEdge(0, 1),),
        coordinates=coordinates,
        embedding_dimension=embedding_dimension,
    )

    expected_distance = float(np.linalg.norm(target))
    assert geometry.coordinates is not None
    assert geometry.coordinates.shape == (2, embedding_dimension)
    assert geometry.dimension == embedding_dimension
    assert np.array_equal(geometry.position(1), target)
    assert np.array_equal(geometry.displacement_between(0, 1), target)
    assert geometry.distance(0, 1) == pytest.approx(expected_distance)
    assert np.allclose(geometry.direction(0, 1), target / expected_distance)


def test_coordinate_free_geometry_remains_a_valid_abstract_graph() -> None:
    geometry = Geometry(
        n_sites=4,
        edges=(
            GeometryEdge(0, 1),
            GeometryEdge(1, 3),
            GeometryEdge(3, 2),
        ),
    )

    assert geometry.coordinates is None
    assert geometry.dimension is None
    assert geometry.neighbors(3) == (1, 2)
    assert geometry.degree(1) == 2

    with pytest.raises(ValueError, match="no spatial coordinates"):
        geometry.position(0)

    with pytest.raises(ValueError, match="no displacement or spatial coordinates"):
        geometry.displacement_between(0, 1)


def test_embedding_dimension_may_be_known_without_site_coordinates() -> None:
    geometry = Geometry(n_sites=3, embedding_dimension=5)

    assert geometry.coordinates is None
    assert geometry.dimension == 5

    with pytest.raises(ValueError, match="no spatial coordinates"):
        geometry.position(0)


def test_edge_displacement_infers_dimension_without_site_coordinates() -> None:
    displacement = (1.0, -2.0, 2.0, 4.0)
    geometry = Geometry(
        n_sites=2,
        edges=(GeometryEdge(0, 1, displacement=displacement),),
    )

    assert geometry.coordinates is None
    assert geometry.dimension == 4
    assert np.array_equal(geometry.displacement_between(0, 1), displacement)
    assert np.array_equal(geometry.displacement_between(1, 0), -np.asarray(displacement))
    assert geometry.distance(0, 1) == pytest.approx(5.0)


def test_edge_displacement_must_match_coordinate_dimension() -> None:
    with pytest.raises(ValueError, match="does not match embedding_dimension"):
        Geometry(
            n_sites=2,
            edges=(GeometryEdge(0, 1, displacement=(1.0, 0.0)),),
            coordinates=np.zeros((2, 3)),
        )
