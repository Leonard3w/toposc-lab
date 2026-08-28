import numpy as np
import pytest

from toposc_lab.geometry import Geometry, GeometryEdge


def test_geometry_stores_an_arbitrary_graph() -> None:
    geometry = Geometry(
        n_sites=4,
        edges=(
            GeometryEdge(0, 2, edge_type="long"),
            GeometryEdge(2, 1),
            GeometryEdge(2, 3, boundary_crossing=True),
        ),
        boundary_sites=frozenset({0, 3}),
        site_types=("A", "B", "A", None),
        metadata={"name": "irregular test graph"},
    )

    assert geometry.site_indices == (0, 1, 2, 3)
    assert geometry.n_edges == 3
    assert geometry.dimension is None
    assert geometry.neighbors(2) == (0, 1, 3)
    assert geometry.degree(2) == 3
    assert geometry.has_edge(2, 0)
    assert geometry.edge_between(2, 0) is geometry.edges[0]
    assert geometry.is_boundary_site(3)
    assert geometry.site_types == ("A", "B", "A", None)
    assert geometry.metadata["name"] == "irregular test graph"


def test_geometry_calculates_spatial_quantities() -> None:
    geometry = Geometry(
        n_sites=3,
        edges=(GeometryEdge(0, 1), GeometryEdge(1, 2)),
        coordinates=np.asarray([(0.0, 0.0), (3.0, 4.0), (3.0, 5.0)]),
    )

    assert geometry.dimension == 2
    assert np.array_equal(geometry.position(1), [3.0, 4.0])
    assert geometry.distance(0, 1) == pytest.approx(5.0)
    assert np.allclose(geometry.direction(0, 1), [0.6, 0.8])


def test_edge_displacement_handles_boundary_crossing_orientation() -> None:
    geometry = Geometry(
        n_sites=3,
        edges=(
            GeometryEdge(
                2,
                0,
                boundary_crossing=True,
                displacement=(1.0,),
            ),
        ),
        coordinates=np.asarray([(0.0,), (1.0,), (2.0,)]),
    )

    assert np.array_equal(geometry.displacement_between(2, 0), [1.0])
    assert np.array_equal(geometry.displacement_between(0, 2), [-1.0])
    assert geometry.distance(2, 0) == pytest.approx(1.0)


def test_geometry_without_coordinates_rejects_spatial_queries() -> None:
    geometry = Geometry(n_sites=2, edges=(GeometryEdge(0, 1),))

    with pytest.raises(ValueError, match="no spatial coordinates"):
        geometry.position(0)

    with pytest.raises(ValueError, match="no displacement"):
        geometry.distance(0, 1)


@pytest.mark.parametrize(
    ("edges", "message"),
    [
        ((GeometryEdge(0, 3),), "outside the geometry"),
        (
            (GeometryEdge(0, 1), GeometryEdge(1, 0)),
            "duplicate edge",
        ),
    ],
)
def test_geometry_rejects_invalid_edges(
    edges: tuple[GeometryEdge, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Geometry(n_sites=3, edges=edges)


def test_geometry_rejects_inconsistent_optional_data() -> None:
    with pytest.raises(ValueError, match="one row per site"):
        Geometry(n_sites=3, coordinates=np.zeros((2, 2)))

    with pytest.raises(ValueError, match="one entry per site"):
        Geometry(n_sites=3, site_types=("A", "B"))

    with pytest.raises(ValueError, match="does not match coordinates"):
        Geometry(
            n_sites=2,
            coordinates=np.zeros((2, 2)),
            embedding_dimension=3,
        )


def test_geometry_copies_and_protects_coordinate_data() -> None:
    coordinates = np.asarray([(0.0,), (1.0,)])
    geometry = Geometry(n_sites=2, coordinates=coordinates)
    coordinates[0, 0] = 10.0

    assert geometry.coordinates is not None
    assert geometry.coordinates[0, 0] == 0.0
    with pytest.raises(ValueError, match="read-only"):
        geometry.coordinates[0, 0] = 5.0
