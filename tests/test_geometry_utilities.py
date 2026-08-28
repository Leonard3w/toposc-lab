from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from toposc_lab.geometry import Geometry, GeometryEdge


def test_neighbors_degree_and_edge_lookup_follow_undirected_semantics() -> None:
    first_edge = GeometryEdge(3, 1)
    geometry = Geometry(
        n_sites=5,
        edges=(first_edge, GeometryEdge(1, 0), GeometryEdge(1, 2)),
    )

    assert geometry.neighbors(1) == (0, 2, 3)
    assert geometry.neighbors(4) == ()
    assert geometry.degree(1) == 3
    assert geometry.degree(4) == 0
    assert geometry.edge_between(3, 1) is first_edge
    assert geometry.edge_between(1, 3) is first_edge
    assert geometry.has_edge(1, 3)
    assert not geometry.has_edge(0, 3)
    assert geometry.edge_between(1, 1) is None


def test_coordinate_displacement_does_not_require_a_graph_edge() -> None:
    geometry = Geometry(
        n_sites=3,
        edges=(GeometryEdge(0, 1),),
        coordinates=np.asarray(
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (2.0, -1.0, 2.0),
            ]
        ),
    )
    expected = np.asarray((2.0, -1.0, 2.0))

    assert not geometry.has_edge(0, 2)
    assert np.array_equal(geometry.displacement_between(0, 2), expected)
    assert np.array_equal(geometry.displacement_between(2, 0), -expected)
    assert geometry.distance(0, 2) == pytest.approx(3.0)
    assert geometry.distance(2, 0) == pytest.approx(3.0)
    assert np.allclose(geometry.direction(0, 2), expected / 3.0)
    assert np.allclose(geometry.direction(2, 0), -expected / 3.0)


def test_explicit_edge_displacement_overrides_endpoint_coordinate_difference() -> None:
    geometry = Geometry(
        n_sites=2,
        edges=(
            GeometryEdge(
                1,
                0,
                boundary_crossing=True,
                displacement=(0.5, 0.0),
            ),
        ),
        coordinates=np.asarray(((0.0, 0.0), (10.0, 0.0))),
    )

    assert np.array_equal(geometry.position(1) - geometry.position(0), [10.0, 0.0])
    assert np.array_equal(geometry.displacement_between(1, 0), [0.5, 0.0])
    assert np.array_equal(geometry.displacement_between(0, 1), [-0.5, 0.0])
    assert geometry.distance(1, 0) == pytest.approx(0.5)


def test_zero_displacement_has_zero_distance_but_no_direction() -> None:
    geometry = Geometry(
        n_sites=2,
        coordinates=np.asarray(((1.0, 2.0), (1.0, 2.0))),
    )

    assert np.array_equal(geometry.displacement_between(0, 0), [0.0, 0.0])
    assert geometry.distance(0, 0) == 0.0
    assert geometry.distance(0, 1) == 0.0

    with pytest.raises(ValueError, match="undefined for zero displacement"):
        geometry.direction(0, 1)


def test_coordinate_free_graph_uses_only_available_edge_displacements() -> None:
    geometry = Geometry(
        n_sites=3,
        edges=(
            GeometryEdge(0, 1, displacement=(3.0, 4.0)),
            GeometryEdge(1, 2),
        ),
    )

    assert geometry.distance(0, 1) == pytest.approx(5.0)
    assert np.allclose(geometry.direction(0, 1), [0.6, 0.8])

    with pytest.raises(ValueError, match="no displacement or spatial coordinates"):
        geometry.distance(1, 2)

    with pytest.raises(ValueError, match="no displacement or spatial coordinates"):
        geometry.displacement_between(0, 2)


def test_utility_methods_reject_sites_outside_geometry() -> None:
    geometry = Geometry(
        n_sites=2,
        edges=(GeometryEdge(0, 1),),
        coordinates=np.asarray(((0.0,), (1.0,))),
    )
    invalid_operations: tuple[Callable[[], object], ...] = (
        lambda: geometry.neighbors(-1),
        lambda: geometry.degree(2),
        lambda: geometry.edge_between(0, 2),
        lambda: geometry.has_edge(2, 0),
        lambda: geometry.position(2),
        lambda: geometry.displacement_between(-1, 0),
        lambda: geometry.distance(0, 2),
        lambda: geometry.direction(2, 0),
    )

    for operation in invalid_operations:
        with pytest.raises(ValueError, match="outside the geometry"):
            operation()
