"""Deterministic Sierpiński-gasket geometry generator."""

from __future__ import annotations

import numpy as np

from toposc_lab.geometry.base import Geometry, GeometryDimension, GeometryEdge
from toposc_lab.geometry.generators._validation import (
    validate_recursion_order,
    validate_spacing,
)

_LatticePoint = tuple[int, int]
_LatticeEdge = tuple[_LatticePoint, _LatticePoint]


def sierpinski_gasket(
    order: int,
    *,
    spacing: float = 1.0,
) -> Geometry:
    """Create a finite Sierpiński gasket of nonnegative recursion order.

    Order zero is one triangle. Each subsequent order joins three translated
    copies of the previous graph at their corner sites. ``spacing`` is the
    smallest edge length; the outer side length is ``2**order * spacing``.
    """
    order = validate_recursion_order(order)
    spacing = validate_spacing(spacing)

    points: set[_LatticePoint] = {(0, 0), (1, 0), (0, 1)}
    edges: set[_LatticeEdge] = {
        _canonical_edge((0, 0), (1, 0)),
        _canonical_edge((0, 0), (0, 1)),
        _canonical_edge((1, 0), (0, 1)),
    }
    side_length = 1
    for _ in range(order):
        shifts = (
            (0, 0),
            (side_length, 0),
            (0, side_length),
        )
        points = {
            _translate_point(point, shift)
            for shift in shifts
            for point in points
        }
        edges = {
            _canonical_edge(
                _translate_point(edge[0], shift),
                _translate_point(edge[1], shift),
            )
            for shift in shifts
            for edge in edges
        }
        side_length *= 2

    ordered_points = tuple(sorted(points, key=lambda point: (point[1], point[0])))
    point_indices = {point: index for index, point in enumerate(ordered_points)}
    height = np.sqrt(3.0) / 2.0
    coordinates = spacing * np.asarray(
        [
            (point[0] + 0.5 * point[1], height * point[1])
            for point in ordered_points
        ],
        dtype=float,
    )
    geometry_edges = tuple(
        GeometryEdge(
            point_indices[source],
            point_indices[target],
            edge_type="nearest_neighbor",
            displacement=(
                spacing
                * (
                    target[0]
                    - source[0]
                    + 0.5 * (target[1] - source[1])
                ),
                spacing * height * (target[1] - source[1]),
            ),
        )
        for source, target in sorted(edges)
    )
    boundary_sites = frozenset(
        point_indices[point]
        for point in ordered_points
        if point[0] == 0
        or point[1] == 0
        or point[0] + point[1] == side_length
    )

    return Geometry(
        n_sites=len(ordered_points),
        edges=geometry_edges,
        coordinates=coordinates,
        boundary_sites=boundary_sites,
        dimension_records=(
            GeometryDimension(
                kind="topological",
                value=1.0,
                scope="infinite_family",
                method="covering_dimension",
                exact=True,
            ),
            GeometryDimension(
                kind="hausdorff",
                value=float(np.log(3.0) / np.log(2.0)),
                scope="infinite_family",
                method="analytic_self_similarity_log_3_over_log_2",
                exact=True,
            ),
        ),
        metadata={
            "generator": "sierpinski_gasket",
            "order": order,
            "spacing": spacing,
            "outer_side_length": side_length * spacing,
            "copies_per_iteration": 3,
            "scale_factor": 2,
            "boundary_definition": "outer_triangle",
        },
    )


def _translate_point(
    point: _LatticePoint,
    shift: _LatticePoint,
) -> _LatticePoint:
    return point[0] + shift[0], point[1] + shift[1]


def _canonical_edge(
    first: _LatticePoint,
    second: _LatticePoint,
) -> _LatticeEdge:
    return (first, second) if first < second else (second, first)
