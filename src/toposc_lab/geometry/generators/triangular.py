"""Two-dimensional triangular-lattice geometry generator."""

from __future__ import annotations

import numpy as np

from toposc_lab.geometry.base import Geometry, GeometryEdge
from toposc_lab.geometry.generators._validation import (
    validate_axis_size,
    validate_boundary,
    validate_spacing,
)


def triangular(
    n_x: int,
    n_y: int,
    *,
    spacing: float = 1.0,
    boundary_x: str = "open",
    boundary_y: str = "open",
) -> Geometry:
    """Create a finite triangular nearest-neighbor lattice.

    Sites use the oblique primitive vectors ``a1 = (spacing, 0)`` and
    ``a2 = (spacing / 2, sqrt(3) * spacing / 2)``. Site ``(x, y)`` has index
    ``x * n_y + y``. The three stored positive bond directions are ``a1``,
    ``a2``, and ``a1 - a2``; their reverses complete the six bulk neighbors.

    A periodic axis requires at least three sites because ``Geometry`` is a
    simple graph and therefore cannot represent the parallel periodic bonds
    that arise for a two-site circumference.
    """
    n_x = validate_axis_size(
        n_x,
        name="n_x",
        geometry_name="triangular lattice",
    )
    n_y = validate_axis_size(
        n_y,
        name="n_y",
        geometry_name="triangular lattice",
    )
    spacing = validate_spacing(spacing)
    boundary_x = validate_boundary(boundary_x, name="boundary_x")
    boundary_y = validate_boundary(boundary_y, name="boundary_y")
    _validate_periodic_axis(n_x, boundary_x, name="n_x")
    _validate_periodic_axis(n_y, boundary_y, name="n_y")

    height = np.sqrt(3.0) * spacing / 2.0
    primitive_x = (spacing, 0.0)
    primitive_y = (spacing / 2.0, height)
    coordinates = np.asarray(
        [
            (
                x * primitive_x[0] + y * primitive_y[0],
                x * primitive_x[1] + y * primitive_y[1],
            )
            for x in range(n_x)
            for y in range(n_y)
        ],
        dtype=float,
    )

    def site_index(x: int, y: int) -> int:
        return x * n_y + y

    directions = (
        (1, 0, primitive_x),
        (0, 1, primitive_y),
        (1, -1, (spacing / 2.0, -height)),
    )
    edges: list[GeometryEdge] = []
    for x in range(n_x):
        for y in range(n_y):
            source = site_index(x, y)
            for delta_x, delta_y, displacement in directions:
                target_x = x + delta_x
                target_y = y + delta_y
                boundary_crossing = False

                if not 0 <= target_x < n_x:
                    if boundary_x == "open":
                        continue
                    target_x %= n_x
                    boundary_crossing = True
                if not 0 <= target_y < n_y:
                    if boundary_y == "open":
                        continue
                    target_y %= n_y
                    boundary_crossing = True

                edges.append(
                    GeometryEdge(
                        source,
                        site_index(target_x, target_y),
                        edge_type="nearest_neighbor",
                        boundary_crossing=boundary_crossing,
                        displacement=displacement,
                        metadata={
                            "lattice_direction": (delta_x, delta_y),
                        },
                    )
                )

    boundary_sites = frozenset(
        site_index(x, y)
        for x in range(n_x)
        for y in range(n_y)
        if (boundary_x == "open" and x in (0, n_x - 1))
        or (boundary_y == "open" and y in (0, n_y - 1))
    )

    return Geometry(
        n_sites=n_x * n_y,
        edges=tuple(edges),
        coordinates=coordinates,
        boundary_sites=boundary_sites,
        metadata={
            "generator": "triangular",
            "shape": (n_x, n_y),
            "boundary_x": boundary_x,
            "boundary_y": boundary_y,
            "intrinsic_dimension": 2,
            "spacing": spacing,
            "primitive_vectors": (primitive_x, primitive_y),
        },
    )


def _validate_periodic_axis(size: int, boundary: str, *, name: str) -> None:
    if boundary == "periodic" and size < 3:
        raise ValueError(f"{name} must be at least three for a periodic triangular lattice")
