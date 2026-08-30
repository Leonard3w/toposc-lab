from __future__ import annotations

import numpy as np

from toposc_lab.geometry.base import Geometry, GeometryEdge
from toposc_lab.geometry.generators._validation import (
    validate_axis_size,
    validate_boundary,
    validate_spacing,
)


def square(
    n_x: int,
    n_y: int,
    *,
    spacing: float = 1.0,
    boundary_x: str = "open",
    boundary_y: str = "open",
) -> Geometry:
    """Create a two-dimensional square nearest-neighbor lattice.

    Site ``(x, y)`` has index ``x * n_y + y``. Periodic closing edges carry
    explicit local displacement vectors, so their geometric length remains
    ``spacing`` instead of spanning the full coordinate extent.
    """
    n_x = validate_axis_size(n_x, name="n_x", geometry_name="square lattice")
    n_y = validate_axis_size(n_y, name="n_y", geometry_name="square lattice")
    spacing = validate_spacing(spacing)
    boundary_x = validate_boundary(boundary_x, name="boundary_x")
    boundary_y = validate_boundary(boundary_y, name="boundary_y")

    def site_index(x: int, y: int) -> int:
        return x * n_y + y

    coordinates = spacing * np.asarray(
        [(x, y) for x in range(n_x) for y in range(n_y)],
        dtype=float,
    )
    edges: list[GeometryEdge] = []

    for x in range(n_x):
        for y in range(n_y):
            site = site_index(x, y)

            if x < n_x - 1:
                edges.append(
                    GeometryEdge(
                        site,
                        site_index(x + 1, y),
                        edge_type="nearest_neighbor",
                        displacement=(spacing, 0.0),
                    )
                )
            elif boundary_x == "periodic" and n_x > 2:
                edges.append(
                    GeometryEdge(
                        site,
                        site_index(0, y),
                        edge_type="nearest_neighbor",
                        boundary_crossing=True,
                        displacement=(spacing, 0.0),
                    )
                )

            if y < n_y - 1:
                edges.append(
                    GeometryEdge(
                        site,
                        site_index(x, y + 1),
                        edge_type="nearest_neighbor",
                        displacement=(0.0, spacing),
                    )
                )
            elif boundary_y == "periodic" and n_y > 2:
                edges.append(
                    GeometryEdge(
                        site,
                        site_index(x, 0),
                        edge_type="nearest_neighbor",
                        boundary_crossing=True,
                        displacement=(0.0, spacing),
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
            "generator": "square",
            "shape": (n_x, n_y),
            "boundary_x": boundary_x,
            "boundary_y": boundary_y,
            "intrinsic_dimension": 2,
            "spacing": spacing,
        },
    )
