"""Two-dimensional honeycomb-lattice geometry generator."""

from __future__ import annotations

import numpy as np

from toposc_lab.geometry.base import Geometry, GeometryEdge
from toposc_lab.geometry.generators._validation import (
    validate_axis_size,
    validate_boundary,
    validate_spacing,
)


def honeycomb(
    n_x: int,
    n_y: int,
    *,
    spacing: float = 1.0,
    boundary_x: str = "open",
    boundary_y: str = "open",
) -> Geometry:
    """Create a honeycomb lattice with two sites per unit cell.

    Site indices are cell-major with sublattice A before B. For cell ``(x, y)``,
    A has index ``2 * (x * n_y + y)`` and B has the following index. Every A
    site connects to B in the same cell and to B in the previous x and y cells.
    Explicit edge displacements preserve the nearest-neighbor distance across
    periodic boundaries.
    """
    n_x = validate_axis_size(
        n_x,
        name="n_x",
        geometry_name="honeycomb lattice",
    )
    n_y = validate_axis_size(
        n_y,
        name="n_y",
        geometry_name="honeycomb lattice",
    )
    spacing = validate_spacing(spacing)
    boundary_x = validate_boundary(boundary_x, name="boundary_x")
    boundary_y = validate_boundary(boundary_y, name="boundary_y")

    sqrt_three = np.sqrt(3.0)
    primitive_x = (sqrt_three * spacing, 0.0)
    primitive_y = (sqrt_three * spacing / 2.0, 1.5 * spacing)
    sublattice_offset = (sqrt_three * spacing / 2.0, 0.5 * spacing)

    def site_index(x: int, y: int, sublattice_index: int) -> int:
        return 2 * (x * n_y + y) + sublattice_index

    coordinates = np.asarray(
        [
            (
                x * primitive_x[0]
                + y * primitive_y[0]
                + sublattice_index * sublattice_offset[0],
                x * primitive_x[1]
                + y * primitive_y[1]
                + sublattice_index * sublattice_offset[1],
            )
            for x in range(n_x)
            for y in range(n_y)
            for sublattice_index in (0, 1)
        ],
        dtype=float,
    )

    neighbor_cells = (
        (0, 0),
        (-1, 0),
        (0, -1),
    )
    edges: list[GeometryEdge] = []
    for x in range(n_x):
        for y in range(n_y):
            source = site_index(x, y, 0)
            for delta_x, delta_y in neighbor_cells:
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

                displacement = (
                    delta_x * primitive_x[0]
                    + delta_y * primitive_y[0]
                    + sublattice_offset[0],
                    delta_x * primitive_x[1]
                    + delta_y * primitive_y[1]
                    + sublattice_offset[1],
                )
                edges.append(
                    GeometryEdge(
                        source,
                        site_index(target_x, target_y, 1),
                        edge_type="nearest_neighbor",
                        boundary_crossing=boundary_crossing,
                        displacement=displacement,
                        metadata={
                            "lattice_direction": (delta_x, delta_y),
                        },
                    )
                )

    degrees = [0] * (2 * n_x * n_y)
    for edge in edges:
        degrees[edge.source] += 1
        degrees[edge.target] += 1
    boundary_sites = frozenset(
        site for site, degree in enumerate(degrees) if degree < 3
    )

    return Geometry(
        n_sites=2 * n_x * n_y,
        edges=tuple(edges),
        coordinates=coordinates,
        boundary_sites=boundary_sites,
        site_types=tuple(
            "A" if site % 2 == 0 else "B"
            for site in range(2 * n_x * n_y)
        ),
        metadata={
            "generator": "honeycomb",
            "shape": (n_x, n_y),
            "sites_per_cell": 2,
            "boundary_x": boundary_x,
            "boundary_y": boundary_y,
            "intrinsic_dimension": 2,
            "spacing": spacing,
            "primitive_vectors": (primitive_x, primitive_y),
            "sublattice_offset": sublattice_offset,
        },
    )
