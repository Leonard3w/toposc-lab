"""Two-dimensional Kagome-lattice geometry generator."""

from __future__ import annotations

import numpy as np

from toposc_lab.geometry.base import Geometry, GeometryEdge
from toposc_lab.geometry.generators._validation import (
    resolve_axis_index,
    validate_axis_size,
    validate_boundary,
    validate_spacing,
)


def kagome(
    n_x: int,
    n_y: int,
    *,
    spacing: float = 1.0,
    boundary_x: str = "open",
    boundary_y: str = "open",
) -> Geometry:
    """Create a Kagome lattice with three sites per triangular unit cell.

    Site indices are cell-major with sublattices A, B, and C. The chosen basis
    is the midpoint representation of the three honeycomb bonds anchored at one
    A site. Consequently, the fully periodic graph is exactly the line graph of
    a fully periodic honeycomb lattice with the same number of unit cells.
    """
    n_x = validate_axis_size(
        n_x,
        name="n_x",
        geometry_name="Kagome lattice",
    )
    n_y = validate_axis_size(
        n_y,
        name="n_y",
        geometry_name="Kagome lattice",
    )
    spacing = validate_spacing(spacing)
    boundary_x = validate_boundary(boundary_x, name="boundary_x")
    boundary_y = validate_boundary(boundary_y, name="boundary_y")

    height = np.sqrt(3.0) * spacing / 2.0
    primitive_x = (2.0 * spacing, 0.0)
    primitive_y = (spacing, 2.0 * height)
    basis_offsets = (
        (spacing, height),
        (0.0, height),
        (spacing / 2.0, 0.0),
    )

    def site_index(x: int, y: int, sublattice_index: int) -> int:
        return 3 * (x * n_y + y) + sublattice_index

    coordinates = np.asarray(
        [
            (
                x * primitive_x[0]
                + y * primitive_y[0]
                + basis_offsets[sublattice_index][0],
                x * primitive_x[1]
                + y * primitive_y[1]
                + basis_offsets[sublattice_index][1],
            )
            for x in range(n_x)
            for y in range(n_y)
            for sublattice_index in (0, 1, 2)
        ],
        dtype=float,
    )

    bonds = (
        (0, 1, 0, 0),
        (0, 2, 0, 0),
        (1, 2, 0, 0),
        (0, 1, 1, 0),
        (0, 2, 0, 1),
        (1, 2, -1, 1),
    )
    edges: list[GeometryEdge] = []
    for x in range(n_x):
        for y in range(n_y):
            for source_type, target_type, delta_x, delta_y in bonds:
                resolved_x = resolve_axis_index(
                    x + delta_x,
                    size=n_x,
                    boundary=boundary_x,
                )
                resolved_y = resolve_axis_index(
                    y + delta_y,
                    size=n_y,
                    boundary=boundary_y,
                )
                if resolved_x is None or resolved_y is None:
                    continue
                target_x, crossed_x = resolved_x
                target_y, crossed_y = resolved_y

                displacement = (
                    delta_x * primitive_x[0]
                    + delta_y * primitive_y[0]
                    + basis_offsets[target_type][0]
                    - basis_offsets[source_type][0],
                    delta_x * primitive_x[1]
                    + delta_y * primitive_y[1]
                    + basis_offsets[target_type][1]
                    - basis_offsets[source_type][1],
                )
                edges.append(
                    GeometryEdge(
                        site_index(x, y, source_type),
                        site_index(target_x, target_y, target_type),
                        edge_type="nearest_neighbor",
                        boundary_crossing=crossed_x or crossed_y,
                        displacement=displacement,
                        metadata={
                            "lattice_direction": (delta_x, delta_y),
                        },
                    )
                )

    degrees = [0] * (3 * n_x * n_y)
    for edge in edges:
        degrees[edge.source] += 1
        degrees[edge.target] += 1
    boundary_sites = frozenset(
        site for site, degree in enumerate(degrees) if degree < 4
    )

    return Geometry(
        n_sites=3 * n_x * n_y,
        edges=tuple(edges),
        coordinates=coordinates,
        boundary_sites=boundary_sites,
        site_types=tuple(
            ("A", "B", "C")[site % 3]
            for site in range(3 * n_x * n_y)
        ),
        metadata={
            "generator": "kagome",
            "shape": (n_x, n_y),
            "sites_per_cell": 3,
            "boundary_x": boundary_x,
            "boundary_y": boundary_y,
            "intrinsic_dimension": 2,
            "spacing": spacing,
            "primitive_vectors": (primitive_x, primitive_y),
            "basis_offsets": basis_offsets,
            "construction": "honeycomb_line_graph",
        },
    )
