"""Three-dimensional body-centered-cubic lattice geometry generator."""

from __future__ import annotations

from itertools import product

import numpy as np

from toposc_lab.geometry.base import Geometry, GeometryEdge
from toposc_lab.geometry.generators._dimensions import regular_lattice_dimensions
from toposc_lab.geometry.generators._validation import (
    resolve_axis_index,
    validate_axis_size,
    validate_boundary,
    validate_spacing,
)


def body_centered_cubic(
    n_x: int,
    n_y: int,
    n_z: int,
    *,
    spacing: float = 1.0,
    boundary_x: str = "open",
    boundary_y: str = "open",
    boundary_z: str = "open",
) -> Geometry:
    """Create a body-centered-cubic nearest-neighbor network.

    Each conventional cubic cell contains corner sublattice A and body-center
    sublattice B. ``spacing`` is the A-B nearest-neighbor distance, so the
    conventional cubic cell length is ``2 * spacing / sqrt(3)``. Every A site
    connects to the eight B sites displaced by all combinations of zero and
    minus one conventional cell along the three axes.
    """
    n_x = validate_axis_size(n_x, name="n_x", geometry_name="BCC lattice")
    n_y = validate_axis_size(n_y, name="n_y", geometry_name="BCC lattice")
    n_z = validate_axis_size(n_z, name="n_z", geometry_name="BCC lattice")
    spacing = validate_spacing(spacing)
    boundary_x = validate_boundary(boundary_x, name="boundary_x")
    boundary_y = validate_boundary(boundary_y, name="boundary_y")
    boundary_z = validate_boundary(boundary_z, name="boundary_z")

    cell_length = 2.0 * spacing / np.sqrt(3.0)
    cell_vectors = (
        (cell_length, 0.0, 0.0),
        (0.0, cell_length, 0.0),
        (0.0, 0.0, cell_length),
    )
    body_center_offset = (
        cell_length / 2.0,
        cell_length / 2.0,
        cell_length / 2.0,
    )

    def site_index(x: int, y: int, z: int, sublattice_index: int) -> int:
        return 2 * ((x * n_y + y) * n_z + z) + sublattice_index

    coordinates = np.asarray(
        [
            (
                x * cell_length + sublattice_index * body_center_offset[0],
                y * cell_length + sublattice_index * body_center_offset[1],
                z * cell_length + sublattice_index * body_center_offset[2],
            )
            for x in range(n_x)
            for y in range(n_y)
            for z in range(n_z)
            for sublattice_index in (0, 1)
        ],
        dtype=float,
    )

    neighbor_cells = tuple(product((0, -1), repeat=3))
    edges: list[GeometryEdge] = []
    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):
                source = site_index(x, y, z, 0)
                for delta_x, delta_y, delta_z in neighbor_cells:
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
                    resolved_z = resolve_axis_index(
                        z + delta_z,
                        size=n_z,
                        boundary=boundary_z,
                    )
                    if resolved_x is None or resolved_y is None or resolved_z is None:
                        continue
                    target_x, crossed_x = resolved_x
                    target_y, crossed_y = resolved_y
                    target_z, crossed_z = resolved_z
                    displacement = (
                        delta_x * cell_length + body_center_offset[0],
                        delta_y * cell_length + body_center_offset[1],
                        delta_z * cell_length + body_center_offset[2],
                    )
                    edges.append(
                        GeometryEdge(
                            source,
                            site_index(target_x, target_y, target_z, 1),
                            edge_type="nearest_neighbor",
                            boundary_crossing=crossed_x or crossed_y or crossed_z,
                            displacement=displacement,
                            metadata={
                                "lattice_direction": (
                                    delta_x,
                                    delta_y,
                                    delta_z,
                                ),
                            },
                        )
                    )

    degrees = [0] * (2 * n_x * n_y * n_z)
    for edge in edges:
        degrees[edge.source] += 1
        degrees[edge.target] += 1
    boundary_sites = frozenset(
        site for site, degree in enumerate(degrees) if degree < 8
    )

    return Geometry(
        n_sites=2 * n_x * n_y * n_z,
        edges=tuple(edges),
        coordinates=coordinates,
        boundary_sites=boundary_sites,
        site_types=tuple(
            "A" if site % 2 == 0 else "B"
            for site in range(2 * n_x * n_y * n_z)
        ),
        dimension_records=regular_lattice_dimensions(3),
        metadata={
            "generator": "body_centered_cubic",
            "lattice_type": "bcc",
            "shape": (n_x, n_y, n_z),
            "sites_per_cell": 2,
            "boundary_x": boundary_x,
            "boundary_y": boundary_y,
            "boundary_z": boundary_z,
            "spacing": spacing,
            "conventional_cell_length": cell_length,
            "cell_vectors": cell_vectors,
            "basis_offsets": ((0.0, 0.0, 0.0), body_center_offset),
            "sublattice_roles": {
                "A": "corner",
                "B": "body_center",
            },
        },
    )
