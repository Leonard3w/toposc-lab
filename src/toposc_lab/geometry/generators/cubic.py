"""Three-dimensional simple-cubic lattice geometry generator."""

from __future__ import annotations

import numpy as np

from toposc_lab.geometry.base import Geometry, GeometryEdge
from toposc_lab.geometry.generators._dimensions import regular_lattice_dimensions
from toposc_lab.geometry.generators._validation import (
    resolve_axis_index,
    validate_axis_size,
    validate_boundary,
    validate_periodic_axis_size,
    validate_spacing,
)


def cubic(
    n_x: int,
    n_y: int,
    n_z: int,
    *,
    spacing: float = 1.0,
    boundary_x: str = "open",
    boundary_y: str = "open",
    boundary_z: str = "open",
) -> Geometry:
    """Create a three-dimensional simple-cubic nearest-neighbor lattice.

    Site ``(x, y, z)`` has index ``(x * n_y + y) * n_z + z``. Only the three
    positive Cartesian bond directions are stored; their reverses complete the
    six bulk neighbors. Periodic closing edges retain explicit local 3D
    displacement vectors.
    """
    n_x = validate_axis_size(n_x, name="n_x", geometry_name="cubic lattice")
    n_y = validate_axis_size(n_y, name="n_y", geometry_name="cubic lattice")
    n_z = validate_axis_size(n_z, name="n_z", geometry_name="cubic lattice")
    spacing = validate_spacing(spacing)
    boundary_x = validate_boundary(boundary_x, name="boundary_x")
    boundary_y = validate_boundary(boundary_y, name="boundary_y")
    boundary_z = validate_boundary(boundary_z, name="boundary_z")
    for size, boundary, name in (
        (n_x, boundary_x, "n_x"),
        (n_y, boundary_y, "n_y"),
        (n_z, boundary_z, "n_z"),
    ):
        validate_periodic_axis_size(
            size,
            boundary,
            name=name,
            geometry_name="cubic lattice",
        )

    def site_index(x: int, y: int, z: int) -> int:
        return (x * n_y + y) * n_z + z

    coordinates = spacing * np.asarray(
        [
            (x, y, z)
            for x in range(n_x)
            for y in range(n_y)
            for z in range(n_z)
        ],
        dtype=float,
    )
    directions = (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    )
    edges: list[GeometryEdge] = []
    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z):
                source = site_index(x, y, z)
                for delta_x, delta_y, delta_z in directions:
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
                        spacing * delta_x,
                        spacing * delta_y,
                        spacing * delta_z,
                    )
                    edges.append(
                        GeometryEdge(
                            source,
                            site_index(target_x, target_y, target_z),
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

    boundary_sites = frozenset(
        site_index(x, y, z)
        for x in range(n_x)
        for y in range(n_y)
        for z in range(n_z)
        if (boundary_x == "open" and x in (0, n_x - 1))
        or (boundary_y == "open" and y in (0, n_y - 1))
        or (boundary_z == "open" and z in (0, n_z - 1))
    )

    return Geometry(
        n_sites=n_x * n_y * n_z,
        edges=tuple(edges),
        coordinates=coordinates,
        boundary_sites=boundary_sites,
        dimension_records=regular_lattice_dimensions(3),
        metadata={
            "generator": "cubic",
            "shape": (n_x, n_y, n_z),
            "boundary_x": boundary_x,
            "boundary_y": boundary_y,
            "boundary_z": boundary_z,
            "spacing": spacing,
            "primitive_vectors": (
                (spacing, 0.0, 0.0),
                (0.0, spacing, 0.0),
                (0.0, 0.0, spacing),
            ),
        },
    )
