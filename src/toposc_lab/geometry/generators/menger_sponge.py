"""Deterministic voxel-centered Menger-sponge geometry generator."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from toposc_lab.geometry.base import (
    Geometry,
    GeometryBoundaryComponent,
    GeometryDimension,
    GeometryEdge,
)
from toposc_lab.geometry.generators._validation import (
    validate_recursion_order,
    validate_spacing,
)

DEFAULT_MENGER_MAX_SITES = 25_000

_Voxel = tuple[int, int, int]


def menger_sponge(
    order: int,
    *,
    spacing: float = 1.0,
    max_sites: int | None = DEFAULT_MENGER_MAX_SITES,
) -> Geometry:
    """Create a finite voxel-centered Menger-sponge graph.

    Order zero is one cubic voxel. Each iteration replaces every retained
    voxel with the 20 cubes of a three-by-three-by-three block that are not in
    a face center or the body center. Face-sharing voxels are nearest
    neighbors. ``max_sites`` rejects oversized requests before construction;
    pass ``None`` to disable that explicit resource guard.
    """
    order = validate_recursion_order(order)
    spacing = validate_spacing(spacing)
    max_sites = _validate_max_sites(max_sites)
    expected_sites = 20**order
    if max_sites is not None and expected_sites > max_sites:
        raise ValueError(
            f"menger_sponge order {order} requires {expected_sites} sites, "
            f"exceeding max_sites={max_sites}"
        )

    voxels: set[_Voxel] = {(0, 0, 0)}
    side_length = 1
    for _ in range(order):
        shifts = tuple(
            (
                block_x * side_length,
                block_y * side_length,
                block_z * side_length,
            )
            for block_z in range(3)
            for block_y in range(3)
            for block_x in range(3)
            if sum(axis == 1 for axis in (block_x, block_y, block_z)) <= 1
        )
        voxels = {
            (
                voxel[0] + shift[0],
                voxel[1] + shift[1],
                voxel[2] + shift[2],
            )
            for shift in shifts
            for voxel in voxels
        }
        side_length *= 3

    ordered_voxels = tuple(
        sorted(voxels, key=lambda voxel: (voxel[2], voxel[1], voxel[0]))
    )
    voxel_indices = {
        voxel: index for index, voxel in enumerate(ordered_voxels)
    }
    coordinates = spacing * np.asarray(
        [
            (voxel[0] + 0.5, voxel[1] + 0.5, voxel[2] + 0.5)
            for voxel in ordered_voxels
        ],
        dtype=float,
    )
    positive_directions = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    edges = tuple(
        GeometryEdge(
            voxel_indices[voxel],
            voxel_indices[neighbor],
            edge_type="nearest_neighbor",
            displacement=(
                spacing * delta_x,
                spacing * delta_y,
                spacing * delta_z,
            ),
        )
        for voxel in ordered_voxels
        for delta_x, delta_y, delta_z in positive_directions
        if (
            neighbor := (
                voxel[0] + delta_x,
                voxel[1] + delta_y,
                voxel[2] + delta_z,
            )
        )
        in voxels
    )
    boundary_sites = frozenset(
        voxel_indices[voxel]
        for voxel in ordered_voxels
        if any(
            (
                voxel[0] + delta_x,
                voxel[1] + delta_y,
                voxel[2] + delta_z,
            )
            not in voxels
            for delta_x, delta_y, delta_z in (
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            )
        )
    )
    boundary_components = (
        GeometryBoundaryComponent("outer", 0, boundary_sites),
    )

    return Geometry(
        n_sites=len(ordered_voxels),
        edges=edges,
        coordinates=coordinates,
        boundary_sites=boundary_sites,
        boundary_components=boundary_components,
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
                value=float(np.log(20.0) / np.log(3.0)),
                scope="infinite_family",
                method="analytic_self_similarity_log_20_over_log_3",
                exact=True,
            ),
        ),
        metadata={
            "generator": "menger_sponge",
            "order": order,
            "spacing": spacing,
            "shape": (side_length, side_length, side_length),
            "outer_side_length": side_length * spacing,
            "copies_per_iteration": 20,
            "scale_factor": 3,
            "site_representation": "retained_voxel_centers",
            "boundary_definition": "exterior_accessible_surface_including_tunnels",
            "max_sites": max_sites,
        },
    )


def _validate_max_sites(max_sites: int | None) -> int | None:
    if max_sites is None:
        return None
    if isinstance(max_sites, bool) or not isinstance(max_sites, Integral):
        raise TypeError("max_sites must be an integer or None")
    result = int(max_sites)
    if result < 1:
        raise ValueError("max_sites must be positive")
    return result
