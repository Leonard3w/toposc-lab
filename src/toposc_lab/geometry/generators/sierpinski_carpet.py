"""Deterministic cell-centered Sierpiński-carpet geometry generator."""

from __future__ import annotations

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

_Cell = tuple[int, int]
_Hole = tuple[int, int, int]


def sierpinski_carpet(
    order: int,
    *,
    spacing: float = 1.0,
) -> Geometry:
    """Create a finite cell-centered Sierpiński carpet.

    Order zero is one square cell. Each iteration replaces every retained
    cell with a three-by-three block whose center cell is removed. Sites lie
    at retained cell centers and orthogonally adjacent cells share one edge.
    ``spacing`` is both the cell size and nearest-neighbor distance.
    """
    order = validate_recursion_order(order)
    spacing = validate_spacing(spacing)

    cells: set[_Cell] = {(0, 0)}
    holes: tuple[_Hole, ...] = ()
    side_length = 1
    for _ in range(order):
        shifts = tuple(
            (block_x * side_length, block_y * side_length)
            for block_y in range(3)
            for block_x in range(3)
            if (block_x, block_y) != (1, 1)
        )
        cells = {
            (cell[0] + shift[0], cell[1] + shift[1])
            for shift in shifts
            for cell in cells
        }
        holes = tuple(
            (hole_x + shift_x, hole_y + shift_y, hole_size)
            for shift_x, shift_y in shifts
            for hole_x, hole_y, hole_size in holes
        ) + ((side_length, side_length, side_length),)
        side_length *= 3

    ordered_cells = tuple(sorted(cells, key=lambda cell: (cell[1], cell[0])))
    cell_indices = {cell: index for index, cell in enumerate(ordered_cells)}
    coordinates = spacing * np.asarray(
        [(cell[0] + 0.5, cell[1] + 0.5) for cell in ordered_cells],
        dtype=float,
    )
    edges = tuple(
        GeometryEdge(
            cell_indices[cell],
            cell_indices[neighbor],
            edge_type="nearest_neighbor",
            displacement=(spacing * delta_x, spacing * delta_y),
        )
        for cell in ordered_cells
        for delta_x, delta_y in ((1, 0), (0, 1))
        if (neighbor := (cell[0] + delta_x, cell[1] + delta_y)) in cells
    )

    outer_sites = frozenset(
        cell_indices[cell]
        for cell in ordered_cells
        if cell[0] in (0, side_length - 1)
        or cell[1] in (0, side_length - 1)
    )
    boundary_components: list[GeometryBoundaryComponent] = [
        GeometryBoundaryComponent("outer", 0, outer_sites)
    ]
    for component_index, hole in enumerate(
        sorted(holes, key=lambda item: (item[1], item[0], item[2]))
    ):
        boundary_components.append(
            GeometryBoundaryComponent(
                "hole",
                component_index,
                frozenset(
                    cell_indices[cell]
                    for cell in _hole_boundary_cells(hole)
                ),
            )
        )
    boundary_sites = frozenset(
        site
        for component in boundary_components
        for site in component.sites
    )

    return Geometry(
        n_sites=len(ordered_cells),
        edges=edges,
        coordinates=coordinates,
        boundary_sites=boundary_sites,
        boundary_components=tuple(boundary_components),
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
                value=float(np.log(8.0) / np.log(3.0)),
                scope="infinite_family",
                method="analytic_self_similarity_log_8_over_log_3",
                exact=True,
            ),
        ),
        metadata={
            "generator": "sierpinski_carpet",
            "order": order,
            "spacing": spacing,
            "shape": (side_length, side_length),
            "outer_side_length": side_length * spacing,
            "copies_per_iteration": 8,
            "scale_factor": 3,
            "site_representation": "retained_cell_centers",
            "boundary_definition": "outer_and_hole_components",
            "n_holes": len(holes),
        },
    )


def _hole_boundary_cells(hole: _Hole) -> frozenset[_Cell]:
    hole_x, hole_y, hole_size = hole
    return frozenset(
        {
            *((hole_x - 1, y) for y in range(hole_y, hole_y + hole_size)),
            *((hole_x + hole_size, y) for y in range(hole_y, hole_y + hole_size)),
            *((x, hole_y - 1) for x in range(hole_x, hole_x + hole_size)),
            *((x, hole_y + hole_size) for x in range(hole_x, hole_x + hole_size)),
        }
    )
