"""Finite Ammann--Beenker tiling patches from a four-dimensional cut-and-project set."""

from __future__ import annotations

from itertools import product
from math import ceil, hypot, sqrt
from numbers import Real

import numpy as np

from toposc_lab.geometry.base import (
    Geometry,
    GeometryBoundaryComponent,
    GeometryDimension,
    GeometryEdge,
    GeometryFace,
)
from toposc_lab.geometry.generators._validation import (
    validate_optional_budget,
    validate_spacing,
)

AMMANN_BEENKER_INFLATION_FACTOR = 1.0 + sqrt(2.0)
DEFAULT_AMMANN_BEENKER_MAX_SITES = 100_000
DEFAULT_AMMANN_BEENKER_MAX_CANDIDATES = 2_000_000

_SQRT_TWO = sqrt(2.0)
_INV_SQRT_TWO = 1.0 / _SQRT_TWO
_WINDOW_APOTHEM = AMMANN_BEENKER_INFLATION_FACTOR / 2.0
_WINDOW_CIRCUMRADIUS = hypot(_WINDOW_APOTHEM, 0.5)
_MINIMUM_TILING_RADIUS = 2.0
_PHYSICAL_BASIS = (
    (1.0, 0.0),
    (_INV_SQRT_TWO, _INV_SQRT_TWO),
    (0.0, 1.0),
    (-_INV_SQRT_TWO, _INV_SQRT_TWO),
)
_INTERNAL_BASIS = (
    (1.0, 0.0),
    (-_INV_SQRT_TWO, _INV_SQRT_TWO),
    (0.0, -1.0),
    (_INV_SQRT_TWO, _INV_SQRT_TWO),
)

_LatticePoint = tuple[int, int, int, int]
_PointRecord = tuple[_LatticePoint, tuple[float, float], tuple[float, float]]


def ammann_beenker_patch(
    radius: float,
    *,
    spacing: float = 1.0,
    max_sites: int | None = DEFAULT_AMMANN_BEENKER_MAX_SITES,
    max_candidates: int | None = DEFAULT_AMMANN_BEENKER_MAX_CANDIDATES,
) -> Geometry:
    """Create a centered radial patch of the Ammann--Beenker tiling.

    Vertices are selected from ``Z^4`` by projection into a centered regular
    octagonal acceptance window. ``radius`` is the physical cutoff measured in
    unit tile edges before multiplication by ``spacing``. A minimum radius of
    two guarantees that the returned patch contains complete tiles.
    """
    radius = _validate_radius(radius)
    spacing = validate_spacing(spacing)
    max_sites = validate_optional_budget(max_sites, name="max_sites")
    max_candidates = validate_optional_budget(
        max_candidates,
        name="max_candidates",
    )
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, radius)
    lattice_bound = int(ceil((radius + _WINDOW_CIRCUMRADIUS) / 2.0))
    n_candidates = (2 * lattice_bound + 1) ** 4
    if max_candidates is not None and n_candidates > max_candidates:
        raise ValueError(
            f"ammann_beenker_patch radius {radius} requires scanning "
            f"{n_candidates} lattice candidates, exceeding "
            f"max_candidates={max_candidates}"
        )

    selected_points: list[_PointRecord] = []
    lattice_range = range(-lattice_bound, lattice_bound + 1)
    for candidate in product(lattice_range, repeat=4):
        lattice_point: _LatticePoint = (
            candidate[0],
            candidate[1],
            candidate[2],
            candidate[3],
        )
        internal = _project(lattice_point, _INTERNAL_BASIS)
        if not _inside_acceptance_window(internal, tolerance=tolerance):
            continue
        physical = _project(lattice_point, _PHYSICAL_BASIS)
        if hypot(*physical) > radius + tolerance:
            continue
        selected_points.append((lattice_point, physical, internal))
        if max_sites is not None and len(selected_points) > max_sites:
            raise ValueError(
                f"ammann_beenker_patch radius {radius} requires more than "
                f"max_sites={max_sites}"
            )

    selected_points.sort(
        key=lambda record: (
            record[1][1],
            record[1][0],
            record[0],
        )
    )
    point_indices = {
        lattice_point: site
        for site, (lattice_point, _, _) in enumerate(selected_points)
    }
    coordinates = spacing * np.asarray(
        [physical for _, physical, _ in selected_points],
        dtype=float,
    )
    edges = _build_edges(
        selected_points=selected_points,
        point_indices=point_indices,
        spacing=spacing,
    )
    faces = _build_faces(
        selected_points=selected_points,
        point_indices=point_indices,
    )
    boundary_sites = _find_cut_boundary(
        selected_points=selected_points,
        point_indices=point_indices,
        radius=radius,
        tolerance=tolerance,
    )
    site_types = tuple(
        f"coordination_{_full_coordination(internal, tolerance=tolerance)}"
        for _, _, internal in selected_points
    )
    n_square_tiles = sum(face.face_type == "square" for face in faces)
    n_rhombus_tiles = sum(face.face_type == "rhombus_45_degree" for face in faces)

    return Geometry(
        n_sites=len(selected_points),
        edges=edges,
        coordinates=coordinates,
        boundary_sites=boundary_sites,
        boundary_components=(
            GeometryBoundaryComponent("outer", 0, boundary_sites),
        ),
        site_types=site_types,
        dimension_records=(
            GeometryDimension(
                kind="topological",
                value=2.0,
                scope="infinite_family",
                method="ammann_beenker_tiling_support_covering_dimension",
                exact=True,
            ),
        ),
        metadata={
            "generator": "ammann_beenker_patch",
            "family": "ammann_beenker_tiling",
            "construction": "canonical_cut_and_project",
            "source_lattice": "Z4",
            "acceptance_window": "centered_regular_octagon",
            "window_boundary_convention": "closed_with_numeric_tolerance",
            "window_apothem": _WINDOW_APOTHEM,
            "radius_in_edge_lengths": radius,
            "cutoff_radius": radius * spacing,
            "spacing": spacing,
            "inflation_factor": AMMANN_BEENKER_INFLATION_FACTOR,
            "n_square_tiles": n_square_tiles,
            "n_rhombus_tiles": n_rhombus_tiles,
            "face_representation": "complete_projected_hypercube_two_faces",
            "boundary_definition": "accepted_neighbor_outside_radial_cutoff",
            "lattice_index_bound": lattice_bound,
            "n_lattice_candidates": n_candidates,
            "max_sites": max_sites,
            "max_candidates": max_candidates,
        },
        faces=faces,
    )


def _validate_radius(radius: float) -> float:
    if isinstance(radius, bool) or not isinstance(radius, Real):
        raise TypeError("radius must be a real number")
    result = float(radius)
    if not np.isfinite(result) or result < _MINIMUM_TILING_RADIUS:
        raise ValueError("radius must be finite and at least two")
    return result


def _project(
    lattice_point: _LatticePoint,
    basis: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    return (
        sum(coefficient * vector[0] for coefficient, vector in zip(lattice_point, basis)),
        sum(coefficient * vector[1] for coefficient, vector in zip(lattice_point, basis)),
    )


def _inside_acceptance_window(
    internal: tuple[float, float],
    *,
    tolerance: float,
) -> bool:
    x_coordinate, y_coordinate = internal
    support = max(
        abs(x_coordinate),
        abs(y_coordinate),
        abs(x_coordinate + y_coordinate) * _INV_SQRT_TWO,
        abs(x_coordinate - y_coordinate) * _INV_SQRT_TWO,
    )
    return support <= _WINDOW_APOTHEM + tolerance


def _shift_lattice_point(
    lattice_point: _LatticePoint,
    axis: int,
    step: int,
) -> _LatticePoint:
    shifted = list(lattice_point)
    shifted[axis] += step
    return shifted[0], shifted[1], shifted[2], shifted[3]


def _build_edges(
    *,
    selected_points: list[_PointRecord],
    point_indices: dict[_LatticePoint, int],
    spacing: float,
) -> tuple[GeometryEdge, ...]:
    edges: list[GeometryEdge] = []
    for lattice_point, _, _ in selected_points:
        for axis, direction in enumerate(_PHYSICAL_BASIS):
            neighbor = _shift_lattice_point(lattice_point, axis, 1)
            if neighbor not in point_indices:
                continue
            edges.append(
                GeometryEdge(
                    point_indices[lattice_point],
                    point_indices[neighbor],
                    edge_type=f"ammann_beenker_axis_{axis}",
                    displacement=(spacing * direction[0], spacing * direction[1]),
                    metadata={
                        "projection_axis": axis,
                        "orientation_degrees": 45 * axis,
                    },
                )
            )
    return tuple(edges)


def _build_faces(
    *,
    selected_points: list[_PointRecord],
    point_indices: dict[_LatticePoint, int],
) -> tuple[GeometryFace, ...]:
    faces: list[GeometryFace] = []
    for lattice_point, _, _ in selected_points:
        for first_axis in range(4):
            first = _shift_lattice_point(lattice_point, first_axis, 1)
            if first not in point_indices:
                continue
            for second_axis in range(first_axis + 1, 4):
                second = _shift_lattice_point(lattice_point, second_axis, 1)
                opposite = _shift_lattice_point(first, second_axis, 1)
                if second not in point_indices or opposite not in point_indices:
                    continue
                is_square = second_axis - first_axis == 2
                faces.append(
                    GeometryFace(
                        sites=(
                            point_indices[lattice_point],
                            point_indices[first],
                            point_indices[opposite],
                            point_indices[second],
                        ),
                        face_type=("square" if is_square else "rhombus_45_degree"),
                        metadata={
                            "projection_axes": (first_axis, second_axis),
                            "interior_angles_degrees": (
                                (90, 90) if is_square else (45, 135)
                            ),
                        },
                    )
                )
    return tuple(faces)


def _find_cut_boundary(
    *,
    selected_points: list[_PointRecord],
    point_indices: dict[_LatticePoint, int],
    radius: float,
    tolerance: float,
) -> frozenset[int]:
    boundary_sites: set[int] = set()
    for lattice_point, physical, internal in selected_points:
        for axis in range(4):
            for step in (-1, 1):
                neighbor_internal = (
                    internal[0] + step * _INTERNAL_BASIS[axis][0],
                    internal[1] + step * _INTERNAL_BASIS[axis][1],
                )
                if not _inside_acceptance_window(
                    neighbor_internal,
                    tolerance=tolerance,
                ):
                    continue
                neighbor_physical = (
                    physical[0] + step * _PHYSICAL_BASIS[axis][0],
                    physical[1] + step * _PHYSICAL_BASIS[axis][1],
                )
                if hypot(*neighbor_physical) > radius + tolerance:
                    boundary_sites.add(point_indices[lattice_point])
                    break
            if point_indices[lattice_point] in boundary_sites:
                break
    return frozenset(boundary_sites)


def _full_coordination(
    internal: tuple[float, float],
    *,
    tolerance: float,
) -> int:
    return sum(
        _inside_acceptance_window(
            (
                internal[0] + step * _INTERNAL_BASIS[axis][0],
                internal[1] + step * _INTERNAL_BASIS[axis][1],
            ),
            tolerance=tolerance,
        )
        for axis in range(4)
        for step in (-1, 1)
    )
