"""Frozen geometry construction for TOPOSC-P10-CAL-001."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from toposc_lab.geometry import Geometry, GeometryBoundaryComponent, GeometryEdge
from toposc_lab.search.geometry_genome import GeometryGenome
from toposc_lab.search.phase_9_8_evaluation import (
    Phase98GeometryApplicability,
    _segments_intersect,
    validate_phase_9_8_geometry,
)
from toposc_lab.search.phase_10_research import square_reference

CALIBRATION_PROTOCOL_ID = "TOPOSC-P10-CAL-001"
CALIBRATION_PROTOCOL_COMMIT = "a1c5a087ed1383f496ebe6a14e1bc7eecaea568a"
CALIBRATION_PROTOCOL_PATH = "docs/decisions/pre_phase_10_calibration_protocol_v1.md"
PREFLIGHT_ROOTS = (10_829_900, 10_829_901)
FULL_ROOTS = tuple(range(10_830_000, 10_830_016))
AMPLITUDES = (0.0, 0.005, 0.025, 0.05, 0.1)
EDGE_STEPS = (0, 1, 2, 4, 8)


def calibration_edge_pool() -> tuple[tuple[int, int], ...]:
    """Return the symmetric, triangulation-neutral 8x8 square neighborhood."""
    coordinates = _base_coordinates()
    return tuple(
        (source, target)
        for source in range(64)
        for target in range(source + 1, 64)
        if np.max(np.abs(coordinates[target] - coordinates[source])) <= 1.0
    )


def derive_calibration_schedule(root: int) -> dict[str, Any]:
    """Derive the one position seed followed by eight edge-step seeds."""
    if type(root) is not int:
        raise TypeError("calibration root seed must be an integer")
    values = tuple(int(value) for value in np.random.PCG64(root).random_raw(9))
    return {"root": root, "position_seed": values[0], "step_seeds": values[1:]}


def calibration_position_field(seed: int) -> np.ndarray:
    """Draw displacements only for the 36 non-boundary sites."""
    base = square_reference()
    field = np.zeros((base.n_sites, 2), dtype=float)
    interior = sorted(set(range(base.n_sites)) - set(base.boundary_sites))
    field[interior] = np.random.Generator(np.random.PCG64(seed)).uniform(
        -1.0, 1.0, size=(len(interior), 2)
    )
    field.setflags(write=False)
    return field


def build_calibration_geometry(
    coordinates: np.ndarray, edge_pairs: tuple[tuple[int, int], ...]
) -> Geometry:
    """Build one neutral snapshot and recompute every physical displacement."""
    points = np.asarray(coordinates, dtype=float)
    base = square_reference()
    edges = tuple(
        GeometryEdge(
            source,
            target,
            edge_type="calibration_coupling",
            displacement=tuple(float(x) for x in points[target] - points[source]),
        )
        for source, target in sorted(edge_pairs)
    )
    return Geometry(
        n_sites=64,
        edges=edges,
        coordinates=points,
        boundary_sites=base.boundary_sites,
        boundary_components=(GeometryBoundaryComponent("outer", 0, base.boundary_sites),),
        dimension_records=(),
        metadata={"generator": "phase10_calibration_square_deformation_v1"},
    )


def legal_calibration_swaps(
    edge_pairs: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int, int, int], ...]:
    """Enumerate all swaps satisfying the complete clean contract on X0."""
    present = tuple(sorted(edge_pairs))
    present_set = set(present)
    absent = tuple(edge for edge in calibration_edge_pool() if edge not in present_set)
    degrees = [0] * 64
    adjacency: list[set[int]] = [set() for _ in range(64)]
    for source, target in present:
        degrees[source] += 1
        degrees[target] += 1
        adjacency[source].add(target)
        adjacency[target].add(source)
    proposals: list[tuple[int, int, int, int]] = []
    coordinates = _base_coordinates()
    for removed_source, removed_target in present:
        for added_source, added_target in absent:
            changes: dict[int, int] = {}
            for site, delta in (
                (removed_source, -1),
                (removed_target, -1),
                (added_source, 1),
                (added_target, 1),
            ):
                changes[site] = changes.get(site, 0) + delta
            if any(not 2 <= degrees[site] + delta <= 4 for site, delta in changes.items()):
                continue
            if not _connected_after_swap(
                adjacency,
                (removed_source, removed_target),
                (added_source, added_target),
            ):
                continue
            retained = present_set - {(removed_source, removed_target)}
            if any(
                not {added_source, added_target}.intersection(edge)
                and _segments_intersect(
                    coordinates[added_source],
                    coordinates[added_target],
                    coordinates[edge[0]],
                    coordinates[edge[1]],
                )
                for edge in retained
            ):
                continue
            proposals.append((removed_source, removed_target, added_source, added_target))
    return tuple(proposals)


def build_calibration_trajectory(root: int) -> dict[str, Any]:
    """Construct and validate the complete paired 5x5 trajectory before physics."""
    schedule = derive_calibration_schedule(root)
    field = calibration_position_field(schedule["position_seed"])
    base = square_reference()
    base_edges = tuple(sorted((edge.source, edge.target) for edge in base.edges))
    edge_states: dict[int, tuple[tuple[int, int], ...]] = {0: base_edges}
    operations = []
    current = base_edges
    for step, seed in enumerate(schedule["step_seeds"], 1):
        legal = legal_calibration_swaps(current)
        chosen = None
        if legal:
            index = int(np.random.Generator(np.random.PCG64(seed)).integers(len(legal)))
            chosen = legal[index]
            removed = (chosen[0], chosen[1])
            added = (chosen[2], chosen[3])
            current = tuple(sorted((set(current) - {removed}) | {added}))
        degrees = [0] * 64
        for source, target in current:
            degrees[source] += 1
            degrees[target] += 1
        operations.append(
            {
                "step": step,
                "seed": seed,
                "legal_swap_count": len(legal),
                "chosen_swap": chosen,
                "reason": "selected" if chosen is not None else "no_legal_swap",
                "affected_sites": () if chosen is None else tuple(sorted(set(chosen))),
                "endpoint_classes": None
                if chosen is None
                else (
                    _edge_boundary_class((chosen[0], chosen[1])),
                    _edge_boundary_class((chosen[2], chosen[3])),
                ),
                "degree_sequence": tuple(degrees),
            }
        )
        if step in EDGE_STEPS:
            edge_states[step] = current
    cells = []
    for amplitude in AMPLITUDES:
        coordinates = (
            _base_coordinates()
            if amplitude == 0.0
            else _base_coordinates() + amplitude * field
        )
        for edge_step in EDGE_STEPS:
            geometry = build_calibration_geometry(coordinates, edge_states[edge_step])
            report = validate_phase_9_8_geometry(
                geometry, applicability=Phase98GeometryApplicability.CLEAN_PRIMARY
            )
            if not report.is_applicable:
                raise ValueError(
                    f"invalid calibration cell root={root}, a={amplitude}, k={edge_step}: "
                    f"{report.issues}"
                )
            distance = len(set(edge_states[edge_step]) ^ set(base_edges)) // 2
            cells.append(
                {
                    "amplitude": amplitude,
                    "edge_step": edge_step,
                    "edge_distance": distance,
                    "genome": GeometryGenome.from_geometry(geometry),
                    "measurements": dict(report.measurements),
                }
            )
    return {
        "schedule": schedule,
        "position_field": field,
        "operations": operations,
        "cells": cells,
    }


def ordered_calibration_cells(plans: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """Apply the frozen control/positions/connectivity/combined ordering."""
    base = build_calibration_geometry(_base_coordinates(), _base_edges())
    result = [{"block": "control_start", "root": None, "amplitude": 0.0,
               "edge_step": 0, "edge_distance": 0,
               "genome": GeometryGenome.from_geometry(base)}]
    definitions: tuple[tuple[str, Callable[[float, int], bool]], ...] = (
        ("positions", lambda a, k: a > 0.0 and k == 0),
        ("connectivity", lambda a, k: a == 0.0 and k > 0),
        ("combined", lambda a, k: a > 0.0 and k > 0),
    )
    for block, predicate in definitions:
        for plan in plans:
            root = plan["schedule"]["root"]
            for cell in plan["cells"]:
                if predicate(cell["amplitude"], cell["edge_step"]):
                    result.append({"block": block, "root": root, **cell})
    result.append({"block": "control_end", "root": None, "amplitude": 0.0,
                   "edge_step": 0, "edge_distance": 0,
                   "genome": GeometryGenome.from_geometry(base)})
    return tuple(result)


def _base_coordinates() -> np.ndarray:
    coordinates = square_reference().coordinates
    assert coordinates is not None
    return coordinates.copy()


def _base_edges() -> tuple[tuple[int, int], ...]:
    return tuple(sorted((edge.source, edge.target) for edge in square_reference().edges))


def _connected_after_swap(
    adjacency: list[set[int]], removed: tuple[int, int], added: tuple[int, int]
) -> bool:
    visited = {0}
    queue = deque([0])
    while queue:
        site = queue.popleft()
        neighbors = set(adjacency[site])
        if site == removed[0]:
            neighbors.discard(removed[1])
        if site == removed[1]:
            neighbors.discard(removed[0])
        if site == added[0]:
            neighbors.add(added[1])
        if site == added[1]:
            neighbors.add(added[0])
        for neighbor in neighbors - visited:
            visited.add(neighbor)
            queue.append(neighbor)
    return len(visited) == 64


def _edge_boundary_class(edge: tuple[int, int]) -> str:
    boundary = square_reference().boundary_sites
    count = int(edge[0] in boundary) + int(edge[1] in boundary)
    return ("interior/interior", "boundary/interior", "boundary/boundary")[count]
