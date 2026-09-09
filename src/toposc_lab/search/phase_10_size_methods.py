"""Frozen geometry and topology-input contract for TOPOSC-P10-SIZE-001."""

from __future__ import annotations

from collections import deque
from functools import cache
from typing import Any

import numpy as np

from toposc_lab.geometry import Geometry, GeometryBoundaryComponent, GeometryEdge
from toposc_lab.geometry.generators.square import square
from toposc_lab.geometry.validation import validate_geometry
from toposc_lab.search.geometry_genome import GeometryGenome
from toposc_lab.search.phase_9_8_evaluation import (
    PHASE_9_8_GEOMETRY_TOLERANCE,
    Phase98TopologyInputs,
    _boundary_graph_distances,
    _clipped_voronoi_areas,
    _segments_intersect,
)

SIZE_METHODS_PROTOCOL_ID = "TOPOSC-P10-SIZE-001"
SIZE_METHODS_PROTOCOL_COMMIT = "795124b0d89cf755f575a0de3049a2f3983aba1b"
SIZE_METHODS_PROTOCOL_PATH = "docs/decisions/pre_phase_10_size_methods_protocol_v1.md"
PREFLIGHT_ROOTS = (10_839_900, 10_839_901)
FULL_ROOTS = tuple(range(10_840_000, 10_840_016))
SIZES = (8, 10, 12)
POSITION_AMPLITUDE = 0.05
MASK_NAMES = (
    "graph_depth_2",
    "graph_depth_3",
    "fixed_depth_2",
    "fixed_depth_3",
    "central_fraction",
)


def edge_step_count(n: int) -> int:
    """Return the frozen approximately edge-density-matched swap count."""
    _require_size(n)
    edge_count = 2 * n * (n - 1)
    return (edge_count + 27) // 28


def size_reference(n: int) -> Geometry:
    """Return one neutral open square reference with the frozen outer boundary."""
    _require_size(n)
    base = square(n, n, spacing=1.0, boundary_x="open", boundary_y="open")
    coordinates = base.coordinates
    assert coordinates is not None
    pairs = tuple(sorted((edge.source, edge.target) for edge in base.edges))
    return build_size_geometry(n, coordinates, pairs)


def size_edge_pool(n: int) -> tuple[tuple[int, int], ...]:
    """Return the square king-neighborhood pool, including both diagonals."""
    coordinates = _base_coordinates(n)
    return tuple(
        (source, target)
        for source in range(n * n)
        for target in range(source + 1, n * n)
        if np.max(np.abs(coordinates[target] - coordinates[source])) <= 1.0
    )


def derive_size_schedule(root: int) -> dict[str, Any]:
    """Derive independent size roles, then a position and K step roles per size."""
    if type(root) is not int:
        raise TypeError("size-methods root seed must be an integer")
    size_seeds = tuple(int(value) for value in np.random.PCG64(root).random_raw(3))
    schedules = []
    for n, size_seed in zip(SIZES, size_seeds, strict=True):
        values = tuple(
            int(value) for value in np.random.PCG64(size_seed).random_raw(edge_step_count(n) + 1)
        )
        schedules.append(
            {
                "n": n,
                "size_seed": size_seed,
                "position_seed": values[0],
                "step_seeds": values[1:],
            }
        )
    return {"root": root, "sizes": schedules}


def size_position_field(n: int, seed: int) -> np.ndarray:
    """Draw a componentwise uniform field while pinning every outer site."""
    _require_size(n)
    field = np.zeros((n * n, 2), dtype=float)
    interior = tuple(x * n + y for x in range(1, n - 1) for y in range(1, n - 1))
    field[np.asarray(interior, dtype=np.intp)] = np.random.Generator(
        np.random.PCG64(seed)
    ).uniform(-1.0, 1.0, size=(len(interior), 2))
    field.setflags(write=False)
    return field


def build_size_geometry(
    n: int,
    coordinates: np.ndarray,
    edge_pairs: tuple[tuple[int, int], ...],
) -> Geometry:
    """Build one neutral snapshot with recomputed physical displacements."""
    _require_size(n)
    points = np.asarray(coordinates, dtype=float)
    boundary = frozenset(
        x * n + y
        for x in range(n)
        for y in range(n)
        if x in (0, n - 1) or y in (0, n - 1)
    )
    edges = tuple(
        GeometryEdge(
            source,
            target,
            edge_type="size_calibration_coupling",
            displacement=tuple(float(value) for value in points[target] - points[source]),
        )
        for source, target in sorted(edge_pairs)
    )
    return Geometry(
        n_sites=n * n,
        edges=edges,
        coordinates=points,
        boundary_sites=boundary,
        boundary_components=(GeometryBoundaryComponent("outer", 0, boundary),),
        dimension_records=(),
        metadata={"generator": "phase10_size_square_deformation_v1"},
    )


def validate_size_geometry(geometry: Geometry, n: int) -> dict[str, Any]:
    """Validate the complete variable-size resource and representation contract."""
    _require_size(n)
    issues: list[str] = []
    base = validate_geometry(geometry, require_connected=True)
    if not base.is_valid:
        issues.append("base_geometry_invalid")
    coordinates = geometry.coordinates
    expected_boundary = frozenset(
        x * n + y
        for x in range(n)
        for y in range(n)
        if x in (0, n - 1) or y in (0, n - 1)
    )
    expected_edges = 2 * n * (n - 1)
    if geometry.n_sites != n * n:
        issues.append("site_count")
    if geometry.n_edges != expected_edges:
        issues.append("edge_count")
    if coordinates is None or coordinates.shape != (n * n, 2):
        issues.append("coordinates")
        minimum_separation = None
        maximum_edge_length = None
        crossing_count = None
    else:
        differences = coordinates[:, None, :] - coordinates[None, :, :]
        distances = np.linalg.norm(differences, axis=2)
        np.fill_diagonal(distances, np.inf)
        minimum_separation = float(np.min(distances))
        maximum_edge_length = max(
            float(np.linalg.norm(coordinates[edge.target] - coordinates[edge.source]))
            for edge in geometry.edges
        )
        edge_pairs = tuple((edge.source, edge.target) for edge in geometry.edges)
        crossing_count = sum(
            _segments_intersect(
                coordinates[first[0]],
                coordinates[first[1]],
                coordinates[second[0]],
                coordinates[second[1]],
            )
            for index, first in enumerate(edge_pairs)
            for second in edge_pairs[index + 1 :]
            if not set(first).intersection(second)
        )
        if minimum_separation < 0.55:
            issues.append("minimum_site_separation")
        if maximum_edge_length > 1.75:
            issues.append("maximum_edge_length")
        if crossing_count:
            issues.append("straight_edge_crossing")
        if not (
            np.allclose(np.min(coordinates, axis=0), (0.0, 0.0), rtol=0.0,
                        atol=PHASE_9_8_GEOMETRY_TOLERANCE)
            and np.allclose(np.max(coordinates, axis=0), (n - 1, n - 1), rtol=0.0,
                            atol=PHASE_9_8_GEOMETRY_TOLERANCE)
        ):
            issues.append("bounding_box")
        shell = frozenset(
            site
            for site, (x_value, y_value) in enumerate(coordinates)
            if min(x_value, y_value, n - 1 - x_value, n - 1 - y_value) <= 0.875
        )
        if shell != expected_boundary:
            issues.append("boundary_shell")
        if any(
            edge.displacement is None
            or not np.allclose(
                np.asarray(edge.displacement),
                coordinates[edge.target] - coordinates[edge.source],
                rtol=0.0,
                atol=PHASE_9_8_GEOMETRY_TOLERANCE,
            )
            for edge in geometry.edges
        ):
            issues.append("edge_displacement")
    degrees = tuple(len(geometry.neighbors(site)) for site in range(geometry.n_sites))
    if min(degrees) < 2 or max(degrees) > 4:
        issues.append("degree_range")
    pairs = tuple((edge.source, edge.target) for edge in geometry.edges)
    if pairs != tuple(sorted(pairs)) or any(source >= target for source, target in pairs):
        issues.append("edge_order")
    if any(
        edge.edge_type != "size_calibration_coupling"
        or edge.boundary_crossing
        or bool(edge.metadata)
        for edge in geometry.edges
    ):
        issues.append("edge_semantics")
    if geometry.boundary_sites != expected_boundary:
        issues.append("boundary_membership")
    if (
        len(geometry.boundary_components) != 1
        or geometry.boundary_components[0]
        != GeometryBoundaryComponent("outer", 0, expected_boundary)
    ):
        issues.append("boundary_component")
    if (
        geometry.dimension_records
        or geometry.faces
        or geometry.rooted_tree is not None
        or geometry.site_types is not None
        or dict(geometry.metadata) != {"generator": "phase10_size_square_deformation_v1"}
    ):
        issues.append("neutral_metadata")
    return {
        "is_valid": not issues,
        "issues": tuple(dict.fromkeys(issues)),
        "measurements": {
            "site_count": geometry.n_sites,
            "edge_count": geometry.n_edges,
            "minimum_degree": min(degrees),
            "maximum_degree": max(degrees),
            "boundary_site_count": len(geometry.boundary_sites),
            "minimum_site_separation": minimum_separation,
            "maximum_edge_length": maximum_edge_length,
            "straight_edge_crossing_count": crossing_count,
            "base_valid": base.is_valid,
        },
    }


def legal_size_swaps(
    n: int, edge_pairs: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int, int, int], ...]:
    """Enumerate the lexicographically ordered legal single-edge swaps on X0."""
    _require_size(n)
    present = tuple(sorted(edge_pairs))
    present_set = set(present)
    absent = tuple(edge for edge in size_edge_pool(n) if edge not in present_set)
    adjacency: list[set[int]] = [set() for _ in range(n * n)]
    for source, target in present:
        adjacency[source].add(target)
        adjacency[target].add(source)
    degrees = tuple(len(neighbors) for neighbors in adjacency)
    crossings = _pool_crossings(n)
    proposals: list[tuple[int, int, int, int]] = []
    for removed_source, removed_target in present:
        removed = (removed_source, removed_target)
        retained = present_set - {removed}
        component = _reachable_without_edge(adjacency, removed)
        disconnected = len(component) != n * n
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
            if disconnected and ((added_source in component) == (added_target in component)):
                continue
            added = (added_source, added_target)
            if crossings[added].intersection(retained):
                continue
            proposals.append((*removed, *added))
    return tuple(proposals)


def build_size_trajectory(root: int) -> dict[str, Any]:
    """Construct all three paired variants at every frozen size before physics."""
    schedule = derive_size_schedule(root)
    size_plans = []
    for size_schedule in schedule["sizes"]:
        n = size_schedule["n"]
        field = size_position_field(n, size_schedule["position_seed"])
        base_coordinates = _base_coordinates(n)
        base_edges = _base_edges(n)
        current = base_edges
        operations = []
        for step, seed in enumerate(size_schedule["step_seeds"], 1):
            legal = legal_size_swaps(n, current)
            chosen = None
            if legal:
                selected = int(np.random.Generator(np.random.PCG64(seed)).integers(len(legal)))
                chosen = legal[selected]
                current = tuple(
                    sorted((set(current) - {(chosen[0], chosen[1])}) | {(chosen[2], chosen[3])})
                )
            geometry = build_size_geometry(n, base_coordinates, current)
            report = validate_size_geometry(geometry, n)
            if not report["is_valid"]:
                raise ValueError(
                    f"invalid size trajectory root={root}, n={n}, step={step}: "
                    f"{report['issues']}"
                )
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
                        _edge_boundary_class(n, (chosen[0], chosen[1])),
                        _edge_boundary_class(n, (chosen[2], chosen[3])),
                    ),
                    "degree_sequence": tuple(
                        len(geometry.neighbors(site)) for site in range(geometry.n_sites)
                    ),
                }
            )
        displaced = base_coordinates + POSITION_AMPLITUDE * field
        variants = (
            ("positions", POSITION_AMPLITUDE, 0, base_edges, displaced),
            ("connectivity", 0.0, edge_step_count(n), current, base_coordinates),
            ("combined", POSITION_AMPLITUDE, edge_step_count(n), current, displaced),
        )
        cells = []
        for block, amplitude, steps, edges, coordinates in variants:
            geometry = build_size_geometry(n, coordinates, edges)
            report = validate_size_geometry(geometry, n)
            if not report["is_valid"]:
                raise ValueError(
                    f"invalid size-methods cell root={root}, n={n}, block={block}: "
                    f"{report['issues']}"
                )
            cells.append(
                {
                    "block": block,
                    "amplitude": amplitude,
                    "edge_step": steps,
                    "edge_distance": len(set(edges) ^ set(base_edges)) // 2,
                    "genome": GeometryGenome.from_geometry(geometry),
                    "measurements": report["measurements"],
                    "degree_sequence": tuple(
                        len(geometry.neighbors(site)) for site in range(geometry.n_sites)
                    ),
                    "boundary_fraction": len(geometry.boundary_sites) / geometry.n_sites,
                    "edge_endpoint_class_counts": _endpoint_class_counts(n, edges),
                }
            )
        size_plans.append(
            {
                "n": n,
                "schedule": size_schedule,
                "position_field": field,
                "operations": operations,
                "cells": cells,
            }
        )
    return {"root": root, "schedule": schedule, "sizes": size_plans}


def ordered_size_cells(plans: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """Order controls and the three frozen blocks independently for every size."""
    result = []
    for n in SIZES:
        reference = GeometryGenome.from_geometry(size_reference(n))
        control = {
            "n": n,
            "root": None,
            "amplitude": 0.0,
            "edge_step": 0,
            "edge_distance": 0,
            "genome": reference,
            "measurements": validate_size_geometry(reference.to_geometry(), n)["measurements"],
            "degree_sequence": tuple(
                len(reference.to_geometry().neighbors(site)) for site in range(n * n)
            ),
            "boundary_fraction": len(reference.boundary_sites) / reference.n_sites,
            "edge_endpoint_class_counts": _endpoint_class_counts(n, _base_edges(n)),
        }
        result.append({"block": "control_start", **control})
        for block in ("positions", "connectivity", "combined"):
            for plan in plans:
                size_plan = next(item for item in plan["sizes"] if item["n"] == n)
                cell = next(item for item in size_plan["cells"] if item["block"] == block)
                result.append({"n": n, "root": plan["root"], **cell})
        result.append({"block": "control_end", **control})
    return tuple(result)


def build_size_topology_inputs(
    geometry: Geometry, n: int
) -> tuple[Phase98TopologyInputs, dict[str, Any]]:
    """Build the five frozen masks and exact variable-size numerical inputs."""
    report = validate_size_geometry(geometry, n)
    if not report["is_valid"]:
        raise ValueError(f"invalid size-methods geometry: {report['issues']}")
    coordinates = geometry.coordinates
    assert coordinates is not None
    areas = _clipped_voronoi_areas(
        coordinates, cell=(-0.5, n - 0.5, -0.5, n - 0.5)
    )
    if np.any(areas <= 0.0) or not np.isclose(
        float(np.sum(areas)), float(n * n), rtol=0.0, atol=1.0e-10
    ):
        raise ValueError("size-methods Voronoi areas are invalid")
    order = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
    graph_distances = _boundary_graph_distances(geometry)
    base = _base_coordinates(n)
    fixed_distances = np.min(
        np.column_stack((base[:, 0], base[:, 1], n - 1 - base[:, 0], n - 1 - base[:, 1])),
        axis=1,
    )
    masks = (
        graph_distances[order] >= 2,
        graph_distances[order] >= 3,
        fixed_distances[order] >= 2,
        fixed_distances[order] >= 3,
        fixed_distances[order] >= n / 4,
    )
    inputs = Phase98TopologyInputs(
        basis_coordinates=np.tile(coordinates, (2, 1)),
        position_areas=areas[order],
        bulk_masks=masks,
        bott_periods=((0.95 * n, 0.95 * n), (float(n), float(n)), (1.05 * n, 1.05 * n)),
        localizer_probe=((n - 1) / 2, (n - 1) / 2),
        localizer_kappas=(0.1, 0.2, 0.3),
    )
    evidence = {
        "mask_names": MASK_NAMES,
        "unique_coordinate_order": order,
        "position_areas_in_coordinate_order": inputs.position_areas,
        "bulk_masks_in_coordinate_order": inputs.bulk_masks,
        "bott_periods": inputs.bott_periods,
        "localizer_probe": inputs.localizer_probe,
        "localizer_kappas": inputs.localizer_kappas,
    }
    return inputs, evidence


def _require_size(n: int) -> None:
    if type(n) is not int or n not in SIZES:
        raise ValueError(f"n must be one of {SIZES}")


def _base_coordinates(n: int) -> np.ndarray:
    coordinates = square(n, n, spacing=1.0).coordinates
    assert coordinates is not None
    return coordinates.copy()


def _base_edges(n: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted((edge.source, edge.target) for edge in square(n, n, spacing=1.0).edges)
    )


def _reachable_without_edge(
    adjacency: list[set[int]], removed: tuple[int, int]
) -> frozenset[int]:
    visited = {0}
    queue = deque([0])
    while queue:
        site = queue.popleft()
        neighbors = adjacency[site]
        for neighbor in neighbors:
            if (site, neighbor) in (removed, removed[::-1]) or neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return frozenset(visited)


def _edge_boundary_class(n: int, edge: tuple[int, int]) -> str:
    def is_boundary(site: int) -> bool:
        x_value, y_value = divmod(site, n)
        return x_value in (0, n - 1) or y_value in (0, n - 1)

    count = int(is_boundary(edge[0])) + int(is_boundary(edge[1]))
    return ("interior/interior", "boundary/interior", "boundary/boundary")[count]


def _endpoint_class_counts(
    n: int, edges: tuple[tuple[int, int], ...]
) -> dict[str, int]:
    counts = {
        "interior/interior": 0,
        "boundary/interior": 0,
        "boundary/boundary": 0,
    }
    for edge in edges:
        counts[_edge_boundary_class(n, edge)] += 1
    return counts


@cache
def _pool_crossings(
    n: int,
) -> dict[tuple[int, int], frozenset[tuple[int, int]]]:
    coordinates = _base_coordinates(n)
    pool = size_edge_pool(n)
    result: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    for first in pool:
        result[first] = frozenset(
            second
            for second in pool
            if not set(first).intersection(second)
            and _segments_intersect(
                coordinates[first[0]],
                coordinates[first[1]],
                coordinates[second[0]],
                coordinates[second[1]],
            )
        )
    return result
