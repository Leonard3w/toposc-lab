"""Frozen geometry contract for TOPOSC-P10-EDGE-LOCATION-001."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.search.geometry_genome import GeometryGenome
from toposc_lab.search.phase_10_size_methods import (
    build_size_geometry,
    size_edge_pool,
    size_reference,
    validate_size_geometry,
)

EDGE_LOCATION_PROTOCOL_ID = "TOPOSC-P10-EDGE-LOCATION-001"
EDGE_LOCATION_PROTOCOL_COMMIT = "acf122f9f8cc4e8848f3508766c127b435083f6e"
EDGE_LOCATION_PROTOCOL_PATH = "docs/decisions/pre_phase_10_edge_location_protocol_v1.md"
N = 12
ANCHORS = (3, 4, 5, 6, 7)
ARMS: tuple[Literal["boundary", "interior"], ...] = ("boundary", "interior")
PREFLIGHT_PAIR_IDS = (2, 7)
FULL_PAIR_IDS = tuple(range(40))

Point = tuple[int, int]
Edge = tuple[int, int]


def pair_definition(pair_id: int, arm: Literal["boundary", "interior"]) -> dict[str, Any]:
    """Return one exact transformed intervention and its fixed local patch."""
    if type(pair_id) is not int or pair_id not in FULL_PAIR_IDS:
        raise ValueError("pair_id must be an integer from 0 through 39")
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    transform_index, anchor_index = divmod(pair_id, len(ANCHORS))
    mirror, rotation = divmod(transform_index, 4)
    anchor_y = ANCHORS[anchor_index]
    anchor_x = 0 if arm == "boundary" else 3
    base_points = (
        (anchor_x, anchor_y),
        (anchor_x + 1, anchor_y),
        (anchor_x + 1, anchor_y + 1),
        (anchor_x + 2, anchor_y + 1),
    )
    points = tuple(_transform(point, mirror, rotation) for point in base_points)
    removed = tuple(sorted((_edge(points[0], points[1]), _edge(points[2], points[3]))))
    added = tuple(sorted((_edge(points[0], points[2]), _edge(points[1], points[3]))))
    patch = tuple(
        sorted(
            _site(_transform((anchor_x + u, anchor_y + v), mirror, rotation))
            for u in range(4)
            for v in range(-1, 3)
        )
    )
    return {
        "pair_id": pair_id,
        "arm": arm,
        "mirror": mirror,
        "rotation": rotation,
        "anchor_y": anchor_y,
        "affected_sites": tuple(sorted(_site(point) for point in points)),
        "removed_edges": removed,
        "added_edges": added,
        "measurement_sites": patch,
        "intervention_depth": min(_depth(point) for point in points),
    }


def build_edge_location_cell(
    pair_id: int, arm: Literal["boundary", "interior"]
) -> dict[str, Any]:
    """Build and fully audit one degree-preserving edge-location cell."""
    definition = pair_definition(pair_id, arm)
    reference = size_reference(N)
    coordinates = reference.coordinates
    assert coordinates is not None
    reference_edges = tuple((edge.source, edge.target) for edge in reference.edges)
    edge_set = set(reference_edges)
    removed = set(definition["removed_edges"])
    added = set(definition["added_edges"])
    if not removed.issubset(edge_set) or added.intersection(edge_set):
        raise ValueError("intervention does not replace exactly the frozen square edges")
    if not added.issubset(set(size_edge_pool(N))):
        raise ValueError("intervention adds an edge outside the frozen local pool")
    final_edges = tuple(sorted((edge_set - removed) | added))
    geometry = build_size_geometry(N, coordinates, final_edges)
    validation = validate_size_geometry(geometry, N)
    if not validation["is_valid"]:
        raise ValueError(f"invalid edge-location geometry: {validation['issues']}")
    if geometry.coordinates is None or not np.array_equal(geometry.coordinates, coordinates):
        raise ValueError("edge-location coordinates differ from the reference")
    degree_sequence = _degrees(geometry)
    if degree_sequence != _degrees(reference):
        raise ValueError("edge-location intervention changed the degree sequence")
    expected_depth = 0 if arm == "boundary" else 3
    if definition["intervention_depth"] != expected_depth:
        raise ValueError("edge-location intervention has the wrong frozen depth")
    if len(definition["measurement_sites"]) != 16:
        raise ValueError("edge-location measurement patch must contain 16 sites")
    removed_lengths = _lengths(coordinates, definition["removed_edges"])
    added_lengths = _lengths(coordinates, definition["added_edges"])
    removed_displacements = _displacements(coordinates, definition["removed_edges"])
    added_displacements = _displacements(coordinates, definition["added_edges"])
    if removed_lengths != (1.0, 1.0) or not np.allclose(
        added_lengths, (np.sqrt(2.0), np.sqrt(2.0)), rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("edge-location intervention lengths differ from the protocol")
    return {
        **definition,
        "n": N,
        "block": "edge_location",
        "genome": GeometryGenome.from_geometry(geometry),
        "geometry_id": exact_geometry_id(geometry),
        "validation": validation,
        "degree_sequence": degree_sequence,
        "edge_distance": len(edge_set.symmetric_difference(final_edges)) // 2,
        "removed_edge_lengths": removed_lengths,
        "added_edge_lengths": added_lengths,
        "removed_edge_displacements": removed_displacements,
        "added_edge_displacements": added_displacements,
        "evaluation_seed": None,
    }


def build_edge_location_plan(*, preflight: bool) -> tuple[dict[str, Any], ...]:
    """Return the frozen stage order, including its two square controls."""
    pair_ids = PREFLIGHT_PAIR_IDS if preflight else FULL_PAIR_IDS
    reference = size_reference(N)
    control = {
        "n": N,
        "pair_id": None,
        "arm": None,
        "mirror": None,
        "rotation": None,
        "anchor_y": None,
        "affected_sites": (),
        "removed_edges": (),
        "added_edges": (),
        "measurement_sites": (),
        "intervention_depth": None,
        "genome": GeometryGenome.from_geometry(reference),
        "geometry_id": exact_geometry_id(reference),
        "validation": validate_size_geometry(reference, N),
        "degree_sequence": _degrees(reference),
        "edge_distance": 0,
        "removed_edge_lengths": (),
        "added_edge_lengths": (),
        "removed_edge_displacements": (),
        "added_edge_displacements": (),
        "evaluation_seed": None,
    }
    cells: list[dict[str, Any]] = [{"block": "control_start", **control}]
    for pair_id in pair_ids:
        arms = ARMS if pair_id % 2 == 0 else tuple(reversed(ARMS))
        cells.extend(build_edge_location_cell(pair_id, arm) for arm in arms)
    cells.append({"block": "control_end", **control})
    expected = 6 if preflight else 82
    if len(cells) != expected:
        raise AssertionError("edge-location plan differs from the frozen budget")
    return tuple(cells)


def _transform(point: Point, mirror: int, rotation: int) -> Point:
    x_value, y_value = point
    if mirror:
        y_value = N - 1 - y_value
    for _ in range(rotation):
        x_value, y_value = N - 1 - y_value, x_value
    if not 0 <= x_value < N or not 0 <= y_value < N:
        raise ValueError("transformed edge-location point lies outside the square")
    return x_value, y_value


def _site(point: Point) -> int:
    return point[0] * N + point[1]


def _edge(first: Point, second: Point) -> Edge:
    first_site, second_site = _site(first), _site(second)
    return (first_site, second_site) if first_site < second_site else (second_site, first_site)


def _depth(point: Point) -> int:
    x_value, y_value = point
    return min(x_value, y_value, N - 1 - x_value, N - 1 - y_value)


def _degrees(geometry: Any) -> tuple[int, ...]:
    return tuple(len(geometry.neighbors(site)) for site in range(geometry.n_sites))


def _lengths(coordinates: np.ndarray, edges: tuple[Edge, ...]) -> tuple[float, ...]:
    return tuple(
        sorted(
            float(np.linalg.norm(coordinates[target] - coordinates[source]))
            for source, target in edges
        )
    )


def _displacements(
    coordinates: np.ndarray, edges: tuple[Edge, ...]
) -> tuple[tuple[float, float], ...]:
    result = []
    for source, target in edges:
        displacement = coordinates[target] - coordinates[source]
        result.append((float(displacement[0]), float(displacement[1])))
    return tuple(result)
