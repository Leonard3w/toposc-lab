"""Declarative artificial geometry growth from integer displacement rules."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

from toposc_lab.geometry.base import (
    Geometry,
    GeometryBoundaryComponent,
    GeometryEdge,
)
from toposc_lab.geometry.generators._validation import (
    validate_optional_budget,
    validate_recursion_order,
    validate_spacing,
)

DEFAULT_ARTIFICIAL_RULE_MAX_SITES = 100_000
DEFAULT_ARTIFICIAL_RULE_MAX_EDGES = 1_000_000
DEFAULT_ARTIFICIAL_RULE_MAX_APPLICATIONS = 2_000_000

_LatticePoint = tuple[int, ...]
_LatticeEdge = tuple[_LatticePoint, _LatticePoint]


def artificial_rule_graph(
    iterations: int,
    displacement_rules: ArrayLike,
    *,
    spacing: float = 1.0,
    max_sites: int | None = DEFAULT_ARTIFICIAL_RULE_MAX_SITES,
    max_edges: int | None = DEFAULT_ARTIFICIAL_RULE_MAX_EDGES,
    max_rule_applications: int | None = DEFAULT_ARTIFICIAL_RULE_MAX_APPLICATIONS,
) -> Geometry:
    """Grow an unnamed finite geometry from integer displacement rules.

    Growth starts at the integer origin and applies every rule breadth-first to
    each newly reached site. Rules must be unique, nonzero, and inverse closed,
    making the induced finite graph unambiguous and undirected. ``iterations``
    is the maximum rule-graph distance from the origin.
    """
    iterations = validate_recursion_order(iterations)
    rules = _validate_displacement_rules(displacement_rules)
    spacing = validate_spacing(spacing)
    max_sites = validate_optional_budget(max_sites, name="max_sites")
    max_edges = validate_optional_budget(max_edges, name="max_edges")
    max_rule_applications = validate_optional_budget(
        max_rule_applications,
        name="max_rule_applications",
    )

    origin = (0,) * len(rules[0])
    depths: dict[_LatticePoint, int] = {origin: 0}
    frontier: tuple[_LatticePoint, ...] = (origin,)
    lattice_edges: set[_LatticeEdge] = set()
    n_rule_applications = 0

    for depth in range(iterations):
        n_rule_applications = _consume_rule_application_budget(
            current=n_rule_applications,
            additional=len(frontier) * len(rules),
            max_rule_applications=max_rule_applications,
        )
        next_frontier: set[_LatticePoint] = set()
        for source in frontier:
            for rule in rules:
                target = _translate(source, rule)
                if target not in depths:
                    if max_sites is not None and len(depths) >= max_sites:
                        raise ValueError(
                            "artificial_rule_graph requires more than "
                            f"max_sites={max_sites} at iteration {depth + 1}"
                        )
                    depths[target] = depth + 1
                    next_frontier.add(target)
                _add_lattice_edge(
                    source,
                    target,
                    lattice_edges=lattice_edges,
                    max_edges=max_edges,
                )
        frontier = tuple(sorted(next_frontier))

    n_rule_applications = _consume_rule_application_budget(
        current=n_rule_applications,
        additional=len(frontier) * len(rules),
        max_rule_applications=max_rule_applications,
    )
    for source in frontier:
        for rule in rules:
            target = _translate(source, rule)
            if target in depths:
                _add_lattice_edge(
                    source,
                    target,
                    lattice_edges=lattice_edges,
                    max_edges=max_edges,
                )

    ordered_points = tuple(sorted(depths, key=lambda point: (depths[point], point)))
    point_indices = {point: site for site, point in enumerate(ordered_points)}
    coordinates = _scaled_coordinates(ordered_points, spacing=spacing)
    canonical_families = tuple(sorted({_canonical_rule(rule) for rule in rules}))
    family_indices = {
        family: family_index
        for family_index, family in enumerate(canonical_families)
    }
    indexed_edges = sorted(
        (
            min(point_indices[first], point_indices[second]),
            max(point_indices[first], point_indices[second]),
            first,
            second,
        )
        for first, second in lattice_edges
    )
    edges = tuple(
        _build_geometry_edge(
            source,
            target,
            first_point=first_point,
            second_point=second_point,
            ordered_points=ordered_points,
            family_indices=family_indices,
            spacing=spacing,
        )
        for source, target, first_point, second_point in indexed_edges
    )
    boundary_sites = frozenset(
        point_indices[point]
        for point, depth in depths.items()
        if depth == iterations
    )
    shell_populations = tuple(
        sum(depth == shell for depth in depths.values())
        for shell in range(iterations + 1)
    )

    return Geometry(
        n_sites=len(ordered_points),
        edges=edges,
        coordinates=coordinates,
        boundary_sites=boundary_sites,
        boundary_components=(
            GeometryBoundaryComponent("outer", 0, boundary_sites),
        ),
        metadata={
            "generator": "artificial_rule_graph",
            "family": "unnamed_integer_displacement_rule_geometry",
            "construction": "breadth_first_displacement_rules",
            "iterations": iterations,
            "displacement_rules": rules,
            "canonical_displacement_families": canonical_families,
            "embedding_dimension": len(origin),
            "spacing": spacing,
            "shell_populations": shell_populations,
            "boundary_definition": "maximum_rule_distance_shell",
            "rule_order": "lexicographic_normalized",
            "n_rule_applications": n_rule_applications,
            "max_sites": max_sites,
            "max_edges": max_edges,
            "max_rule_applications": max_rule_applications,
        },
    )


def _validate_displacement_rules(rules: ArrayLike) -> tuple[_LatticePoint, ...]:
    try:
        source = np.asarray(rules, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError("displacement_rules must be a rectangular integer array") from error
    if source.ndim != 2:
        raise ValueError("displacement_rules must be a two-dimensional array")
    if source.shape[0] < 1:
        raise ValueError("displacement_rules must contain at least one rule")
    if source.shape[1] < 1:
        raise ValueError("displacement rules must contain at least one dimension")

    normalized_rules: list[_LatticePoint] = []
    for row in source:
        normalized_rule: list[int] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("displacement rule components must be integers")
            normalized_rule.append(int(value))
        rule = tuple(normalized_rule)
        if not any(rule):
            raise ValueError("displacement rules must not contain the zero vector")
        normalized_rules.append(rule)

    if len(set(normalized_rules)) != len(normalized_rules):
        raise ValueError("displacement rules must be unique")
    rule_set = set(normalized_rules)
    for rule in normalized_rules:
        inverse = tuple(-component for component in rule)
        if inverse not in rule_set:
            raise ValueError(
                f"displacement rule {rule} is missing inverse rule {inverse}"
            )
    return tuple(sorted(normalized_rules))


def _consume_rule_application_budget(
    *,
    current: int,
    additional: int,
    max_rule_applications: int | None,
) -> int:
    result = current + additional
    if max_rule_applications is not None and result > max_rule_applications:
        raise ValueError(
            f"artificial_rule_graph requires at least {result} rule applications, "
            f"exceeding max_rule_applications={max_rule_applications}"
        )
    return result


def _translate(point: _LatticePoint, rule: _LatticePoint) -> _LatticePoint:
    return tuple(coordinate + displacement for coordinate, displacement in zip(point, rule))


def _canonical_rule(rule: _LatticePoint) -> _LatticePoint:
    inverse = tuple(-component for component in rule)
    return min(rule, inverse)


def _canonical_lattice_edge(
    first: _LatticePoint,
    second: _LatticePoint,
) -> _LatticeEdge:
    return (first, second) if first < second else (second, first)


def _add_lattice_edge(
    source: _LatticePoint,
    target: _LatticePoint,
    *,
    lattice_edges: set[_LatticeEdge],
    max_edges: int | None,
) -> None:
    edge = _canonical_lattice_edge(source, target)
    if edge in lattice_edges:
        return
    if max_edges is not None and len(lattice_edges) >= max_edges:
        raise ValueError(
            f"artificial_rule_graph requires more than max_edges={max_edges}"
        )
    lattice_edges.add(edge)


def _scaled_coordinates(
    ordered_points: tuple[_LatticePoint, ...],
    *,
    spacing: float,
) -> np.ndarray:
    try:
        coordinates = spacing * np.asarray(ordered_points, dtype=float)
    except (OverflowError, ValueError) as error:
        raise ValueError("generated coordinates exceed floating-point range") from error
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("generated coordinates exceed floating-point range")
    return coordinates


def _build_geometry_edge(
    source: int,
    target: int,
    *,
    first_point: _LatticePoint,
    second_point: _LatticePoint,
    ordered_points: tuple[_LatticePoint, ...],
    family_indices: dict[_LatticePoint, int],
    spacing: float,
) -> GeometryEdge:
    source_point = ordered_points[source]
    target_point = ordered_points[target]
    displacement_rule = tuple(
        second - first for first, second in zip(first_point, second_point)
    )
    family = _canonical_rule(displacement_rule)
    displacement = tuple(
        spacing * (target_coordinate - source_coordinate)
        for source_coordinate, target_coordinate in zip(source_point, target_point)
    )
    return GeometryEdge(
        source,
        target,
        edge_type=f"rule_displacement_{family_indices[family]}",
        displacement=displacement,
        metadata={
            "canonical_displacement_rule": family,
        },
    )
