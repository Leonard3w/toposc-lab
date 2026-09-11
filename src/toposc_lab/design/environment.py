"""Finite constructive episodes with continuous point proposals and exact edge masks.

Free placement selects from freshly sampled continuous proposals, not from a lattice.
Proposal exhaustion is an explicit failed episode, never a hidden retry or repair.
No boundary or topological meaning is inferred from these engineering geometries.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import hypot, isfinite

import numpy as np
from numpy.typing import NDArray

from toposc_lab.geometry import Geometry, GeometryEdge

Point = tuple[float, float]
Edge = tuple[int, int]
FEATURE_COUNT = 12


@dataclass(frozen=True)
class BuildRules:
    site_count: int = 12
    edge_count: int = 16
    width: float = 4.0
    height: float = 4.0
    minimum_separation: float = 0.4
    maximum_edge_length: float = 2.0
    minimum_degree: int = 1
    maximum_degree: int = 4
    point_proposals: int = 32
    fixed_points: tuple[Point, ...] = ()
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        for name in (
            "site_count",
            "edge_count",
            "minimum_degree",
            "maximum_degree",
            "point_proposals",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.site_count < 2 or self.point_proposals < 1:
            raise ValueError("need at least two sites and one proposal")
        if not 0 <= self.minimum_degree <= self.maximum_degree < self.site_count:
            raise ValueError("invalid degree bounds")
        if not self.site_count - 1 <= self.edge_count <= 3 * self.site_count - 6 and not (
            self.site_count == 2 and self.edge_count == 1
        ):
            raise ValueError("edge budget cannot describe a connected simple planar graph")
        if not (
            self.site_count * self.minimum_degree
            <= 2 * self.edge_count
            <= self.site_count * self.maximum_degree
        ):
            raise ValueError("degree bounds and edge budget disagree")
        for name in ("width", "height", "minimum_separation", "maximum_edge_length", "tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.tolerance >= min(self.minimum_separation, self.maximum_edge_length):
            raise ValueError("tolerance must be smaller than length constraints")
        points = tuple(tuple(float(v) for v in p) for p in self.fixed_points)
        if any(len(p) != 2 for p in points):
            raise ValueError("points need two coordinates")
        object.__setattr__(self, "fixed_points", points)
        if points:
            if len(points) != self.site_count:
                raise ValueError("fixed points must match site_count")
            if any(not self.contains(p) for p in self.fixed_points):
                raise ValueError("fixed point outside domain")
            if any(
                hypot(a[0] - b[0], a[1] - b[1]) < self.minimum_separation
                for a, b in combinations(self.fixed_points, 2)
            ):
                raise ValueError("fixed points violate minimum separation")

    def contains(self, point: Point) -> bool:
        return (
            all(isfinite(v) for v in point)
            and 0 <= point[0] <= self.width
            and 0 <= point[1] <= self.height
        )


@dataclass(frozen=True)
class BuildState:
    points: tuple[Point, ...] = ()
    edges: tuple[Edge, ...] = ()
    failure: str | None = None

    def geometry(self) -> Geometry:
        """Materialize a snapshot without inventing a physical boundary."""
        return Geometry(
            n_sites=len(self.points),
            coordinates=np.asarray(self.points, dtype=float),
            embedding_dimension=2,
            edges=tuple(
                GeometryEdge(
                    a,
                    b,
                    displacement=(
                        self.points[b][0] - self.points[a][0],
                        self.points[b][1] - self.points[a][1],
                    ),
                )
                for a, b in sorted(self.edges)
            ),
        )


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point, tolerance: float) -> bool:
    return abs(_orientation(a, b, p)) <= tolerance and all(
        min(a[k], b[k]) - tolerance <= p[k] <= max(a[k], b[k]) + tolerance for k in (0, 1)
    )


def _intersects(a: Point, b: Point, c: Point, d: Point, tolerance: float) -> bool:
    turns = (
        _orientation(a, b, c),
        _orientation(a, b, d),
        _orientation(c, d, a),
        _orientation(c, d, b),
    )
    return (
        (turns[0] * turns[1] < -(tolerance**2) and turns[2] * turns[3] < -(tolerance**2))
        or _on_segment(a, b, c, tolerance)
        or _on_segment(a, b, d, tolerance)
        or _on_segment(c, d, a, tolerance)
        or _on_segment(c, d, b, tolerance)
    )


def components(count: int, edges: tuple[Edge, ...]) -> tuple[int, ...]:
    parents = list(range(count))

    def root(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for a, b in edges:
        parents[root(a)] = root(b)
    return tuple(root(i) for i in range(count))


class GraphBuilder:
    """Replaceable environment; all state transitions are explicit and immutable."""

    identifier = "continuous_proposal_planar_builder_v1"

    def __init__(self, rules: BuildRules) -> None:
        self.rules = rules
        self._inventories: dict[
            tuple[Point, ...], tuple[tuple[Edge, ...], dict[Edge, frozenset[Edge]]]
        ] = {}

    def initial(self) -> BuildState:
        return BuildState(points=self.rules.fixed_points)

    def stage(self, state: BuildState) -> str:
        if state.failure:
            return "failed"
        if len(state.points) < self.rules.site_count:
            return "points"
        return "complete" if len(state.edges) == self.rules.edge_count else "edges"

    def choices(self, state: BuildState, rng: np.random.Generator) -> tuple[Point | Edge, ...]:
        stage = self.stage(state)
        if stage == "points":
            proposals = rng.uniform(
                (0.0, 0.0),
                (self.rules.width, self.rules.height),
                size=(self.rules.point_proposals, 2),
            )
            return tuple(
                (float(x), float(y))
                for x, y in proposals
                if all(
                    hypot(x - a, y - b) >= self.rules.minimum_separation for a, b in state.points
                )
            )
        if stage == "edges":
            return tuple(
                edge
                for edge in self._local_edges(state)
                if self._can_finish(BuildState(state.points, (*state.edges, edge)))
            )
        return ()

    def _local_edges(self, state: BuildState) -> tuple[Edge, ...]:
        degrees = [0] * len(state.points)
        for a, b in state.edges:
            degrees[a] += 1
            degrees[b] += 1
        inventory, conflicts = self._inventory(state.points)
        present = set(state.edges)
        return tuple(
            (a, b)
            for a, b in inventory
            if (a, b) not in present
            and max(degrees[a], degrees[b]) < self.rules.maximum_degree
            and not conflicts[(a, b)].intersection(present)
        )

    def _inventory(
        self, points: tuple[Point, ...]
    ) -> tuple[tuple[Edge, ...], dict[Edge, frozenset[Edge]]]:
        """Compute length/site/crossing geometry once per point set, not per action."""
        if points in self._inventories:
            return self._inventories[points]
        accepted: list[Edge] = []
        for a, b in combinations(range(len(points)), 2):
            start, end = points[a], points[b]
            if hypot(start[0] - end[0], start[1] - end[1]) > self.rules.maximum_edge_length:
                continue
            # Also forbid a segment passing through an unconnected third site.
            if any(
                _on_segment(start, end, p, self.rules.tolerance)
                for i, p in enumerate(points)
                if i not in (a, b)
            ):
                continue
            accepted.append((a, b))
        conflicts: dict[Edge, set[Edge]] = {e: set() for e in accepted}
        for first, second in combinations(accepted, 2):
            if set(first).intersection(second):
                continue
            if _intersects(
                points[first[0]],
                points[first[1]],
                points[second[0]],
                points[second[1]],
                self.rules.tolerance,
            ):
                conflicts[first].add(second)
                conflicts[second].add(first)
        result = tuple(accepted), {e: frozenset(v) for e, v in conflicts.items()}
        if len(self._inventories) >= 4:
            self._inventories.pop(next(iter(self._inventories)))
        self._inventories[points] = result
        return result

    def _can_finish(self, state: BuildState) -> bool:
        """Necessary conditions only: never claim a complete feasibility solver."""
        remaining = self.rules.edge_count - len(state.edges)
        labels = components(len(state.points), state.edges)
        if len(set(labels)) - 1 > remaining:
            return False
        possible = self._local_edges(state) if remaining else ()
        if len(possible) < remaining:
            return False
        degrees = [0] * len(state.points)
        available = [0] * len(state.points)
        for a, b in state.edges:
            degrees[a] += 1
            degrees[b] += 1
        for a, b in possible:
            available[a] += 1
            available[b] += 1
        if sum(max(0, self.rules.minimum_degree - d) for d in degrees) > 2 * remaining:
            return False
        if any(d + a < self.rules.minimum_degree for d, a in zip(degrees, available)):
            return False
        return len(set(components(len(state.points), (*state.edges, *possible)))) == 1

    def step(self, state: BuildState, choice: Point | Edge) -> BuildState:
        stage = self.stage(state)
        if stage == "points":
            point = (float(choice[0]), float(choice[1]))
            if not self.rules.contains(point) or any(
                hypot(point[0] - x, point[1] - y) < self.rules.minimum_separation
                for x, y in state.points
            ):
                raise ValueError("invalid point action")
            return BuildState((*state.points, point), state.edges)
        if stage != "edges" or any(type(i) is not int for i in choice):
            raise ValueError("invalid edge action or terminal state")
        edge = (int(choice[0]), int(choice[1]))
        candidate = BuildState(state.points, (*state.edges, edge))
        if edge not in self._local_edges(state) or not self._can_finish(candidate):
            raise ValueError("masked edge action")
        return candidate

    def features(self, state: BuildState, choices: tuple[Point | Edge, ...]) -> NDArray[np.float64]:
        """Version-1 shared action features; no site-ID embedding or physics input."""
        rows: list[list[float]] = []
        labels = components(len(state.points), state.edges)
        degrees = [sum(i in e for e in state.edges) for i in range(len(state.points))]
        for choice in choices:
            if self.stage(state) == "points":
                x, y = choice
                distances = [hypot(x - a, y - b) for a, b in state.points]
                rows.append(
                    [
                        1,
                        x / self.rules.width,
                        y / self.rules.height,
                        min(distances, default=0) / hypot(self.rules.width, self.rules.height),
                        sum(d <= self.rules.maximum_edge_length for d in distances)
                        / self.rules.site_count,
                        0,
                        0,
                        0,
                        0,
                        0,
                        len(state.points) / self.rules.site_count,
                        0,
                    ]
                )
            else:
                a, b = int(choice[0]), int(choice[1])
                x, y = state.points[a]
                u, v = state.points[b]
                rows.append(
                    [
                        0,
                        (x + u) / (2 * self.rules.width),
                        (y + v) / (2 * self.rules.height),
                        hypot(u - x, v - y) / self.rules.maximum_edge_length,
                        float(labels[a] != labels[b]),
                        min(degrees[a], degrees[b]) / self.rules.maximum_degree,
                        max(degrees[a], degrees[b]) / self.rules.maximum_degree,
                        abs(u - x) / self.rules.width,
                        abs(v - y) / self.rules.height,
                        (u - x) * (v - y) / (self.rules.width * self.rules.height),
                        1,
                        len(state.edges) / self.rules.edge_count,
                    ]
                )
        return np.asarray(rows, dtype=float).reshape((-1, FEATURE_COUNT))

    def validate_final(self, state: BuildState) -> tuple[str, ...]:
        """Independent replay checks the complete construction contract."""
        if state.failure:
            return (state.failure,)
        if len(state.points) != self.rules.site_count or len(state.edges) != self.rules.edge_count:
            return ("incomplete",)
        if self.rules.fixed_points and state.points != self.rules.fixed_points:
            return ("fixed_points_changed",)
        check = BuildState()
        try:
            for point in state.points:
                check = self.step(check, point)
            for edge in state.edges:
                check = self.step(check, edge)
        except ValueError as error:
            return (str(error),)
        return () if self._can_finish(check) else ("invalid_final_graph",)
