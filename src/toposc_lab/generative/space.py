"""Fixed-resource planar wiring space and embedding-aware duplicate rejection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache

import numpy as np

from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.search.geometry_genome import geometry_to_genome
from toposc_lab.search.mutation_validity import MutationValidityPolicy, validate_geometry_mutation

Edge = tuple[int, int]


@dataclass(frozen=True, slots=True)
class NoveltyConfig:
    minimum_distance: float = 0.06

    def __post_init__(self) -> None:
        if not np.isfinite(self.minimum_distance) or not 0 <= self.minimum_distance < 1:
            raise ValueError("novelty distance must be in [0,1)")


@dataclass(frozen=True)
class GeometrySearchSpace:
    """Version-one fabrication proxy, deliberately fixed to one matched stratum."""

    side: int = 6
    edge_count: int = 60

    def __post_init__(self) -> None:
        if self.side != 6 or self.edge_count != 60:
            raise ValueError("v1 supports only the frozen 36-site, 60-edge stratum")

    @property
    def coordinates(self) -> np.ndarray:
        return np.array([(x, y) for y in range(self.side) for x in range(self.side)], float)

    @property
    def boundary(self) -> frozenset[int]:
        return frozenset(
            i for i, (x, y) in enumerate(self.coordinates) if x in (0, 5) or y in (0, 5)
        )

    @cached_property
    def edge_pool(self) -> tuple[Edge, ...]:
        xy = self.coordinates
        return tuple(
            (i, j)
            for i in range(36)
            for j in range(i + 1, 36)
            if np.max(np.abs(xy[i] - xy[j])) == 1
        )

    @cached_property
    def perimeter(self) -> frozenset[Edge]:
        xy = self.coordinates
        return frozenset(
            (i, j)
            for i, j in self.edge_pool
            if np.linalg.norm(xy[i] - xy[j]) == 1
            and (
                (xy[i, 0] == xy[j, 0] and xy[i, 0] in (0, 5))
                or (xy[i, 1] == xy[j, 1] and xy[i, 1] in (0, 5))
            )
        )

    @cached_property
    def policy(self) -> MutationValidityPolicy:
        return MutationValidityPolicy(
            require_connected=True,
            require_coordinates=True,
            required_embedding_dimension=2,
            minimum_site_count=36,
            maximum_site_count=36,
            minimum_edge_count=60,
            maximum_edge_count=60,
            minimum_degree=2,
            maximum_degree=6,
            coordinate_lower_bounds=(0.0, 0.0),
            coordinate_upper_bounds=(5.0, 5.0),
            minimum_site_separation=1.0,
            maximum_edge_length=float(np.sqrt(2)),
            forbid_straight_edge_crossings=True,
            numerical_tolerance=1e-12,
        )

    def build(self, edges: tuple[Edge, ...] | frozenset[Edge]) -> Geometry:
        return Geometry(
            n_sites=36,
            edges=tuple(GeometryEdge(*e) for e in sorted(edges)),
            coordinates=self.coordinates,
            embedding_dimension=2,
            boundary_sites=self.boundary,
        )

    def reasons(self, geometry: Geometry) -> tuple[str, ...]:
        genome = geometry_to_genome(geometry)
        report = validate_geometry_mutation(genome, genome, policy=self.policy)
        reasons = [i.code for i in report.issues]
        if geometry.coordinates is None or not np.array_equal(
            geometry.coordinates, self.coordinates
        ):
            reasons.append("fixed_lattice_required")
        edges = frozenset(tuple(sorted((e.source, e.target))) for e in geometry.edges)
        if not edges <= frozenset(self.edge_pool):
            reasons.append("edge_outside_pool")
        if not self.perimeter <= edges or geometry.boundary_sites != self.boundary:
            reasons.append("outer_boundary_contract")
        if any(
            e.boundary_crossing
            or e.displacement is not None
            or e.edge_type is not None
            or e.metadata
            for e in geometry.edges
        ):
            reasons.append("unsupported_edge_semantics")
        if geometry.faces or geometry.site_types is not None or geometry.rooted_tree is not None:
            reasons.append("unsupported_geometry_semantics")
        return tuple(reasons)

    def sample(self, rng: np.random.Generator, *, patch: bool = False) -> Geometry:
        """One raw proposal; validity and duplicate rejection belong to the caller."""
        xy = self.coordinates
        axial = [
            e
            for e in self.edge_pool
            if np.linalg.norm(xy[e[0]] - xy[e[1]]) == 1 and e not in self.perimeter
        ]
        diagonals = []
        for y in range(5):
            for x in range(5):
                i = y * 6 + x
                diagonals.append((i, i + 7) if rng.integers(2) else (i + 1, i + 6))
        available = axial + diagonals
        priority = rng.random(len(available))
        if patch:
            center = rng.uniform(1, 4, 2)
            radius = float(rng.uniform(1, 3))
            preference = int(rng.choice([-1, 1]))
            midpoints = np.array([(xy[i] + xy[j]) / 2 for i, j in available])
            inside = np.linalg.norm(midpoints - center, axis=1) <= radius
            diagonal = np.array([abs(i - j) in (5, 7) for i, j in available])
            priority += preference * inside * np.where(diagonal, 0.75, -0.75)
        chosen = np.argsort(priority, kind="stable")[-40:]
        return self.build(self.perimeter | frozenset(available[int(i)] for i in chosen))


def embedded_edges(geometry: Geometry) -> tuple[Edge, ...]:
    """Coordinates anchor identity, so relabeling sites cannot create novelty."""
    xy = geometry.coordinates
    if xy is None or xy.shape != (36, 2):
        raise ValueError("structural distance requires the frozen 6x6 lattice")
    grid = {(float(x), float(y)): y * 6 + x for y in range(6) for x in range(6)}
    try:
        labels = [grid[tuple(p)] for p in xy]
    except KeyError as error:
        raise ValueError("coordinates outside the fixed lattice") from error
    if len(set(labels)) != 36:
        raise ValueError("duplicate lattice position")
    return tuple(
        sorted(
            (min(labels[e.source], labels[e.target]), max(labels[e.source], labels[e.target]))
            for e in geometry.edges
        )
    )


@lru_cache(maxsize=16384)
def _orbit(edges: tuple[Edge, ...]) -> tuple[frozenset[Edge], ...]:
    variants = []
    for mirror in (False, True):
        for turns in range(4):
            labels = []
            for i in range(36):
                x, y = i % 6, i // 6
                if mirror:
                    x = 5 - x
                for _ in range(turns):
                    x, y = 5 - y, x
                labels.append(y * 6 + x)
            variants.append(
                frozenset((min(labels[i], labels[j]), max(labels[i], labels[j])) for i, j in edges)
            )
    return tuple(variants)


def structural_distance(first: Geometry, second: Geometry) -> float:
    a = frozenset(embedded_edges(first))
    return min(len(a ^ b) / max(1, len(a | b)) for b in _orbit(embedded_edges(second)))


def structural_key(geometry: Geometry) -> tuple[Edge, ...]:
    return min(tuple(sorted(v)) for v in _orbit(embedded_edges(geometry)))
