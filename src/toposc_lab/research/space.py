"""Fixed-position connectivity search, independent of any physical model.

Validity uses the established mutation policy. Canonical identities quotient the
square's dihedral symmetries for acquisition diversity only, never physics reuse.
"""
from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.geometry.serialization import geometry_from_bytes, geometry_to_bytes
from toposc_lab.search.geometry_genome import geometry_to_genome
from toposc_lab.search.mutation_validity import MutationValidityPolicy, validate_geometry_mutation

Edge = tuple[int, int]
Mutation = Callable[["FixedConnectivitySpace", set[Edge], np.random.Generator], set[Edge]]
MUTATION_REGISTRY: dict[str, Mutation] = {}
SPACE_REGISTRY: dict[str, type] = {}


def _edge(first: int, second: int) -> Edge:
    return (first, second) if first < second else (second, first)


def register_mutation(name: str, mutation: Mutation) -> None:
    if not name or name in MUTATION_REGISTRY:
        raise ValueError(f"duplicate/empty mutation name: {name}")
    MUTATION_REGISTRY[name] = mutation


def geometry_to_payload(geometry: Geometry) -> dict[str, Any]:
    """Lossless established serialization, with a cheap JSON preview for the UI."""
    return {
        "format": "toposc_geometry_v1",
        "archive": base64.b64encode(geometry_to_bytes(geometry)).decode("ascii"),
        "n_sites": geometry.n_sites,
        "coordinates": None if geometry.coordinates is None else geometry.coordinates.tolist(),
        "edges": [[edge.source, edge.target] for edge in geometry.edges],
        "boundary_sites": sorted(geometry.boundary_sites),
    }


def geometry_from_payload(payload: dict[str, Any] | Geometry) -> Geometry:
    if isinstance(payload, Geometry):
        return payload
    if payload.get("format") != "toposc_geometry_v1":
        raise ValueError("unsupported research geometry payload")
    return geometry_from_bytes(base64.b64decode(payload["archive"], validate=True))


@dataclass(frozen=True)
class FixedConnectivitySpace:
    side: int = 10
    min_degree: int = 1
    max_degree: int = 8
    max_bond_length: float = 1.5
    bond_tolerance: float = 0.05
    forbid_crossings: bool = False
    minimum_distance: float = 0.002

    def __post_init__(self) -> None:
        if isinstance(self.side, bool) or not isinstance(self.side, int) or self.side < 2:
            raise ValueError("side must be an integer >= 2")
        if not 1 <= self.min_degree <= self.max_degree:
            raise ValueError("invalid degree constraints")
        if not np.isfinite(self.max_bond_length) or self.max_bond_length < 1:
            raise ValueError("maximum bond length must be finite and >= 1")
        if not np.isfinite(self.bond_tolerance) or not 0 <= self.bond_tolerance < 1:
            raise ValueError("bond_tolerance must be a fraction in [0,1)")
        if not np.isfinite(self.minimum_distance) or not 0 <= self.minimum_distance < 1:
            raise ValueError("minimum_distance must be in [0,1)")
        if not isinstance(self.forbid_crossings, bool):
            raise TypeError("forbid_crossings must be boolean")
        # Building the policy also validates integer-valued degree bounds.
        _ = self.policy

    @cached_property
    def coordinates(self) -> np.ndarray:
        result = np.array([(x, y) for y in range(self.side) for x in range(self.side)], float)
        result.setflags(write=False)
        return result

    @cached_property
    def boundary(self) -> frozenset[int]:
        return frozenset(i for i, p in enumerate(self.coordinates) if
                         any(v in (0, self.side - 1) for v in p))

    @property
    def edge_count(self) -> int:
        return 2 * self.side * (self.side - 1)

    @cached_property
    def edge_pool(self) -> tuple[Edge, ...]:
        # A bond passing through another fixed site is not a local simple bond.
        from math import gcd
        return tuple((i, j) for i in range(self.side**2) for j in range(i + 1, self.side**2)
                     if np.linalg.norm(self.coordinates[i] - self.coordinates[j])
                     <= self.max_bond_length + 1e-12
                     and gcd(*(int(abs(v)) for v in self.coordinates[i] - self.coordinates[j])) == 1)

    @cached_property
    def policy(self) -> MutationValidityPolicy:
        allowance = int(np.floor(self.edge_count * self.bond_tolerance + 1e-12))
        return MutationValidityPolicy(
            require_connected=True, require_coordinates=True, required_embedding_dimension=2,
            minimum_site_count=self.side**2, maximum_site_count=self.side**2,
            minimum_edge_count=self.edge_count - allowance,
            maximum_edge_count=self.edge_count + allowance,
            minimum_degree=self.min_degree, maximum_degree=self.max_degree,
            maximum_edge_length=self.max_bond_length, minimum_site_separation=1.0,
            forbid_straight_edge_crossings=self.forbid_crossings, numerical_tolerance=1e-12,
        )

    def build(self, edges: set[Edge] | tuple[Edge, ...]) -> Geometry:
        return Geometry(n_sites=self.side**2, coordinates=self.coordinates,
                        edges=tuple(GeometryEdge(*e) for e in sorted(edges)),
                        boundary_sites=self.boundary, embedding_dimension=2)

    def reference(self) -> Geometry:
        return self.build({e for e in self.edge_pool if
                           np.linalg.norm(self.coordinates[e[0]] - self.coordinates[e[1]]) == 1})

    def validate(self, geometry: Geometry) -> list[str]:
        genome = geometry_to_genome(geometry)
        reasons = [issue.code for issue in
                   validate_geometry_mutation(genome, genome, policy=self.policy).issues]
        if geometry.coordinates is None or not np.array_equal(geometry.coordinates, self.coordinates):
            reasons.append("fixed_sites_required")
        if geometry.boundary_sites != self.boundary:
            reasons.append("fixed_boundary_required")
        try:
            local_edges = set(self.edges(geometry))
        except ValueError:
            local_edges = None
        if local_edges is None or not local_edges <= set(self.edge_pool):
            reasons.append("edge_outside_local_pool")
        if any(e.displacement is not None or e.boundary_crossing or e.edge_type or e.metadata
               for e in geometry.edges):
            reasons.append("unsupported_edge_semantics")
        if geometry.faces or geometry.rooted_tree is not None or geometry.site_types is not None:
            reasons.append("unsupported_geometry_semantics")
        return reasons

    def edges(self, geometry: Geometry) -> tuple[Edge, ...]:
        """Embedded identity survives consistent relabeling of a geometry."""
        if geometry.coordinates is None or geometry.n_sites != self.side**2:
            raise ValueError("geometry must contain the fixed square sites")
        lookup = {tuple(p): i for i, p in enumerate(self.coordinates)}
        try:
            labels = [lookup[tuple(p)] for p in geometry.coordinates]
        except KeyError as error:
            raise ValueError("geometry changes fixed sites") from error
        if len(set(labels)) != self.side**2:
            raise ValueError("geometry repeats fixed sites")
        return tuple(sorted(_edge(labels[e.source], labels[e.target])
                            for e in geometry.edges))

    def orbit(self, geometry: Geometry) -> tuple[frozenset[Edge], ...]:
        edges = self.edges(geometry)
        return tuple(frozenset(_edge(labels[i], labels[j]) for i, j in edges)
                     for labels in self.symmetry_labels)

    @cached_property
    def symmetry_labels(self) -> tuple[tuple[int, ...], ...]:
        variants = []
        for mirror in (False, True):
            for turns in range(4):
                labels = []
                for i in range(self.side**2):
                    x, y = i % self.side, i // self.side
                    if mirror:
                        x = self.side - 1 - x
                    for _ in range(turns):
                        x, y = self.side - 1 - y, x
                    labels.append(y * self.side + x)
                variants.append(tuple(labels))
        return tuple(variants)

    def hash(self, geometry: Geometry) -> str:
        key = min(tuple(sorted(v)) for v in self.orbit(geometry))
        return hashlib.sha256(json.dumps([self.side, key], separators=(",", ":")).encode()).hexdigest()

    def distance(self, first: Geometry, second: Geometry) -> float:
        edges = frozenset(self.edges(first))
        return min(len(edges ^ other) / max(1, len(edges | other)) for other in self.orbit(second))

    def sample(self, rng: np.random.Generator) -> Geometry:
        """Random wiring at fixed budget; caller validates every draw.

        The initializer retains the outer ring to avoid isolated corners.
        Mutations can alter that ring. One diagonal per square improves planar
        acceptance without changing the configurable crossing validity check.
        """
        pool = list(self.edge_pool)
        if self.forbid_crossings:
            for y in range(self.side - 1):
                for x in range(self.side - 1):
                    a = y * self.side + x
                    diagonals = ((a, a + self.side + 1), (a + 1, a + self.side))
                    excluded = diagonals[int(rng.integers(2))]
                    if excluded in pool:
                        pool.remove(excluded)
        edges = {e for e in pool if np.linalg.norm(self.coordinates[e[0]] - self.coordinates[e[1]]) == 1
                 and all(i in self.boundary for i in e)
                 and (self.coordinates[e[0], 0] == self.coordinates[e[1], 0]
                      and self.coordinates[e[0], 0] in (0, self.side - 1)
                      or self.coordinates[e[0], 1] == self.coordinates[e[1], 1]
                      and self.coordinates[e[0], 1] in (0, self.side - 1))}
        rng.shuffle(pool)
        components = list(range(self.side**2))
        degrees = np.zeros(self.side**2, dtype=int)

        def root(site: int) -> int:
            while components[site] != site:
                components[site] = components[components[site]]
                site = components[site]
            return site

        for i, j in sorted(edges):
            components[root(i)] = root(j)
            degrees[i] += 1
            degrees[j] += 1
        # Random spanning forest first, then repair low coordination and fill.
        # This remains a proposal sampler: impossible constraints are reported
        # through validate rather than relaxed to make a draw pass.
        for i, j in pool:
            if len(edges) >= self.edge_count:
                break
            if root(i) != root(j) and max(degrees[i], degrees[j]) < self.max_degree:
                edges.add((i, j))
                components[root(i)] = root(j)
                degrees[i] += 1
                degrees[j] += 1
        prioritized = sorted(pool, key=lambda edge:
                             int(degrees[edge[0]] >= self.min_degree) +
                             int(degrees[edge[1]] >= self.min_degree))
        for i, j in prioritized:
            if len(edges) >= self.edge_count:
                break
            if (i, j) not in edges and max(degrees[i], degrees[j]) < self.max_degree:
                edges.add((i, j))
                degrees[i] += 1
                degrees[j] += 1
        return self.build(edges)

    def mutate(self, parent: Geometry, rng: np.random.Generator,
               operator: str | None = None) -> tuple[Geometry, dict[str, Any]]:
        selected = operator or str(rng.choice(tuple(MUTATION_REGISTRY)))
        if selected not in MUTATION_REGISTRY:
            raise ValueError(f"unknown mutation {selected}")
        before = set(self.edges(parent))
        after = MUTATION_REGISTRY[selected](self, before.copy(), rng)
        return self.build(after), {"operator": selected, "removed": sorted(before - after),
                                   "added": sorted(after - before)}


def _pick(values: list[Edge], rng: np.random.Generator) -> Edge | None:
    return values[int(rng.integers(len(values)))] if values else None


def _remove(space: FixedConnectivitySpace, edges: set[Edge], rng: np.random.Generator) -> set[Edge]:
    edge = _pick(sorted(edges), rng)
    if edge is not None:
        edges.remove(edge)
    return edges


def _add(space: FixedConnectivitySpace, edges: set[Edge], rng: np.random.Generator) -> set[Edge]:
    edge = _pick(sorted(set(space.edge_pool) - edges), rng)
    if edge is not None:
        edges.add(edge)
    return edges


def _rewire(space: FixedConnectivitySpace, edges: set[Edge], rng: np.random.Generator,
            boundary: bool = False) -> set[Edge]:
    removed = _pick([e for e in sorted(edges) if not boundary or
                     bool(set(e) & space.boundary)], rng)
    if removed is None:
        return edges
    replacements = [e for e in space.edge_pool if e not in edges and set(e) & set(removed)
                    and (not boundary or bool(set(e) & space.boundary))]
    replacement = _pick(replacements, rng)
    if replacement is not None:
        edges.remove(removed)
        edges.add(replacement)
    return edges


def _swap(space: FixedConnectivitySpace, edges: set[Edge], rng: np.random.Generator) -> set[Edge]:
    # Degree-preserving two-edge swap.
    if len(edges) < 2:
        return edges
    ordered = sorted(edges)
    selected = rng.choice(len(ordered), 2, replace=False)
    (a, b), (c, d) = (ordered[int(i)] for i in selected)
    if len({a, b, c, d}) != 4:
        return edges
    replacements = {_edge(a, c), _edge(b, d)}
    if len(replacements) == 2 and replacements <= set(space.edge_pool) and not replacements & edges:
        edges.difference_update(((a, b), (c, d)))
        edges.update(replacements)
    return edges


def _flip(space: FixedConnectivitySpace, edges: set[Edge], rng: np.random.Generator) -> set[Edge]:
    valid = []
    for y in range(space.side - 1):
        for x in range(space.side - 1):
            a = y * space.side + x
            b, c, d = a + 1, a + space.side, a + space.side + 1
            perimeter = {(a, b), (a, c), (b, d), (c, d)}
            if not perimeter <= edges:
                continue
            for old, new in (((a, d), (b, c)), ((b, c), (a, d))):
                if old in edges and new not in edges and new in space.edge_pool:
                    valid.append((old, new))
    if valid:
        old, new = valid[int(rng.integers(len(valid)))]
        edges.remove(old)
        edges.add(new)
    return edges


def _motif(space: FixedConnectivitySpace, edges: set[Edge], rng: np.random.Generator) -> set[Edge]:
    center = int(rng.integers(space.side**2))
    sites = {center} | {i for edge in edges if center in edge for i in edge}
    old = sorted(e for e in edges if set(e) <= sites)
    new = sorted(e for e in space.edge_pool if e not in edges and set(e) <= sites)
    count = min(2, len(old), len(new))
    if count:
        removed = rng.choice(len(old), count, replace=False)
        added = rng.choice(len(new), count, replace=False)
        edges.difference_update(old[int(i)] for i in removed)
        edges.update(new[int(i)] for i in added)
    return edges


for _name, _function in {
    "remove_local_bond": _remove, "add_local_bond": _add, "edge_swap": _swap,
    "local_rewiring": _rewire, "diagonal_flip": _flip, "local_motif_replacement": _motif,
    "boundary_rewiring": lambda space, edges, rng: _rewire(space, edges, rng, True),
}.items():
    register_mutation(_name, _function)
SPACE_REGISTRY["fixed_connectivity"] = FixedConnectivitySpace
