"""Outcome-blind degree-preserving motif interventions in the frozen wiring space."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np

from toposc_lab.generative.space import GeometrySearchSpace, embedded_edges
from toposc_lab.geometry import Geometry
from toposc_lab.patterns.features import adjacency, motif_instances
from toposc_lab.search.geometry_genome import geometry_to_genome
from toposc_lab.search.mutation_validity import validate_geometry_mutation

MOTIFS = (
    "triangle_count",
    "chordless_square_count",
    "boundary_bulk_edges",
    "central_diagonal_count",
)


def motif_counts(geometry: Geometry) -> dict[str, int]:
    triangles, squares = motif_instances(adjacency(geometry))
    xy = geometry.coordinates
    if xy is None:
        raise ValueError("embedded motifs require coordinates")
    edges = embedded_edges(geometry)
    # embedded_edges uses canonical lattice labels, so use canonical coordinates too.
    grid = GeometrySearchSpace().coordinates
    boundary = GeometrySearchSpace().boundary
    diagonals = [e for e in edges if np.linalg.norm(grid[e[0]] - grid[e[1]]) > 1.1]
    return {
        "triangle_count": len(triangles),
        "chordless_square_count": len(squares),
        "boundary_bulk_edges": sum((a in boundary) != (b in boundary) for a, b in edges),
        "central_diagonal_count": sum(
            int(np.linalg.norm((grid[a] + grid[b]) / 2 - (2.5, 2.5)) <= 1.5)
            for a, b in diagonals
        ),
        "diagonal_count": len(diagonals),
    }


@dataclass(frozen=True)
class EdgeSwap:
    removed: tuple[tuple[int, int], ...]
    added: tuple[tuple[int, int], ...]
    counts: tuple[int, ...]

    def apply(self, geometry: Geometry) -> Geometry:
        """Apply this edit only to a compatible valid parent, preserving its degrees."""
        space = GeometrySearchSpace()
        if space.reasons(geometry):
            raise ValueError("swap parent must belong to the validated wiring stratum")
        original = frozenset(embedded_edges(geometry))
        removed, added = frozenset(self.removed), frozenset(self.added)
        old_vertices = sorted(i for edge in self.removed for i in edge)
        new_vertices = sorted(i for edge in self.added for i in edge)
        if (
            len(removed) != 2
            or len(added) != 2
            or len(set(old_vertices)) != 4
            or old_vertices != new_vertices
            or not removed <= original - space.perimeter
            or added & original
        ):
            raise ValueError("edit is not a compatible degree-preserving double-edge swap")
        result = space.build((original - removed) | added)
        if space.reasons(result):
            raise ValueError("swap violates the validated wiring stratum")
        counts = motif_counts(result)
        if self.counts != tuple(counts[k] for k in (*MOTIFS, "diagonal_count")):
            raise ValueError("swap motif counts do not match the resulting geometry")
        return result


def valid_swaps(geometry: Geometry) -> tuple[EdgeSwap, ...]:
    space = GeometrySearchSpace()
    if space.reasons(geometry):
        raise ValueError("parent must belong to the validated wiring stratum")
    edges = frozenset(embedded_edges(geometry))
    pool = frozenset(space.edge_pool)
    swaps = []
    seen = set()
    for first, second in combinations(sorted(edges - space.perimeter), 2):
        a, b = first
        c, d = second
        if len({a, b, c, d}) != 4:
            continue
        for additions in (((a, c), (b, d)), ((a, d), (b, c))):
            added = tuple(sorted((min(e), max(e)) for e in additions))
            if any(e in edges or e not in pool for e in added):
                continue
            proposed = (edges - {first, second}) | frozenset(added)
            if proposed in seen:
                continue
            seen.add(proposed)
            g = space.build(proposed)
            if space.reasons(g):
                continue
            counts = motif_counts(g)
            swaps.append(
                EdgeSwap(
                    (first, second),
                    added,
                    tuple(int(counts[k]) for k in (*MOTIFS, "diagonal_count")),
                )
            )
    return tuple(swaps)


def matched_swaps(
    geometry: Geometry,
    motif: str,
    swaps: tuple[EdgeSwap, ...],
    seed: int,
    replicates: int = 2,
    *,
    replicate_seeds: tuple[int, ...] | None = None,
) -> tuple[tuple[EdgeSwap, EdgeSwap], ...]:
    """Match using structure only, standardized over the parent's valid edit pool.

    Each replicate uses its explicitly supplied perturbation seed, or ``seed + r``.
    The minimum-cost remaining matched pair is selected, breaking ties by that
    replicate's RNG. No target or control edit is reused within a parent/target.
    Standard deviations describe the structural edit pool, not physics labels.
    """
    if motif not in MOTIFS:
        raise ValueError("unknown motif")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seeds = (
        replicate_seeds
        if replicate_seeds is not None
        else tuple(seed + r for r in range(replicates))
    )
    if len(seeds) != replicates or any(
        isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds
    ):
        raise ValueError("replicate_seeds must contain one nonnegative integer per replicate")
    if len(set(seeds)) != len(seeds):
        raise ValueError("perturbation seeds must be distinct")
    if GeometrySearchSpace().reasons(geometry):
        raise ValueError("matching parent must belong to the validated wiring stratum")
    index = MOTIFS.index(motif)
    before = motif_counts(geometry)
    base = np.array([before[k] for k in (*MOTIFS, "diagonal_count")])
    delta = np.array([s.counts for s in swaps]) - base if swaps else np.empty((0, 5))
    targets = [int(i) for i in np.flatnonzero(delta[:, index] < 0)]
    controls = [int(i) for i in np.flatnonzero(delta[:, index] == 0)]
    if not len(targets) or not len(controls):
        return ()
    scales = np.std(delta, axis=0)
    scales[scales == 0] = 1.0
    nuisance = [k for k in range(4) if k != index]
    matches = []
    for i in targets:
        for j in controls:
            if delta[i, 4] != delta[j, 4]:
                continue
            if motif != "boundary_bulk_edges" and delta[i, 2] != delta[j, 2]:
                continue
            cost = float(
                np.sum(((delta[i, nuisance] - delta[j, nuisance]) / scales[nuisance]) ** 2)
            )
            matches.append((cost, int(i), int(j)))
    chosen = []
    used: set[int] = set()
    for perturbation_seed in seeds:
        remaining = [(cost, i, j) for cost, i, j in matches if i not in used and j not in used]
        if not remaining:
            break
        minimum_cost = min(cost for cost, _, _ in remaining)
        ties = [(i, j) for cost, i, j in remaining if cost == minimum_cost]
        rng = np.random.default_rng(perturbation_seed)
        i, j = ties[int(rng.integers(len(ties)))]
        chosen.append((swaps[i], swaps[j]))
        used.update((i, j))
    return tuple(chosen)


def deletion_candidates(geometry: Geometry) -> tuple[tuple[int, int], ...]:
    """Single-edge deletions; explicit 59-edge diagnostic extension, no node edits."""
    space = GeometrySearchSpace()
    if space.reasons(geometry):
        raise ValueError("deletion parent must be valid")
    edges = frozenset((min(e.source, e.target), max(e.source, e.target)) for e in geometry.edges)
    result = []
    for e in sorted(edges - space.perimeter):
        if not deletion_reasons(space.build(edges - {e})):
            result.append(e)
    return tuple(result)


def deletion_reasons(geometry: Geometry) -> tuple[str, ...]:
    space = GeometrySearchSpace()
    genome = geometry_to_genome(geometry)
    report = validate_geometry_mutation(
        genome,
        genome,
        policy=replace(
            space.policy,
            minimum_edge_count=59,
            maximum_edge_count=59,
        ),
    )
    reasons = [i.code for i in report.issues]
    edges = frozenset((min(e.source, e.target), max(e.source, e.target)) for e in geometry.edges)
    if geometry.coordinates is None or not np.array_equal(geometry.coordinates, space.coordinates):
        reasons.append("fixed_coordinates_required")
    if not edges <= frozenset(space.edge_pool) or not space.perimeter <= edges:
        reasons.append("fixed_edge_pool_and_perimeter_required")
    if geometry.boundary_sites != space.boundary:
        reasons.append("fixed_boundary_required")
    if (
        geometry.faces
        or geometry.site_types is not None
        or geometry.rooted_tree is not None
        or any(
            e.boundary_crossing
            or e.displacement is not None
            or e.edge_type is not None
            or e.metadata
            for e in geometry.edges
        )
    ):
        reasons.append("unsupported_geometry_semantics")
    return tuple(reasons)
