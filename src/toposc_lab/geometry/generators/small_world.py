"""Seeded Watts–Strogatz small-world network generator."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from toposc_lab.geometry.base import Geometry, GeometryEdge
from toposc_lab.geometry.generators._validation import (
    validate_graph_size,
    validate_optional_budget,
    validate_probability,
    validate_seed,
)

DEFAULT_SMALL_WORLD_MAX_EDGES = 1_000_000

_EdgeKey = tuple[int, int]


def small_world_network(
    n_sites: int,
    neighbor_degree: int,
    rewiring_probability: float,
    *,
    seed: int,
    max_edges: int | None = DEFAULT_SMALL_WORLD_MAX_EDGES,
) -> Geometry:
    """Draw a Watts–Strogatz small-world network.

    ``neighbor_degree`` must be even. The initial ring connects each site to
    half that many neighbors in each cyclic direction. Every forward ring edge
    is considered once for rewiring in stable source/offset order. Rewiring
    preserves the total edge count and rejects self-edges and duplicate edges,
    but intentionally does not preserve every individual site degree.
    """
    n_sites = validate_graph_size(n_sites)
    neighbor_degree = _validate_neighbor_degree(
        neighbor_degree,
        n_sites=n_sites,
    )
    rewiring_probability = validate_probability(
        rewiring_probability,
        name="rewiring_probability",
    )
    seed = validate_seed(seed)
    max_edges = validate_optional_budget(max_edges, name="max_edges")
    n_edges = n_sites * neighbor_degree // 2
    if max_edges is not None and n_edges > max_edges:
        raise ValueError(
            f"small_world_network with n_sites={n_sites} and "
            f"neighbor_degree={neighbor_degree} requires {n_edges} edges, "
            f"exceeding max_edges={max_edges}"
        )

    forward_offsets = range(1, neighbor_degree // 2 + 1)
    original_edges = tuple(
        (source, (source + offset) % n_sites)
        for source in range(n_sites)
        for offset in forward_offsets
    )
    edge_types: dict[_EdgeKey, str] = {
        _edge_key(source, target): "ring_lattice"
        for source, target in original_edges
    }
    neighbors: list[set[int]] = [set() for _ in range(n_sites)]
    for source, target in edge_types:
        neighbors[source].add(target)
        neighbors[target].add(source)

    random_number_generator = np.random.Generator(np.random.PCG64(seed))
    rewiring_decisions = 0
    rewired_edges = 0
    for source, original_target in original_edges:
        if random_number_generator.random() >= rewiring_probability:
            continue
        rewiring_decisions += 1

        original_edge = _edge_key(source, original_target)
        del edge_types[original_edge]
        neighbors[source].remove(original_target)
        neighbors[original_target].remove(source)

        available_targets = tuple(
            target
            for target in range(n_sites)
            if target != source and target not in neighbors[source]
        )
        target = available_targets[
            int(random_number_generator.integers(len(available_targets)))
        ]
        new_edge = _edge_key(source, target)
        if target == original_target:
            edge_types[new_edge] = "ring_lattice"
        else:
            edge_types[new_edge] = "rewired"
            rewired_edges += 1
        neighbors[source].add(target)
        neighbors[target].add(source)

    return Geometry(
        n_sites=n_sites,
        edges=tuple(
            GeometryEdge(source, target, edge_type=edge_types[(source, target)])
            for source, target in sorted(edge_types)
        ),
        metadata={
            "generator": "small_world_network",
            "model": "watts_strogatz",
            "n_sites": n_sites,
            "neighbor_degree": neighbor_degree,
            "rewiring_probability": rewiring_probability,
            "seed": seed,
            "rng": "numpy_pcg64",
            "rewiring_order": "source_then_positive_ring_offset",
            "rewiring_decisions": rewiring_decisions,
            "rewired_edges": rewired_edges,
            "max_edges": max_edges,
        },
    )


def _validate_neighbor_degree(neighbor_degree: int, *, n_sites: int) -> int:
    if isinstance(neighbor_degree, bool) or not isinstance(neighbor_degree, Integral):
        raise TypeError("neighbor_degree must be an integer")
    result = int(neighbor_degree)
    if result < 0:
        raise ValueError("neighbor_degree must be nonnegative")
    if result >= n_sites:
        raise ValueError("neighbor_degree must be smaller than n_sites")
    if result % 2 != 0:
        raise ValueError("neighbor_degree must be even")
    return result


def _edge_key(source: int, target: int) -> _EdgeKey:
    return (source, target) if source < target else (target, source)
