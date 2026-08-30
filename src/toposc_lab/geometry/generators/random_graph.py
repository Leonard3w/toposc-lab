"""Seeded Erdős–Rényi random-graph geometry generator."""

from __future__ import annotations

import numpy as np

from toposc_lab.geometry.base import Geometry, GeometryEdge
from toposc_lab.geometry.generators._validation import (
    validate_graph_size,
    validate_optional_budget,
    validate_probability,
    validate_seed,
)

DEFAULT_RANDOM_GRAPH_MAX_CANDIDATE_EDGES = 1_000_000


def random_graph(
    n_sites: int,
    edge_probability: float,
    *,
    seed: int,
    max_candidate_edges: int | None = DEFAULT_RANDOM_GRAPH_MAX_CANDIDATE_EDGES,
) -> Geometry:
    """Draw an undirected simple graph from the ``G(n, p)`` ensemble.

    Every unordered pair of distinct sites is included independently with
    probability ``edge_probability``. Candidate pairs are visited in stable
    lexicographic order using an explicitly selected NumPy PCG64 stream.
    ``max_candidate_edges`` guards the quadratic enumeration before sampling.
    """
    n_sites = validate_graph_size(n_sites)
    edge_probability = validate_probability(
        edge_probability,
        name="edge_probability",
    )
    seed = validate_seed(seed)
    max_candidate_edges = validate_optional_budget(
        max_candidate_edges,
        name="max_candidate_edges",
    )
    candidate_edges = n_sites * (n_sites - 1) // 2
    if (
        max_candidate_edges is not None
        and candidate_edges > max_candidate_edges
    ):
        raise ValueError(
            f"random_graph with {n_sites} sites requires {candidate_edges} "
            "candidate edges, exceeding "
            f"max_candidate_edges={max_candidate_edges}"
        )

    random_number_generator = np.random.Generator(np.random.PCG64(seed))
    edges: list[GeometryEdge] = []
    for source in range(n_sites - 1):
        draws = random_number_generator.random(n_sites - source - 1)
        edges.extend(
            GeometryEdge(source, target, edge_type="random")
            for target, draw in enumerate(draws, start=source + 1)
            if draw < edge_probability
        )

    return Geometry(
        n_sites=n_sites,
        edges=tuple(edges),
        metadata={
            "generator": "random_graph",
            "ensemble": "erdos_renyi_gnp",
            "n_sites": n_sites,
            "edge_probability": edge_probability,
            "seed": seed,
            "rng": "numpy_pcg64",
            "candidate_edges": candidate_edges,
            "max_candidate_edges": max_candidate_edges,
        },
    )
