"""Seeded uniform random-regular graph geometry generator."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from toposc_lab.geometry.base import Geometry, GeometryEdge
from toposc_lab.geometry.generators._validation import (
    validate_graph_size,
    validate_optional_budget,
    validate_positive_integer,
    validate_seed,
)

DEFAULT_RANDOM_REGULAR_MAX_EDGES = 1_000_000
DEFAULT_RANDOM_REGULAR_MAX_ATTEMPTS = 1_000

_EdgeKey = tuple[int, int]


def random_regular_graph(
    n_sites: int,
    degree: int,
    *,
    seed: int,
    max_edges: int | None = DEFAULT_RANDOM_REGULAR_MAX_EDGES,
    max_attempts: int = DEFAULT_RANDOM_REGULAR_MAX_ATTEMPTS,
) -> Geometry:
    """Draw a uniform undirected simple ``degree``-regular graph.

    A shuffled configuration model is rejected whenever it contains a
    self-edge or duplicate edge. Conditional on acceptance, simple regular
    graphs are sampled uniformly. Dense degrees use the complement of the
    corresponding sparse regular graph. Exhausting ``max_attempts`` raises an
    error rather than returning a biased fallback.
    """
    n_sites = validate_graph_size(n_sites)
    degree = _validate_degree(degree, n_sites=n_sites)
    seed = validate_seed(seed)
    max_edges = validate_optional_budget(max_edges, name="max_edges")
    max_attempts = validate_positive_integer(max_attempts, name="max_attempts")
    n_edges = n_sites * degree // 2
    if max_edges is not None and n_edges > max_edges:
        raise ValueError(
            f"random_regular_graph with n_sites={n_sites} and degree={degree} "
            f"requires {n_edges} edges, exceeding max_edges={max_edges}"
        )

    complement_degree = n_sites - 1 - degree
    complemented = complement_degree < degree
    effective_degree = complement_degree if complemented else degree
    random_number_generator = np.random.Generator(np.random.PCG64(seed))
    sampled_edges, attempts_used = _sample_simple_regular_edges(
        n_sites=n_sites,
        degree=effective_degree,
        random_number_generator=random_number_generator,
        max_attempts=max_attempts,
        seed=seed,
    )
    if complemented:
        final_edges = tuple(
            (source, target)
            for source in range(n_sites)
            for target in range(source + 1, n_sites)
            if (source, target) not in sampled_edges
        )
    else:
        final_edges = tuple(sorted(sampled_edges))

    return Geometry(
        n_sites=n_sites,
        edges=tuple(
            GeometryEdge(source, target, edge_type="random_regular")
            for source, target in final_edges
        ),
        metadata={
            "generator": "random_regular_graph",
            "ensemble": "uniform_simple_regular",
            "n_sites": n_sites,
            "degree": degree,
            "seed": seed,
            "rng": "numpy_pcg64",
            "sampler": "configuration_model_rejection",
            "effective_degree": effective_degree,
            "complemented": complemented,
            "attempts_used": attempts_used,
            "max_attempts": max_attempts,
            "max_edges": max_edges,
        },
    )


def _validate_degree(degree: int, *, n_sites: int) -> int:
    if isinstance(degree, bool) or not isinstance(degree, Integral):
        raise TypeError("degree must be an integer")
    result = int(degree)
    if result < 0:
        raise ValueError("degree must be nonnegative")
    if result >= n_sites:
        raise ValueError("degree must be smaller than n_sites")
    if n_sites * result % 2 != 0:
        raise ValueError("n_sites * degree must be even")
    return result


def _sample_simple_regular_edges(
    *,
    n_sites: int,
    degree: int,
    random_number_generator: np.random.Generator,
    max_attempts: int,
    seed: int,
) -> tuple[set[_EdgeKey], int]:
    if degree == 0:
        return set(), 0

    stubs = np.repeat(np.arange(n_sites, dtype=np.int64), degree)
    for attempt in range(1, max_attempts + 1):
        shuffled_stubs = random_number_generator.permutation(stubs)
        edges: set[_EdgeKey] = set()
        valid = True
        for offset in range(0, shuffled_stubs.size, 2):
            first = int(shuffled_stubs[offset])
            second = int(shuffled_stubs[offset + 1])
            if first == second:
                valid = False
                break
            edge = (first, second) if first < second else (second, first)
            if edge in edges:
                valid = False
                break
            edges.add(edge)
        if valid:
            return edges, attempt

    raise RuntimeError(
        f"failed to sample a simple {degree}-regular graph on {n_sites} sites "
        f"after {max_attempts} attempts using seed {seed}"
    )
