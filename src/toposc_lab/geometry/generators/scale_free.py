"""Seeded Barabási–Albert scale-free network generator."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from toposc_lab.geometry.base import Geometry, GeometryEdge
from toposc_lab.geometry.generators._validation import (
    validate_graph_size,
    validate_optional_budget,
    validate_seed,
)

DEFAULT_SCALE_FREE_MAX_EDGES = 1_000_000


def scale_free_graph(
    n_sites: int,
    attachments_per_site: int,
    *,
    seed: int,
    max_edges: int | None = DEFAULT_SCALE_FREE_MAX_EDGES,
) -> Geometry:
    """Draw a Barabási–Albert preferential-attachment graph.

    The initial graph is the complete graph on ``attachments_per_site + 1``
    sites. Every later site attaches to that many distinct existing sites,
    selected sequentially without replacement with probabilities proportional
    to their current degrees.
    """
    n_sites = validate_graph_size(n_sites)
    attachments_per_site = _validate_attachments(
        attachments_per_site,
        n_sites=n_sites,
    )
    seed = validate_seed(seed)
    max_edges = validate_optional_budget(max_edges, name="max_edges")
    initial_sites = attachments_per_site + 1
    n_edges = (
        attachments_per_site * n_sites
        - attachments_per_site * initial_sites // 2
    )
    if max_edges is not None and n_edges > max_edges:
        raise ValueError(
            f"scale_free_graph with n_sites={n_sites} and "
            f"attachments_per_site={attachments_per_site} requires "
            f"{n_edges} edges, exceeding max_edges={max_edges}"
        )

    typed_edges: list[tuple[int, int, str]] = [
        (source, target, "initial_clique")
        for source in range(initial_sites)
        for target in range(source + 1, initial_sites)
    ]
    repeated_nodes = [
        site
        for site in range(initial_sites)
        for _ in range(attachments_per_site)
    ]
    random_number_generator = np.random.Generator(np.random.PCG64(seed))

    for new_site in range(initial_sites, n_sites):
        targets: list[int] = []
        selected_targets: set[int] = set()
        while len(targets) < attachments_per_site:
            candidate = repeated_nodes[
                int(random_number_generator.integers(len(repeated_nodes)))
            ]
            if candidate in selected_targets:
                continue
            selected_targets.add(candidate)
            targets.append(candidate)

        typed_edges.extend(
            (target, new_site, "preferential_attachment")
            for target in targets
        )
        repeated_nodes.extend(targets)
        repeated_nodes.extend([new_site] * attachments_per_site)

    typed_edges.sort(key=lambda edge: (edge[0], edge[1]))
    return Geometry(
        n_sites=n_sites,
        edges=tuple(
            GeometryEdge(source, target, edge_type=edge_type)
            for source, target, edge_type in typed_edges
        ),
        metadata={
            "generator": "scale_free_graph",
            "model": "barabasi_albert",
            "n_sites": n_sites,
            "attachments_per_site": attachments_per_site,
            "initial_graph": "complete_m_plus_one",
            "initial_sites": initial_sites,
            "attachment_sampling": (
                "sequential_degree_weighted_without_replacement"
            ),
            "seed": seed,
            "rng": "numpy_pcg64",
            "max_edges": max_edges,
        },
    )


def _validate_attachments(attachments_per_site: int, *, n_sites: int) -> int:
    if (
        isinstance(attachments_per_site, bool)
        or not isinstance(attachments_per_site, Integral)
    ):
        raise TypeError("attachments_per_site must be an integer")
    result = int(attachments_per_site)
    if result < 1:
        raise ValueError("attachments_per_site must be positive")
    if result >= n_sites:
        raise ValueError("attachments_per_site must be smaller than n_sites")
    return result
