"""Deterministic scalar descriptors of a geometry's undirected graph."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from types import MappingProxyType
from typing import TypeAlias

import numpy as np

from toposc_lab.geometry.base import Geometry

GeometryDescriptorScalar: TypeAlias = bool | int | float | None


def extract_geometry_descriptors(
    geometry: Geometry,
) -> Mapping[str, GeometryDescriptorScalar]:
    """Return immutable scalar descriptors of the undirected one-skeleton.

    Stored edge orientation, coordinates, boundary labels, site/edge types,
    faces, and metadata do not affect these descriptors. Shortest-path values
    include only distinct, mutually reachable unordered site pairs.
    ``reachable_pair_fraction`` records their fraction among all unordered
    pairs so disconnected geometries remain explicit.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be a Geometry")

    neighbor_sets = tuple(
        frozenset(geometry.neighbors(site)) for site in geometry.site_indices
    )
    degrees = np.asarray([len(neighbors) for neighbors in neighbor_sets], dtype=float)
    components = _connected_components(neighbor_sets)
    linked_neighbor_pairs = tuple(
        _linked_neighbor_pair_count(neighbors, neighbor_sets)
        for neighbors in neighbor_sets
    )
    local_clustering = tuple(
        0.0
        if degree < 2
        else linked_pairs / (degree * (degree - 1) / 2.0)
        for degree, linked_pairs in zip(degrees, linked_neighbor_pairs, strict=True)
    )
    finite_distances = _finite_pair_distances(neighbor_sets)
    total_pair_count = geometry.n_sites * (geometry.n_sites - 1) // 2
    connected_pair_count = len(finite_distances)

    descriptors: dict[str, GeometryDescriptorScalar] = {
        "site_count": geometry.n_sites,
        "edge_count": geometry.n_edges,
        "mean_degree": float(np.mean(degrees)),
        "degree_variance": float(np.var(degrees)),
        "component_count": len(components),
        "largest_component_site_count": max(len(component) for component in components),
        "is_connected": len(components) == 1,
        "cycle_rank": geometry.n_edges - geometry.n_sites + len(components),
        "has_cycles": geometry.n_edges - geometry.n_sites + len(components) > 0,
        "triangle_count": sum(linked_neighbor_pairs) // 3,
        "mean_local_clustering": float(np.mean(local_clustering)),
        "connected_pair_count": connected_pair_count,
        "reachable_pair_fraction": (
            1.0 if total_pair_count == 0 else connected_pair_count / total_pair_count
        ),
        "mean_finite_shortest_path_length": (
            None if not finite_distances else float(np.mean(finite_distances))
        ),
        "maximum_finite_shortest_path_length": (
            None if not finite_distances else max(finite_distances)
        ),
    }
    return MappingProxyType(descriptors)


def _connected_components(
    neighbor_sets: tuple[frozenset[int], ...],
) -> tuple[frozenset[int], ...]:
    unseen = set(range(len(neighbor_sets)))
    components: list[frozenset[int]] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        component: set[int] = set()
        unseen.remove(root)
        while stack:
            site = stack.pop()
            component.add(site)
            new_neighbors = unseen.intersection(neighbor_sets[site])
            unseen.difference_update(new_neighbors)
            stack.extend(sorted(new_neighbors, reverse=True))
        components.append(frozenset(component))
    return tuple(components)


def _linked_neighbor_pair_count(
    neighbors: frozenset[int],
    neighbor_sets: tuple[frozenset[int], ...],
) -> int:
    ordered_neighbors = sorted(neighbors)
    return sum(
        target in neighbor_sets[source]
        for position, source in enumerate(ordered_neighbors)
        for target in ordered_neighbors[position + 1 :]
    )


def _finite_pair_distances(
    neighbor_sets: tuple[frozenset[int], ...],
) -> tuple[int, ...]:
    distances: list[int] = []
    for source in range(len(neighbor_sets)):
        source_distances = {source: 0}
        queue = deque([source])
        while queue:
            site = queue.popleft()
            for neighbor in neighbor_sets[site]:
                if neighbor in source_distances:
                    continue
                source_distances[neighbor] = source_distances[site] + 1
                queue.append(neighbor)
        distances.extend(
            source_distances[target]
            for target in range(source + 1, len(neighbor_sets))
            if target in source_distances
        )
    return tuple(distances)
