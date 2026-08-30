"""Generic deterministic k-nearest-neighbor geometry builder."""

from __future__ import annotations

from math import dist
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from toposc_lab.geometry.base import Geometry, GeometryEdge
from toposc_lab.geometry.generators._point_cloud import validate_point_coordinates
from toposc_lab.geometry.generators._validation import (
    validate_optional_budget,
    validate_positive_integer,
)

KNearestSymmetrization: TypeAlias = Literal["union", "mutual"]

DEFAULT_K_NEAREST_MAX_NEIGHBOR_SELECTIONS = 1_000_000
DEFAULT_K_NEAREST_MAX_EDGES = 1_000_000


def k_nearest_neighbor_graph(
    coordinates: ArrayLike,
    k: int,
    *,
    symmetrization: KNearestSymmetrization = "union",
    max_neighbor_selections: int | None = (
        DEFAULT_K_NEAREST_MAX_NEIGHBOR_SELECTIONS
    ),
    max_edges: int | None = DEFAULT_K_NEAREST_MAX_EDGES,
) -> Geometry:
    """Build an undirected graph from exact deterministic kNN selections.

    Each site ranks every other site by ``(Euclidean distance, site index)``
    and selects exactly ``k`` entries. ``union`` retains an edge selected by
    either endpoint; ``mutual`` retains only edges selected by both endpoints.
    Edge metadata records which endpoints made the directed selection.
    """
    point_coordinates = validate_point_coordinates(coordinates)
    n_sites = int(point_coordinates.shape[0])
    k = validate_positive_integer(k, name="k")
    if k >= n_sites:
        raise ValueError("k must be smaller than the number of sites")
    symmetrization = _validate_symmetrization(symmetrization)
    max_neighbor_selections = validate_optional_budget(
        max_neighbor_selections,
        name="max_neighbor_selections",
    )
    max_edges = validate_optional_budget(max_edges, name="max_edges")

    n_neighbor_selections = n_sites * k
    if (
        max_neighbor_selections is not None
        and n_neighbor_selections > max_neighbor_selections
    ):
        raise ValueError(
            f"k_nearest_neighbor_graph with {n_sites} sites and k={k} requires "
            f"{n_neighbor_selections} directed neighbor selections, exceeding "
            f"max_neighbor_selections={max_neighbor_selections}"
        )

    selected_neighbors = _select_nearest_neighbors(point_coordinates, k=k)
    selected_sets = tuple(frozenset(neighbors) for neighbors in selected_neighbors)
    union_edge_keys = {
        _edge_key(source, target)
        for source, neighbors in enumerate(selected_neighbors)
        for target in neighbors
    }
    if symmetrization == "union":
        edge_keys = tuple(sorted(union_edge_keys))
    else:
        edge_keys = tuple(
            edge_key
            for edge_key in sorted(union_edge_keys)
            if edge_key[1] in selected_sets[edge_key[0]]
            and edge_key[0] in selected_sets[edge_key[1]]
        )
    if max_edges is not None and len(edge_keys) > max_edges:
        raise ValueError(
            f"k_nearest_neighbor_graph requires {len(edge_keys)} edges, "
            f"exceeding max_edges={max_edges}"
        )

    edges = tuple(
        _build_edge(
            source,
            target,
            point_coordinates=point_coordinates,
            selected_sets=selected_sets,
        )
        for source, target in edge_keys
    )
    n_mutual_edges = sum(
        len(edge.metadata["selected_by"]) == 2 for edge in edges
    )

    return Geometry(
        n_sites=n_sites,
        edges=edges,
        coordinates=point_coordinates,
        metadata={
            "generator": "k_nearest_neighbor_graph",
            "construction": "euclidean_k_nearest_neighbors",
            "distance_metric": "euclidean_l2",
            "k": k,
            "tie_break": "ascending_site_index",
            "symmetrization": symmetrization,
            "coincident_site_policy": "rank_distinct_indices_at_zero_distance",
            "edge_order": "lexicographic_site_indices",
            "edge_selection_provenance": "edge_metadata_selected_by",
            "spatial_index": "scipy_ckdtree_with_exact_distance_ranking",
            "boundary_inference": "not_performed",
            "n_sites": n_sites,
            "embedding_dimension": int(point_coordinates.shape[1]),
            "n_directed_neighbor_selections": n_neighbor_selections,
            "n_mutual_edges": n_mutual_edges,
            "n_one_sided_edges": len(edges) - n_mutual_edges,
            "max_neighbor_selections": max_neighbor_selections,
            "max_edges": max_edges,
        },
    )


def _validate_symmetrization(value: str) -> KNearestSymmetrization:
    if not isinstance(value, str):
        raise TypeError("symmetrization must be a string")
    if value not in ("union", "mutual"):
        raise ValueError("symmetrization must be either 'union' or 'mutual'")
    return cast(KNearestSymmetrization, value)


def _select_nearest_neighbors(
    point_coordinates: NDArray[np.float64],
    *,
    k: int,
) -> tuple[tuple[int, ...], ...]:
    tree = cKDTree(point_coordinates)
    selections: list[tuple[int, ...]] = []
    for source, source_coordinates in enumerate(point_coordinates):
        _, initial_indices = tree.query(
            source_coordinates,
            k=k + 1,
            p=2.0,
            eps=0.0,
        )
        initial_candidates = {
            int(target)
            for target in np.atleast_1d(initial_indices)
            if int(target) != source
        }
        initial_ranking = _rank_candidates(
            source,
            initial_candidates,
            point_coordinates=point_coordinates,
        )
        if len(initial_ranking) < k:
            raise RuntimeError("spatial index returned too few nearest neighbors")
        kth_distance = initial_ranking[k - 1][0]
        tie_radius = float(np.nextafter(kth_distance, np.inf))
        tie_candidates = {
            int(target)
            for target in tree.query_ball_point(
                source_coordinates,
                tie_radius,
                p=2.0,
                eps=0.0,
            )
            if int(target) != source
        }
        ranked_candidates = _rank_candidates(
            source,
            initial_candidates | tie_candidates,
            point_coordinates=point_coordinates,
        )
        selections.append(tuple(target for _, target in ranked_candidates[:k]))
    return tuple(selections)


def _rank_candidates(
    source: int,
    candidates: set[int],
    *,
    point_coordinates: NDArray[np.float64],
) -> list[tuple[float, int]]:
    return sorted(
        (
            dist(point_coordinates[source], point_coordinates[target]),
            target,
        )
        for target in candidates
    )


def _build_edge(
    source: int,
    target: int,
    *,
    point_coordinates: NDArray[np.float64],
    selected_sets: tuple[frozenset[int], ...],
) -> GeometryEdge:
    displacement = point_coordinates[target] - point_coordinates[source]
    selected_by = tuple(
        selector
        for selector, selected in (
            (source, target in selected_sets[source]),
            (target, source in selected_sets[target]),
        )
        if selected
    )
    return GeometryEdge(
        source,
        target,
        edge_type="k_nearest_neighbor",
        displacement=tuple(float(value) for value in displacement),
        metadata={
            "selected_by": selected_by,
            "mutual": len(selected_by) == 2,
        },
    )


def _edge_key(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first < second else (second, first)
