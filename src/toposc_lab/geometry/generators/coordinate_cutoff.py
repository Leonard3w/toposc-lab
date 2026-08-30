"""Generic Euclidean coordinate-cutoff graph builder."""

from __future__ import annotations

from math import dist

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from toposc_lab.geometry.base import Geometry, GeometryEdge
from toposc_lab.geometry.generators._point_cloud import (
    validate_point_coordinates,
    validate_positive_distance,
)
from toposc_lab.geometry.generators._validation import validate_optional_budget

DEFAULT_COORDINATE_CUTOFF_MAX_EDGES = 1_000_000


def coordinate_cutoff_graph(
    coordinates: ArrayLike,
    cutoff: float,
    *,
    max_edges: int | None = DEFAULT_COORDINATE_CUTOFF_MAX_EDGES,
) -> Geometry:
    """Connect every distinct site pair at Euclidean distance ``<= cutoff``.

    Coordinates may have any positive embedding dimension. A spatial index
    finds candidate neighbors, after which the public inclusive distance rule
    is evaluated exactly in deterministic lexicographic site order. Distinct
    sites with coincident coordinates are connected by a zero-length edge.
    """
    point_coordinates = validate_point_coordinates(coordinates)
    cutoff = validate_positive_distance(cutoff, name="cutoff")
    max_edges = validate_optional_budget(max_edges, name="max_edges")

    tree = cKDTree(point_coordinates)
    search_radius = float(np.nextafter(cutoff, np.inf))
    edges: list[GeometryEdge] = []
    for source, source_coordinates in enumerate(point_coordinates):
        candidate_targets = sorted(
            int(target)
            for target in tree.query_ball_point(source_coordinates, search_radius)
            if int(target) > source
        )
        for target in candidate_targets:
            target_coordinates = point_coordinates[target]
            if dist(source_coordinates, target_coordinates) > cutoff:
                continue
            displacement = target_coordinates - source_coordinates
            edges.append(
                GeometryEdge(
                    source,
                    target,
                    edge_type="distance_cutoff",
                    displacement=tuple(float(value) for value in displacement),
                )
            )
            if max_edges is not None and len(edges) > max_edges:
                raise ValueError(
                    "coordinate_cutoff_graph requires more than "
                    f"max_edges={max_edges}"
                )

    return Geometry(
        n_sites=int(point_coordinates.shape[0]),
        edges=tuple(edges),
        coordinates=point_coordinates,
        metadata={
            "generator": "coordinate_cutoff_graph",
            "construction": "euclidean_coordinate_cutoff",
            "distance_metric": "euclidean_l2",
            "distance_relation": "less_than_or_equal",
            "cutoff": cutoff,
            "coincident_site_policy": "connect_distinct_indices",
            "edge_order": "lexicographic_site_indices",
            "spatial_index": "scipy_ckdtree_with_exact_distance_filter",
            "boundary_inference": "not_performed",
            "n_sites": int(point_coordinates.shape[0]),
            "embedding_dimension": int(point_coordinates.shape[1]),
            "max_edges": max_edges,
        },
    )
