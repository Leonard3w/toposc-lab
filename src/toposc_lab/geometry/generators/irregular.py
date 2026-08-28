from __future__ import annotations

import numpy as np

from toposc_lab.geometry.base import Geometry, GeometryEdge


def irregular_cluster() -> Geometry:
    """Create a deterministic seven-site irregular reference geometry.

    The graph has nonuniform coordinates and coordination numbers. Most edges
    describe local connections, while one explicitly typed long-range edge
    demonstrates that graph connectivity need not be inferred from Euclidean
    distance.
    """
    coordinates = np.asarray(
        [
            (0.0, 0.0),
            (1.1, 0.2),
            (0.1, 1.3),
            (1.4, 1.1),
            (2.6, 0.4),
            (2.8, 1.8),
            (1.0, 2.5),
        ],
        dtype=float,
    )
    local_connections = (
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 6),
        (3, 4),
        (3, 5),
        (3, 6),
        (4, 5),
        (5, 6),
    )
    edges = tuple(
        GeometryEdge(source, target, edge_type="local")
        for source, target in local_connections
    ) + (GeometryEdge(0, 5, edge_type="long_range"),)

    return Geometry(
        n_sites=7,
        edges=edges,
        coordinates=coordinates,
        boundary_sites=frozenset({0, 2, 4, 5, 6}),
        site_types=("A", "A", "B", "A", "B", "B", "C"),
        metadata={
            "generator": "irregular_cluster",
            "name": "seven_site_reference",
            "intrinsic_dimension": 2,
        },
    )
