"""General, model-independent discrete geometries."""

from toposc_lab.geometry.base import Geometry, GeometryEdge
from toposc_lab.geometry.generators import chain, irregular_cluster, ring, square

__all__ = [
    "Geometry",
    "GeometryEdge",
    "chain",
    "irregular_cluster",
    "ring",
    "square",
]
