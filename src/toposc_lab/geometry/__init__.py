"""General, model-independent discrete geometries."""

from toposc_lab.geometry.base import Geometry, GeometryEdge
from toposc_lab.geometry.generators import (
    BUILTIN_GEOMETRY_GENERATORS,
    CHAIN_GENERATOR,
    IRREGULAR_CLUSTER_GENERATOR,
    RING_GENERATOR,
    SQUARE_GENERATOR,
    TRIANGULAR_GENERATOR,
    FunctionGeometryGenerator,
    GeometryGenerationRequest,
    GeometryGenerator,
    GeometryGeneratorRegistry,
    chain,
    irregular_cluster,
    ring,
    square,
    triangular,
)

__all__ = [
    "BUILTIN_GEOMETRY_GENERATORS",
    "CHAIN_GENERATOR",
    "FunctionGeometryGenerator",
    "Geometry",
    "GeometryEdge",
    "GeometryGenerationRequest",
    "GeometryGenerator",
    "GeometryGeneratorRegistry",
    "IRREGULAR_CLUSTER_GENERATOR",
    "RING_GENERATOR",
    "SQUARE_GENERATOR",
    "TRIANGULAR_GENERATOR",
    "chain",
    "irregular_cluster",
    "ring",
    "square",
    "triangular",
]
