"""Generators for model-independent discrete geometries."""

from toposc_lab.geometry.generators.chain import chain
from toposc_lab.geometry.generators.irregular import irregular_cluster
from toposc_lab.geometry.generators.protocol import (
    FunctionGeometryGenerator,
    GeometryGenerationRequest,
    GeometryGenerator,
    GeometryGeneratorRegistry,
)
from toposc_lab.geometry.generators.ring import ring
from toposc_lab.geometry.generators.square import square

CHAIN_GENERATOR = FunctionGeometryGenerator(key="chain", builder=chain)
IRREGULAR_CLUSTER_GENERATOR = FunctionGeometryGenerator(
    key="irregular_cluster",
    builder=irregular_cluster,
)
RING_GENERATOR = FunctionGeometryGenerator(key="ring", builder=ring)
SQUARE_GENERATOR = FunctionGeometryGenerator(key="square", builder=square)

BUILTIN_GEOMETRY_GENERATORS = GeometryGeneratorRegistry(
    (
        CHAIN_GENERATOR,
        IRREGULAR_CLUSTER_GENERATOR,
        RING_GENERATOR,
        SQUARE_GENERATOR,
    )
)

__all__ = [
    "BUILTIN_GEOMETRY_GENERATORS",
    "CHAIN_GENERATOR",
    "FunctionGeometryGenerator",
    "GeometryGenerationRequest",
    "GeometryGenerator",
    "GeometryGeneratorRegistry",
    "IRREGULAR_CLUSTER_GENERATOR",
    "RING_GENERATOR",
    "SQUARE_GENERATOR",
    "chain",
    "irregular_cluster",
    "ring",
    "square",
]
