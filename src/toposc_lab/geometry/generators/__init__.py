"""Generators for model-independent discrete geometries."""

from toposc_lab.geometry.generators.chain import chain
from toposc_lab.geometry.generators.cubic import cubic
from toposc_lab.geometry.generators.honeycomb import honeycomb
from toposc_lab.geometry.generators.irregular import irregular_cluster
from toposc_lab.geometry.generators.kagome import kagome
from toposc_lab.geometry.generators.protocol import (
    FunctionGeometryGenerator,
    GeometryGenerationRequest,
    GeometryGenerator,
    GeometryGeneratorRegistry,
)
from toposc_lab.geometry.generators.ring import ring
from toposc_lab.geometry.generators.square import square
from toposc_lab.geometry.generators.triangular import triangular

CHAIN_GENERATOR = FunctionGeometryGenerator(key="chain", builder=chain)
CUBIC_GENERATOR = FunctionGeometryGenerator(key="cubic", builder=cubic)
HONEYCOMB_GENERATOR = FunctionGeometryGenerator(key="honeycomb", builder=honeycomb)
IRREGULAR_CLUSTER_GENERATOR = FunctionGeometryGenerator(
    key="irregular_cluster",
    builder=irregular_cluster,
)
KAGOME_GENERATOR = FunctionGeometryGenerator(key="kagome", builder=kagome)
RING_GENERATOR = FunctionGeometryGenerator(key="ring", builder=ring)
SQUARE_GENERATOR = FunctionGeometryGenerator(key="square", builder=square)
TRIANGULAR_GENERATOR = FunctionGeometryGenerator(
    key="triangular",
    builder=triangular,
)

BUILTIN_GEOMETRY_GENERATORS = GeometryGeneratorRegistry(
    (
        CHAIN_GENERATOR,
        CUBIC_GENERATOR,
        HONEYCOMB_GENERATOR,
        IRREGULAR_CLUSTER_GENERATOR,
        KAGOME_GENERATOR,
        RING_GENERATOR,
        SQUARE_GENERATOR,
        TRIANGULAR_GENERATOR,
    )
)

__all__ = [
    "BUILTIN_GEOMETRY_GENERATORS",
    "CHAIN_GENERATOR",
    "CUBIC_GENERATOR",
    "FunctionGeometryGenerator",
    "GeometryGenerationRequest",
    "GeometryGenerator",
    "GeometryGeneratorRegistry",
    "HONEYCOMB_GENERATOR",
    "IRREGULAR_CLUSTER_GENERATOR",
    "KAGOME_GENERATOR",
    "RING_GENERATOR",
    "SQUARE_GENERATOR",
    "TRIANGULAR_GENERATOR",
    "chain",
    "cubic",
    "honeycomb",
    "irregular_cluster",
    "kagome",
    "ring",
    "square",
    "triangular",
]
