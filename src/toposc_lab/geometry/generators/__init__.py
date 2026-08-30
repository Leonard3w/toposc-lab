"""Generators for model-independent discrete geometries."""

from toposc_lab.geometry.generators.body_centered_cubic import body_centered_cubic
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
from toposc_lab.geometry.generators.sierpinski import sierpinski_gasket
from toposc_lab.geometry.generators.sierpinski_carpet import sierpinski_carpet
from toposc_lab.geometry.generators.square import square
from toposc_lab.geometry.generators.triangular import triangular

BODY_CENTERED_CUBIC_GENERATOR = FunctionGeometryGenerator(
    key="body_centered_cubic",
    builder=body_centered_cubic,
)
CHAIN_GENERATOR = FunctionGeometryGenerator(key="chain", builder=chain)
CUBIC_GENERATOR = FunctionGeometryGenerator(key="cubic", builder=cubic)
HONEYCOMB_GENERATOR = FunctionGeometryGenerator(key="honeycomb", builder=honeycomb)
IRREGULAR_CLUSTER_GENERATOR = FunctionGeometryGenerator(
    key="irregular_cluster",
    builder=irregular_cluster,
)
KAGOME_GENERATOR = FunctionGeometryGenerator(key="kagome", builder=kagome)
RING_GENERATOR = FunctionGeometryGenerator(key="ring", builder=ring)
SIERPINSKI_CARPET_GENERATOR = FunctionGeometryGenerator(
    key="sierpinski_carpet",
    builder=sierpinski_carpet,
)
SIERPINSKI_GASKET_GENERATOR = FunctionGeometryGenerator(
    key="sierpinski_gasket",
    builder=sierpinski_gasket,
)
SQUARE_GENERATOR = FunctionGeometryGenerator(key="square", builder=square)
TRIANGULAR_GENERATOR = FunctionGeometryGenerator(
    key="triangular",
    builder=triangular,
)

BUILTIN_GEOMETRY_GENERATORS = GeometryGeneratorRegistry(
    (
        BODY_CENTERED_CUBIC_GENERATOR,
        CHAIN_GENERATOR,
        CUBIC_GENERATOR,
        HONEYCOMB_GENERATOR,
        IRREGULAR_CLUSTER_GENERATOR,
        KAGOME_GENERATOR,
        RING_GENERATOR,
        SIERPINSKI_CARPET_GENERATOR,
        SIERPINSKI_GASKET_GENERATOR,
        SQUARE_GENERATOR,
        TRIANGULAR_GENERATOR,
    )
)

__all__ = [
    "BUILTIN_GEOMETRY_GENERATORS",
    "BODY_CENTERED_CUBIC_GENERATOR",
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
    "SIERPINSKI_CARPET_GENERATOR",
    "SIERPINSKI_GASKET_GENERATOR",
    "SQUARE_GENERATOR",
    "TRIANGULAR_GENERATOR",
    "body_centered_cubic",
    "chain",
    "cubic",
    "honeycomb",
    "irregular_cluster",
    "kagome",
    "ring",
    "sierpinski_carpet",
    "sierpinski_gasket",
    "square",
    "triangular",
]
