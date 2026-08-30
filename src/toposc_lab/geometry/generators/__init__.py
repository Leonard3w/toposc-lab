"""Generators for model-independent discrete geometries."""

from toposc_lab.geometry.generators.body_centered_cubic import body_centered_cubic
from toposc_lab.geometry.generators.chain import chain
from toposc_lab.geometry.generators.cubic import cubic
from toposc_lab.geometry.generators.honeycomb import honeycomb
from toposc_lab.geometry.generators.irregular import irregular_cluster
from toposc_lab.geometry.generators.kagome import kagome
from toposc_lab.geometry.generators.menger_sponge import (
    DEFAULT_MENGER_MAX_SITES,
    menger_sponge,
)
from toposc_lab.geometry.generators.protocol import (
    FunctionGeometryGenerator,
    GeometryGenerationRequest,
    GeometryGenerator,
    GeometryGeneratorRegistry,
)
from toposc_lab.geometry.generators.random_graph import (
    DEFAULT_RANDOM_GRAPH_MAX_CANDIDATE_EDGES,
    random_graph,
)
from toposc_lab.geometry.generators.random_regular import (
    DEFAULT_RANDOM_REGULAR_MAX_ATTEMPTS,
    DEFAULT_RANDOM_REGULAR_MAX_EDGES,
    random_regular_graph,
)
from toposc_lab.geometry.generators.ring import ring
from toposc_lab.geometry.generators.sierpinski import sierpinski_gasket
from toposc_lab.geometry.generators.sierpinski_carpet import sierpinski_carpet
from toposc_lab.geometry.generators.small_world import (
    DEFAULT_SMALL_WORLD_MAX_EDGES,
    small_world_network,
)
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
MENGER_SPONGE_GENERATOR = FunctionGeometryGenerator(
    key="menger_sponge",
    builder=menger_sponge,
)
RANDOM_GRAPH_GENERATOR = FunctionGeometryGenerator(
    key="random_graph",
    builder=random_graph,
    stochastic=True,
)
RANDOM_REGULAR_GRAPH_GENERATOR = FunctionGeometryGenerator(
    key="random_regular_graph",
    builder=random_regular_graph,
    stochastic=True,
)
RING_GENERATOR = FunctionGeometryGenerator(key="ring", builder=ring)
SIERPINSKI_CARPET_GENERATOR = FunctionGeometryGenerator(
    key="sierpinski_carpet",
    builder=sierpinski_carpet,
)
SIERPINSKI_GASKET_GENERATOR = FunctionGeometryGenerator(
    key="sierpinski_gasket",
    builder=sierpinski_gasket,
)
SMALL_WORLD_NETWORK_GENERATOR = FunctionGeometryGenerator(
    key="small_world_network",
    builder=small_world_network,
    stochastic=True,
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
        MENGER_SPONGE_GENERATOR,
        RANDOM_GRAPH_GENERATOR,
        RANDOM_REGULAR_GRAPH_GENERATOR,
        RING_GENERATOR,
        SIERPINSKI_CARPET_GENERATOR,
        SIERPINSKI_GASKET_GENERATOR,
        SMALL_WORLD_NETWORK_GENERATOR,
        SQUARE_GENERATOR,
        TRIANGULAR_GENERATOR,
    )
)

__all__ = [
    "BUILTIN_GEOMETRY_GENERATORS",
    "BODY_CENTERED_CUBIC_GENERATOR",
    "CHAIN_GENERATOR",
    "CUBIC_GENERATOR",
    "DEFAULT_MENGER_MAX_SITES",
    "DEFAULT_RANDOM_GRAPH_MAX_CANDIDATE_EDGES",
    "DEFAULT_RANDOM_REGULAR_MAX_ATTEMPTS",
    "DEFAULT_RANDOM_REGULAR_MAX_EDGES",
    "DEFAULT_SMALL_WORLD_MAX_EDGES",
    "FunctionGeometryGenerator",
    "GeometryGenerationRequest",
    "GeometryGenerator",
    "GeometryGeneratorRegistry",
    "HONEYCOMB_GENERATOR",
    "IRREGULAR_CLUSTER_GENERATOR",
    "KAGOME_GENERATOR",
    "MENGER_SPONGE_GENERATOR",
    "RANDOM_GRAPH_GENERATOR",
    "RANDOM_REGULAR_GRAPH_GENERATOR",
    "RING_GENERATOR",
    "SIERPINSKI_CARPET_GENERATOR",
    "SIERPINSKI_GASKET_GENERATOR",
    "SMALL_WORLD_NETWORK_GENERATOR",
    "SQUARE_GENERATOR",
    "TRIANGULAR_GENERATOR",
    "body_centered_cubic",
    "chain",
    "cubic",
    "honeycomb",
    "irregular_cluster",
    "kagome",
    "menger_sponge",
    "random_graph",
    "random_regular_graph",
    "ring",
    "sierpinski_carpet",
    "sierpinski_gasket",
    "small_world_network",
    "square",
    "triangular",
]
