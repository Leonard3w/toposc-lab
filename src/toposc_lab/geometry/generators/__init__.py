"""Generators for model-independent discrete geometries."""

from toposc_lab.geometry.generators.ammann_beenker import (
    AMMANN_BEENKER_INFLATION_FACTOR,
    DEFAULT_AMMANN_BEENKER_MAX_CANDIDATES,
    DEFAULT_AMMANN_BEENKER_MAX_SITES,
    ammann_beenker_patch,
)
from toposc_lab.geometry.generators.artificial_rule import (
    DEFAULT_ARTIFICIAL_RULE_MAX_APPLICATIONS,
    DEFAULT_ARTIFICIAL_RULE_MAX_EDGES,
    DEFAULT_ARTIFICIAL_RULE_MAX_SITES,
    artificial_rule_graph,
)
from toposc_lab.geometry.generators.body_centered_cubic import body_centered_cubic
from toposc_lab.geometry.generators.cayley_tree import (
    DEFAULT_CAYLEY_TREE_MAX_SITES,
    cayley_tree,
)
from toposc_lab.geometry.generators.chain import chain
from toposc_lab.geometry.generators.coordinate_cutoff import (
    DEFAULT_COORDINATE_CUTOFF_MAX_EDGES,
    coordinate_cutoff_graph,
)
from toposc_lab.geometry.generators.cubic import cubic
from toposc_lab.geometry.generators.fibonacci_chain import (
    DEFAULT_FIBONACCI_CHAIN_MAX_SITES,
    FIBONACCI_GOLDEN_RATIO,
    fibonacci_chain,
)
from toposc_lab.geometry.generators.honeycomb import honeycomb
from toposc_lab.geometry.generators.irregular import irregular_cluster
from toposc_lab.geometry.generators.k_nearest_neighbor import (
    DEFAULT_K_NEAREST_MAX_EDGES,
    DEFAULT_K_NEAREST_MAX_NEIGHBOR_SELECTIONS,
    KNearestSymmetrization,
    k_nearest_neighbor_graph,
)
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
from toposc_lab.geometry.generators.scale_free import (
    DEFAULT_SCALE_FREE_MAX_EDGES,
    scale_free_graph,
)
from toposc_lab.geometry.generators.sierpinski import sierpinski_gasket
from toposc_lab.geometry.generators.sierpinski_carpet import sierpinski_carpet
from toposc_lab.geometry.generators.silver_mean_chain import (
    DEFAULT_SILVER_MEAN_CHAIN_MAX_SITES,
    SILVER_MEAN_RATIO,
    silver_mean_chain,
)
from toposc_lab.geometry.generators.small_world import (
    DEFAULT_SMALL_WORLD_MAX_EDGES,
    small_world_network,
)
from toposc_lab.geometry.generators.square import square
from toposc_lab.geometry.generators.triangular import triangular
from toposc_lab.geometry.generators.tree import tree

AMMANN_BEENKER_PATCH_GENERATOR = FunctionGeometryGenerator(
    key="ammann_beenker_patch",
    builder=ammann_beenker_patch,
)
ARTIFICIAL_RULE_GRAPH_GENERATOR = FunctionGeometryGenerator(
    key="artificial_rule_graph",
    builder=artificial_rule_graph,
)
BODY_CENTERED_CUBIC_GENERATOR = FunctionGeometryGenerator(
    key="body_centered_cubic",
    builder=body_centered_cubic,
)
CAYLEY_TREE_GENERATOR = FunctionGeometryGenerator(
    key="cayley_tree",
    builder=cayley_tree,
)
CHAIN_GENERATOR = FunctionGeometryGenerator(key="chain", builder=chain)
COORDINATE_CUTOFF_GRAPH_GENERATOR = FunctionGeometryGenerator(
    key="coordinate_cutoff_graph",
    builder=coordinate_cutoff_graph,
)
CUBIC_GENERATOR = FunctionGeometryGenerator(key="cubic", builder=cubic)
FIBONACCI_CHAIN_GENERATOR = FunctionGeometryGenerator(
    key="fibonacci_chain",
    builder=fibonacci_chain,
)
HONEYCOMB_GENERATOR = FunctionGeometryGenerator(key="honeycomb", builder=honeycomb)
IRREGULAR_CLUSTER_GENERATOR = FunctionGeometryGenerator(
    key="irregular_cluster",
    builder=irregular_cluster,
)
K_NEAREST_NEIGHBOR_GRAPH_GENERATOR = FunctionGeometryGenerator(
    key="k_nearest_neighbor_graph",
    builder=k_nearest_neighbor_graph,
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
SCALE_FREE_GRAPH_GENERATOR = FunctionGeometryGenerator(
    key="scale_free_graph",
    builder=scale_free_graph,
    stochastic=True,
)
SIERPINSKI_CARPET_GENERATOR = FunctionGeometryGenerator(
    key="sierpinski_carpet",
    builder=sierpinski_carpet,
)
SIERPINSKI_GASKET_GENERATOR = FunctionGeometryGenerator(
    key="sierpinski_gasket",
    builder=sierpinski_gasket,
)
SILVER_MEAN_CHAIN_GENERATOR = FunctionGeometryGenerator(
    key="silver_mean_chain",
    builder=silver_mean_chain,
)
SMALL_WORLD_NETWORK_GENERATOR = FunctionGeometryGenerator(
    key="small_world_network",
    builder=small_world_network,
    stochastic=True,
)
SQUARE_GENERATOR = FunctionGeometryGenerator(key="square", builder=square)
TREE_GENERATOR = FunctionGeometryGenerator(key="tree", builder=tree)
TRIANGULAR_GENERATOR = FunctionGeometryGenerator(
    key="triangular",
    builder=triangular,
)

BUILTIN_GEOMETRY_GENERATORS = GeometryGeneratorRegistry(
    (
        AMMANN_BEENKER_PATCH_GENERATOR,
        ARTIFICIAL_RULE_GRAPH_GENERATOR,
        BODY_CENTERED_CUBIC_GENERATOR,
        CAYLEY_TREE_GENERATOR,
        CHAIN_GENERATOR,
        COORDINATE_CUTOFF_GRAPH_GENERATOR,
        CUBIC_GENERATOR,
        FIBONACCI_CHAIN_GENERATOR,
        HONEYCOMB_GENERATOR,
        IRREGULAR_CLUSTER_GENERATOR,
        K_NEAREST_NEIGHBOR_GRAPH_GENERATOR,
        KAGOME_GENERATOR,
        MENGER_SPONGE_GENERATOR,
        RANDOM_GRAPH_GENERATOR,
        RANDOM_REGULAR_GRAPH_GENERATOR,
        RING_GENERATOR,
        SCALE_FREE_GRAPH_GENERATOR,
        SIERPINSKI_CARPET_GENERATOR,
        SIERPINSKI_GASKET_GENERATOR,
        SILVER_MEAN_CHAIN_GENERATOR,
        SMALL_WORLD_NETWORK_GENERATOR,
        SQUARE_GENERATOR,
        TREE_GENERATOR,
        TRIANGULAR_GENERATOR,
    )
)

__all__ = [
    "AMMANN_BEENKER_INFLATION_FACTOR",
    "AMMANN_BEENKER_PATCH_GENERATOR",
    "ARTIFICIAL_RULE_GRAPH_GENERATOR",
    "BUILTIN_GEOMETRY_GENERATORS",
    "BODY_CENTERED_CUBIC_GENERATOR",
    "CAYLEY_TREE_GENERATOR",
    "CHAIN_GENERATOR",
    "COORDINATE_CUTOFF_GRAPH_GENERATOR",
    "CUBIC_GENERATOR",
    "DEFAULT_CAYLEY_TREE_MAX_SITES",
    "DEFAULT_ARTIFICIAL_RULE_MAX_APPLICATIONS",
    "DEFAULT_ARTIFICIAL_RULE_MAX_EDGES",
    "DEFAULT_ARTIFICIAL_RULE_MAX_SITES",
    "DEFAULT_COORDINATE_CUTOFF_MAX_EDGES",
    "DEFAULT_AMMANN_BEENKER_MAX_CANDIDATES",
    "DEFAULT_AMMANN_BEENKER_MAX_SITES",
    "DEFAULT_FIBONACCI_CHAIN_MAX_SITES",
    "DEFAULT_K_NEAREST_MAX_EDGES",
    "DEFAULT_K_NEAREST_MAX_NEIGHBOR_SELECTIONS",
    "DEFAULT_MENGER_MAX_SITES",
    "DEFAULT_RANDOM_GRAPH_MAX_CANDIDATE_EDGES",
    "DEFAULT_RANDOM_REGULAR_MAX_ATTEMPTS",
    "DEFAULT_RANDOM_REGULAR_MAX_EDGES",
    "DEFAULT_SCALE_FREE_MAX_EDGES",
    "DEFAULT_SILVER_MEAN_CHAIN_MAX_SITES",
    "DEFAULT_SMALL_WORLD_MAX_EDGES",
    "FunctionGeometryGenerator",
    "FIBONACCI_CHAIN_GENERATOR",
    "FIBONACCI_GOLDEN_RATIO",
    "GeometryGenerationRequest",
    "GeometryGenerator",
    "GeometryGeneratorRegistry",
    "HONEYCOMB_GENERATOR",
    "IRREGULAR_CLUSTER_GENERATOR",
    "KNearestSymmetrization",
    "K_NEAREST_NEIGHBOR_GRAPH_GENERATOR",
    "KAGOME_GENERATOR",
    "MENGER_SPONGE_GENERATOR",
    "RANDOM_GRAPH_GENERATOR",
    "RANDOM_REGULAR_GRAPH_GENERATOR",
    "RING_GENERATOR",
    "SCALE_FREE_GRAPH_GENERATOR",
    "SIERPINSKI_CARPET_GENERATOR",
    "SIERPINSKI_GASKET_GENERATOR",
    "SILVER_MEAN_CHAIN_GENERATOR",
    "SILVER_MEAN_RATIO",
    "SMALL_WORLD_NETWORK_GENERATOR",
    "SQUARE_GENERATOR",
    "TREE_GENERATOR",
    "TRIANGULAR_GENERATOR",
    "body_centered_cubic",
    "ammann_beenker_patch",
    "artificial_rule_graph",
    "cayley_tree",
    "chain",
    "coordinate_cutoff_graph",
    "cubic",
    "fibonacci_chain",
    "honeycomb",
    "irregular_cluster",
    "k_nearest_neighbor_graph",
    "kagome",
    "menger_sponge",
    "random_graph",
    "random_regular_graph",
    "ring",
    "scale_free_graph",
    "sierpinski_carpet",
    "sierpinski_gasket",
    "silver_mean_chain",
    "small_world_network",
    "square",
    "tree",
    "triangular",
]
