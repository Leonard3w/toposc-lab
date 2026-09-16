"""Extensible graph-only descriptors; no exact labels enter search features."""
from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from scipy.sparse.csgraph import shortest_path

from toposc_lab.geometry import Geometry
from toposc_lab.geometry.descriptors import extract_geometry_descriptors

DESCRIPTOR_REGISTRY: dict[str, Callable[[Geometry], float]] = {}
DEFAULT_BEHAVIOR_DESCRIPTORS = ("coordination_variance", "clustering_coefficient")
DEFAULT_DESCRIPTOR_BOUNDS = ((0.0, 8.0), (0.0, 1.0))


def register_descriptor(name: str, function: Callable[[Geometry], float]) -> None:
    if not name or name in DESCRIPTOR_REGISTRY:
        raise ValueError(f"duplicate/empty descriptor name: {name}")
    DESCRIPTOR_REGISTRY[name] = function


def _all(geometry: Geometry) -> dict[str, float]:
    # Preserve validated graph descriptor definitions, extend with explicit metrics.
    result = {name: float(value) for name, value in extract_geometry_descriptors(geometry).items()
              if value is not None}
    n = geometry.n_sites
    adjacency = np.zeros((n, n), dtype=float)
    for edge in geometry.edges:
        adjacency[edge.source, edge.target] = adjacency[edge.target, edge.source] = 1
    degrees = adjacency.sum(axis=1)
    common = adjacency @ adjacency
    upper_common = common[np.triu_indices(n, 1)]
    four_cycles = np.sum(upper_common * (upper_common - 1) / 2) / 2
    laplacian = np.diag(degrees) - adjacency
    eigenvalues = np.linalg.eigvalsh(laplacian)
    boundary = np.array(sorted(geometry.boundary_sites), dtype=int)
    bulk = np.array(sorted(set(range(n)) - geometry.boundary_sites), dtype=int)
    distances = shortest_path(adjacency, directed=False, unweighted=True)
    paths = distances[np.triu_indices(n, 1)]
    paths = paths[np.isfinite(paths)]
    result.update({
        "mean_coordination": float(degrees.mean()),
        "coordination_variance": float(degrees.var()),
        "clustering_coefficient": result["mean_local_clustering"],
        "four_cycle_count": float(four_cycles),
        "edge_density": float(2 * geometry.n_edges / max(1, n * (n - 1))),
        "minimum_degree": float(degrees.min()), "maximum_degree": float(degrees.max()),
        "boundary_degree_mean": float(degrees[boundary].mean()) if len(boundary) else 0.0,
        "boundary_degree_variance": float(degrees[boundary].var()) if len(boundary) else 0.0,
        "bulk_degree_mean": float(degrees[bulk].mean()) if len(bulk) else 0.0,
        "bulk_degree_variance": float(degrees[bulk].var()) if len(bulk) else 0.0,
        "boundary_site_count": float(len(boundary)), "bulk_site_count": float(len(bulk)),
        "graph_diameter": float(paths.max()) if len(paths) else 0.0,
        "shortest_path_mean": float(paths.mean()) if len(paths) else 0.0,
        "shortest_path_variance": float(paths.var()) if len(paths) else 0.0,
        "laplacian_algebraic_connectivity": float(max(0, eigenvalues[1])) if n > 1 else 0,
        "laplacian_spectral_radius": float(eigenvalues[-1]),
        "local_connectivity_disorder": float(np.mean([
            (degrees[e.source] - degrees[e.target])**2 for e in geometry.edges
        ])) if geometry.n_edges else 0.0,
    })
    # Loop summary explicitly describes only enumerated triangles and 4-cycles.
    triangles = result["triangle_count"]
    loops = triangles + four_cycles
    result["short_loop_count"] = float(loops)
    result["short_loop_mean_size"] = float((3 * triangles + 4 * four_cycles) / loops) if loops else 0
    if geometry.coordinates is not None and geometry.n_edges:
        vectors = np.array([geometry.coordinates[e.target] - geometry.coordinates[e.source]
                            for e in geometry.edges])
        lengths = np.linalg.norm(vectors, axis=1)
        directions = vectors / np.maximum(lengths[:, None], np.finfo(float).eps)
        tensor = directions.T @ directions / len(directions)
        eigen = np.linalg.eigvalsh(tensor)
        result.update({"bond_length_mean": float(lengths.mean()),
                       "bond_length_variance": float(lengths.var()),
                       "bond_length_minimum": float(lengths.min()),
                       "bond_length_maximum": float(lengths.max()),
                       "bond_length_q25": float(np.quantile(lengths, 0.25)),
                       "bond_length_median": float(np.median(lengths)),
                       "bond_length_q75": float(np.quantile(lengths, 0.75)),
                       "anisotropy": float(eigen[-1] - eigen[0])})
    return result


def compute_descriptors(geometry: Geometry, names: Iterable[str] | None = None) -> dict[str, float]:
    result = _all(geometry)
    for name, function in DESCRIPTOR_REGISTRY.items():
        if name not in _BUILTIN_NAMES:
            result[name] = float(function(geometry))
    if names is not None:
        result = {name: result[name] for name in names}
    if any(not np.isfinite(value) for value in result.values()):
        raise ValueError("geometry descriptor returned NaN or infinity")
    return result


# Registry entries are individually callable, while compute_descriptors shares work.
_BUILTIN_NAMES = (
    "mean_coordination", "coordination_variance", "clustering_coefficient", "triangle_count",
    "four_cycle_count", "cycle_rank", "short_loop_count", "short_loop_mean_size", "edge_density",
    "boundary_degree_mean", "boundary_degree_variance", "bulk_degree_mean", "bulk_degree_variance",
    "graph_diameter", "shortest_path_mean", "shortest_path_variance",
    "laplacian_algebraic_connectivity", "laplacian_spectral_radius", "bond_length_mean",
    "bond_length_variance", "bond_length_minimum", "bond_length_maximum", "anisotropy",
    "local_connectivity_disorder", "mean_degree", "degree_variance", "mean_local_clustering",
)
def _registered_builtin(name: str) -> Callable[[Geometry], float]:
    def compute(geometry: Geometry) -> float:
        return _all(geometry)[name]
    return compute


for _name in _BUILTIN_NAMES:
    register_descriptor(_name, _registered_builtin(_name))
