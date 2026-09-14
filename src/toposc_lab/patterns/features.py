"""Label-free descriptors; graph spectra are not Hamiltonian spectra."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from types import MappingProxyType

import numpy as np
from scipy.sparse.csgraph import shortest_path

from toposc_lab.geometry import Geometry, extract_geometry_descriptors


@dataclass(frozen=True)
class PatternFeatures:
    values: Mapping[str, float | None]
    adjacency_spectrum: tuple[float, ...]
    laplacian_spectrum: tuple[float, ...]
    normalized_laplacian_spectrum: tuple[float, ...]
    degree_sequence: tuple[int, ...]
    triangles: tuple[tuple[int, ...], ...]
    chordless_squares: tuple[tuple[int, ...], ...]
    ball_growth: tuple[float, ...]
    dimension_status: str = "unavailable: finite graph, no scaling family or justified fit range"
    schema_version: int = 1


def adjacency(geometry: Geometry) -> np.ndarray:
    matrix = np.zeros((geometry.n_sites, geometry.n_sites), dtype=float)
    for edge in geometry.edges:
        matrix[edge.source, edge.target] = matrix[edge.target, edge.source] = 1
    return matrix


def motif_instances(matrix: np.ndarray) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Unordered vertex sets inducing K3 and chordless C4, counted once each."""
    n = len(matrix)
    neighbors = [set(np.flatnonzero(row)) for row in matrix]
    triangles = tuple(
        (a, int(b), int(c))
        for a in range(n)
        for b in sorted(neighbors[a])
        if b > a
        for c in sorted(neighbors[a] & neighbors[b])
        if c > b
    )
    squares = set()
    for a, c in combinations(range(n), 2):
        if matrix[a, c]:
            continue
        for b, d in combinations(sorted(neighbors[a] & neighbors[c]), 2):
            if not matrix[b, d]:
                squares.add(tuple(sorted((a, int(b), c, int(d)))))
    return triangles, tuple(sorted(squares))


def extract_pattern_features(geometry: Geometry) -> PatternFeatures:
    """Simple undirected one-skeleton; path means use reachable unordered pairs.

    Geometry coordinates and explicit boundary labels enter only named features.
    Boundary distance is the mean distance to each bulk site's nearest declared
    boundary; it is unavailable if any bulk site cannot reach a boundary. Edge
    lengths and angles describe endpoint coordinates, not periodic displacements.
    No fractal or effective dimension is fitted from a single finite graph.
    Spectral gaps use ordered algebraic eigenvalues, not absolute eigenvalues.
    """
    matrix = adjacency(geometry)
    n = len(matrix)
    degree = matrix.sum(axis=1)
    values: dict[str, float | None] = {
        k: None if v is None else float(v)
        for k, v in extract_geometry_descriptors(geometry).items()
    }
    for k in range(7):
        values[f"degree_fraction_{k}"] = float(np.mean(degree == k))
    values["degree_fraction_above_6"] = float(np.mean(degree > 6))
    values["degree_minimum"] = float(degree.min())
    values["degree_maximum"] = float(degree.max())
    boundary = np.array([i in geometry.boundary_sites for i in range(n)])
    values["boundary_fraction"] = float(boundary.mean())
    values["boundary_bulk_ratio"] = (
        float(boundary.sum() / (~boundary).sum()) if (~boundary).any() else None
    )
    values["boundary_degree_mean"] = float(degree[boundary].mean()) if boundary.any() else None
    values["bulk_degree_mean"] = float(degree[~boundary].mean()) if (~boundary).any() else None
    values["boundary_bulk_edges"] = float(matrix[np.ix_(boundary, ~boundary)].sum())
    triangles, squares = motif_instances(matrix)
    values["chordless_square_count"] = float(len(squares))
    values["boundary_triangle_count"] = float(sum(any(boundary[i] for i in t) for t in triangles))
    distances = shortest_path(matrix, directed=False, unweighted=True)
    upper = distances[np.triu_indices(n, 1)]
    finite = upper[np.isfinite(upper)]
    values["path_standard_deviation"] = float(finite.std()) if len(finite) else None
    values["global_efficiency"] = float(np.sum(1 / finite) / len(upper)) if len(upper) else None
    values["bulk_boundary_path_mean"] = None
    if boundary.any() and (~boundary).any():
        nearest_boundary = np.min(distances[np.ix_(~boundary, boundary)], axis=1)
        if np.isfinite(nearest_boundary).all():
            values["bulk_boundary_path_mean"] = float(nearest_boundary.mean())
    a_spectrum = np.linalg.eigvalsh(matrix)
    laplacian = np.diag(degree) - matrix
    l_spectrum = np.linalg.eigvalsh(laplacian)
    inverse = np.zeros(n)
    inverse[degree > 0] = 1 / np.sqrt(degree[degree > 0])
    normalized = inverse[:, None] * laplacian * inverse[None, :]
    nl_spectrum = np.linalg.eigvalsh(normalized)
    values.update(
        {
            "adjacency_radius": float(np.max(np.abs(a_spectrum))),
            "adjacency_top_gap": float(a_spectrum[-1] - a_spectrum[-2]) if n > 1 else None,
            "adjacency_energy_per_site": float(np.mean(np.abs(a_spectrum))),
            "algebraic_connectivity": max(0.0, float(l_spectrum[1])) if n > 1 else None,
            "laplacian_radius": float(l_spectrum[-1]),
            "normalized_algebraic_connectivity": max(0.0, float(nl_spectrum[1])) if n > 1 else None,
        }
    )
    growth = tuple(float(np.mean(np.sum(distances <= r, axis=1))) for r in range(1, 5))
    values.update({f"mean_ball_size_r{r}": v for r, v in enumerate(growth, 1)})
    xy = geometry.coordinates
    names = (
        "embedding_dimension",
        "edge_length_mean",
        "edge_length_sd",
        "edge_length_max",
        "coordinate_anisotropy",
        "edge_orientation_anisotropy",
        "bond_angle_cosine_mean",
        "mean_pair_distance",
        "bbox_density",
        "declared_hole_count",
        "diagonal_count",
        "central_diagonal_count",
    )
    values.update(dict.fromkeys(names))
    values["embedding_dimension"] = (
        float(geometry.embedding_dimension) if geometry.embedding_dimension is not None else None
    )
    # Count declarations without inferring holes from graph cycles or coordinates.
    values["declared_hole_count"] = float(
        sum(c.kind == "hole" for c in geometry.boundary_components)
    )
    if xy is not None:
        vectors = np.array([xy[e.source] - xy[e.target] for e in geometry.edges])
        lengths = np.linalg.norm(vectors, axis=1) if len(vectors) else np.array([])
        if len(lengths):
            values.update(
                edge_length_mean=float(lengths.mean()),
                edge_length_sd=float(lengths.std()),
                edge_length_max=float(lengths.max()),
            )
        covariance = (xy - xy.mean(axis=0)).T @ (xy - xy.mean(axis=0)) / n
        eig = np.linalg.eigvalsh(covariance)
        values["coordinate_anisotropy"] = (
            float((eig[-1] - eig[0]) / eig.sum()) if eig.sum() else 0.0
        )
        delta = xy[:, None, :] - xy[None, :, :]
        pair = np.linalg.norm(delta, axis=2)[np.triu_indices(n, 1)]
        values["mean_pair_distance"] = float(pair.mean()) if len(pair) else None
        volume = float(np.prod(np.ptp(xy, axis=0)))
        values["bbox_density"] = n / volume if volume > 0 else None
        nonzero = lengths > 0
        if nonzero.any():
            unit = vectors[nonzero] / lengths[nonzero, None]
            orient = np.linalg.eigvalsh(unit.T @ unit / len(unit))
            values["edge_orientation_anisotropy"] = float(orient[-1] - orient[0])
        cosines = []
        for i in range(n):
            for a, b in combinations(np.flatnonzero(matrix[i]), 2):
                u, v = xy[a] - xy[i], xy[b] - xy[i]
                denominator = np.linalg.norm(u) * np.linalg.norm(v)
                if denominator:
                    cosines.append(float(np.clip(u @ v / denominator, -1, 1)))
        values["bond_angle_cosine_mean"] = float(np.mean(cosines)) if cosines else None
        grid = {(float(x), float(y)) for y in range(6) for x in range(6)}
        if xy.shape == (36, 2) and {tuple(p) for p in xy} == grid:
            diagonal = np.isclose(lengths, np.sqrt(2), rtol=0, atol=1e-12)
            central = np.array(
                [
                    np.linalg.norm((xy[e.source] + xy[e.target]) / 2 - (2.5, 2.5)) <= 1.5
                    for e in geometry.edges
                ],
                dtype=bool,
            )
            values["diagonal_count"] = float(diagonal.sum())
            values["central_diagonal_count"] = float(np.sum(diagonal & central))
    return PatternFeatures(
        MappingProxyType(values),
        tuple(a_spectrum.tolist()),
        tuple(l_spectrum.tolist()),
        tuple(nl_spectrum.tolist()),
        tuple(int(d) for d in degree),
        triangles,
        squares,
        growth,
    )
