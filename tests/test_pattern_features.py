from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest

from toposc_lab.generative.space import GeometrySearchSpace
from toposc_lab.geometry import Geometry, GeometryBoundaryComponent, GeometryEdge
from toposc_lab.patterns.features import extract_pattern_features


def graph(n, edges, **kwargs):
    return Geometry(n_sites=n, edges=tuple(GeometryEdge(*e) for e in edges), **kwargs)


def test_graph_motifs_paths_and_disconnection():
    square = extract_pattern_features(graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)]))
    assert square.chordless_squares == ((0, 1, 2, 3),)
    assert square.values["triangle_count"] == 0
    assert square.values["mean_finite_shortest_path_length"] == pytest.approx(4 / 3)
    chorded = extract_pattern_features(graph(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]))
    assert len(chorded.triangles) == 2
    assert not chorded.chordless_squares
    disconnected = extract_pattern_features(graph(4, [(0, 1), (2, 3)]))
    assert disconnected.values["reachable_pair_fraction"] == pytest.approx(1 / 3)
    assert disconnected.values["global_efficiency"] == pytest.approx(1 / 3)


def test_spectra_known_cycle_and_isolated_vertices():
    result = extract_pattern_features(graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)]))
    np.testing.assert_allclose(result.laplacian_spectrum, [0, 2, 2, 4], atol=1e-12)
    np.testing.assert_allclose(result.normalized_laplacian_spectrum, [0, 1, 1, 2], atol=1e-12)
    assert result.values["algebraic_connectivity"] == pytest.approx(2)
    disconnected = extract_pattern_features(graph(3, [(0, 1)]))
    np.testing.assert_allclose(disconnected.normalized_laplacian_spectrum, [0, 0, 2], atol=1e-12)


def test_coordinate_and_boundary_features_and_relabeling():
    space = GeometrySearchSpace()
    g = space.sample(np.random.default_rng(16001))
    first = extract_pattern_features(g)
    permutation = np.random.default_rng(3).permutation(36)
    xy = np.empty_like(g.coordinates)
    xy[permutation] = g.coordinates
    other = graph(
        36,
        [(int(permutation[e.source]), int(permutation[e.target])) for e in g.edges],
        coordinates=xy,
        boundary_sites=frozenset(int(permutation[i]) for i in g.boundary_sites),
    )
    second = extract_pattern_features(other)
    for k, value in first.values.items():
        assert (
            second.values[k] == pytest.approx(value, abs=1e-12)
            if value is not None
            else second.values[k] is None
        )
    assert first.values["boundary_fraction"] == pytest.approx(20 / 36)
    assert first.values["bbox_density"] == pytest.approx(36 / 25)
    assert first.values["coordinate_anisotropy"] == 0
    assert extract_pattern_features(graph(2, [(0, 1)])).values["bbox_density"] is None


def test_dimension_is_unavailable_and_growth_is_finite():
    result = extract_pattern_features(graph(5, [(0, 1), (1, 2), (2, 3), (3, 4)]))
    assert result.dimension_status.startswith("unavailable")
    assert result.ball_growth == pytest.approx((2.6, 3.8, 4.6, 5))
    assert all("fractal" not in key for key in result.values)


def test_bulk_distance_uses_nearest_boundary_in_each_component():
    disconnected = graph(4, [(0, 1), (2, 3)], boundary_sites=frozenset({0, 2}))
    assert extract_pattern_features(disconnected).values["bulk_boundary_path_mean"] == 1
    no_boundary_in_second_component = replace(disconnected, boundary_sites=frozenset({0}))
    assert (
        extract_pattern_features(no_boundary_in_second_component).values["bulk_boundary_path_mean"]
        is None
    )


def test_declared_geometry_labels_do_not_require_coordinates():
    geometry = graph(
        4,
        [(0, 1), (1, 2), (2, 3), (3, 0)],
        embedding_dimension=2,
        boundary_sites=frozenset({0, 1}),
        boundary_components=(GeometryBoundaryComponent("hole", 0, frozenset({0, 1})),),
    )
    result = extract_pattern_features(geometry)
    assert result.values["declared_hole_count"] == 1
    assert result.values["embedding_dimension"] == 2
    assert result.values["edge_length_mean"] is None
    assert result.dimension_status.startswith("unavailable")
    undeclared = extract_pattern_features(graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)]))
    assert undeclared.values["declared_hole_count"] == 0


def test_edgeless_embedding_and_singleton_have_no_undefined_numeric_values():
    xy = np.array([(x, y) for y in range(6) for x in range(6)], dtype=float)
    edgeless = extract_pattern_features(graph(36, [], coordinates=xy))
    assert edgeless.values["central_diagonal_count"] == 0
    assert edgeless.values["diagonal_count"] == 0
    assert edgeless.values["edge_length_mean"] is None
    assert edgeless.values["global_efficiency"] == 0
    assert edgeless.ball_growth == (1, 1, 1, 1)
    singleton = extract_pattern_features(graph(1, [], coordinates=np.zeros((1, 2))))
    assert singleton.values["global_efficiency"] is None
    assert singleton.values["bulk_boundary_path_mean"] is None
    assert singleton.values["algebraic_connectivity"] is None
    assert singleton.values["bbox_density"] is None
    for result in (edgeless, singleton):
        assert all(value is None or np.isfinite(value) for value in result.values.values())


@pytest.mark.parametrize("reflection", [False, True])
@pytest.mark.parametrize("quarter_turns", range(4))
def test_all_scalar_features_are_d4_invariant(reflection, quarter_turns):
    geometry = GeometrySearchSpace().sample(np.random.default_rng(16001))
    xy = geometry.coordinates.copy()
    for _ in range(quarter_turns):
        xy = np.column_stack((5 - xy[:, 1], xy[:, 0]))
    if reflection:
        xy[:, 0] = 5 - xy[:, 0]
    original = extract_pattern_features(geometry)
    transformed = extract_pattern_features(replace(geometry, coordinates=xy))
    for name, value in original.values.items():
        if value is None:
            assert transformed.values[name] is None
        else:
            assert transformed.values[name] == pytest.approx(value, abs=1e-12)


def test_motif_counts_match_induced_vertex_sets_for_every_five_vertex_graph():
    # Exhaust all 1024 simple graphs: a separate induced-subgraph definition
    # catches both missed instances and duplicate counts from opposite corners.
    possible_edges = tuple(combinations(range(5), 2))
    for mask in range(1 << len(possible_edges)):
        edges = {edge for index, edge in enumerate(possible_edges) if mask & (1 << index)}
        result = extract_pattern_features(graph(5, sorted(edges)))
        triangles = []
        squares = []
        for size, instances in ((3, triangles), (4, squares)):
            for sites in combinations(range(5), size):
                induced = {edge for edge in combinations(sites, 2) if edge in edges}
                degrees = [sum(site in edge for edge in induced) for site in sites]
                if degrees == [2] * size:
                    instances.append(sites)
        assert result.triangles == tuple(triangles)
        assert result.values["triangle_count"] == len(triangles)
        assert result.chordless_squares == tuple(squares)
