import numpy as np

from toposc_lab.geometry import Geometry, irregular_cluster


def test_irregular_cluster_has_nonuniform_graph_structure() -> None:
    geometry = irregular_cluster()

    assert isinstance(geometry, Geometry)
    assert geometry.n_sites == 7
    assert geometry.n_edges == 13
    assert geometry.dimension == 2
    assert geometry.neighbors(0) == (1, 2, 5)
    assert [geometry.degree(site) for site in geometry.site_indices] == [3, 4, 4, 5, 3, 4, 3]
    assert geometry.metadata["generator"] == "irregular_cluster"


def test_irregular_cluster_uses_nonuniform_coordinates() -> None:
    geometry = irregular_cluster()
    local_distances = [
        geometry.distance(edge.source, edge.target)
        for edge in geometry.edges
        if edge.edge_type == "local"
    ]

    assert geometry.coordinates is not None
    assert not np.allclose(geometry.coordinates, np.round(geometry.coordinates))
    assert np.ptp(local_distances) > 0.5


def test_irregular_cluster_carries_optional_site_and_edge_information() -> None:
    geometry = irregular_cluster()
    long_range_edge = geometry.edge_between(0, 5)

    assert geometry.boundary_sites == frozenset({0, 2, 4, 5, 6})
    assert geometry.site_types == ("A", "A", "B", "A", "B", "B", "C")
    assert long_range_edge is not None
    assert long_range_edge.edge_type == "long_range"
    assert geometry.distance(0, 5) > max(
        geometry.distance(edge.source, edge.target)
        for edge in geometry.edges
        if edge.edge_type == "local"
    )


def test_irregular_cluster_is_connected() -> None:
    geometry = irregular_cluster()
    visited = {0}
    pending = [0]

    while pending:
        site = pending.pop()
        for neighbor in geometry.neighbors(site):
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)

    assert visited == set(geometry.site_indices)


def test_irregular_cluster_is_deterministic() -> None:
    first = irregular_cluster()
    second = irregular_cluster()

    assert np.array_equal(first.coordinates, second.coordinates)
    assert first.edges == second.edges
    assert first.boundary_sites == second.boundary_sites
    assert first.site_types == second.site_types
