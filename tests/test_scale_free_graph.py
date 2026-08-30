from __future__ import annotations

import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    DEFAULT_SCALE_FREE_MAX_EDGES,
    SCALE_FREE_GRAPH_GENERATOR,
    Geometry,
    scale_free_graph,
)


def test_scale_free_graph_has_fixed_pcg64_reference_sample() -> None:
    geometry = scale_free_graph(10, 2, seed=42)

    assert isinstance(geometry, Geometry)
    assert tuple(
        (edge.source, edge.target, edge.edge_type) for edge in geometry.edges
    ) == (
        (0, 1, "initial_clique"),
        (0, 2, "initial_clique"),
        (0, 3, "preferential_attachment"),
        (0, 4, "preferential_attachment"),
        (0, 5, "preferential_attachment"),
        (0, 6, "preferential_attachment"),
        (1, 2, "initial_clique"),
        (1, 7, "preferential_attachment"),
        (2, 3, "preferential_attachment"),
        (2, 4, "preferential_attachment"),
        (2, 7, "preferential_attachment"),
        (2, 9, "preferential_attachment"),
        (4, 5, "preferential_attachment"),
        (4, 6, "preferential_attachment"),
        (4, 8, "preferential_attachment"),
        (6, 9, "preferential_attachment"),
        (7, 8, "preferential_attachment"),
    )
    assert [geometry.degree(site) for site in geometry.site_indices] == [
        6,
        3,
        6,
        2,
        5,
        2,
        3,
        3,
        2,
        2,
    ]


@pytest.mark.parametrize(
    ("n_sites", "attachments_per_site"),
    ((2, 1), (10, 2), (30, 3), (20, 5)),
)
def test_scale_free_edge_count_follows_exact_growth_formula(
    n_sites: int,
    attachments_per_site: int,
) -> None:
    geometry = scale_free_graph(n_sites, attachments_per_site, seed=17)

    assert geometry.n_edges == (
        attachments_per_site * n_sites
        - attachments_per_site * (attachments_per_site + 1) // 2
    )


def test_each_growth_site_attaches_to_exactly_m_distinct_older_sites() -> None:
    n_sites = 100
    attachments_per_site = 4
    geometry = scale_free_graph(n_sites, attachments_per_site, seed=8)
    initial_sites = attachments_per_site + 1

    for site in range(initial_sites, n_sites):
        older_neighbors = {
            neighbor for neighbor in geometry.neighbors(site) if neighbor < site
        }
        assert len(older_neighbors) == attachments_per_site


def test_initial_graph_is_complete_m_plus_one_clique() -> None:
    attachments_per_site = 4
    initial_sites = attachments_per_site + 1
    geometry = scale_free_graph(20, attachments_per_site, seed=9)

    assert {
        (edge.source, edge.target)
        for edge in geometry.edges
        if edge.edge_type == "initial_clique"
    } == {
        (source, target)
        for source in range(initial_sites)
        for target in range(source + 1, initial_sites)
    }


def test_complete_initial_graph_without_growth_is_supported() -> None:
    geometry = scale_free_graph(6, 5, seed=42)

    assert geometry.n_edges == 15
    assert all(geometry.degree(site) == 5 for site in geometry.site_indices)
    assert all(edge.edge_type == "initial_clique" for edge in geometry.edges)


def test_scale_free_graph_is_connected_by_construction() -> None:
    geometry = scale_free_graph(200, 3, seed=33)
    visited = {0}
    pending = [0]
    while pending:
        site = pending.pop()
        for neighbor in geometry.neighbors(site):
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)

    assert visited == set(geometry.site_indices)


def test_preferential_attachment_favors_early_sites_across_seeds() -> None:
    first_ten_degree_sum = 0
    last_ten_degree_sum = 0
    for seed in range(30):
        geometry = scale_free_graph(100, 2, seed=seed)
        first_ten_degree_sum += sum(geometry.degree(site) for site in range(10))
        last_ten_degree_sum += sum(geometry.degree(site) for site in range(90, 100))

    assert first_ten_degree_sum > 3 * last_ten_degree_sum


def test_every_site_has_at_least_attachment_degree() -> None:
    attachments_per_site = 3
    geometry = scale_free_graph(1000, attachments_per_site, seed=2026)

    assert min(geometry.degree(site) for site in geometry.site_indices) == 3
    assert max(geometry.degree(site) for site in geometry.site_indices) == 114


def test_scale_free_edges_are_canonical_simple_and_unique() -> None:
    geometry = scale_free_graph(100, 4, seed=19)
    endpoint_pairs = tuple((edge.source, edge.target) for edge in geometry.edges)

    assert endpoint_pairs == tuple(sorted(endpoint_pairs))
    assert all(source < target for source, target in endpoint_pairs)
    assert len(endpoint_pairs) == len(set(endpoint_pairs))


def test_scale_free_graph_is_reproducible_for_same_seed() -> None:
    first = scale_free_graph(100, 3, seed=12345)
    second = scale_free_graph(100, 3, seed=12345)

    assert first.edges == second.edges
    assert first.metadata == second.metadata


def test_scale_free_graph_changes_with_seed() -> None:
    first = scale_free_graph(100, 3, seed=1)
    second = scale_free_graph(100, 3, seed=2)

    assert first.edges != second.edges


def test_abstract_scale_free_graph_has_no_invented_geometry_data() -> None:
    geometry = scale_free_graph(20, 2, seed=3)

    assert geometry.coordinates is None
    assert geometry.embedding_dimension is None
    assert geometry.boundary_sites == frozenset()
    assert geometry.boundary_components == ()
    assert geometry.site_types is None
    assert geometry.dimension_records == ()


def test_scale_free_graph_records_model_metadata() -> None:
    geometry = scale_free_graph(10, 2, seed=42)

    assert geometry.metadata == {
        "generator": "scale_free_graph",
        "model": "barabasi_albert",
        "n_sites": 10,
        "attachments_per_site": 2,
        "initial_graph": "complete_m_plus_one",
        "initial_sites": 3,
        "attachment_sampling": (
            "sequential_degree_weighted_without_replacement"
        ),
        "seed": 42,
        "rng": "numpy_pcg64",
        "max_edges": DEFAULT_SCALE_FREE_MAX_EDGES,
    }


def test_builtin_scale_free_generator_requires_seed() -> None:
    assert SCALE_FREE_GRAPH_GENERATOR.stochastic
    with pytest.raises(ValueError, match="require an explicit seed"):
        BUILTIN_GEOMETRY_GENERATORS.generate(
            "scale_free_graph",
            parameters={"n_sites": 10, "attachments_per_site": 2},
        )


def test_builtin_scale_free_generator_records_provenance() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "scale_free_graph",
        parameters={"n_sites": 10, "attachments_per_site": 2},
        seed=42,
    )

    assert geometry.metadata["generation"] == {
        "generator_key": "scale_free_graph",
        "generator_version": 1,
        "parameters": {"n_sites": 10, "attachments_per_site": 2},
        "seed": 42,
    }


def test_edge_budget_is_checked_before_growth() -> None:
    assert scale_free_graph(10, 2, seed=1, max_edges=17).n_edges == 17
    with pytest.raises(ValueError, match=r"requires 17 edges.*max_edges=16"):
        scale_free_graph(10, 2, seed=1, max_edges=16)


def test_edge_budget_can_be_explicitly_disabled() -> None:
    geometry = scale_free_graph(10, 2, seed=1, max_edges=None)

    assert geometry.n_edges == 17
    assert geometry.metadata["max_edges"] is None


@pytest.mark.parametrize("n_sites", (0, -1, -10))
def test_scale_free_rejects_nonpositive_site_count(n_sites: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        scale_free_graph(n_sites, 1, seed=1)


@pytest.mark.parametrize("n_sites", (True, 3.5, "4"))
def test_scale_free_rejects_non_integer_site_count(n_sites: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        scale_free_graph(n_sites, 1, seed=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("attachments_per_site", (0, -1, -10))
def test_scale_free_rejects_nonpositive_attachment_count(
    attachments_per_site: int,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        scale_free_graph(10, attachments_per_site, seed=1)


@pytest.mark.parametrize("attachments_per_site", (True, 2.5, "2"))
def test_scale_free_rejects_non_integer_attachment_count(
    attachments_per_site: object,
) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        scale_free_graph(
            10,
            attachments_per_site,  # type: ignore[arg-type]
            seed=1,
        )


def test_scale_free_rejects_attachment_count_not_smaller_than_size() -> None:
    with pytest.raises(ValueError, match="smaller than n_sites"):
        scale_free_graph(10, 10, seed=1)


def test_scale_free_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        scale_free_graph(10, 2, seed=-1)


@pytest.mark.parametrize("seed", (True, 1.5, "1"))
def test_scale_free_rejects_non_integer_seed(seed: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        scale_free_graph(10, 2, seed=seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_edges", (0, -1, -10))
def test_scale_free_rejects_nonpositive_edge_budget(max_edges: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        scale_free_graph(10, 2, seed=1, max_edges=max_edges)


@pytest.mark.parametrize("max_edges", (True, 1.5, "20"))
def test_scale_free_rejects_non_integer_edge_budget(max_edges: object) -> None:
    with pytest.raises(TypeError, match="must be an integer or None"):
        scale_free_graph(
            10,
            2,
            seed=1,
            max_edges=max_edges,  # type: ignore[arg-type]
        )
