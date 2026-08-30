from __future__ import annotations

import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    DEFAULT_RANDOM_REGULAR_MAX_ATTEMPTS,
    DEFAULT_RANDOM_REGULAR_MAX_EDGES,
    RANDOM_REGULAR_GRAPH_GENERATOR,
    Geometry,
    random_regular_graph,
)


def test_random_regular_graph_has_fixed_pcg64_reference_sample() -> None:
    geometry = random_regular_graph(8, 2, seed=42)

    assert isinstance(geometry, Geometry)
    assert tuple((edge.source, edge.target) for edge in geometry.edges) == (
        (0, 1),
        (0, 6),
        (1, 6),
        (2, 3),
        (2, 5),
        (3, 4),
        (4, 7),
        (5, 7),
    )
    assert all(edge.edge_type == "random_regular" for edge in geometry.edges)


@pytest.mark.parametrize(
    ("n_sites", "degree", "seed"),
    ((1, 0, 0), (6, 1, 1), (8, 2, 42), (12, 3, 9)),
)
def test_every_site_has_exact_requested_degree(
    n_sites: int,
    degree: int,
    seed: int,
) -> None:
    geometry = random_regular_graph(n_sites, degree, seed=seed)

    assert geometry.n_edges == n_sites * degree // 2
    assert all(
        geometry.degree(site) == degree for site in geometry.site_indices
    )


def test_zero_regular_graph_is_not_forced_to_be_connected() -> None:
    geometry = random_regular_graph(8, 0, seed=7)

    assert geometry.n_edges == 0
    assert all(geometry.degree(site) == 0 for site in geometry.site_indices)
    assert geometry.metadata["attempts_used"] == 0


def test_complete_regular_graph_uses_zero_degree_complement() -> None:
    n_sites = 7
    geometry = random_regular_graph(n_sites, n_sites - 1, seed=7)

    assert geometry.n_edges == n_sites * (n_sites - 1) // 2
    assert all(
        geometry.degree(site) == n_sites - 1 for site in geometry.site_indices
    )
    assert geometry.metadata["complemented"] is True
    assert geometry.metadata["effective_degree"] == 0
    assert geometry.metadata["attempts_used"] == 0


def test_dense_sample_is_exact_complement_of_sparse_sample() -> None:
    n_sites = 10
    sparse = random_regular_graph(n_sites, 2, seed=17)
    dense = random_regular_graph(n_sites, 7, seed=17)
    sparse_edges = {(edge.source, edge.target) for edge in sparse.edges}
    dense_edges = {(edge.source, edge.target) for edge in dense.edges}
    complete_edges = {
        (source, target)
        for source in range(n_sites)
        for target in range(source + 1, n_sites)
    }

    assert sparse_edges.isdisjoint(dense_edges)
    assert sparse_edges | dense_edges == complete_edges
    assert dense.metadata["complemented"] is True
    assert dense.metadata["effective_degree"] == 2


def test_sampler_reaches_every_perfect_matching_on_four_sites() -> None:
    matchings = {
        tuple(
            sorted((edge.source, edge.target) for edge in random_regular_graph(4, 1, seed=seed).edges)
        )
        for seed in range(30)
    }

    assert matchings == {
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    }


def test_random_regular_graph_is_reproducible_for_same_seed() -> None:
    first = random_regular_graph(20, 3, seed=12345)
    second = random_regular_graph(20, 3, seed=12345)

    assert first.edges == second.edges
    assert first.metadata == second.metadata


def test_random_regular_graph_changes_with_seed() -> None:
    first = random_regular_graph(20, 3, seed=1)
    second = random_regular_graph(20, 3, seed=2)

    assert first.edges != second.edges


def test_random_regular_edges_are_canonical_and_unique() -> None:
    geometry = random_regular_graph(30, 3, seed=17)
    endpoint_pairs = tuple((edge.source, edge.target) for edge in geometry.edges)

    assert endpoint_pairs == tuple(sorted(endpoint_pairs))
    assert all(source < target for source, target in endpoint_pairs)
    assert len(endpoint_pairs) == len(set(endpoint_pairs))


def test_abstract_regular_graph_has_no_invented_geometry_data() -> None:
    geometry = random_regular_graph(12, 3, seed=3)

    assert geometry.coordinates is None
    assert geometry.embedding_dimension is None
    assert geometry.boundary_sites == frozenset()
    assert geometry.boundary_components == ()
    assert geometry.site_types is None
    assert geometry.dimension_records == ()


def test_random_regular_graph_records_sampler_metadata() -> None:
    geometry = random_regular_graph(8, 2, seed=42)

    assert geometry.metadata == {
        "generator": "random_regular_graph",
        "ensemble": "uniform_simple_regular",
        "n_sites": 8,
        "degree": 2,
        "seed": 42,
        "rng": "numpy_pcg64",
        "sampler": "configuration_model_rejection",
        "effective_degree": 2,
        "complemented": False,
        "attempts_used": 6,
        "max_attempts": DEFAULT_RANDOM_REGULAR_MAX_ATTEMPTS,
        "max_edges": DEFAULT_RANDOM_REGULAR_MAX_EDGES,
    }


def test_builtin_random_regular_generator_requires_seed() -> None:
    assert RANDOM_REGULAR_GRAPH_GENERATOR.stochastic
    with pytest.raises(ValueError, match="require an explicit seed"):
        BUILTIN_GEOMETRY_GENERATORS.generate(
            "random_regular_graph",
            parameters={"n_sites": 8, "degree": 2},
        )


def test_builtin_random_regular_generator_records_provenance() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "random_regular_graph",
        parameters={"n_sites": 8, "degree": 2},
        seed=42,
    )

    assert geometry.metadata["generation"] == {
        "generator_key": "random_regular_graph",
        "generator_version": 1,
        "parameters": {"n_sites": 8, "degree": 2},
        "seed": 42,
    }


def test_exhausted_attempt_limit_raises_instead_of_returning_fallback() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"failed to sample.*after 1 attempts using seed 0",
    ):
        random_regular_graph(8, 2, seed=0, max_attempts=1)


def test_edge_budget_is_checked_before_sampling() -> None:
    assert random_regular_graph(10, 3, seed=1, max_edges=15).n_edges == 15
    with pytest.raises(ValueError, match=r"requires 15 edges.*max_edges=14"):
        random_regular_graph(10, 3, seed=1, max_edges=14)


def test_edge_budget_can_be_explicitly_disabled() -> None:
    geometry = random_regular_graph(8, 2, seed=1, max_edges=None)

    assert geometry.n_edges == 8
    assert geometry.metadata["max_edges"] is None


@pytest.mark.parametrize("n_sites", (0, -1, -10))
def test_random_regular_rejects_nonpositive_site_count(n_sites: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        random_regular_graph(n_sites, 0, seed=1)


@pytest.mark.parametrize("n_sites", (True, 3.5, "4"))
def test_random_regular_rejects_non_integer_site_count(n_sites: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        random_regular_graph(n_sites, 0, seed=1)  # type: ignore[arg-type]


def test_random_regular_rejects_negative_degree() -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        random_regular_graph(8, -1, seed=1)


@pytest.mark.parametrize("degree", (True, 2.5, "2"))
def test_random_regular_rejects_non_integer_degree(degree: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        random_regular_graph(8, degree, seed=1)  # type: ignore[arg-type]


def test_random_regular_rejects_degree_not_smaller_than_site_count() -> None:
    with pytest.raises(ValueError, match="smaller than n_sites"):
        random_regular_graph(8, 8, seed=1)


def test_random_regular_rejects_odd_stub_count() -> None:
    with pytest.raises(ValueError, match=r"n_sites \* degree must be even"):
        random_regular_graph(5, 3, seed=1)


def test_random_regular_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        random_regular_graph(8, 2, seed=-1)


@pytest.mark.parametrize("seed", (True, 1.5, "1"))
def test_random_regular_rejects_non_integer_seed(seed: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        random_regular_graph(8, 2, seed=seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_attempts", (0, -1, -10))
def test_random_regular_rejects_nonpositive_attempt_limit(
    max_attempts: int,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        random_regular_graph(8, 2, seed=1, max_attempts=max_attempts)


@pytest.mark.parametrize("max_attempts", (True, 1.5, "10"))
def test_random_regular_rejects_non_integer_attempt_limit(
    max_attempts: object,
) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        random_regular_graph(
            8,
            2,
            seed=1,
            max_attempts=max_attempts,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("max_edges", (0, -1, -10))
def test_random_regular_rejects_nonpositive_edge_budget(max_edges: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        random_regular_graph(8, 2, seed=1, max_edges=max_edges)


@pytest.mark.parametrize("max_edges", (True, 1.5, "10"))
def test_random_regular_rejects_non_integer_edge_budget(
    max_edges: object,
) -> None:
    with pytest.raises(TypeError, match="must be an integer or None"):
        random_regular_graph(
            8,
            2,
            seed=1,
            max_edges=max_edges,  # type: ignore[arg-type]
        )
