from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    DEFAULT_RANDOM_GRAPH_MAX_CANDIDATE_EDGES,
    RANDOM_GRAPH_GENERATOR,
    Geometry,
    random_graph,
)


def test_random_graph_has_fixed_pcg64_reference_sample() -> None:
    geometry = random_graph(6, 0.35, seed=42)

    assert isinstance(geometry, Geometry)
    assert tuple((edge.source, edge.target) for edge in geometry.edges) == (
        (0, 5),
        (1, 5),
    )
    assert all(edge.edge_type == "random" for edge in geometry.edges)


def test_zero_probability_returns_empty_graph_without_connectivity_retry() -> None:
    geometry = random_graph(8, 0.0, seed=7)

    assert geometry.n_edges == 0
    assert all(geometry.degree(site) == 0 for site in geometry.site_indices)


def test_unit_probability_returns_complete_graph() -> None:
    n_sites = 8
    geometry = random_graph(n_sites, 1.0, seed=7)

    assert geometry.n_edges == n_sites * (n_sites - 1) // 2
    assert all(
        geometry.degree(site) == n_sites - 1 for site in geometry.site_indices
    )
    assert tuple((edge.source, edge.target) for edge in geometry.edges) == tuple(
        (source, target)
        for source in range(n_sites)
        for target in range(source + 1, n_sites)
    )


def test_random_graph_is_reproducible_for_same_seed() -> None:
    first = random_graph(30, 0.2, seed=12345)
    second = random_graph(30, 0.2, seed=12345)

    assert first.edges == second.edges
    assert first.metadata == second.metadata


def test_random_graph_changes_with_seed() -> None:
    first = random_graph(30, 0.2, seed=1)
    second = random_graph(30, 0.2, seed=2)

    assert first.edges != second.edges


def test_random_graph_edges_are_canonical_and_unique() -> None:
    geometry = random_graph(50, 0.4, seed=17)
    endpoint_pairs = tuple((edge.source, edge.target) for edge in geometry.edges)

    assert endpoint_pairs == tuple(sorted(endpoint_pairs))
    assert all(source < target for source, target in endpoint_pairs)
    assert len(endpoint_pairs) == len(set(endpoint_pairs))


def test_abstract_random_graph_has_no_invented_geometry_data() -> None:
    geometry = random_graph(12, 0.3, seed=3)

    assert geometry.coordinates is None
    assert geometry.embedding_dimension is None
    assert geometry.boundary_sites == frozenset()
    assert geometry.boundary_components == ()
    assert geometry.site_types is None
    assert geometry.dimension_records == ()


def test_random_graph_records_ensemble_and_rng_metadata() -> None:
    geometry = random_graph(6, 0.35, seed=42)

    assert geometry.metadata == {
        "generator": "random_graph",
        "ensemble": "erdos_renyi_gnp",
        "n_sites": 6,
        "edge_probability": 0.35,
        "seed": 42,
        "rng": "numpy_pcg64",
        "candidate_edges": 15,
        "max_candidate_edges": DEFAULT_RANDOM_GRAPH_MAX_CANDIDATE_EDGES,
    }


def test_builtin_random_generator_requires_seed() -> None:
    assert RANDOM_GRAPH_GENERATOR.stochastic
    with pytest.raises(ValueError, match="require an explicit seed"):
        BUILTIN_GEOMETRY_GENERATORS.generate(
            "random_graph",
            parameters={"n_sites": 8, "edge_probability": 0.25},
        )


def test_builtin_random_generator_records_seed_separately_in_provenance() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "random_graph",
        parameters={"n_sites": 8, "edge_probability": 0.25},
        seed=99,
    )

    assert geometry.metadata["seed"] == 99
    assert geometry.metadata["generation"] == {
        "generator_key": "random_graph",
        "generator_version": 1,
        "parameters": {"n_sites": 8, "edge_probability": 0.25},
        "seed": 99,
    }


def test_candidate_budget_is_checked_before_sampling() -> None:
    assert DEFAULT_RANDOM_GRAPH_MAX_CANDIDATE_EDGES < 1415 * 1414 // 2
    with pytest.raises(
        ValueError,
        match=r"requires 1000405 candidate edges.*max_candidate_edges=1000000",
    ):
        random_graph(1415, 0.1, seed=1)


def test_explicit_candidate_budget_accepts_exact_limit() -> None:
    assert random_graph(5, 0.5, seed=1, max_candidate_edges=10).n_sites == 5
    with pytest.raises(ValueError, match=r"requires 10 candidate edges"):
        random_graph(5, 0.5, seed=1, max_candidate_edges=9)


def test_candidate_budget_can_be_explicitly_disabled() -> None:
    geometry = random_graph(5, 0.5, seed=1, max_candidate_edges=None)

    assert geometry.n_sites == 5
    assert geometry.metadata["max_candidate_edges"] is None


@pytest.mark.parametrize("n_sites", (0, -1, -10))
def test_random_graph_rejects_nonpositive_site_count(n_sites: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        random_graph(n_sites, 0.5, seed=1)


@pytest.mark.parametrize("n_sites", (True, 3.5, "4"))
def test_random_graph_rejects_non_integer_site_count(n_sites: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        random_graph(n_sites, 0.5, seed=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("edge_probability", (-0.1, 1.1, np.inf, np.nan))
def test_random_graph_rejects_invalid_probability(
    edge_probability: float,
) -> None:
    with pytest.raises(ValueError, match="finite and between zero and one"):
        random_graph(5, edge_probability, seed=1)


@pytest.mark.parametrize("edge_probability", (True, "0.5"))
def test_random_graph_rejects_non_real_probability(
    edge_probability: object,
) -> None:
    with pytest.raises(TypeError, match="must be a real number"):
        random_graph(5, edge_probability, seed=1)  # type: ignore[arg-type]


def test_random_graph_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        random_graph(5, 0.5, seed=-1)


@pytest.mark.parametrize("seed", (True, 1.5, "1"))
def test_random_graph_rejects_non_integer_seed(seed: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        random_graph(5, 0.5, seed=seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_candidate_edges", (0, -1, -10))
def test_random_graph_rejects_nonpositive_candidate_budget(
    max_candidate_edges: int,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        random_graph(
            5,
            0.5,
            seed=1,
            max_candidate_edges=max_candidate_edges,
        )


@pytest.mark.parametrize("max_candidate_edges", (True, 1.5, "10"))
def test_random_graph_rejects_non_integer_candidate_budget(
    max_candidate_edges: object,
) -> None:
    with pytest.raises(TypeError, match="must be an integer or None"):
        random_graph(
            5,
            0.5,
            seed=1,
            max_candidate_edges=max_candidate_edges,  # type: ignore[arg-type]
        )
