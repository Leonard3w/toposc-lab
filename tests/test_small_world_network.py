from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    DEFAULT_SMALL_WORLD_MAX_EDGES,
    SMALL_WORLD_NETWORK_GENERATOR,
    Geometry,
    small_world_network,
)


def test_zero_rewiring_probability_returns_exact_ring_lattice() -> None:
    n_sites = 10
    neighbor_degree = 4
    geometry = small_world_network(
        n_sites,
        neighbor_degree,
        0.0,
        seed=42,
    )
    expected_edges = {
        tuple(sorted((source, (source + offset) % n_sites)))
        for source in range(n_sites)
        for offset in range(1, neighbor_degree // 2 + 1)
    }

    assert isinstance(geometry, Geometry)
    assert {(edge.source, edge.target) for edge in geometry.edges} == expected_edges
    assert all(edge.edge_type == "ring_lattice" for edge in geometry.edges)
    assert all(
        geometry.degree(site) == neighbor_degree for site in geometry.site_indices
    )
    assert geometry.metadata["rewiring_decisions"] == 0
    assert geometry.metadata["rewired_edges"] == 0


def test_small_world_has_fixed_partial_rewiring_reference_sample() -> None:
    geometry = small_world_network(10, 4, 0.5, seed=42)

    assert tuple(
        (edge.source, edge.target, edge.edge_type) for edge in geometry.edges
    ) == (
        (0, 1, "ring_lattice"),
        (0, 4, "rewired"),
        (0, 8, "ring_lattice"),
        (0, 9, "ring_lattice"),
        (1, 2, "ring_lattice"),
        (1, 7, "rewired"),
        (1, 8, "rewired"),
        (1, 9, "ring_lattice"),
        (2, 3, "ring_lattice"),
        (2, 4, "ring_lattice"),
        (2, 6, "rewired"),
        (3, 4, "ring_lattice"),
        (3, 8, "rewired"),
        (4, 6, "ring_lattice"),
        (4, 7, "rewired"),
        (5, 6, "ring_lattice"),
        (5, 7, "ring_lattice"),
        (6, 8, "ring_lattice"),
        (7, 9, "ring_lattice"),
        (8, 9, "ring_lattice"),
    )
    assert geometry.metadata["rewiring_decisions"] == 7
    assert geometry.metadata["rewired_edges"] == 6
    assert sum(edge.edge_type == "rewired" for edge in geometry.edges) == 6


@pytest.mark.parametrize("rewiring_probability", (0.0, 0.25, 0.5, 1.0))
def test_rewiring_preserves_edge_count_and_mean_degree(
    rewiring_probability: float,
) -> None:
    n_sites = 30
    neighbor_degree = 6
    geometry = small_world_network(
        n_sites,
        neighbor_degree,
        rewiring_probability,
        seed=17,
    )

    assert geometry.n_edges == n_sites * neighbor_degree // 2
    assert 2 * geometry.n_edges / geometry.n_sites == neighbor_degree


def test_full_rewiring_changes_individual_degrees_but_not_their_sum() -> None:
    geometry = small_world_network(10, 4, 1.0, seed=42)
    degrees = sorted(geometry.degree(site) for site in geometry.site_indices)

    assert degrees == [3, 3, 3, 4, 4, 4, 4, 4, 5, 6]
    assert sum(degrees) == 2 * geometry.n_edges == 40
    assert geometry.metadata["rewiring_decisions"] == 20
    assert geometry.metadata["rewired_edges"] == 17


def test_complete_ring_has_no_alternative_rewiring_target() -> None:
    geometry = small_world_network(5, 4, 1.0, seed=42)

    assert geometry.n_edges == 10
    assert all(geometry.degree(site) == 4 for site in geometry.site_indices)
    assert all(edge.edge_type == "ring_lattice" for edge in geometry.edges)
    assert geometry.metadata["rewiring_decisions"] == 10
    assert geometry.metadata["rewired_edges"] == 0


def test_small_world_edges_are_canonical_simple_and_unique() -> None:
    geometry = small_world_network(50, 6, 1.0, seed=19)
    endpoint_pairs = tuple((edge.source, edge.target) for edge in geometry.edges)

    assert endpoint_pairs == tuple(sorted(endpoint_pairs))
    assert all(source < target for source, target in endpoint_pairs)
    assert len(endpoint_pairs) == len(set(endpoint_pairs))


def test_small_world_does_not_condition_on_connectivity() -> None:
    generated = (
        small_world_network(12, 2, 1.0, seed=seed)
        for seed in range(50)
    )

    assert any(not _is_connected(geometry) for geometry in generated)


def test_small_world_is_reproducible_for_same_seed() -> None:
    first = small_world_network(30, 4, 0.4, seed=12345)
    second = small_world_network(30, 4, 0.4, seed=12345)

    assert first.edges == second.edges
    assert first.metadata == second.metadata


def test_small_world_changes_with_seed() -> None:
    first = small_world_network(30, 4, 0.4, seed=1)
    second = small_world_network(30, 4, 0.4, seed=2)

    assert first.edges != second.edges


def test_abstract_small_world_has_no_invented_geometry_data() -> None:
    geometry = small_world_network(12, 4, 0.3, seed=3)

    assert geometry.coordinates is None
    assert geometry.embedding_dimension is None
    assert geometry.boundary_sites == frozenset()
    assert geometry.boundary_components == ()
    assert geometry.site_types is None
    assert geometry.dimension_records == ()


def test_small_world_records_model_and_rewiring_metadata() -> None:
    geometry = small_world_network(10, 4, 0.5, seed=42)

    assert geometry.metadata == {
        "generator": "small_world_network",
        "model": "watts_strogatz",
        "n_sites": 10,
        "neighbor_degree": 4,
        "rewiring_probability": 0.5,
        "seed": 42,
        "rng": "numpy_pcg64",
        "rewiring_order": "source_then_positive_ring_offset",
        "rewiring_decisions": 7,
        "rewired_edges": 6,
        "max_edges": DEFAULT_SMALL_WORLD_MAX_EDGES,
    }


def test_builtin_small_world_generator_requires_seed() -> None:
    assert SMALL_WORLD_NETWORK_GENERATOR.stochastic
    with pytest.raises(ValueError, match="require an explicit seed"):
        BUILTIN_GEOMETRY_GENERATORS.generate(
            "small_world_network",
            parameters={
                "n_sites": 10,
                "neighbor_degree": 4,
                "rewiring_probability": 0.5,
            },
        )


def test_builtin_small_world_generator_records_provenance() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "small_world_network",
        parameters={
            "n_sites": 10,
            "neighbor_degree": 4,
            "rewiring_probability": 0.5,
        },
        seed=42,
    )

    assert geometry.metadata["generation"] == {
        "generator_key": "small_world_network",
        "generator_version": 1,
        "parameters": {
            "n_sites": 10,
            "neighbor_degree": 4,
            "rewiring_probability": 0.5,
        },
        "seed": 42,
    }


def test_edge_budget_is_checked_before_rewiring() -> None:
    assert small_world_network(10, 4, 0.5, seed=1, max_edges=20).n_edges == 20
    with pytest.raises(ValueError, match=r"requires 20 edges.*max_edges=19"):
        small_world_network(10, 4, 0.5, seed=1, max_edges=19)


def test_edge_budget_can_be_explicitly_disabled() -> None:
    geometry = small_world_network(10, 4, 0.5, seed=1, max_edges=None)

    assert geometry.n_edges == 20
    assert geometry.metadata["max_edges"] is None


@pytest.mark.parametrize("n_sites", (0, -1, -10))
def test_small_world_rejects_nonpositive_site_count(n_sites: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        small_world_network(n_sites, 0, 0.5, seed=1)


@pytest.mark.parametrize("n_sites", (True, 3.5, "4"))
def test_small_world_rejects_non_integer_site_count(n_sites: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        small_world_network(n_sites, 0, 0.5, seed=1)  # type: ignore[arg-type]


def test_small_world_rejects_negative_neighbor_degree() -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        small_world_network(10, -2, 0.5, seed=1)


@pytest.mark.parametrize("neighbor_degree", (True, 2.5, "2"))
def test_small_world_rejects_non_integer_neighbor_degree(
    neighbor_degree: object,
) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        small_world_network(
            10,
            neighbor_degree,  # type: ignore[arg-type]
            0.5,
            seed=1,
        )


def test_small_world_rejects_odd_neighbor_degree() -> None:
    with pytest.raises(ValueError, match="must be even"):
        small_world_network(10, 3, 0.5, seed=1)


def test_small_world_rejects_neighbor_degree_not_smaller_than_size() -> None:
    with pytest.raises(ValueError, match="smaller than n_sites"):
        small_world_network(10, 10, 0.5, seed=1)


@pytest.mark.parametrize("rewiring_probability", (-0.1, 1.1, np.inf, np.nan))
def test_small_world_rejects_invalid_rewiring_probability(
    rewiring_probability: float,
) -> None:
    with pytest.raises(ValueError, match="finite and between zero and one"):
        small_world_network(10, 4, rewiring_probability, seed=1)


@pytest.mark.parametrize("rewiring_probability", (True, "0.5"))
def test_small_world_rejects_non_real_rewiring_probability(
    rewiring_probability: object,
) -> None:
    with pytest.raises(TypeError, match="must be a real number"):
        small_world_network(
            10,
            4,
            rewiring_probability,  # type: ignore[arg-type]
            seed=1,
        )


def test_small_world_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        small_world_network(10, 4, 0.5, seed=-1)


@pytest.mark.parametrize("seed", (True, 1.5, "1"))
def test_small_world_rejects_non_integer_seed(seed: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        small_world_network(10, 4, 0.5, seed=seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_edges", (0, -1, -10))
def test_small_world_rejects_nonpositive_edge_budget(max_edges: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        small_world_network(10, 4, 0.5, seed=1, max_edges=max_edges)


@pytest.mark.parametrize("max_edges", (True, 1.5, "20"))
def test_small_world_rejects_non_integer_edge_budget(max_edges: object) -> None:
    with pytest.raises(TypeError, match="must be an integer or None"):
        small_world_network(
            10,
            4,
            0.5,
            seed=1,
            max_edges=max_edges,  # type: ignore[arg-type]
        )


def _is_connected(geometry: Geometry) -> bool:
    visited = {0}
    pending = [0]
    while pending:
        site = pending.pop()
        for neighbor in geometry.neighbors(site):
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return len(visited) == geometry.n_sites
