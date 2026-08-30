from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    DEFAULT_SILVER_MEAN_CHAIN_MAX_SITES,
    SILVER_MEAN_RATIO,
    Geometry,
    GeometryDimension,
    silver_mean_chain,
)


def _bond_word(geometry: Geometry) -> str:
    return "".join(
        "L" if edge.edge_type == "silver_mean_long" else "S"
        for edge in geometry.edges
    )


@pytest.mark.parametrize(
    ("order", "expected_word", "expected_counts"),
    (
        (0, "S", (0, 1)),
        (1, "L", (1, 0)),
        (2, "LSL", (2, 1)),
        (3, "LSLLLSL", (5, 2)),
        (4, "LSLLLSLLSLLSLLLSL", (12, 5)),
    ),
)
def test_silver_mean_substitution_words_and_counts_are_exact(
    order: int,
    expected_word: str,
    expected_counts: tuple[int, int],
) -> None:
    geometry = silver_mean_chain(order)

    assert _bond_word(geometry) == expected_word
    assert geometry.n_sites == len(expected_word) + 1
    assert geometry.metadata["n_long_bonds"] == expected_counts[0]
    assert geometry.metadata["n_short_bonds"] == expected_counts[1]


def test_finite_approximants_are_palindromic() -> None:
    for order in range(10):
        word = _bond_word(silver_mean_chain(order))
        assert word == word[::-1]


def test_bond_counts_follow_the_pell_type_recurrence() -> None:
    bond_counts = [silver_mean_chain(order).n_edges for order in range(9)]

    assert bond_counts[:7] == [1, 1, 3, 7, 17, 41, 99]
    assert all(
        bond_counts[order] == 2 * bond_counts[order - 1] + bond_counts[order - 2]
        for order in range(2, len(bond_counts))
    )


def test_custom_lengths_define_coordinates_and_edge_displacements() -> None:
    geometry = silver_mean_chain(3, spacing=2.0, long_short_ratio=1.5)

    assert geometry.coordinates is not None
    assert geometry.coordinates[:, 0] == pytest.approx(
        [0.0, 3.0, 5.0, 8.0, 11.0, 14.0, 16.0, 19.0]
    )
    assert [edge.displacement for edge in geometry.edges] == [
        (3.0,),
        (2.0,),
        (3.0,),
        (3.0,),
        (3.0,),
        (2.0,),
        (3.0,),
    ]
    assert [geometry.distance(site, site + 1) for site in range(7)] == (
        pytest.approx([3.0, 2.0, 3.0, 3.0, 3.0, 2.0, 3.0])
    )


def test_chain_connectivity_degrees_and_open_boundary_are_explicit() -> None:
    geometry = silver_mean_chain(6)

    assert tuple((edge.source, edge.target) for edge in geometry.edges) == tuple(
        (site, site + 1) for site in range(geometry.n_sites - 1)
    )
    assert geometry.n_edges == geometry.n_sites - 1
    assert geometry.boundary_sites == frozenset({0, geometry.n_sites - 1})
    assert geometry.boundary_components == ()
    assert geometry.degree(0) == geometry.degree(geometry.n_sites - 1) == 1
    assert all(
        geometry.degree(site) == 2 for site in range(1, geometry.n_sites - 1)
    )


def test_default_lengths_have_silver_mean_inflation() -> None:
    outer_lengths = []
    for order in range(8):
        geometry = silver_mean_chain(order)
        assert geometry.coordinates is not None
        outer_lengths.append(float(geometry.coordinates[-1, 0]))

    assert np.asarray(outer_lengths[1:]) / np.asarray(outer_lengths[:-1]) == (
        pytest.approx(SILVER_MEAN_RATIO)
    )


def test_dimension_semantics_do_not_infer_a_lattice_dimension() -> None:
    geometry = silver_mean_chain(4)

    assert geometry.embedding_dimension == 1
    assert geometry.dimension_records == (
        GeometryDimension(
            kind="topological",
            value=1.0,
            scope="infinite_family",
            method="covering_dimension",
            exact=True,
        ),
    )
    assert all(record.kind != "lattice" for record in geometry.dimension_records)


def test_silver_and_nonsilver_metric_regimes_are_distinguished() -> None:
    silver = silver_mean_chain(3)
    nonsilver = silver_mean_chain(3, long_short_ratio=1.5)

    assert silver.metadata["geometric_regime"] == "silver_mean_self_similar"
    assert (
        nonsilver.metadata["geometric_regime"]
        == "octonacci_symbolic_nonsilver_lengths"
    )
    assert silver.metadata["substitution_seed"] == "S"
    assert silver.metadata["substitution_rule"] == {"L": "LSL", "S": "L"}
    assert silver.metadata["word_symmetry"] == "palindromic"
    assert "word" not in silver.metadata


def test_silver_mean_chain_is_deterministic() -> None:
    first = silver_mean_chain(7, spacing=1.25)
    second = silver_mean_chain(7, spacing=1.25)

    assert first.edges == second.edges
    assert first.coordinates is not None
    assert second.coordinates is not None
    assert np.array_equal(first.coordinates, second.coordinates)
    assert first.boundary_sites == second.boundary_sites
    assert first.dimension_records == second.dimension_records
    assert first.metadata == second.metadata


def test_silver_mean_chain_is_available_through_common_registry() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "silver_mean_chain",
        parameters={"order": 3, "spacing": 2.0},
    )

    assert _bond_word(geometry) == "LSLLLSL"
    assert geometry.metadata["generation"] == {
        "generator_key": "silver_mean_chain",
        "generator_version": 1,
        "parameters": {"order": 3, "spacing": 2.0},
        "seed": None,
    }


def test_metadata_contains_geometry_but_no_model_parameters() -> None:
    geometry = silver_mean_chain(3)

    assert geometry.metadata["generator"] == "silver_mean_chain"
    assert geometry.metadata["family"] == "silver_mean_substitution_tiling"
    assert geometry.metadata["alternative_name"] == "octonacci_chain"
    assert geometry.metadata["boundary_condition"] == "open"
    assert not {"hopping", "onsite", "pairing"} & geometry.metadata.keys()


def test_site_budget_accepts_exact_count_and_rejects_next_site() -> None:
    assert silver_mean_chain(10, max_sites=3364).n_sites == 3364
    with pytest.raises(
        ValueError,
        match=r"at least 3364 sites at substitution step 10, exceeding max_sites=3363",
    ):
        silver_mean_chain(10, max_sites=3363)


def test_default_budget_stops_huge_order_before_materializing_word() -> None:
    with pytest.raises(
        ValueError,
        match=(
            rf"at least 114244 sites at substitution step 14, exceeding "
            rf"max_sites={DEFAULT_SILVER_MEAN_CHAIN_MAX_SITES}"
        ),
    ):
        silver_mean_chain(1_000_000)


def test_site_budget_can_be_explicitly_disabled() -> None:
    geometry = silver_mean_chain(8, max_sites=None)

    assert geometry.n_sites == 578
    assert geometry.metadata["max_sites"] is None


@pytest.mark.parametrize("order", (-1, -5))
def test_silver_mean_chain_rejects_negative_order(order: int) -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        silver_mean_chain(order)


@pytest.mark.parametrize("order", (True, 2.5, "3"))
def test_silver_mean_chain_rejects_non_integer_order(order: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        silver_mean_chain(order)  # type: ignore[arg-type]


@pytest.mark.parametrize("spacing", (0.0, -1.0, np.inf, -np.inf, np.nan))
def test_silver_mean_chain_rejects_invalid_spacing(spacing: float) -> None:
    with pytest.raises(ValueError, match="spacing must be finite and positive"):
        silver_mean_chain(3, spacing=spacing)


@pytest.mark.parametrize("spacing", (True, "1.0"))
def test_silver_mean_chain_rejects_nonreal_spacing(spacing: object) -> None:
    with pytest.raises(TypeError, match="spacing must be a real number"):
        silver_mean_chain(3, spacing=spacing)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "long_short_ratio", (1.0, 0.5, -1.0, np.inf, -np.inf, np.nan)
)
def test_silver_mean_chain_rejects_invalid_length_ratio(
    long_short_ratio: float,
) -> None:
    with pytest.raises(ValueError, match="must be finite and greater than one"):
        silver_mean_chain(3, long_short_ratio=long_short_ratio)


@pytest.mark.parametrize("long_short_ratio", (True, "1.5"))
def test_silver_mean_chain_rejects_nonreal_length_ratio(
    long_short_ratio: object,
) -> None:
    with pytest.raises(TypeError, match="must be a real number"):
        silver_mean_chain(3, long_short_ratio=long_short_ratio)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_sites", (0, -1, -10))
def test_silver_mean_chain_rejects_nonpositive_site_budget(max_sites: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        silver_mean_chain(3, max_sites=max_sites)


@pytest.mark.parametrize("max_sites", (True, 1.5, "10"))
def test_silver_mean_chain_rejects_non_integer_site_budget(
    max_sites: object,
) -> None:
    with pytest.raises(TypeError, match="must be an integer or None"):
        silver_mean_chain(3, max_sites=max_sites)  # type: ignore[arg-type]
