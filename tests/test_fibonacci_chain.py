from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    DEFAULT_FIBONACCI_CHAIN_MAX_SITES,
    FIBONACCI_GOLDEN_RATIO,
    Geometry,
    GeometryDimension,
    fibonacci_chain,
)


def _bond_word(geometry: Geometry) -> str:
    return "".join(
        "L" if edge.edge_type == "fibonacci_long" else "S"
        for edge in geometry.edges
    )


@pytest.mark.parametrize(
    ("order", "expected_word", "expected_counts"),
    (
        (0, "L", (1, 0)),
        (1, "LS", (1, 1)),
        (2, "LSL", (2, 1)),
        (3, "LSLLS", (3, 2)),
        (4, "LSLLSLSL", (5, 3)),
        (5, "LSLLSLSLLSLLS", (8, 5)),
    ),
)
def test_fibonacci_substitution_words_and_counts_are_exact(
    order: int,
    expected_word: str,
    expected_counts: tuple[int, int],
) -> None:
    geometry = fibonacci_chain(order)

    assert _bond_word(geometry) == expected_word
    assert geometry.n_sites == len(expected_word) + 1
    assert geometry.metadata["n_long_bonds"] == expected_counts[0]
    assert geometry.metadata["n_short_bonds"] == expected_counts[1]


def test_custom_lengths_define_coordinates_and_edge_displacements() -> None:
    geometry = fibonacci_chain(3, spacing=2.0, long_short_ratio=1.5)

    assert geometry.coordinates is not None
    assert geometry.coordinates[:, 0] == pytest.approx(
        [0.0, 3.0, 5.0, 8.0, 11.0, 13.0]
    )
    assert [edge.displacement for edge in geometry.edges] == [
        (3.0,),
        (2.0,),
        (3.0,),
        (3.0,),
        (2.0,),
    ]
    assert [geometry.distance(site, site + 1) for site in range(5)] == (
        pytest.approx([3.0, 2.0, 3.0, 3.0, 2.0])
    )


def test_chain_connectivity_degrees_and_open_boundary_are_explicit() -> None:
    geometry = fibonacci_chain(6)

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


def test_default_lengths_have_golden_ratio_inflation() -> None:
    outer_lengths = []
    for order in range(7):
        geometry = fibonacci_chain(order)
        assert geometry.coordinates is not None
        outer_lengths.append(float(geometry.coordinates[-1, 0]))

    assert np.asarray(outer_lengths[1:]) / np.asarray(outer_lengths[:-1]) == (
        pytest.approx(FIBONACCI_GOLDEN_RATIO)
    )


def test_dimension_semantics_do_not_infer_a_lattice_dimension() -> None:
    geometry = fibonacci_chain(4)

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


def test_golden_and_nongolden_metric_regimes_are_distinguished() -> None:
    golden = fibonacci_chain(3)
    nongolden = fibonacci_chain(3, long_short_ratio=1.5)

    assert golden.metadata["geometric_regime"] == "golden_ratio_self_similar"
    assert (
        nongolden.metadata["geometric_regime"]
        == "fibonacci_symbolic_nongolden_lengths"
    )
    assert golden.metadata["substitution_rule"] == {"L": "LS", "S": "L"}
    assert "word" not in golden.metadata


def test_fibonacci_chain_is_deterministic() -> None:
    first = fibonacci_chain(7, spacing=1.25)
    second = fibonacci_chain(7, spacing=1.25)

    assert first.edges == second.edges
    assert first.coordinates is not None
    assert second.coordinates is not None
    assert np.array_equal(first.coordinates, second.coordinates)
    assert first.boundary_sites == second.boundary_sites
    assert first.dimension_records == second.dimension_records
    assert first.metadata == second.metadata


def test_fibonacci_chain_is_available_through_common_registry() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "fibonacci_chain",
        parameters={"order": 3, "spacing": 2.0},
    )

    assert _bond_word(geometry) == "LSLLS"
    assert geometry.metadata["generation"] == {
        "generator_key": "fibonacci_chain",
        "generator_version": 1,
        "parameters": {"order": 3, "spacing": 2.0},
        "seed": None,
    }


def test_metadata_contains_geometry_but_no_model_parameters() -> None:
    geometry = fibonacci_chain(3)

    assert geometry.metadata["generator"] == "fibonacci_chain"
    assert geometry.metadata["family"] == "fibonacci_substitution_tiling"
    assert geometry.metadata["boundary_condition"] == "open"
    assert not {"hopping", "onsite", "pairing"} & geometry.metadata.keys()


def test_site_budget_accepts_exact_count_and_rejects_next_site() -> None:
    assert fibonacci_chain(10, max_sites=145).n_sites == 145
    with pytest.raises(
        ValueError,
        match=r"at least 145 sites at substitution step 10, exceeding max_sites=144",
    ):
        fibonacci_chain(10, max_sites=144)


def test_default_budget_stops_huge_order_before_materializing_word() -> None:
    with pytest.raises(
        ValueError,
        match=(
            rf"at least 121394 sites at substitution step 24, exceeding "
            rf"max_sites={DEFAULT_FIBONACCI_CHAIN_MAX_SITES}"
        ),
    ):
        fibonacci_chain(1_000_000)


def test_site_budget_can_be_explicitly_disabled() -> None:
    geometry = fibonacci_chain(10, max_sites=None)

    assert geometry.n_sites == 145
    assert geometry.metadata["max_sites"] is None


@pytest.mark.parametrize("order", (-1, -5))
def test_fibonacci_chain_rejects_negative_order(order: int) -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        fibonacci_chain(order)


@pytest.mark.parametrize("order", (True, 2.5, "3"))
def test_fibonacci_chain_rejects_non_integer_order(order: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        fibonacci_chain(order)  # type: ignore[arg-type]


@pytest.mark.parametrize("spacing", (0.0, -1.0, np.inf, -np.inf, np.nan))
def test_fibonacci_chain_rejects_invalid_spacing(spacing: float) -> None:
    with pytest.raises(ValueError, match="spacing must be finite and positive"):
        fibonacci_chain(3, spacing=spacing)


@pytest.mark.parametrize("spacing", (True, "1.0"))
def test_fibonacci_chain_rejects_nonreal_spacing(spacing: object) -> None:
    with pytest.raises(TypeError, match="spacing must be a real number"):
        fibonacci_chain(3, spacing=spacing)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "long_short_ratio", (1.0, 0.5, -1.0, np.inf, -np.inf, np.nan)
)
def test_fibonacci_chain_rejects_invalid_length_ratio(
    long_short_ratio: float,
) -> None:
    with pytest.raises(ValueError, match="must be finite and greater than one"):
        fibonacci_chain(3, long_short_ratio=long_short_ratio)


@pytest.mark.parametrize("long_short_ratio", (True, "1.5"))
def test_fibonacci_chain_rejects_nonreal_length_ratio(
    long_short_ratio: object,
) -> None:
    with pytest.raises(TypeError, match="must be a real number"):
        fibonacci_chain(3, long_short_ratio=long_short_ratio)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_sites", (0, -1, -10))
def test_fibonacci_chain_rejects_nonpositive_site_budget(max_sites: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        fibonacci_chain(3, max_sites=max_sites)


@pytest.mark.parametrize("max_sites", (True, 1.5, "10"))
def test_fibonacci_chain_rejects_non_integer_site_budget(
    max_sites: object,
) -> None:
    with pytest.raises(TypeError, match="must be an integer or None"):
        fibonacci_chain(3, max_sites=max_sites)  # type: ignore[arg-type]
