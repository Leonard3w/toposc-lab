from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    Geometry,
    artificial_rule_graph,
)


LINE_RULES = ((-1,), (1,))
CARDINAL_RULES = ((1, 0), (0, -1), (-1, 0), (0, 1))


def _edge_pairs(geometry: Geometry) -> tuple[tuple[int, int], ...]:
    return tuple((edge.source, edge.target) for edge in geometry.edges)


@pytest.mark.parametrize(
    ("iterations", "n_sites", "n_edges", "n_boundary", "shells"),
    (
        (0, 1, 0, 1, (1,)),
        (1, 3, 2, 2, (1, 2)),
        (3, 7, 6, 2, (1, 2, 2, 2)),
    ),
)
def test_inverse_unit_rules_generate_exact_open_chains(
    iterations: int,
    n_sites: int,
    n_edges: int,
    n_boundary: int,
    shells: tuple[int, ...],
) -> None:
    geometry = artificial_rule_graph(iterations, LINE_RULES)

    assert geometry.n_sites == n_sites
    assert geometry.n_edges == n_edges
    assert len(geometry.boundary_sites) == n_boundary
    assert geometry.metadata["shell_populations"] == shells
    assert geometry.metadata["n_rule_applications"] == 2 * n_sites


def test_cardinal_rules_generate_exact_manhattan_ball() -> None:
    geometry = artificial_rule_graph(2, CARDINAL_RULES)

    assert geometry.n_sites == 13
    assert geometry.n_edges == 16
    assert len(geometry.boundary_sites) == 8
    assert geometry.metadata["shell_populations"] == (1, 4, 8)
    assert geometry.metadata["n_rule_applications"] == 52
    assert geometry.coordinates is not None
    assert {
        tuple(int(value) for value in coordinate)
        for coordinate in geometry.coordinates
    } == {
        (x, y)
        for x in range(-2, 3)
        for y in range(-2, 3)
        if abs(x) + abs(y) <= 2
    }


def test_final_shell_contains_every_induced_rule_edge() -> None:
    geometry = artificial_rule_graph(2, CARDINAL_RULES)
    assert geometry.coordinates is not None
    coordinate_sites = {
        tuple(int(value) for value in coordinate): site
        for site, coordinate in enumerate(geometry.coordinates)
    }
    expected_edges = {
        tuple(sorted((source, coordinate_sites[target])))
        for point, source in coordinate_sites.items()
        for rule in CARDINAL_RULES
        if (target := tuple(a + b for a, b in zip(point, rule)))
        in coordinate_sites
    }

    assert set(_edge_pairs(geometry)) == expected_edges


def test_rule_order_is_canonical_and_does_not_change_output() -> None:
    first = artificial_rule_graph(2, CARDINAL_RULES)
    second = artificial_rule_graph(2, tuple(reversed(CARDINAL_RULES)))

    assert first.edges == second.edges
    assert first.metadata == second.metadata
    assert first.coordinates is not None
    assert second.coordinates is not None
    assert np.array_equal(first.coordinates, second.coordinates)
    assert first.metadata["displacement_rules"] == (
        (-1, 0),
        (0, -1),
        (0, 1),
        (1, 0),
    )


def test_opposite_rules_share_stable_edge_families() -> None:
    geometry = artificial_rule_graph(1, CARDINAL_RULES)

    assert geometry.metadata["canonical_displacement_families"] == (
        (-1, 0),
        (0, -1),
    )
    assert {edge.edge_type for edge in geometry.edges} == {
        "rule_displacement_0",
        "rule_displacement_1",
    }
    for edge in geometry.edges:
        family = edge.metadata["canonical_displacement_rule"]
        assert family in geometry.metadata["canonical_displacement_families"]


def test_spacing_scales_coordinates_and_edge_displacements() -> None:
    geometry = artificial_rule_graph(1, ((-2,), (2,)), spacing=0.25)

    assert geometry.coordinates is not None
    assert set(geometry.coordinates[:, 0]) == {-0.5, 0.0, 0.5}
    assert all(edge.displacement is not None for edge in geometry.edges)
    assert all(
        abs(edge.displacement[0]) == 0.5
        for edge in geometry.edges
        if edge.displacement is not None
    )


@pytest.mark.parametrize("dimension", (1, 2, 3, 7))
def test_builder_supports_arbitrary_embedding_dimension(dimension: int) -> None:
    rules = tuple(
        rule
        for axis in range(dimension)
        for sign in (-1, 1)
        for rule in (
            tuple(sign if component == axis else 0 for component in range(dimension)),
        )
    )
    geometry = artificial_rule_graph(1, rules)

    assert geometry.embedding_dimension == dimension
    assert geometry.n_sites == 1 + 2 * dimension
    assert geometry.n_edges == 2 * dimension


def test_last_rule_distance_shell_is_an_explicit_procedural_boundary() -> None:
    geometry = artificial_rule_graph(2, CARDINAL_RULES)

    assert len(geometry.boundary_components) == 1
    component = geometry.boundary_components[0]
    assert component.kind == "outer"
    assert component.component_index == 0
    assert component.sites == geometry.boundary_sites
    assert geometry.metadata["boundary_definition"] == "maximum_rule_distance_shell"


def test_unnamed_geometry_does_not_invent_physical_or_dimensional_semantics() -> None:
    geometry = artificial_rule_graph(2, CARDINAL_RULES)

    assert geometry.site_types is None
    assert geometry.dimension_records == ()
    assert geometry.faces == ()
    assert geometry.rooted_tree is None
    assert not {"hopping", "onsite", "pairing"} & geometry.metadata.keys()


def test_builder_is_available_through_common_registry() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "artificial_rule_graph",
        parameters={
            "iterations": 1,
            "displacement_rules": [[-1], [1]],
            "spacing": 0.5,
        },
    )

    assert geometry.n_sites == 3
    assert geometry.metadata["generation"] == {
        "generator_key": "artificial_rule_graph",
        "generator_version": 1,
        "parameters": {
            "displacement_rules": ((-1,), (1,)),
            "iterations": 1,
            "spacing": 0.5,
        },
        "seed": None,
    }


def test_metadata_records_declarative_construction_only() -> None:
    geometry = artificial_rule_graph(1, LINE_RULES)

    assert geometry.metadata["generator"] == "artificial_rule_graph"
    assert geometry.metadata["family"] == "unnamed_integer_displacement_rule_geometry"
    assert geometry.metadata["construction"] == "breadth_first_displacement_rules"
    assert geometry.metadata["rule_order"] == "lexicographic_normalized"
    assert geometry.metadata["embedding_dimension"] == 1


def test_site_budget_accepts_exact_graph_and_rejects_next() -> None:
    assert artificial_rule_graph(2, CARDINAL_RULES, max_sites=13).n_sites == 13
    with pytest.raises(ValueError, match=r"more than max_sites=12 at iteration 2"):
        artificial_rule_graph(2, CARDINAL_RULES, max_sites=12)


def test_edge_budget_accepts_exact_graph_and_rejects_next() -> None:
    assert artificial_rule_graph(2, CARDINAL_RULES, max_edges=16).n_edges == 16
    with pytest.raises(ValueError, match=r"more than max_edges=15"):
        artificial_rule_graph(2, CARDINAL_RULES, max_edges=15)


def test_rule_application_budget_accepts_exact_work_and_rejects_next() -> None:
    geometry = artificial_rule_graph(
        2,
        CARDINAL_RULES,
        max_rule_applications=52,
    )
    assert geometry.metadata["n_rule_applications"] == 52
    with pytest.raises(
        ValueError,
        match=r"requires at least 52 rule applications.*max_rule_applications=51",
    ):
        artificial_rule_graph(2, CARDINAL_RULES, max_rule_applications=51)


def test_resource_budgets_can_be_explicitly_disabled() -> None:
    geometry = artificial_rule_graph(
        2,
        CARDINAL_RULES,
        max_sites=None,
        max_edges=None,
        max_rule_applications=None,
    )

    assert geometry.n_sites == 13
    assert geometry.metadata["max_sites"] is None
    assert geometry.metadata["max_edges"] is None
    assert geometry.metadata["max_rule_applications"] is None


@pytest.mark.parametrize(
    "rules",
    (
        1,
        [1, -1],
        [[[1]], [[-1]]],
        [],
        np.empty((2, 0), dtype=int),
        [[1], [-1, 0]],
    ),
)
def test_builder_rejects_nonrectangular_or_nonmatrix_rules(rules: object) -> None:
    with pytest.raises(ValueError):
        artificial_rule_graph(1, rules)  # type: ignore[arg-type]


@pytest.mark.parametrize("rules", (((0,),), ((-1,), (-1,), (1,))))
def test_builder_rejects_zero_or_duplicate_rules(
    rules: tuple[tuple[int, ...], ...],
) -> None:
    with pytest.raises(ValueError):
        artificial_rule_graph(1, rules)


def test_builder_requires_an_inverse_for_every_rule() -> None:
    with pytest.raises(ValueError, match=r"missing inverse rule"):
        artificial_rule_graph(1, ((1, 0), (-1, 0), (0, 1)))


@pytest.mark.parametrize("component", (True, 1.5, "1"))
def test_builder_rejects_noninteger_rule_components(component: object) -> None:
    with pytest.raises(TypeError, match="components must be integers"):
        artificial_rule_graph(1, ((component,), (-1,)))  # type: ignore[arg-type]


@pytest.mark.parametrize("iterations", (-1, -10))
def test_builder_rejects_negative_iterations(iterations: int) -> None:
    with pytest.raises(ValueError):
        artificial_rule_graph(iterations, LINE_RULES)


@pytest.mark.parametrize("iterations", (True, 1.5, "1"))
def test_builder_rejects_noninteger_iterations(iterations: object) -> None:
    with pytest.raises(TypeError):
        artificial_rule_graph(iterations, LINE_RULES)  # type: ignore[arg-type]


@pytest.mark.parametrize("spacing", (0.0, -1.0, np.inf, np.nan))
def test_builder_rejects_invalid_spacing(spacing: float) -> None:
    with pytest.raises(ValueError):
        artificial_rule_graph(1, LINE_RULES, spacing=spacing)


@pytest.mark.parametrize("spacing", (True, "1"))
def test_builder_rejects_nonreal_spacing(spacing: object) -> None:
    with pytest.raises(TypeError):
        artificial_rule_graph(1, LINE_RULES, spacing=spacing)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "keyword_arguments",
    (
        {"max_sites": 0},
        {"max_edges": -1},
        {"max_rule_applications": 0},
    ),
)
def test_builder_rejects_nonpositive_resource_budgets(
    keyword_arguments: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        artificial_rule_graph(1, LINE_RULES, **keyword_arguments)


@pytest.mark.parametrize(
    "keyword_arguments",
    (
        {"max_sites": True},
        {"max_edges": 1.5},
        {"max_rule_applications": "10"},
    ),
)
def test_builder_rejects_noninteger_resource_budgets(
    keyword_arguments: dict[str, Any],
) -> None:
    with pytest.raises(TypeError):
        artificial_rule_graph(1, LINE_RULES, **keyword_arguments)
