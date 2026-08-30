from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    Geometry,
    GeometryBoundaryComponent,
    GeometryDimension,
    sierpinski_carpet,
)


@pytest.mark.parametrize("order", range(5))
def test_sierpinski_carpet_counts_follow_exact_recursion(order: int) -> None:
    geometry = sierpinski_carpet(order)

    assert isinstance(geometry, Geometry)
    assert geometry.n_sites == 8**order
    assert geometry.n_edges == 8 * (8**order - 3**order) // 5
    assert geometry.metadata["n_holes"] == (8**order - 1) // 7


def test_order_zero_is_one_cell_with_only_an_outer_boundary() -> None:
    geometry = sierpinski_carpet(0, spacing=2.0)

    assert geometry.n_sites == 1
    assert geometry.n_edges == 0
    assert geometry.coordinates is not None
    assert np.array_equal(geometry.coordinates, ((1.0, 1.0),))
    assert geometry.boundary_sites == frozenset({0})
    assert geometry.boundary_components == (
        GeometryBoundaryComponent("outer", 0, frozenset({0})),
    )


def test_sierpinski_carpet_is_connected_with_square_grid_degrees() -> None:
    geometry = sierpinski_carpet(4)
    visited = {0}
    pending = [0]
    while pending:
        site = pending.pop()
        for neighbor in geometry.neighbors(site):
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)

    assert visited == set(geometry.site_indices)
    assert {geometry.degree(site) for site in geometry.site_indices} == {2, 3, 4}


def test_edges_join_only_orthogonally_adjacent_cell_centers() -> None:
    spacing = 0.25
    geometry = sierpinski_carpet(3, spacing=spacing)

    assert all(
        geometry.distance(edge.source, edge.target) == pytest.approx(spacing)
        for edge in geometry.edges
    )
    assert all(
        sum(abs(value) > 0.0 for value in edge.displacement or ()) == 1
        for edge in geometry.edges
    )


def test_coordinates_cover_the_cell_centers_of_the_outer_square() -> None:
    order = 3
    spacing = 0.4
    geometry = sierpinski_carpet(order, spacing=spacing)
    side_length = 3**order

    assert geometry.coordinates is not None
    assert geometry.coordinates.shape == (8**order, 2)
    assert geometry.embedding_dimension == 2
    assert np.min(geometry.coordinates) == pytest.approx(0.5 * spacing)
    assert np.max(geometry.coordinates) == pytest.approx(
        (side_length - 0.5) * spacing
    )
    assert geometry.metadata["shape"] == (side_length, side_length)
    assert geometry.metadata["outer_side_length"] == pytest.approx(
        side_length * spacing
    )
    assert geometry.metadata["site_representation"] == "retained_cell_centers"


@pytest.mark.parametrize("order", range(1, 5))
def test_boundary_components_preserve_outer_and_each_hole(order: int) -> None:
    geometry = sierpinski_carpet(order)
    outer, *holes = geometry.boundary_components
    side_length = 3**order

    assert outer.kind == "outer"
    assert outer.component_index == 0
    assert len(outer.sites) == 4 * side_length - 4
    assert len(holes) == (8**order - 1) // 7
    assert all(component.kind == "hole" for component in holes)
    assert [component.component_index for component in holes] == list(
        range(len(holes))
    )
    assert Counter(len(component.sites) for component in holes) == Counter(
        {
            4 * 3**level: 8 ** (order - level - 1)
            for level in range(order)
        }
    )
    assert geometry.boundary_sites == frozenset(
        site
        for component in geometry.boundary_components
        for site in component.sites
    )
    assert geometry.metadata["boundary_definition"] == "outer_and_hole_components"


def test_distinct_hole_boundaries_may_overlap_at_one_retained_cell() -> None:
    geometry = sierpinski_carpet(2)
    holes = tuple(
        component
        for component in geometry.boundary_components
        if component.kind == "hole"
    )
    memberships = Counter(
        site
        for component in holes
        for site in component.sites
    )

    assert max(memberships.values()) == 2
    assert sum(count == 2 for count in memberships.values()) == 4


def test_sierpinski_carpet_dimensions_are_explicit_and_separate() -> None:
    geometry = sierpinski_carpet(3)

    assert geometry.embedding_dimension == 2
    assert geometry.dimension_records == (
        GeometryDimension(
            kind="topological",
            value=1.0,
            scope="infinite_family",
            method="covering_dimension",
            exact=True,
        ),
        GeometryDimension(
            kind="hausdorff",
            value=float(np.log(8.0) / np.log(3.0)),
            scope="infinite_family",
            method="analytic_self_similarity_log_8_over_log_3",
            exact=True,
        ),
    )


def test_sierpinski_carpet_generator_is_deterministic() -> None:
    first = sierpinski_carpet(3, spacing=0.4)
    second = sierpinski_carpet(3, spacing=0.4)

    assert first.edges == second.edges
    assert first.coordinates is not None
    assert second.coordinates is not None
    assert np.array_equal(first.coordinates, second.coordinates)
    assert first.boundary_sites == second.boundary_sites
    assert first.boundary_components == second.boundary_components
    assert first.dimension_records == second.dimension_records


def test_sierpinski_carpet_is_available_through_common_registry() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "sierpinski_carpet",
        parameters={"order": 2, "spacing": 0.5},
    )

    assert geometry.n_sites == 64
    assert geometry.n_edges == 88
    assert geometry.metadata["generation"] == {
        "generator_key": "sierpinski_carpet",
        "generator_version": 1,
        "parameters": {"order": 2, "spacing": 0.5},
        "seed": None,
    }


@pytest.mark.parametrize("order", (-1, -5))
def test_sierpinski_carpet_rejects_negative_order(order: int) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        sierpinski_carpet(order)


@pytest.mark.parametrize("order", (True, 1.5, "2"))
def test_sierpinski_carpet_rejects_non_integer_order(order: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        sierpinski_carpet(order)  # type: ignore[arg-type]


@pytest.mark.parametrize("spacing", (0.0, -1.0, np.inf, np.nan))
def test_sierpinski_carpet_rejects_invalid_spacing(spacing: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        sierpinski_carpet(2, spacing=spacing)
