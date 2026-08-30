from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    Geometry,
    GeometryDimension,
    sierpinski_gasket,
)


@pytest.mark.parametrize("order", range(5))
def test_sierpinski_counts_follow_exact_recursion(order: int) -> None:
    geometry = sierpinski_gasket(order)

    assert isinstance(geometry, Geometry)
    assert geometry.n_sites == (3 ** (order + 1) + 3) // 2
    assert geometry.n_edges == 3 ** (order + 1)
    assert len(geometry.boundary_sites) == 3 * 2**order


def test_order_zero_is_one_equilateral_triangle() -> None:
    geometry = sierpinski_gasket(0, spacing=2.0)

    assert geometry.n_sites == 3
    assert geometry.n_edges == 3
    assert geometry.boundary_sites == frozenset({0, 1, 2})
    assert geometry.coordinates is not None
    assert all(
        geometry.distance(edge.source, edge.target) == pytest.approx(2.0)
        for edge in geometry.edges
    )


def test_sierpinski_is_connected_with_expected_vertex_degrees() -> None:
    geometry = sierpinski_gasket(4)
    visited = {0}
    pending = [0]
    while pending:
        site = pending.pop()
        for neighbor in geometry.neighbors(site):
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)

    assert visited == set(geometry.site_indices)
    assert {geometry.degree(site) for site in geometry.site_indices} == {2, 4}


def test_sierpinski_has_expected_upright_triangles_and_hole_boundaries() -> None:
    order = 3
    geometry = sierpinski_gasket(order)
    triangles = {
        tuple(sorted((first, second, third)))
        for first in geometry.site_indices
        for second, third in combinations(geometry.neighbors(first), 2)
        if geometry.has_edge(second, third)
    }

    assert geometry.coordinates is not None
    signed_double_areas = tuple(
        (
            geometry.coordinates[second, 0] - geometry.coordinates[first, 0]
        )
        * (
            geometry.coordinates[third, 1] - geometry.coordinates[first, 1]
        )
        - (
            geometry.coordinates[second, 1] - geometry.coordinates[first, 1]
        )
        * (
            geometry.coordinates[third, 0] - geometry.coordinates[first, 0]
        )
        for first, second, third in triangles
    )

    assert sum(area > 0.0 for area in signed_double_areas) == 3**order
    assert sum(area < 0.0 for area in signed_double_areas) == 3 ** (order - 1)


def test_sierpinski_coordinates_and_boundary_use_outer_triangle() -> None:
    order = 3
    spacing = 0.25
    geometry = sierpinski_gasket(order, spacing=spacing)
    outer_length = 2**order * spacing

    assert geometry.coordinates is not None
    assert geometry.coordinates.shape == (geometry.n_sites, 2)
    assert geometry.embedding_dimension == 2
    assert np.min(geometry.coordinates) == pytest.approx(0.0)
    assert np.max(geometry.coordinates[:, 0]) == pytest.approx(outer_length)
    assert np.max(geometry.coordinates[:, 1]) == pytest.approx(
        np.sqrt(3.0) * outer_length / 2.0
    )
    assert geometry.metadata["boundary_definition"] == "outer_triangle"
    assert geometry.metadata["outer_side_length"] == pytest.approx(outer_length)


def test_sierpinski_dimensions_are_explicit_and_separate_from_embedding() -> None:
    geometry = sierpinski_gasket(3)

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
            value=float(np.log(3.0) / np.log(2.0)),
            scope="infinite_family",
            method="analytic_self_similarity_log_3_over_log_2",
            exact=True,
        ),
    )
    assert "intrinsic_dimension" not in geometry.metadata


def test_sierpinski_generator_is_deterministic() -> None:
    first = sierpinski_gasket(3, spacing=0.4)
    second = sierpinski_gasket(3, spacing=0.4)

    assert first.edges == second.edges
    assert first.coordinates is not None
    assert second.coordinates is not None
    assert np.array_equal(first.coordinates, second.coordinates)
    assert first.boundary_sites == second.boundary_sites
    assert first.dimension_records == second.dimension_records


def test_sierpinski_generator_is_available_through_common_registry() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "sierpinski_gasket",
        parameters={"order": 2, "spacing": 0.5},
    )

    assert geometry.n_sites == 15
    assert geometry.metadata["generation"] == {
        "generator_key": "sierpinski_gasket",
        "generator_version": 1,
        "parameters": {"order": 2, "spacing": 0.5},
        "seed": None,
    }


@pytest.mark.parametrize("order", (-1, -5))
def test_sierpinski_rejects_negative_order(order: int) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        sierpinski_gasket(order)


@pytest.mark.parametrize("order", (True, 1.5, "2"))
def test_sierpinski_rejects_non_integer_order(order: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        sierpinski_gasket(order)  # type: ignore[arg-type]


@pytest.mark.parametrize("spacing", (0.0, -1.0, np.inf, np.nan))
def test_sierpinski_rejects_invalid_spacing(spacing: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        sierpinski_gasket(2, spacing=spacing)
