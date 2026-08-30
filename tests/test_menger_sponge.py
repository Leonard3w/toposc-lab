from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    DEFAULT_MENGER_MAX_SITES,
    Geometry,
    GeometryBoundaryComponent,
    GeometryDimension,
    menger_sponge,
)


@pytest.mark.parametrize("order", range(4))
def test_menger_counts_follow_exact_recursion(order: int) -> None:
    geometry = menger_sponge(order)

    assert isinstance(geometry, Geometry)
    assert geometry.n_sites == 20**order
    assert geometry.n_edges == 2 * (20**order - 8**order)


def test_order_zero_is_one_voxel_with_an_outer_boundary() -> None:
    geometry = menger_sponge(0, spacing=2.0)

    assert geometry.n_sites == 1
    assert geometry.n_edges == 0
    assert geometry.coordinates is not None
    assert np.array_equal(geometry.coordinates, ((1.0, 1.0, 1.0),))
    assert geometry.boundary_sites == frozenset({0})
    assert geometry.boundary_components == (
        GeometryBoundaryComponent("outer", 0, frozenset({0})),
    )


def test_menger_is_connected_with_cubic_grid_degrees() -> None:
    geometry = menger_sponge(3)
    visited = {0}
    pending = [0]
    while pending:
        site = pending.pop()
        for neighbor in geometry.neighbors(site):
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)

    assert visited == set(geometry.site_indices)
    assert {geometry.degree(site) for site in geometry.site_indices} == {
        2,
        3,
        4,
        5,
        6,
    }


def test_edges_join_only_face_adjacent_voxel_centers() -> None:
    spacing = 0.25
    geometry = menger_sponge(2, spacing=spacing)

    assert all(
        geometry.distance(edge.source, edge.target) == pytest.approx(spacing)
        for edge in geometry.edges
    )
    assert all(
        sum(abs(value) > 0.0 for value in edge.displacement or ()) == 1
        for edge in geometry.edges
    )


def test_coordinates_obey_the_menger_digit_rule() -> None:
    order = 2
    geometry = menger_sponge(order)
    side_length = 3**order

    assert geometry.coordinates is not None
    assert geometry.coordinates.shape == (20**order, 3)
    assert geometry.embedding_dimension == 3
    assert np.min(geometry.coordinates) == pytest.approx(0.5)
    assert np.max(geometry.coordinates) == pytest.approx(side_length - 0.5)
    voxel_coordinates = np.rint(geometry.coordinates - 0.5).astype(int)
    assert all(
        all(
            sum((coordinate // 3**level) % 3 == 1 for coordinate in voxel) <= 1
            for level in range(order)
        )
        for voxel in voxel_coordinates
    )
    assert geometry.metadata["shape"] == (
        side_length,
        side_length,
        side_length,
    )
    assert geometry.metadata["outer_side_length"] == pytest.approx(side_length)
    assert geometry.metadata["site_representation"] == "retained_voxel_centers"


@pytest.mark.parametrize("order", (1, 2, 3))
def test_removed_voxels_form_one_exterior_accessible_tunnel_system(
    order: int,
) -> None:
    geometry = menger_sponge(order)
    assert geometry.coordinates is not None
    side_length = 3**order
    retained = {
        tuple(int(round(value - 0.5)) for value in position)
        for position in geometry.coordinates
    }
    removed = set(product(range(side_length), repeat=3)) - retained
    exterior_openings = {
        voxel
        for voxel in removed
        if any(axis in (0, side_length - 1) for axis in voxel)
    }
    visited = set(exterior_openings)
    pending = list(exterior_openings)
    while pending:
        x, y, z = pending.pop()
        for delta_x, delta_y, delta_z in (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ):
            neighbor = x + delta_x, y + delta_y, z + delta_z
            if neighbor in removed and neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)

    assert exterior_openings
    assert visited == removed


def test_boundary_is_one_exterior_component_including_tunnel_surfaces() -> None:
    geometry = menger_sponge(2)

    assert geometry.boundary_components == (
        GeometryBoundaryComponent("outer", 0, geometry.boundary_sites),
    )
    assert geometry.boundary_sites == frozenset(
        site
        for site in geometry.site_indices
        if geometry.degree(site) < 6
    )
    assert len(geometry.boundary_sites) == 392
    assert (
        geometry.metadata["boundary_definition"]
        == "exterior_accessible_surface_including_tunnels"
    )


def test_menger_dimensions_are_explicit_and_separate_from_embedding() -> None:
    geometry = menger_sponge(2)

    assert geometry.embedding_dimension == 3
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
            value=float(np.log(20.0) / np.log(3.0)),
            scope="infinite_family",
            method="analytic_self_similarity_log_20_over_log_3",
            exact=True,
        ),
    )


def test_menger_generator_is_deterministic() -> None:
    first = menger_sponge(2, spacing=0.4)
    second = menger_sponge(2, spacing=0.4)

    assert first.edges == second.edges
    assert first.coordinates is not None
    assert second.coordinates is not None
    assert np.array_equal(first.coordinates, second.coordinates)
    assert first.boundary_sites == second.boundary_sites
    assert first.boundary_components == second.boundary_components
    assert first.dimension_records == second.dimension_records


def test_menger_generator_is_available_through_common_registry() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "menger_sponge",
        parameters={"order": 2, "spacing": 0.5, "max_sites": 400},
    )

    assert geometry.n_sites == 400
    assert geometry.n_edges == 672
    assert geometry.metadata["generation"] == {
        "generator_key": "menger_sponge",
        "generator_version": 1,
        "parameters": {"order": 2, "spacing": 0.5, "max_sites": 400},
        "seed": None,
    }


def test_default_site_budget_rejects_order_four_before_construction() -> None:
    assert DEFAULT_MENGER_MAX_SITES < 20**4
    with pytest.raises(ValueError, match=r"requires 160000 sites.*max_sites=25000"):
        menger_sponge(4)


def test_explicit_site_budget_is_applied_to_estimated_site_count() -> None:
    assert menger_sponge(2, max_sites=400).n_sites == 400
    with pytest.raises(ValueError, match=r"requires 400 sites.*max_sites=399"):
        menger_sponge(2, max_sites=399)


@pytest.mark.parametrize("order", (-1, -5))
def test_menger_rejects_negative_order(order: int) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        menger_sponge(order)


@pytest.mark.parametrize("order", (True, 1.5, "2"))
def test_menger_rejects_non_integer_order(order: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        menger_sponge(order)  # type: ignore[arg-type]


@pytest.mark.parametrize("spacing", (0.0, -1.0, np.inf, np.nan))
def test_menger_rejects_invalid_spacing(spacing: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        menger_sponge(2, spacing=spacing)


@pytest.mark.parametrize("max_sites", (0, -1, -10))
def test_menger_rejects_nonpositive_site_budget(max_sites: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        menger_sponge(1, max_sites=max_sites)


@pytest.mark.parametrize("max_sites", (True, 1.5, "400"))
def test_menger_rejects_non_integer_site_budget(max_sites: object) -> None:
    with pytest.raises(TypeError, match="must be an integer or None"):
        menger_sponge(1, max_sites=max_sites)  # type: ignore[arg-type]
