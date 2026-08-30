from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    Geometry,
    honeycomb,
)
from toposc_lab.lattices.honeycomb import HoneycombLattice


@pytest.mark.parametrize(
    ("boundary_x", "boundary_y"),
    (
        ("open", "open"),
        ("periodic", "open"),
        ("open", "periodic"),
        ("periodic", "periodic"),
    ),
)
def test_honeycomb_geometry_matches_legacy_coordinates_and_nearest_neighbors(
    boundary_x: str,
    boundary_y: str,
) -> None:
    geometry = honeycomb(
        3,
        4,
        boundary_x=boundary_x,
        boundary_y=boundary_y,
    )
    legacy = HoneycombLattice(
        n_x=3,
        n_y=4,
        boundary_x=boundary_x,
        boundary_y=boundary_y,
    )

    assert geometry.coordinates is not None
    assert np.allclose(geometry.coordinates, legacy.coordinates)
    assert tuple(
        (
            edge.source,
            edge.target,
            edge.metadata["lattice_direction"],
        )
        for edge in geometry.edges
    ) == tuple(
        (bond.source, bond.target, bond.direction)
        for bond in legacy.bonds
    )
    assert all(
        geometry.neighbors(site) == legacy.neighbors(site)
        for site in geometry.site_indices
    )


def test_open_honeycomb_has_two_sublattices_and_expected_boundary() -> None:
    geometry = honeycomb(3, 4)

    assert isinstance(geometry, Geometry)
    assert geometry.n_sites == 24
    assert geometry.n_edges == 29
    assert geometry.embedding_dimension == 2
    assert geometry.site_types == tuple("A" if site % 2 == 0 else "B" for site in range(24))
    assert geometry.boundary_sites == frozenset(
        {0, 2, 4, 6, 7, 8, 15, 16, 17, 19, 21, 23}
    )
    assert geometry.neighbors(10) == (3, 9, 11)
    assert geometry.degree(10) == 3


def test_fully_periodic_honeycomb_is_three_regular() -> None:
    geometry = honeycomb(
        3,
        4,
        boundary_x="periodic",
        boundary_y="periodic",
    )

    assert geometry.n_edges == 36
    assert geometry.boundary_sites == frozenset()
    assert all(geometry.degree(site) == 3 for site in geometry.site_indices)
    assert geometry.neighbors(0) == (1, 7, 17)
    assert sum(edge.boundary_crossing for edge in geometry.edges) == 7


def test_mixed_honeycomb_boundaries_mark_only_open_edges() -> None:
    geometry = honeycomb(3, 4, boundary_x="periodic", boundary_y="open")

    assert geometry.n_edges == 33
    assert geometry.boundary_sites == frozenset({0, 7, 8, 15, 16, 23})
    assert all(
        not edge.boundary_crossing
        or edge.metadata["lattice_direction"] == (-1, 0)
        for edge in geometry.edges
    )


def test_honeycomb_spacing_controls_all_nearest_neighbor_distances() -> None:
    geometry = honeycomb(
        3,
        4,
        spacing=0.4,
        boundary_x="periodic",
        boundary_y="periodic",
    )

    assert all(
        geometry.distance(edge.source, edge.target) == pytest.approx(0.4)
        for edge in geometry.edges
    )
    assert geometry.metadata["spacing"] == 0.4
    assert geometry.metadata["sublattice_offset"] == pytest.approx(
        (np.sqrt(3.0) * 0.2, 0.2)
    )


def test_periodic_honeycomb_edges_retain_oriented_local_displacements() -> None:
    geometry = honeycomb(
        3,
        3,
        boundary_x="periodic",
        boundary_y="periodic",
    )

    assert np.allclose(
        geometry.displacement_between(0, 13),
        [-np.sqrt(3.0) / 2.0, 0.5],
    )
    assert np.allclose(
        geometry.displacement_between(0, 5),
        [0.0, -1.0],
    )
    assert np.allclose(
        geometry.displacement_between(13, 0),
        [np.sqrt(3.0) / 2.0, -0.5],
    )


def test_haldane_next_nearest_chirality_is_derivable_from_geometry() -> None:
    geometry = honeycomb(
        3,
        3,
        boundary_x="periodic",
        boundary_y="periodic",
    )
    legacy = HoneycombLattice(
        n_x=3,
        n_y=3,
        boundary_x="periodic",
        boundary_y="periodic",
    )

    derived = _derive_next_nearest_neighbors(geometry)
    expected = {
        (bond.source, bond.target): (
            bond.intermediate,
            bond.chirality,
        )
        for bond in legacy.next_nearest_neighbor_bonds
    }

    assert len(derived) == 54
    assert derived == expected


def test_honeycomb_generator_is_available_through_common_registry() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "honeycomb",
        parameters={
            "n_x": 2,
            "n_y": 3,
            "spacing": 0.75,
            "boundary_y": "periodic",
        },
    )

    assert geometry.n_sites == 12
    assert geometry.metadata["generation"] == {
        "generator_key": "honeycomb",
        "generator_version": 1,
        "parameters": {
            "boundary_y": "periodic",
            "n_x": 2,
            "n_y": 3,
            "spacing": 0.75,
        },
        "seed": None,
    }


@pytest.mark.parametrize(("n_x", "n_y"), ((1, 3), (3, 1), (0, 3), (3, -1)))
def test_honeycomb_rejects_too_small_axes(n_x: int, n_y: int) -> None:
    with pytest.raises(ValueError, match="at least two"):
        honeycomb(n_x, n_y)


@pytest.mark.parametrize(("n_x", "n_y"), ((2.5, 3), (3, True)))
def test_honeycomb_rejects_non_integer_axes(n_x: object, n_y: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        honeycomb(n_x, n_y)  # type: ignore[arg-type]


@pytest.mark.parametrize("spacing", (0.0, -1.0, np.inf, np.nan))
def test_honeycomb_rejects_invalid_spacing(spacing: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        honeycomb(2, 3, spacing=spacing)


@pytest.mark.parametrize("boundary", ("closed", "", "OPEN"))
def test_honeycomb_rejects_invalid_boundary(boundary: str) -> None:
    with pytest.raises(ValueError, match="either open or periodic"):
        honeycomb(3, 3, boundary_x=boundary)
    with pytest.raises(ValueError, match="either open or periodic"):
        honeycomb(3, 3, boundary_y=boundary)


def _derive_next_nearest_neighbors(
    geometry: Geometry,
) -> Mapping[tuple[int, int], tuple[int, int]]:
    adjacency: dict[int, list[tuple[int, np.ndarray]]] = {
        site: [] for site in geometry.site_indices
    }
    for edge in geometry.edges:
        displacement = geometry.displacement_between(edge.source, edge.target)
        adjacency[edge.source].append((edge.target, displacement))
        adjacency[edge.target].append((edge.source, -displacement))

    bonds: dict[tuple[int, int], tuple[int, int]] = {}
    for source in geometry.site_indices:
        for intermediate, first_step in adjacency[source]:
            for target, second_step in adjacency[intermediate]:
                if target <= source or geometry.site_types is None:
                    continue
                if geometry.site_types[source] != geometry.site_types[target]:
                    continue
                cross_product_z = (
                    first_step[0] * second_step[1]
                    - first_step[1] * second_step[0]
                )
                key = (source, target)
                if key not in bonds:
                    bonds[key] = (
                        intermediate,
                        1 if cross_product_z > 0.0 else -1,
                    )
    return bonds
