from __future__ import annotations

from math import sqrt

import numpy as np
import pytest

from toposc_lab.geometry import (
    AMMANN_BEENKER_INFLATION_FACTOR,
    BUILTIN_GEOMETRY_GENERATORS,
    DEFAULT_AMMANN_BEENKER_MAX_CANDIDATES,
    Geometry,
    GeometryBoundaryComponent,
    GeometryDimension,
    GeometryFace,
    ammann_beenker_patch,
)


@pytest.mark.parametrize(
    ("radius", "expected_counts"),
    (
        (2.0, (17, 24, 8, 0, 8, 8)),
        (2.5, (25, 40, 16, 8, 8, 8)),
        (3.0, (33, 56, 24, 8, 16, 16)),
        (5.0, (89, 160, 72, 32, 40, 32)),
        (10.0, (385, 712, 328, 136, 192, 64)),
    ),
)
def test_centered_reference_patches_have_exact_counts(
    radius: float,
    expected_counts: tuple[int, int, int, int, int, int],
) -> None:
    geometry = ammann_beenker_patch(radius)

    assert (
        geometry.n_sites,
        geometry.n_edges,
        geometry.n_faces,
        geometry.metadata["n_square_tiles"],
        geometry.metadata["n_rhombus_tiles"],
        len(geometry.boundary_sites),
    ) == expected_counts


def test_patch_is_a_connected_planar_cell_complex() -> None:
    geometry = ammann_beenker_patch(8.0)
    visited = {0}
    pending = [0]
    while pending:
        site = pending.pop()
        for neighbor in geometry.neighbors(site):
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)

    assert len(visited) == geometry.n_sites
    assert geometry.n_sites - geometry.n_edges + geometry.n_faces == 1


def test_all_edges_have_unit_scaled_length_and_four_orientations() -> None:
    spacing = 1.75
    geometry = ammann_beenker_patch(5.0, spacing=spacing)

    assert {edge.edge_type for edge in geometry.edges} == {
        "ammann_beenker_axis_0",
        "ammann_beenker_axis_1",
        "ammann_beenker_axis_2",
        "ammann_beenker_axis_3",
    }
    assert {edge.metadata["projection_axis"] for edge in geometry.edges} == {
        0,
        1,
        2,
        3,
    }
    assert all(
        geometry.distance(edge.source, edge.target) == pytest.approx(spacing)
        for edge in geometry.edges
    )


def test_faces_are_explicit_unit_squares_and_45_degree_rhombi() -> None:
    geometry = ammann_beenker_patch(5.0)
    assert geometry.coordinates is not None

    assert all(isinstance(face, GeometryFace) for face in geometry.faces)
    assert {face.face_type for face in geometry.faces} == {
        "square",
        "rhombus_45_degree",
    }
    for face in geometry.faces:
        assert len(face.sites) == 4
        cyclic_sites = face.sites[1:] + face.sites[:1]
        assert all(
            geometry.has_edge(source, target)
            for source, target in zip(face.sites, cyclic_sites, strict=True)
        )
        polygon = geometry.coordinates[np.asarray(face.sites)]
        side_lengths = np.linalg.norm(
            np.roll(polygon, -1, axis=0) - polygon,
            axis=1,
        )
        area = 0.5 * abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
        assert side_lengths == pytest.approx(np.ones(4))
        expected_area = 1.0 if face.face_type == "square" else 1.0 / sqrt(2.0)
        assert area == pytest.approx(expected_area)


def test_centered_patch_has_eightfold_rotational_site_symmetry() -> None:
    geometry = ammann_beenker_patch(8.0)
    assert geometry.coordinates is not None
    angle = np.pi / 4.0
    rotation = np.asarray(
        ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    )
    coordinate_set = {
        tuple(np.round(coordinate, decimals=12)) for coordinate in geometry.coordinates
    }
    rotated_set = {
        tuple(np.round(rotation @ coordinate, decimals=12))
        for coordinate in geometry.coordinates
    }

    assert rotated_set == coordinate_set


def test_site_types_record_infinite_tiling_coordination_not_cut_degree() -> None:
    geometry = ammann_beenker_patch(10.0)
    assert geometry.coordinates is not None
    assert geometry.site_types is not None
    full_degrees: list[int] = []
    for site_type in geometry.site_types:
        assert site_type is not None
        full_degrees.append(int(site_type.removeprefix("coordination_")))
    origin = int(np.argmin(np.linalg.norm(geometry.coordinates, axis=1)))

    assert set(full_degrees) == {3, 4, 5, 6, 7, 8}
    assert full_degrees[origin] == 8
    assert geometry.degree(origin) == 8
    assert geometry.boundary_sites == frozenset(
        site
        for site, full_degree in enumerate(full_degrees)
        if geometry.degree(site) < full_degree
    )


def test_outer_boundary_is_one_explicit_radial_cut_component() -> None:
    geometry = ammann_beenker_patch(5.0)

    assert geometry.boundary_components == (
        GeometryBoundaryComponent("outer", 0, geometry.boundary_sites),
    )
    assert geometry.metadata["boundary_definition"] == (
        "accepted_neighbor_outside_radial_cutoff"
    )


def test_coordinates_obey_radial_cutoff_and_spacing() -> None:
    geometry = ammann_beenker_patch(5.0, spacing=2.0)
    assert geometry.coordinates is not None

    radii = np.linalg.norm(geometry.coordinates, axis=1)
    assert np.all(radii <= 10.0 + 1.0e-12)
    assert np.max(radii) > 10.0 - 2.0
    assert geometry.metadata["radius_in_edge_lengths"] == 5.0
    assert geometry.metadata["cutoff_radius"] == 10.0


def test_dimension_semantics_do_not_claim_a_translation_lattice() -> None:
    geometry = ammann_beenker_patch(5.0)

    assert geometry.embedding_dimension == 2
    assert geometry.dimension_records == (
        GeometryDimension(
            kind="topological",
            value=2.0,
            scope="infinite_family",
            method="ammann_beenker_tiling_support_covering_dimension",
            exact=True,
        ),
    )
    assert all(record.kind != "lattice" for record in geometry.dimension_records)


def test_ammann_beenker_patch_is_deterministic() -> None:
    first = ammann_beenker_patch(6.0, spacing=1.25)
    second = ammann_beenker_patch(6.0, spacing=1.25)

    assert first.edges == second.edges
    assert first.faces == second.faces
    assert first.coordinates is not None
    assert second.coordinates is not None
    assert np.array_equal(first.coordinates, second.coordinates)
    assert first.boundary_sites == second.boundary_sites
    assert first.site_types == second.site_types
    assert first.dimension_records == second.dimension_records
    assert first.metadata == second.metadata


def test_ammann_beenker_patch_is_available_through_common_registry() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "ammann_beenker_patch",
        parameters={"radius": 3.0, "spacing": 2.0},
    )

    assert isinstance(geometry, Geometry)
    assert geometry.n_sites == 33
    assert geometry.n_faces == 24
    assert geometry.metadata["generation"] == {
        "generator_key": "ammann_beenker_patch",
        "generator_version": 1,
        "parameters": {"radius": 3.0, "spacing": 2.0},
        "seed": None,
    }


def test_metadata_records_construction_without_model_parameters() -> None:
    geometry = ammann_beenker_patch(3.0)

    assert geometry.metadata["generator"] == "ammann_beenker_patch"
    assert geometry.metadata["family"] == "ammann_beenker_tiling"
    assert geometry.metadata["construction"] == "canonical_cut_and_project"
    assert geometry.metadata["source_lattice"] == "Z4"
    assert geometry.metadata["acceptance_window"] == "centered_regular_octagon"
    assert geometry.metadata["inflation_factor"] == pytest.approx(
        AMMANN_BEENKER_INFLATION_FACTOR
    )
    assert not {"hopping", "onsite", "pairing"} & geometry.metadata.keys()


def test_site_budget_accepts_exact_count_and_rejects_next_site() -> None:
    assert ammann_beenker_patch(5.0, max_sites=89).n_sites == 89
    with pytest.raises(ValueError, match=r"requires more than max_sites=88"):
        ammann_beenker_patch(5.0, max_sites=88)


def test_candidate_budget_is_checked_before_four_dimensional_scan() -> None:
    geometry = ammann_beenker_patch(10.0, max_candidates=28_561)
    assert geometry.metadata["n_lattice_candidates"] == 28_561

    with pytest.raises(
        ValueError,
        match=r"requires scanning 28561 lattice candidates, exceeding max_candidates=28560",
    ):
        ammann_beenker_patch(10.0, max_candidates=28_560)


def test_default_candidate_budget_stops_huge_radius() -> None:
    with pytest.raises(
        ValueError,
        match=(
            rf"requires scanning 112550881 lattice candidates, exceeding "
            rf"max_candidates={DEFAULT_AMMANN_BEENKER_MAX_CANDIDATES}"
        ),
    ):
        ammann_beenker_patch(100.0)


def test_resource_budgets_can_be_explicitly_disabled() -> None:
    geometry = ammann_beenker_patch(
        5.0,
        max_sites=None,
        max_candidates=None,
    )

    assert geometry.n_sites == 89
    assert geometry.metadata["max_sites"] is None
    assert geometry.metadata["max_candidates"] is None


@pytest.mark.parametrize("radius", (0.0, 1.0, 1.999, -1.0, np.inf, -np.inf, np.nan))
def test_ammann_beenker_rejects_invalid_radius(radius: float) -> None:
    with pytest.raises(ValueError, match="radius must be finite and at least two"):
        ammann_beenker_patch(radius)


@pytest.mark.parametrize("radius", (True, "5.0"))
def test_ammann_beenker_rejects_nonreal_radius(radius: object) -> None:
    with pytest.raises(TypeError, match="radius must be a real number"):
        ammann_beenker_patch(radius)  # type: ignore[arg-type]


@pytest.mark.parametrize("spacing", (0.0, -1.0, np.inf, -np.inf, np.nan))
def test_ammann_beenker_rejects_invalid_spacing(spacing: float) -> None:
    with pytest.raises(ValueError, match="spacing must be finite and positive"):
        ammann_beenker_patch(3.0, spacing=spacing)


@pytest.mark.parametrize("spacing", (True, "1.0"))
def test_ammann_beenker_rejects_nonreal_spacing(spacing: object) -> None:
    with pytest.raises(TypeError, match="spacing must be a real number"):
        ammann_beenker_patch(3.0, spacing=spacing)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ("max_sites", "max_candidates"))
@pytest.mark.parametrize("value", (0, -1, -10))
def test_ammann_beenker_rejects_nonpositive_budgets(name: str, value: int) -> None:
    with pytest.raises(ValueError, match=rf"{name} must be positive"):
        ammann_beenker_patch(3.0, **{name: value})


@pytest.mark.parametrize("name", ("max_sites", "max_candidates"))
@pytest.mark.parametrize("value", (True, 1.5, "10"))
def test_ammann_beenker_rejects_noninteger_budgets(
    name: str,
    value: object,
) -> None:
    with pytest.raises(TypeError, match=rf"{name} must be an integer or None"):
        ammann_beenker_patch(3.0, **{name: value})  # type: ignore[arg-type]
