from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from toposc_lab.evaluation import exact_geometry_id
from toposc_lab.geometry import (
    Geometry,
    GeometryBoundaryComponent,
    GeometryDimension,
    GeometryEdge,
    GeometryFace,
    square,
    tree,
    validate_geometry,
)
from toposc_lab.robustness import (
    UNIFORM_COORDINATE_PERTURBATION_KEY,
    UNIFORM_COORDINATE_PERTURBATION_VERSION,
    apply_uniform_coordinate_perturbation,
)


def _annotated_triangle() -> Geometry:
    return Geometry(
        n_sites=3,
        edges=(
            GeometryEdge(
                2,
                0,
                edge_type="periodic_reference",
                boundary_crossing=True,
                displacement=(0.0, 1.0),
                metadata={"image": "local"},
            ),
            GeometryEdge(0, 1, displacement=(1.0, 0.0)),
            GeometryEdge(1, 2),
        ),
        coordinates=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        boundary_sites=frozenset({0, 1, 2}),
        boundary_components=(
            GeometryBoundaryComponent("outer", 0, frozenset({0, 1, 2})),
        ),
        site_types=("a", "b", "c"),
        dimension_records=(
            GeometryDimension(
                kind="topological",
                value=2.0,
                scope="finite_geometry",
                method="fixture declaration",
                exact=True,
            ),
        ),
        metadata={"name": "coordinate_fixture"},
        faces=(GeometryFace((0, 1, 2), face_type="triangle"),),
    )


def test_fixed_pcg64_sample_perturbs_components_in_site_major_order() -> None:
    geometry = _annotated_triangle()

    realization = apply_uniform_coordinate_perturbation(
        geometry,
        width=0.4,
        seed=42,
    )

    assert isinstance(realization.state, Geometry)
    result = realization.state
    expected_offsets = np.asarray(
        [
            [0.10958241942238534, -0.024448624099179083],
            [0.14343916796455296, 0.07894721162374556],
            [-0.1623290608449402, 0.1902489406547024],
        ]
    )
    assert result.coordinates is not None
    assert geometry.coordinates is not None
    assert np.allclose(result.coordinates, geometry.coordinates + expected_offsets)
    assert validate_geometry(result).errors == ()


def test_explicit_displacements_follow_relative_offsets_and_orientation() -> None:
    geometry = _annotated_triangle()
    realization = apply_uniform_coordinate_perturbation(
        geometry,
        width=0.4,
        seed=42,
    )

    assert isinstance(realization.state, Geometry)
    result = realization.state
    assert result.coordinates is not None
    assert geometry.coordinates is not None
    offsets = result.coordinates - geometry.coordinates

    first = result.edges[0]
    assert (first.source, first.target) == (2, 0)
    assert first.edge_type == "periodic_reference"
    assert first.boundary_crossing is True
    assert first.metadata == {"image": "local"}
    assert first.displacement is not None
    assert np.allclose(
        first.displacement,
        np.asarray((0.0, 1.0)) + offsets[0] - offsets[2],
    )
    second = result.edges[1]
    assert second.displacement is not None
    assert np.allclose(
        second.displacement,
        np.asarray((1.0, 0.0)) + offsets[1] - offsets[0],
    )


def test_implicit_displacement_remains_implicit_and_uses_new_coordinates() -> None:
    geometry = _annotated_triangle()
    realization = apply_uniform_coordinate_perturbation(
        geometry,
        width=0.4,
        seed=42,
    )

    assert isinstance(realization.state, Geometry)
    result = realization.state
    assert result.edges[2] is geometry.edges[2]
    assert result.edges[2].displacement is None
    assert result.coordinates is not None
    assert np.allclose(
        result.displacement_between(1, 2),
        result.coordinates[2] - result.coordinates[1],
    )


def test_periodic_image_displacements_remain_local_under_perturbation() -> None:
    geometry = square(
        3,
        3,
        boundary_x="periodic",
        boundary_y="periodic",
    )
    realization = apply_uniform_coordinate_perturbation(
        geometry,
        width=0.2,
        seed=7,
    )

    assert isinstance(realization.state, Geometry)
    result = realization.state
    assert geometry.coordinates is not None
    assert result.coordinates is not None
    offsets = result.coordinates - geometry.coordinates
    assert any(edge.boundary_crossing for edge in geometry.edges)
    for original, perturbed in zip(geometry.edges, result.edges, strict=True):
        assert original.displacement is not None
        assert perturbed.displacement is not None
        assert (perturbed.source, perturbed.target) == (
            original.source,
            original.target,
        )
        assert np.allclose(
            perturbed.displacement,
            np.asarray(original.displacement)
            + offsets[original.target]
            - offsets[original.source],
        )


def test_graph_and_geometry_annotations_are_not_reclassified() -> None:
    geometry = _annotated_triangle()
    realization = apply_uniform_coordinate_perturbation(
        geometry,
        width=0.1,
        seed=19,
    )

    assert isinstance(realization.state, Geometry)
    result = realization.state
    assert result.n_sites == geometry.n_sites
    assert result.boundary_sites == geometry.boundary_sites
    assert result.boundary_components == geometry.boundary_components
    assert result.site_types == geometry.site_types
    assert result.dimension_records == geometry.dimension_records
    assert result.metadata == geometry.metadata
    assert result.faces == geometry.faces
    assert result.faces[0] is geometry.faces[0]


def test_rooted_tree_structure_survives_coordinate_only_perturbation() -> None:
    geometry = replace(
        tree((0, 0, 1, 1)),
        coordinates=np.arange(5, dtype=float).reshape(5, 1),
    )

    realization = apply_uniform_coordinate_perturbation(
        geometry,
        width=0.1,
        seed=3,
    )

    assert isinstance(realization.state, Geometry)
    assert realization.state.rooted_tree is geometry.rooted_tree
    assert realization.state.edges == geometry.edges
    assert validate_geometry(realization.state).errors == ()


def test_same_seed_reproduces_geometry_and_provenance() -> None:
    geometry = _annotated_triangle()

    first = apply_uniform_coordinate_perturbation(
        geometry,
        width=0.3,
        seed=1729,
    )
    second = apply_uniform_coordinate_perturbation(
        geometry,
        width=0.3,
        seed=1729,
    )

    assert isinstance(first.state, Geometry)
    assert isinstance(second.state, Geometry)
    assert exact_geometry_id(first.state) == exact_geometry_id(second.state)
    assert first.provenance == second.provenance


def test_zero_width_returns_exact_source_snapshot() -> None:
    geometry = _annotated_triangle()

    realization = apply_uniform_coordinate_perturbation(
        geometry,
        width=0.0,
        seed=11,
    )

    assert realization.state is geometry
    assert realization.provenance.source == realization.provenance.result


def test_provenance_records_distribution_and_coordinate_relationship() -> None:
    geometry = _annotated_triangle()
    realization = apply_uniform_coordinate_perturbation(
        geometry,
        width=0.25,
        seed=17,
    )

    assert realization.provenance.disorder_key == (
        UNIFORM_COORDINATE_PERTURBATION_KEY
    )
    assert realization.provenance.disorder_version == (
        UNIFORM_COORDINATE_PERTURBATION_VERSION
    )
    assert realization.provenance.seed == 17
    assert realization.provenance.parameters == {
        "coordinate_iteration_order": "site_major_then_embedding_axis",
        "coordinate_update_rule": "source_coordinate_plus_sampled_offset",
        "distribution": "uniform",
        "embedding_dimension": 2,
        "explicit_edge_displacement_policy": (
            "source_displacement_plus_target_offset_minus_source_offset"
        ),
        "geometry_id": exact_geometry_id(geometry),
        "implicit_edge_displacement_policy": (
            "remain_implicit_and_follow_perturbed_coordinates"
        ),
        "input_edge_count": 3,
        "n_sites": 3,
        "sampling_rule": "independent_offset_per_coordinate_component",
        "support": "[-width/2, width/2]",
        "topology_policy": "preserve_graph_and_geometry_annotations",
        "width": 0.25,
        "zero_width_policy": "return_exact_source_snapshot",
    }
    assert realization.provenance.source.identifier == exact_geometry_id(geometry)
    assert isinstance(realization.state, Geometry)
    assert realization.provenance.result.identifier == exact_geometry_id(
        realization.state
    )


def test_coordinate_free_geometry_is_rejected_without_inventing_an_embedding() -> None:
    with pytest.raises(ValueError, match="requires explicit geometry coordinates"):
        apply_uniform_coordinate_perturbation(
            Geometry(n_sites=2, edges=(GeometryEdge(0, 1),)),
            width=0.2,
            seed=1,
        )


@pytest.mark.parametrize("width", [-0.1, np.inf, np.nan])
def test_invalid_width_is_rejected(width: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        apply_uniform_coordinate_perturbation(
            _annotated_triangle(),
            width=width,
            seed=1,
        )


@pytest.mark.parametrize("width", [True, "wide"])
def test_nonreal_width_is_rejected(width: object) -> None:
    with pytest.raises(TypeError, match="must be a real number"):
        apply_uniform_coordinate_perturbation(
            _annotated_triangle(),
            width=width,  # type: ignore[arg-type]
            seed=1,
        )
