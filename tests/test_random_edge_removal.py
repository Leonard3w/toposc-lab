from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.evaluation import exact_geometry_id
from toposc_lab.geometry import (
    Geometry,
    GeometryEdge,
    GeometryFace,
    chain,
    tree,
    validate_geometry,
)
from toposc_lab.robustness import (
    RANDOM_EDGE_REMOVAL_KEY,
    RANDOM_EDGE_REMOVAL_VERSION,
    apply_random_edge_removal,
)


def _geometry_with_faces() -> Geometry:
    edges = (
        GeometryEdge(3, 0, edge_type="oriented_boundary"),
        GeometryEdge(0, 1),
        GeometryEdge(1, 2),
        GeometryEdge(2, 3),
        GeometryEdge(0, 2),
        GeometryEdge(1, 3),
    )
    return Geometry(
        n_sites=4,
        edges=edges,
        coordinates=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
        ),
        boundary_sites=frozenset({0, 1, 2, 3}),
        site_types=("a", "b", "c", "d"),
        faces=(
            GeometryFace((0, 1, 2, 3), face_type="square"),
            GeometryFace((1, 2, 3), face_type="retained_triangle"),
        ),
        metadata={"name": "face_fixture"},
    )


def test_fixed_pcg64_sample_preserves_surviving_edge_objects_and_orientation() -> None:
    geometry = _geometry_with_faces()

    realization = apply_random_edge_removal(
        geometry,
        removal_probability=0.5,
        seed=42,
    )

    assert isinstance(realization.state, Geometry)
    assert realization.state.edges == (
        geometry.edges[0],
        geometry.edges[2],
        geometry.edges[3],
        geometry.edges[5],
    )
    assert realization.state.edges[0] is geometry.edges[0]
    assert (realization.state.edges[0].source, realization.state.edges[0].target) == (
        3,
        0,
    )
    assert realization.state.faces == (geometry.faces[1],)
    assert realization.state.faces[0] is geometry.faces[1]
    assert validate_geometry(realization.state).errors == ()


def test_same_seed_reproduces_geometry_and_provenance() -> None:
    geometry = _geometry_with_faces()

    first = apply_random_edge_removal(
        geometry,
        removal_probability=0.4,
        seed=1729,
    )
    second = apply_random_edge_removal(
        geometry,
        removal_probability=0.4,
        seed=1729,
    )

    assert isinstance(first.state, Geometry)
    assert isinstance(second.state, Geometry)
    assert first.state.edges == second.state.edges
    assert first.state.faces == second.state.faces
    assert first.provenance == second.provenance


def test_zero_probability_preserves_exact_source_snapshot() -> None:
    geometry = _geometry_with_faces()

    realization = apply_random_edge_removal(
        geometry,
        removal_probability=0.0,
        seed=9,
    )

    assert realization.state is geometry
    assert realization.provenance.source == realization.provenance.result


def test_full_removal_keeps_site_geometry_data_without_forcing_connectivity() -> None:
    geometry = _geometry_with_faces()

    realization = apply_random_edge_removal(
        geometry,
        removal_probability=1.0,
        seed=3,
    )

    assert isinstance(realization.state, Geometry)
    result = realization.state
    assert result.n_sites == geometry.n_sites
    assert result.edges == ()
    assert result.faces == ()
    assert result.coordinates is not None
    assert geometry.coordinates is not None
    assert np.array_equal(result.coordinates, geometry.coordinates)
    assert result.boundary_sites == geometry.boundary_sites
    assert result.site_types == geometry.site_types
    assert result.dimension_records == geometry.dimension_records
    assert result.metadata == geometry.metadata
    assert validate_geometry(result).errors == ()


def test_edge_removal_clears_invalidated_rooted_tree_without_relabeling_sites() -> None:
    geometry = tree((0, 0, 1, 1))
    original_site_types = geometry.site_types

    realization = apply_random_edge_removal(
        geometry,
        removal_probability=1.0,
        seed=5,
    )

    assert isinstance(realization.state, Geometry)
    assert realization.state.rooted_tree is None
    assert realization.state.site_types == original_site_types
    assert realization.state.boundary_sites == geometry.boundary_sites
    assert validate_geometry(realization.state).errors == ()


def test_edgeless_geometry_remains_an_exact_snapshot() -> None:
    geometry = Geometry(n_sites=3, metadata={"kind": "edgeless"})

    realization = apply_random_edge_removal(
        geometry,
        removal_probability=0.75,
        seed=7,
    )

    assert realization.state is geometry
    assert realization.provenance.source == realization.provenance.result


def test_provenance_records_sampling_and_geometry_relationship() -> None:
    geometry = chain(4)
    realization = apply_random_edge_removal(
        geometry,
        removal_probability=0.25,
        seed=17,
    )

    assert realization.provenance.disorder_key == RANDOM_EDGE_REMOVAL_KEY
    assert realization.provenance.disorder_version == RANDOM_EDGE_REMOVAL_VERSION
    assert realization.provenance.seed == 17
    assert realization.provenance.parameters == {
        "edge_iteration_order": "stored_geometry_edges",
        "face_policy": "drop_if_any_boundary_edge_is_removed",
        "geometry_id": exact_geometry_id(geometry),
        "input_edge_count": 3,
        "n_sites": 4,
        "removal_probability": 0.25,
        "rooted_tree_policy": "clear_if_any_edge_is_removed",
        "sampling_rule": "independent_bernoulli_remove_if_draw_below_probability",
        "surviving_edge_rule": "preserve_object_orientation_and_order",
    }
    assert realization.provenance.source.identifier == exact_geometry_id(geometry)
    assert isinstance(realization.state, Geometry)
    assert realization.provenance.result.identifier == exact_geometry_id(
        realization.state
    )


@pytest.mark.parametrize("probability", [-0.1, 1.1, np.inf, np.nan])
def test_invalid_removal_probability_is_rejected(probability: float) -> None:
    with pytest.raises(ValueError, match="finite and between zero and one"):
        apply_random_edge_removal(
            chain(2),
            removal_probability=probability,
            seed=1,
        )


@pytest.mark.parametrize("probability", [True, "half"])
def test_nonreal_removal_probability_is_rejected(probability: object) -> None:
    with pytest.raises(TypeError, match="must be a real number"):
        apply_random_edge_removal(
            chain(2),
            removal_probability=probability,  # type: ignore[arg-type]
            seed=1,
        )
