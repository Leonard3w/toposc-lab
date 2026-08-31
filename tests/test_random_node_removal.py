from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.evaluation import exact_geometry_id
from toposc_lab.geometry import (
    Geometry,
    GeometryBoundaryComponent,
    GeometryDimension,
    GeometryEdge,
    GeometryFace,
    tree,
    validate_geometry,
)
from toposc_lab.robustness import (
    RANDOM_NODE_REMOVAL_KEY,
    RANDOM_NODE_REMOVAL_VERSION,
    apply_random_node_removal,
)


def _annotated_geometry() -> Geometry:
    return Geometry(
        n_sites=6,
        edges=(
            GeometryEdge(
                5,
                0,
                edge_type="oriented_boundary",
                boundary_crossing=True,
                displacement=(-2.0, 0.0),
                metadata={"phase_reference": "5_to_0"},
            ),
            GeometryEdge(0, 1),
            GeometryEdge(1, 2),
            GeometryEdge(2, 3),
            GeometryEdge(3, 0),
            GeometryEdge(0, 2),
            GeometryEdge(2, 5),
            GeometryEdge(3, 5),
            GeometryEdge(3, 4),
        ),
        coordinates=np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [-1.0, 1.0],
                [-2.0, 0.0],
            ]
        ),
        boundary_sites=frozenset(range(6)),
        boundary_components=(
            GeometryBoundaryComponent("outer", 0, frozenset(range(6))),
            GeometryBoundaryComponent("hole", 0, frozenset({1, 4})),
        ),
        site_types=("a", "b", "c", "d", "e", "f"),
        dimension_records=(
            GeometryDimension(
                kind="topological",
                value=2.0,
                scope="finite_geometry",
                method="fixture declaration",
                exact=True,
            ),
        ),
        metadata={"name": "annotated_fixture"},
        faces=(
            GeometryFace((0, 1, 2, 3), face_type="removed_face"),
            GeometryFace(
                (0, 2, 3, 5),
                face_type="surviving_face",
                metadata={"orientation": "stored"},
            ),
        ),
    )


def test_fixed_pcg64_sample_reindexes_every_geometry_field() -> None:
    geometry = _annotated_geometry()

    result = apply_random_node_removal(
        geometry,
        removal_probability=0.5,
        seed=42,
    )

    assert result.surviving_sites == (0, 2, 3, 5)
    assert result.removed_sites == (1, 4)
    assert result.old_to_new == {0: 0, 2: 1, 3: 2, 5: 3}
    assert result.state.n_sites == 4
    assert tuple((edge.source, edge.target) for edge in result.state.edges) == (
        (3, 0),
        (1, 2),
        (2, 0),
        (0, 1),
        (1, 3),
        (2, 3),
    )
    first_edge = result.state.edges[0]
    assert first_edge.edge_type == "oriented_boundary"
    assert first_edge.boundary_crossing is True
    assert first_edge.displacement == (-2.0, 0.0)
    assert first_edge.metadata == {"phase_reference": "5_to_0"}
    assert result.state.coordinates is not None
    assert geometry.coordinates is not None
    assert np.array_equal(
        result.state.coordinates,
        geometry.coordinates[[0, 2, 3, 5]],
    )
    assert result.state.boundary_sites == frozenset(range(4))
    assert result.state.boundary_components == (
        GeometryBoundaryComponent("outer", 0, frozenset(range(4))),
    )
    assert result.state.site_types == ("a", "c", "d", "f")
    assert result.state.dimension_records == geometry.dimension_records
    assert result.state.metadata == geometry.metadata
    assert result.state.faces == (
        GeometryFace(
            (0, 1, 2, 3),
            face_type="surviving_face",
            metadata={"orientation": "stored"},
        ),
    )
    assert validate_geometry(result.state).errors == ()


def test_same_seed_reproduces_mapping_geometry_and_provenance() -> None:
    geometry = _annotated_geometry()

    first = apply_random_node_removal(
        geometry,
        removal_probability=0.4,
        seed=1729,
    )
    second = apply_random_node_removal(
        geometry,
        removal_probability=0.4,
        seed=1729,
    )

    assert first.surviving_sites == second.surviving_sites
    assert first.removed_sites == second.removed_sites
    assert first.old_to_new == second.old_to_new
    assert exact_geometry_id(first.state) == exact_geometry_id(second.state)
    assert first.provenance == second.provenance


def test_zero_probability_returns_exact_source_and_identity_mapping() -> None:
    geometry = tree((0, 0, 1, 1))

    result = apply_random_node_removal(
        geometry,
        removal_probability=0.0,
        seed=9,
    )

    assert result.state is geometry
    assert result.surviving_sites == tuple(range(geometry.n_sites))
    assert result.removed_sites == ()
    assert result.old_to_new == dict(enumerate(range(geometry.n_sites)))
    assert result.state.rooted_tree is geometry.rooted_tree
    assert result.provenance.source == result.provenance.result


def test_full_probability_uses_recorded_minimum_survivor_policy() -> None:
    geometry = Geometry(
        n_sites=4,
        coordinates=np.arange(8, dtype=float).reshape(4, 2),
    )

    result = apply_random_node_removal(
        geometry,
        removal_probability=1.0,
        seed=42,
    )

    assert result.surviving_sites == (2,)
    assert result.removed_sites == (0, 1, 3)
    assert result.old_to_new == {2: 0}
    assert result.state.n_sites == 1
    assert result.state.edges == ()
    assert result.state.coordinates is not None
    assert np.array_equal(result.state.coordinates, [[4.0, 5.0]])
    assert validate_geometry(result.state).errors == ()


def test_actual_node_removal_clears_rooted_tree_without_requiring_connectivity() -> None:
    geometry = tree((0, 0, 1, 1))

    result = apply_random_node_removal(
        geometry,
        removal_probability=1.0,
        seed=5,
    )

    assert result.state.rooted_tree is None
    assert result.state.n_sites == 1
    assert validate_geometry(result.state).errors == ()


def test_edgeless_geometry_sites_are_still_sampled() -> None:
    geometry = Geometry(n_sites=4, metadata={"kind": "edgeless"})

    result = apply_random_node_removal(
        geometry,
        removal_probability=0.5,
        seed=42,
    )

    assert result.surviving_sites == (0, 2, 3)
    assert result.removed_sites == (1,)
    assert result.state.n_sites == 3
    assert result.state.edges == ()


def test_provenance_records_sampling_reindexing_and_geometry_relationship() -> None:
    geometry = _annotated_geometry()
    result = apply_random_node_removal(
        geometry,
        removal_probability=0.25,
        seed=17,
    )

    assert result.provenance.disorder_key == RANDOM_NODE_REMOVAL_KEY
    assert result.provenance.disorder_version == RANDOM_NODE_REMOVAL_VERSION
    assert result.provenance.seed == 17
    assert result.provenance.parameters == {
        "boundary_policy": "remap_survivors_and_drop_empty_components",
        "edge_policy": "retain_if_both_endpoints_survive_and_preserve_orientation",
        "face_policy": "retain_if_all_sites_survive_and_preserve_order",
        "geometry_id": exact_geometry_id(geometry),
        "input_edge_count": geometry.n_edges,
        "input_site_count": geometry.n_sites,
        "minimum_survivor_policy": (
            "retain_largest_draw_then_lowest_site_on_tie"
        ),
        "reindexing_rule": "surviving_original_order_to_contiguous_indices",
        "removal_probability": 0.25,
        "rooted_tree_policy": "clear_if_any_site_is_removed",
        "sampling_rule": "independent_bernoulli_remove_if_draw_below_probability",
        "site_iteration_order": "ascending_original_site_index",
    }
    assert result.provenance.source.identifier == exact_geometry_id(geometry)
    assert result.provenance.result.identifier == exact_geometry_id(result.state)


def test_old_to_new_mapping_is_immutable() -> None:
    result = apply_random_node_removal(
        Geometry(n_sites=3),
        removal_probability=0.5,
        seed=42,
    )

    with pytest.raises(TypeError):
        result.old_to_new[0] = 10  # type: ignore[index]


@pytest.mark.parametrize("probability", [-0.1, 1.1, np.inf, np.nan])
def test_invalid_removal_probability_is_rejected(probability: float) -> None:
    with pytest.raises(ValueError, match="finite and between zero and one"):
        apply_random_node_removal(
            Geometry(n_sites=2),
            removal_probability=probability,
            seed=1,
        )


@pytest.mark.parametrize("probability", [True, "half"])
def test_nonreal_removal_probability_is_rejected(probability: object) -> None:
    with pytest.raises(TypeError, match="must be a real number"):
        apply_random_node_removal(
            Geometry(n_sites=2),
            removal_probability=probability,  # type: ignore[arg-type]
            seed=1,
        )
