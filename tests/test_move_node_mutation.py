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
    RootedTreeStructure,
    canonical_graph_hash,
)
from toposc_lab.search import geometry_to_genome, move_node_mutation


def _annotated_triangle() -> Geometry:
    return Geometry(
        n_sites=3,
        edges=(
            GeometryEdge(0, 1, displacement=(1.0, 0.0)),
            GeometryEdge(
                2,
                1,
                edge_type="periodic",
                boundary_crossing=True,
                displacement=(1.0, -1.0),
                metadata={"weight": 2.0},
            ),
            GeometryEdge(2, 0),
        ),
        coordinates=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        boundary_sites=frozenset({0, 1, 2}),
        boundary_components=(
            GeometryBoundaryComponent("outer", 0, frozenset({0, 1, 2})),
        ),
        site_types=("A", "B", "C"),
        dimension_records=(
            GeometryDimension(
                "lattice",
                2.0,
                "finite_geometry",
                "move-node test",
                exact=True,
            ),
        ),
        metadata={"family": "annotated_triangle"},
        faces=(GeometryFace((0, 1, 2), face_type="triangle"),),
    )


def test_move_node_replaces_one_coordinate_without_mutating_source() -> None:
    source = geometry_to_genome(_annotated_triangle())
    assert source.coordinates is not None
    original_coordinates = source.coordinates.copy()

    moved = move_node_mutation(source, 1, coordinate=(1.5, -0.25))

    assert moved is not source
    assert moved.coordinates is not None
    assert np.array_equal(source.coordinates, original_coordinates)
    assert np.array_equal(
        moved.coordinates,
        np.asarray(((0.0, 0.0), (1.5, -0.25), (0.0, 1.0))),
    )
    assert not moved.coordinates.flags.writeable
    assert moved.n_sites == source.n_sites
    assert moved.boundary_sites == source.boundary_sites
    assert moved.boundary_components == source.boundary_components
    assert moved.site_types == source.site_types
    assert moved.dimension_records == source.dimension_records
    assert moved.metadata == source.metadata
    assert moved.faces == source.faces


def test_explicit_incident_displacements_follow_oriented_node_offset() -> None:
    source = geometry_to_genome(_annotated_triangle())

    moved = move_node_mutation(source, 1, coordinate=(1.5, -0.25))

    assert moved.edges[0].displacement == pytest.approx((1.5, -0.25))
    assert moved.edges[1].displacement == pytest.approx((1.5, -1.25))
    assert moved.edges[1].source == 2
    assert moved.edges[1].target == 1
    assert moved.edges[1].edge_type == "periodic"
    assert moved.edges[1].boundary_crossing
    assert moved.edges[1].metadata == source.edges[1].metadata


def test_implicit_displacement_remains_implicit_and_uses_new_coordinates() -> None:
    source = geometry_to_genome(_annotated_triangle())

    moved = move_node_mutation(source, 0, coordinate=(-0.5, 0.25))
    geometry = moved.to_geometry()

    assert moved.edges[2] is source.edges[2]
    assert moved.edges[2].displacement is None
    assert np.allclose(geometry.displacement_between(2, 0), (-0.5, -0.75))


def test_nonincident_explicit_edge_is_retained_exactly() -> None:
    source = geometry_to_genome(_annotated_triangle())

    moved = move_node_mutation(source, 0, coordinate=(-0.5, 0.25))

    assert moved.edges[1] is source.edges[1]


def test_move_node_preserves_rooted_tree_structure() -> None:
    source = geometry_to_genome(
        Geometry(
            n_sites=3,
            edges=(GeometryEdge(0, 1), GeometryEdge(0, 2)),
            coordinates=np.asarray(((0.0,), (1.0,), (2.0,))),
            rooted_tree=RootedTreeStructure(0, (None, 0, 0)),
        )
    )

    moved = move_node_mutation(source, 2, coordinate=(3.0,))

    assert moved.rooted_tree is source.rooted_tree


def test_move_node_supports_higher_dimensional_coordinates() -> None:
    source = geometry_to_genome(
        Geometry(
            n_sites=2,
            edges=(GeometryEdge(0, 1, displacement=(1.0, 0.0, 0.0, 0.0)),),
            coordinates=np.asarray(((0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))),
        )
    )

    moved = move_node_mutation(source, 1, coordinate=(1.0, 2.0, 3.0, 4.0))

    assert moved.embedding_dimension == 4
    assert moved.coordinates is not None
    assert np.array_equal(moved.coordinates[1], (1.0, 2.0, 3.0, 4.0))
    assert moved.edges[0].displacement == pytest.approx((1.0, 2.0, 3.0, 4.0))


def test_coordinate_free_genome_is_rejected_without_inventing_embedding() -> None:
    source = geometry_to_genome(Geometry(n_sites=2, edges=(GeometryEdge(0, 1),)))

    with pytest.raises(ValueError, match="requires an explicit coordinate table"):
        move_node_mutation(source, 0, coordinate=(0.0, 0.0))


@pytest.mark.parametrize(
    "coordinate",
    (
        (),
        (1.0,),
        (1.0, 2.0, 3.0),
        (np.nan, 0.0),
        (np.inf, 0.0),
        (1.0 + 2.0j, 0.0),
    ),
)
def test_move_node_rejects_invalid_coordinate(coordinate: object) -> None:
    source = geometry_to_genome(_annotated_triangle())

    with pytest.raises(ValueError, match="coordinate"):
        move_node_mutation(source, 1, coordinate=coordinate)  # type: ignore[arg-type]


@pytest.mark.parametrize("site_index", (-1, 3))
def test_move_node_rejects_site_index_outside_stored_range(site_index: int) -> None:
    source = geometry_to_genome(_annotated_triangle())

    with pytest.raises(IndexError, match="site_index"):
        move_node_mutation(source, site_index, coordinate=(0.0, 0.0))


@pytest.mark.parametrize("site_index", (True, 1.5))
def test_move_node_rejects_noninteger_site_index(site_index: object) -> None:
    source = geometry_to_genome(_annotated_triangle())

    with pytest.raises(TypeError, match="site_index must be an integer"):
        move_node_mutation(
            source,
            site_index,  # type: ignore[arg-type]
            coordinate=(0.0, 0.0),
        )


def test_nonfinite_derived_explicit_displacement_is_rejected() -> None:
    maximum = np.finfo(np.float64).max
    source = geometry_to_genome(
        Geometry(
            n_sites=2,
            edges=(GeometryEdge(0, 1, displacement=(maximum,)),),
            coordinates=np.asarray(((0.0,), (1.0,))),
        )
    )

    with pytest.raises(ValueError, match="non-finite edge displacement"):
        move_node_mutation(source, 1, coordinate=(maximum,))


def test_moving_node_changes_snapshot_but_not_graph_fingerprint() -> None:
    source = geometry_to_genome(_annotated_triangle())

    moved = move_node_mutation(source, 1, coordinate=(1.5, -0.25))

    assert exact_geometry_id(moved.to_geometry()) != exact_geometry_id(source.to_geometry())
    assert canonical_graph_hash(moved.to_geometry()) == canonical_graph_hash(
        source.to_geometry()
    )


def test_move_node_mutation_requires_genome() -> None:
    with pytest.raises(TypeError, match="GeometryGenome instance"):
        move_node_mutation(object(), 0, coordinate=(0.0,))  # type: ignore[arg-type]


def test_move_node_rejects_invalid_source_genome() -> None:
    source = replace(geometry_to_genome(_annotated_triangle()), n_sites=2)

    with pytest.raises(ValueError, match="geometry genome validation failed"):
        move_node_mutation(source, 1, coordinate=(0.0, 0.0))
