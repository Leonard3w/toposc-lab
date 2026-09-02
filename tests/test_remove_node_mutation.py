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
from toposc_lab.geometry.hashing import canonical_graph_hash
from toposc_lab.search import (
    geometry_from_genome,
    geometry_to_genome,
    remove_node_mutation,
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
            (
                (0.0, -0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
                (-1.0, 1.0),
                (-2.0, 0.0),
            )
        ),
        boundary_sites=frozenset(range(6)),
        boundary_components=(
            GeometryBoundaryComponent("outer", 0, frozenset(range(6))),
            GeometryBoundaryComponent("hole", 0, frozenset((1,))),
            GeometryBoundaryComponent("hole", 1, frozenset((1, 4))),
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


def test_remove_node_reindexes_every_site_bound_field() -> None:
    source_geometry = _annotated_geometry()
    source = geometry_to_genome(source_geometry)
    source_snapshot_id = exact_geometry_id(source_geometry)

    mutated = remove_node_mutation(source, 1)
    restored = geometry_from_genome(mutated)

    assert exact_geometry_id(geometry_from_genome(source)) == source_snapshot_id
    assert restored.n_sites == 5
    assert tuple((edge.source, edge.target) for edge in restored.edges) == (
        (4, 0),
        (1, 2),
        (2, 0),
        (0, 1),
        (1, 4),
        (2, 4),
        (2, 3),
    )
    first_edge = restored.edges[0]
    assert first_edge.edge_type == "oriented_boundary"
    assert first_edge.boundary_crossing is True
    assert first_edge.displacement == (-2.0, 0.0)
    assert first_edge.metadata == {"phase_reference": "5_to_0"}
    assert restored.coordinates is not None
    assert source_geometry.coordinates is not None
    assert (
        restored.coordinates.tobytes()
        == source_geometry.coordinates[[0, 2, 3, 4, 5]].tobytes()
    )
    assert restored.embedding_dimension == source_geometry.embedding_dimension
    assert restored.boundary_sites == frozenset(range(5))
    assert restored.boundary_components == (
        GeometryBoundaryComponent("outer", 0, frozenset(range(5))),
        GeometryBoundaryComponent("hole", 1, frozenset((3,))),
    )
    assert restored.site_types == ("a", "c", "d", "e", "f")
    assert restored.dimension_records == source_geometry.dimension_records
    assert restored.metadata == source_geometry.metadata
    assert restored.faces == (
        GeometryFace(
            (0, 1, 2, 4),
            face_type="surviving_face",
            metadata={"orientation": "stored"},
        ),
    )
    assert validate_geometry(restored).errors == ()


def test_remove_node_preserves_absence_for_abstract_unannotated_graph() -> None:
    source = geometry_to_genome(
        Geometry(
            n_sites=3,
            edges=(GeometryEdge(2, 0, edge_type="oriented"),),
        )
    )

    mutated = remove_node_mutation(source, 1)

    assert mutated.coordinates is None
    assert mutated.embedding_dimension is None
    assert mutated.site_types is None
    assert mutated.boundary_sites == frozenset()
    assert mutated.boundary_components == ()
    assert mutated.edges == (GeometryEdge(1, 0, edge_type="oriented"),)


def test_remove_node_drops_incident_edges_without_requiring_connectivity() -> None:
    source = geometry_to_genome(
        Geometry(
            n_sites=4,
            edges=(GeometryEdge(0, 1), GeometryEdge(1, 2), GeometryEdge(2, 3)),
        )
    )

    restored = geometry_from_genome(remove_node_mutation(source, 1))
    report = validate_geometry(restored)

    assert restored.edges == (GeometryEdge(1, 2),)
    assert report.is_valid
    assert not report.is_connected
    assert tuple(issue.code for issue in report.warnings) == ("disconnected_components",)


def test_remove_node_clears_rooted_tree() -> None:
    source = geometry_to_genome(tree((0, 0, 1, 1)))

    mutated = remove_node_mutation(source, 4)

    assert mutated.rooted_tree is None
    assert geometry_from_genome(mutated).n_sites == 4


def test_remove_node_rejects_removing_the_only_site() -> None:
    source = geometry_to_genome(Geometry(n_sites=1))

    with pytest.raises(ValueError, match="cannot remove the only site"):
        remove_node_mutation(source, 0)


@pytest.mark.parametrize("site_index", (-1, 3))
def test_remove_node_rejects_site_index_outside_stored_range(site_index: int) -> None:
    source = geometry_to_genome(Geometry(n_sites=3))

    with pytest.raises(IndexError, match="outside the stored site sequence"):
        remove_node_mutation(source, site_index)


@pytest.mark.parametrize("site_index", (True, 1.5, "1"))
def test_remove_node_rejects_noninteger_site_index(site_index: object) -> None:
    source = geometry_to_genome(Geometry(n_sites=3))

    with pytest.raises(TypeError, match="site_index must be an integer"):
        remove_node_mutation(source, site_index)  # type: ignore[arg-type]


def test_removed_node_changes_snapshot_and_graph_fingerprint_without_using_either() -> None:
    original = Geometry(
        n_sites=3,
        edges=(GeometryEdge(0, 1), GeometryEdge(1, 2)),
    )
    mutated = geometry_from_genome(
        remove_node_mutation(geometry_to_genome(original), 2)
    )

    assert exact_geometry_id(mutated) != exact_geometry_id(original)
    assert canonical_graph_hash(mutated) != canonical_graph_hash(original)


def test_remove_node_mutation_requires_genome() -> None:
    with pytest.raises(TypeError, match="GeometryGenome instance"):
        remove_node_mutation(object(), 0)  # type: ignore[arg-type]
