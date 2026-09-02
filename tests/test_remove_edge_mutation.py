from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.evaluation import exact_geometry_id
from toposc_lab.geometry import (
    Geometry,
    GeometryBoundaryComponent,
    GeometryEdge,
    GeometryFace,
    canonical_graph_hash,
    geometry_to_bytes,
    tree,
    validate_geometry,
)
from toposc_lab.search import (
    geometry_from_genome,
    geometry_to_genome,
    remove_edge_mutation,
)


def test_remove_edge_uses_stored_index_and_preserves_survivor_orientation() -> None:
    edges = (
        GeometryEdge(2, 0, edge_type="first", metadata={"label": "oriented"}),
        GeometryEdge(0, 1, edge_type="removed"),
        GeometryEdge(1, 2, edge_type="last"),
    )
    source = geometry_to_genome(Geometry(n_sites=3, edges=edges, metadata={"source": "unchanged"}))
    source_bytes = geometry_to_bytes(geometry_from_genome(source))

    mutated = remove_edge_mutation(source, 1)

    assert mutated is not source
    assert mutated.edges == (edges[0], edges[2])
    assert mutated.edges[0] is edges[0]
    assert mutated.edges[1] is edges[2]
    assert (mutated.edges[0].source, mutated.edges[0].target) == (2, 0)
    assert mutated.metadata == source.metadata
    assert geometry_to_bytes(geometry_from_genome(source)) == source_bytes


def test_remove_bridge_may_disconnect_base_valid_genome() -> None:
    source = geometry_to_genome(
        Geometry(
            n_sites=3,
            edges=(GeometryEdge(0, 1), GeometryEdge(1, 2)),
        )
    )

    restored = geometry_from_genome(remove_edge_mutation(source, 1))
    report = validate_geometry(restored)

    assert report.is_valid
    assert not report.is_connected
    assert restored.edges == (GeometryEdge(0, 1),)
    assert tuple(issue.code for issue in report.warnings) == ("disconnected_components",)


def test_embedded_removal_preserves_spatial_and_boundary_data_exactly() -> None:
    coordinates = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.5, 1.0)))
    boundary = frozenset({0, 1, 2})
    geometry = Geometry(
        n_sites=3,
        edges=(
            GeometryEdge(0, 1, displacement=(1.0, 0.0)),
            GeometryEdge(1, 2, displacement=(-0.5, 1.0)),
            GeometryEdge(2, 0, displacement=(-0.5, -1.0)),
        ),
        coordinates=coordinates,
        boundary_sites=boundary,
        boundary_components=(GeometryBoundaryComponent("outer", 0, boundary),),
    )

    restored = geometry_from_genome(remove_edge_mutation(geometry_to_genome(geometry), 1))

    assert restored.coordinates is not None
    assert restored.coordinates.tobytes() == coordinates.tobytes()
    assert restored.embedding_dimension == 2
    assert restored.boundary_sites == boundary
    assert restored.boundary_components == geometry.boundary_components
    assert restored.edges == (geometry.edges[0], geometry.edges[2])


def test_remove_edge_drops_dependent_faces_and_retains_unrelated_faces() -> None:
    first = GeometryFace((0, 1, 2), face_type="first")
    second = GeometryFace((3, 4, 5), face_type="second")
    geometry = Geometry(
        n_sites=6,
        edges=(
            GeometryEdge(0, 1),
            GeometryEdge(1, 2),
            GeometryEdge(2, 0),
            GeometryEdge(3, 4),
            GeometryEdge(4, 5),
            GeometryEdge(5, 3),
        ),
        faces=(first, second),
    )

    mutated = remove_edge_mutation(geometry_to_genome(geometry), 0)

    assert mutated.faces == (second,)
    assert geometry_from_genome(mutated).faces == (second,)


def test_remove_tree_edge_clears_rooted_tree_annotation() -> None:
    geometry = tree((0, 0, 1))

    mutated = remove_edge_mutation(geometry_to_genome(geometry), 1)

    assert mutated.rooted_tree is None
    assert geometry_from_genome(mutated).n_edges == geometry.n_edges - 1


@pytest.mark.parametrize("edge_index", (-1, 3, 10))
def test_remove_edge_rejects_index_outside_stored_sequence(edge_index: int) -> None:
    genome = geometry_to_genome(
        Geometry(
            n_sites=3,
            edges=(GeometryEdge(0, 1), GeometryEdge(1, 2), GeometryEdge(2, 0)),
        )
    )

    with pytest.raises(IndexError, match="outside the stored edge sequence"):
        remove_edge_mutation(genome, edge_index)


@pytest.mark.parametrize("edge_index", (True, 1.0, "1"))
def test_remove_edge_rejects_noninteger_index(edge_index: object) -> None:
    genome = geometry_to_genome(Geometry(n_sites=2, edges=(GeometryEdge(0, 1),)))

    with pytest.raises(TypeError, match="edge_index must be an integer"):
        remove_edge_mutation(genome, edge_index)  # type: ignore[arg-type]


def test_remove_edge_from_edgeless_genome_is_rejected() -> None:
    genome = geometry_to_genome(Geometry(n_sites=1))

    with pytest.raises(IndexError, match="sequence of length 0"):
        remove_edge_mutation(genome, 0)


def test_removed_edge_changes_snapshot_and_graph_fingerprint_without_using_either() -> None:
    original = Geometry(
        n_sites=3,
        edges=(GeometryEdge(0, 1), GeometryEdge(1, 2), GeometryEdge(2, 0)),
    )
    mutated = geometry_from_genome(remove_edge_mutation(geometry_to_genome(original), 2))

    assert exact_geometry_id(mutated) != exact_geometry_id(original)
    assert canonical_graph_hash(mutated) != canonical_graph_hash(original)


def test_remove_edge_mutation_requires_genome() -> None:
    with pytest.raises(TypeError, match="GeometryGenome instance"):
        remove_edge_mutation(object(), 0)  # type: ignore[arg-type]
