from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.evaluation import exact_geometry_id
from toposc_lab.geometry import (
    Geometry,
    GeometryEdge,
    GeometryFace,
    canonical_graph_hash,
    geometry_to_bytes,
    tree,
)
from toposc_lab.search import (
    GeometryGenomeValidationError,
    add_edge_mutation,
    geometry_from_genome,
    geometry_to_genome,
)


def test_add_edge_appends_exact_orientation_without_mutating_source() -> None:
    source = geometry_to_genome(
        Geometry(
            n_sites=3,
            edges=(GeometryEdge(0, 1, edge_type="existing"),),
            metadata={"source": "unchanged"},
        )
    )
    source_bytes = geometry_to_bytes(geometry_from_genome(source))
    added = GeometryEdge(
        2,
        0,
        edge_type="added",
        boundary_crossing=True,
        metadata={"label": "caller_supplied"},
    )

    mutated = add_edge_mutation(source, added)

    assert mutated is not source
    assert mutated.edges == source.edges + (added,)
    assert mutated.edges[-1].source == 2
    assert mutated.edges[-1].target == 0
    assert mutated.edges[-1].displacement is None
    assert mutated.metadata == source.metadata
    assert geometry_to_bytes(geometry_from_genome(source)) == source_bytes


def test_abstract_mutation_does_not_invent_spatial_or_boundary_data() -> None:
    source = geometry_to_genome(Geometry(n_sites=3, edges=(GeometryEdge(0, 1),)))

    restored = geometry_from_genome(add_edge_mutation(source, GeometryEdge(1, 2)))

    assert restored.coordinates is None
    assert restored.embedding_dimension is None
    assert restored.boundary_sites == frozenset()
    assert restored.boundary_components == ()
    assert restored.dimension_records == ()


def test_embedded_mutation_preserves_coordinates_without_inferring_displacement() -> None:
    geometry = Geometry(
        n_sites=3,
        edges=(GeometryEdge(0, 1, displacement=(1.0, 0.0)),),
        coordinates=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 2.0))),
    )
    source = geometry_to_genome(geometry)

    mutated = add_edge_mutation(source, GeometryEdge(2, 0))
    restored = geometry_from_genome(mutated)

    assert restored.coordinates is not None
    assert geometry.coordinates is not None
    assert restored.coordinates.tobytes() == geometry.coordinates.tobytes()
    assert restored.embedding_dimension == 2
    assert restored.edges[-1].displacement is None
    assert np.array_equal(restored.displacement_between(2, 0), (0.0, -2.0))


def test_add_edge_retains_faces_but_clears_invalidated_rooted_tree() -> None:
    square = Geometry(
        n_sites=4,
        edges=(
            GeometryEdge(0, 1),
            GeometryEdge(1, 2),
            GeometryEdge(2, 3),
            GeometryEdge(3, 0),
        ),
        faces=(GeometryFace((0, 1, 2, 3), face_type="square"),),
    )
    square_mutation = add_edge_mutation(
        geometry_to_genome(square),
        GeometryEdge(2, 0),
    )

    assert square_mutation.faces == square.faces

    rooted = tree((0, 0, 1))
    rooted_mutation = add_edge_mutation(
        geometry_to_genome(rooted),
        GeometryEdge(2, 3),
    )

    assert rooted_mutation.rooted_tree is None
    assert geometry_from_genome(rooted_mutation).n_edges == rooted.n_edges + 1


def test_representation_invalid_added_edges_are_rejected() -> None:
    source = geometry_to_genome(Geometry(n_sites=3, edges=(GeometryEdge(1, 0),)))

    with pytest.raises(GeometryGenomeValidationError, match="duplicate edge"):
        add_edge_mutation(source, GeometryEdge(0, 1))
    with pytest.raises(GeometryGenomeValidationError, match="outside the geometry"):
        add_edge_mutation(source, GeometryEdge(2, 3))

    assert source.edges == (GeometryEdge(1, 0),)


def test_added_edge_changes_snapshot_and_graph_fingerprint_without_using_either() -> None:
    original = Geometry(
        n_sites=3,
        edges=(GeometryEdge(0, 1), GeometryEdge(1, 2)),
    )
    mutated = geometry_from_genome(
        add_edge_mutation(geometry_to_genome(original), GeometryEdge(2, 0))
    )

    assert exact_geometry_id(mutated) != exact_geometry_id(original)
    assert canonical_graph_hash(mutated) != canonical_graph_hash(original)


def test_add_edge_mutation_requires_domain_objects() -> None:
    genome = geometry_to_genome(Geometry(n_sites=2))

    with pytest.raises(TypeError, match="GeometryGenome instance"):
        add_edge_mutation(object(), GeometryEdge(0, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="GeometryEdge instance"):
        add_edge_mutation(genome, object())  # type: ignore[arg-type]
