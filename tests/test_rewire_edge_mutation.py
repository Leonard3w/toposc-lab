from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from toposc_lab.evaluation import exact_geometry_id
from toposc_lab.geometry import (
    Geometry,
    GeometryEdge,
    GeometryFace,
    RootedTreeStructure,
    canonical_graph_hash,
    validate_geometry,
)
from toposc_lab.search import geometry_to_genome, rewire_edge_mutation


def test_rewire_replaces_stored_index_and_preserves_survivor_order() -> None:
    source = geometry_to_genome(
        Geometry(
            n_sites=4,
            edges=(
                GeometryEdge(0, 1, edge_type="first"),
                GeometryEdge(2, 1, edge_type="old"),
                GeometryEdge(2, 3, edge_type="last"),
            ),
        )
    )
    replacement = GeometryEdge(
        3,
        1,
        edge_type="replacement",
        boundary_crossing=True,
        displacement=(-2.0, 0.5),
        metadata={"weight": 3.0},
    )

    rewired = rewire_edge_mutation(source, 1, replacement)

    assert rewired is not source
    assert source.edges[1].source == 2
    assert source.edges[1].target == 1
    assert rewired.edges == (source.edges[0], replacement, source.edges[2])
    assert rewired.edges[0] is source.edges[0]
    assert rewired.edges[1] is replacement
    assert rewired.edges[2] is source.edges[2]
    assert rewired.n_sites == source.n_sites
    assert len(rewired.edges) == len(source.edges)


def test_abstract_rewire_does_not_invent_spatial_or_boundary_data() -> None:
    source = geometry_to_genome(
        Geometry(n_sites=3, edges=(GeometryEdge(0, 1), GeometryEdge(1, 2)))
    )

    rewired = rewire_edge_mutation(source, 1, GeometryEdge(0, 2))

    assert rewired.coordinates is None
    assert rewired.embedding_dimension is None
    assert rewired.boundary_sites == frozenset()
    assert rewired.boundary_components == ()
    assert rewired.dimension_records == ()
    assert rewired.edges[1].displacement is None


def test_embedded_rewire_preserves_coordinates_without_inferring_displacement() -> None:
    source = geometry_to_genome(
        Geometry(
            n_sites=3,
            edges=(GeometryEdge(0, 1, displacement=(1.0, 0.0)),),
            coordinates=np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 2.0))),
            boundary_sites=frozenset({0, 1}),
            site_types=("A", "B", "C"),
            metadata={"family": "embedded"},
        )
    )
    assert source.coordinates is not None

    rewired = rewire_edge_mutation(source, 0, GeometryEdge(2, 1))

    assert rewired.coordinates is not None
    assert np.array_equal(rewired.coordinates, source.coordinates)
    assert rewired.edges[0].source == 2
    assert rewired.edges[0].target == 1
    assert rewired.edges[0].displacement is None
    assert rewired.boundary_sites == source.boundary_sites
    assert rewired.site_types == source.site_types
    assert rewired.metadata == source.metadata


def test_rewire_drops_dependent_faces_and_retains_unrelated_faces() -> None:
    first = GeometryFace((0, 1, 2), face_type="first")
    second = GeometryFace((2, 3, 4), face_type="second")
    source = geometry_to_genome(
        Geometry(
            n_sites=5,
            edges=(
                GeometryEdge(0, 1),
                GeometryEdge(1, 2),
                GeometryEdge(2, 0),
                GeometryEdge(2, 3),
                GeometryEdge(3, 4),
                GeometryEdge(4, 2),
            ),
            faces=(first, second),
        )
    )

    rewired = rewire_edge_mutation(source, 0, GeometryEdge(0, 3))

    assert rewired.faces == (second,)
    assert rewired.faces[0] is second


def test_rewire_clears_rooted_tree_annotation() -> None:
    source = geometry_to_genome(
        Geometry(
            n_sites=4,
            edges=(GeometryEdge(0, 1), GeometryEdge(0, 2), GeometryEdge(2, 3)),
            rooted_tree=RootedTreeStructure(0, (None, 0, 0, 2)),
        )
    )
    assert source.rooted_tree is not None

    rewired = rewire_edge_mutation(source, 2, GeometryEdge(1, 3))

    assert rewired.rooted_tree is None


def test_rewire_may_disconnect_a_base_valid_genome() -> None:
    source = geometry_to_genome(
        Geometry(
            n_sites=5,
            edges=(
                GeometryEdge(0, 1),
                GeometryEdge(1, 2),
                GeometryEdge(2, 3),
                GeometryEdge(3, 4),
            ),
        )
    )

    rewired = rewire_edge_mutation(source, 1, GeometryEdge(2, 4))
    report = validate_geometry(rewired.to_geometry())

    assert not report.is_connected
    assert tuple(issue.code for issue in report.warnings) == ("disconnected_components",)


@pytest.mark.parametrize("edge_index", (-1, 2))
def test_rewire_rejects_index_outside_stored_sequence(edge_index: int) -> None:
    source = geometry_to_genome(
        Geometry(n_sites=3, edges=(GeometryEdge(0, 1), GeometryEdge(1, 2)))
    )

    with pytest.raises(IndexError, match="edge_index"):
        rewire_edge_mutation(source, edge_index, GeometryEdge(0, 2))


@pytest.mark.parametrize("edge_index", (True, 0.5))
def test_rewire_rejects_noninteger_index(edge_index: object) -> None:
    source = geometry_to_genome(Geometry(n_sites=3, edges=(GeometryEdge(0, 1),)))

    with pytest.raises(TypeError, match="edge_index must be an integer"):
        rewire_edge_mutation(
            source,
            edge_index,  # type: ignore[arg-type]
            GeometryEdge(0, 2),
        )


@pytest.mark.parametrize("replacement", (GeometryEdge(0, 1), GeometryEdge(1, 0)))
def test_rewire_requires_different_undirected_endpoints(replacement: GeometryEdge) -> None:
    source = geometry_to_genome(Geometry(n_sites=3, edges=(GeometryEdge(0, 1),)))

    with pytest.raises(ValueError, match="must change an undirected endpoint"):
        rewire_edge_mutation(source, 0, replacement)


def test_representation_invalid_replacements_are_rejected() -> None:
    source = geometry_to_genome(
        Geometry(n_sites=3, edges=(GeometryEdge(0, 1), GeometryEdge(1, 2)))
    )

    with pytest.raises(ValueError, match="geometry genome validation failed"):
        rewire_edge_mutation(source, 0, GeometryEdge(1, 2))
    with pytest.raises(ValueError, match="geometry genome validation failed"):
        rewire_edge_mutation(source, 0, GeometryEdge(0, 3))


def test_rewire_changes_snapshot_and_graph_fingerprint_without_using_either() -> None:
    source = geometry_to_genome(
        Geometry(
            n_sites=5,
            edges=(
                GeometryEdge(0, 1),
                GeometryEdge(1, 2),
                GeometryEdge(2, 3),
                GeometryEdge(3, 4),
            ),
        )
    )

    rewired = rewire_edge_mutation(source, 2, GeometryEdge(1, 4))

    assert exact_geometry_id(rewired.to_geometry()) != exact_geometry_id(source.to_geometry())
    assert canonical_graph_hash(rewired.to_geometry()) != canonical_graph_hash(
        source.to_geometry()
    )


def test_rewire_requires_domain_objects() -> None:
    source = geometry_to_genome(Geometry(n_sites=2, edges=(GeometryEdge(0, 1),)))

    with pytest.raises(TypeError, match="GeometryGenome instance"):
        rewire_edge_mutation(object(), 0, GeometryEdge(0, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="GeometryEdge instance"):
        rewire_edge_mutation(source, 0, object())  # type: ignore[arg-type]


def test_rewire_rejects_invalid_source_genome() -> None:
    source = replace(
        geometry_to_genome(Geometry(n_sites=3, edges=(GeometryEdge(0, 1),))),
        n_sites=1,
    )

    with pytest.raises(ValueError, match="geometry genome validation failed"):
        rewire_edge_mutation(source, 0, GeometryEdge(0, 2))
