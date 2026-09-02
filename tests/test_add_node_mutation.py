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
    add_node_mutation,
    geometry_from_genome,
    geometry_to_genome,
)


def test_add_abstract_node_is_isolated_and_does_not_mutate_source() -> None:
    geometry = Geometry(
        n_sites=2,
        edges=(GeometryEdge(1, 0, edge_type="oriented"),),
        metadata={"source": "unchanged"},
    )
    source = geometry_to_genome(geometry)
    source_bytes = geometry_to_bytes(geometry_from_genome(source))

    mutated = add_node_mutation(source)
    restored = geometry_from_genome(mutated)

    assert mutated is not source
    assert mutated.n_sites == 3
    assert mutated.edges == source.edges
    assert mutated.edges[0] is source.edges[0]
    assert restored.neighbors(2) == ()
    assert restored.coordinates is None
    assert restored.embedding_dimension is None
    assert restored.boundary_sites == frozenset()
    assert restored.site_types is None
    assert restored.metadata == geometry.metadata
    assert geometry_to_bytes(geometry_from_genome(source)) == source_bytes


def test_add_embedded_node_requires_and_appends_exact_coordinate() -> None:
    coordinates = np.asarray(
        ((0.0, -0.0, 1.0, 2.0), (1.0, 0.0, 2.0, 3.0)),
        dtype=np.float64,
    )
    source = geometry_to_genome(Geometry(n_sites=2, coordinates=coordinates, embedding_dimension=4))

    restored = geometry_from_genome(add_node_mutation(source, coordinate=(-0.0, 2.0, 3.0, 4.0)))

    assert restored.coordinates is not None
    assert restored.coordinates.shape == (3, 4)
    assert restored.coordinates[:2].tobytes() == coordinates.tobytes()
    assert tuple(value.hex() for value in restored.coordinates[2]) == (
        "-0x0.0p+0",
        "0x1.0000000000000p+1",
        "0x1.8000000000000p+1",
        "0x1.0000000000000p+2",
    )
    assert restored.embedding_dimension == 4


def test_add_node_preserves_declared_dimension_without_coordinate_table() -> None:
    source = geometry_to_genome(
        Geometry(
            n_sites=2,
            edges=(GeometryEdge(0, 1, displacement=(1.0, 0.0)),),
        )
    )

    restored = geometry_from_genome(add_node_mutation(source))

    assert restored.coordinates is None
    assert restored.embedding_dimension == 2
    assert restored.neighbors(2) == ()


def test_add_node_extends_existing_site_type_column() -> None:
    source = geometry_to_genome(Geometry(n_sites=2, site_types=("A", "B")))

    typed = add_node_mutation(source, site_type="C")
    unspecified = add_node_mutation(source)

    assert typed.site_types == ("A", "B", "C")
    assert unspecified.site_types == ("A", "B", None)


def test_add_node_does_not_partially_invent_site_type_column() -> None:
    source = geometry_to_genome(Geometry(n_sites=2))

    with pytest.raises(ValueError, match="no site-type column"):
        add_node_mutation(source, site_type="new")

    assert add_node_mutation(source).site_types is None


def test_add_node_can_join_aggregate_boundary_without_components() -> None:
    source = geometry_to_genome(Geometry(n_sites=2, boundary_sites=frozenset({0})))

    mutated = add_node_mutation(source, boundary=True)

    assert mutated.boundary_sites == frozenset({0, 2})
    assert mutated.boundary_components == ()
    assert geometry_from_genome(mutated).is_boundary_site(2)


def test_add_node_updates_explicit_overlapping_boundary_components() -> None:
    outer = GeometryBoundaryComponent("outer", 0, frozenset({0, 1}))
    hole = GeometryBoundaryComponent("hole", 0, frozenset({1, 2}))
    source = geometry_to_genome(
        Geometry(
            n_sites=3,
            boundary_sites=frozenset({0, 1, 2}),
            boundary_components=(outer, hole),
        )
    )

    mutated = add_node_mutation(
        source,
        boundary=True,
        boundary_component_keys=(("outer", 0), ("hole", 0)),
    )

    assert mutated.boundary_sites == frozenset({0, 1, 2, 3})
    assert mutated.boundary_components == (
        GeometryBoundaryComponent("outer", 0, frozenset({0, 1, 3})),
        GeometryBoundaryComponent("hole", 0, frozenset({1, 2, 3})),
    )
    geometry_from_genome(mutated)


def test_add_node_retains_faces_but_clears_rooted_tree() -> None:
    face = GeometryFace((0, 1, 2), face_type="triangle")
    faced = Geometry(
        n_sites=3,
        edges=(GeometryEdge(0, 1), GeometryEdge(1, 2), GeometryEdge(2, 0)),
        faces=(face,),
    )

    faced_mutation = add_node_mutation(geometry_to_genome(faced))

    assert faced_mutation.faces == (face,)

    rooted = tree((0, 0, 1))
    rooted_mutation = add_node_mutation(geometry_to_genome(rooted))

    assert rooted_mutation.rooted_tree is None
    assert rooted_mutation.edges == rooted.edges
    assert geometry_from_genome(rooted_mutation).n_sites == rooted.n_sites + 1


def test_added_node_may_leave_base_valid_geometry_disconnected() -> None:
    source = geometry_to_genome(Geometry(n_sites=2, edges=(GeometryEdge(0, 1),)))

    report = validate_geometry(geometry_from_genome(add_node_mutation(source)))

    assert report.is_valid
    assert not report.is_connected
    assert tuple(issue.code for issue in report.warnings) == ("disconnected_components",)


@pytest.mark.parametrize(
    "coordinate",
    (None, (1.0,), (1.0, 2.0, 3.0), (np.nan, 1.0), (1.0 + 2.0j, 0.0)),
)
def test_add_embedded_node_rejects_missing_or_invalid_coordinate(
    coordinate: object,
) -> None:
    source = geometry_to_genome(Geometry(n_sites=1, coordinates=np.asarray(((0.0, 0.0),))))

    with pytest.raises(ValueError, match="coordinate"):
        add_node_mutation(source, coordinate=coordinate)  # type: ignore[arg-type]


def test_add_abstract_node_rejects_coordinate() -> None:
    source = geometry_to_genome(Geometry(n_sites=1))

    with pytest.raises(ValueError, match="no coordinate table"):
        add_node_mutation(source, coordinate=(0.0, 0.0))


@pytest.mark.parametrize(
    ("boundary", "keys", "message"),
    (
        (False, (("outer", 0),), "require boundary=True"),
        (True, (), "requires component keys"),
        (True, (("hole", 0),), "unknown boundary component"),
        (True, (("outer", 0), ("outer", 0)), "must not contain duplicates"),
    ),
)
def test_add_node_rejects_inconsistent_component_membership(
    boundary: bool,
    keys: tuple[tuple[str, int], ...],
    message: str,
) -> None:
    outer = GeometryBoundaryComponent("outer", 0, frozenset({0}))
    source = geometry_to_genome(
        Geometry(
            n_sites=1,
            boundary_sites=frozenset({0}),
            boundary_components=(outer,),
        )
    )

    with pytest.raises(ValueError, match=message):
        add_node_mutation(
            source,
            boundary=boundary,
            boundary_component_keys=keys,  # type: ignore[arg-type]
        )


def test_add_node_rejects_component_key_without_component_model() -> None:
    source = geometry_to_genome(Geometry(n_sites=1))

    with pytest.raises(ValueError, match="unknown boundary component"):
        add_node_mutation(
            source,
            boundary=True,
            boundary_component_keys=(("outer", 0),),
        )


def test_added_node_changes_snapshot_and_graph_fingerprint_without_using_either() -> None:
    original = Geometry(n_sites=2, edges=(GeometryEdge(0, 1),))
    mutated = geometry_from_genome(add_node_mutation(geometry_to_genome(original)))

    assert exact_geometry_id(mutated) != exact_geometry_id(original)
    assert canonical_graph_hash(mutated) != canonical_graph_hash(original)


def test_add_node_mutation_requires_domain_values() -> None:
    genome = geometry_to_genome(Geometry(n_sites=1))

    with pytest.raises(TypeError, match="GeometryGenome instance"):
        add_node_mutation(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="site_type must be a string"):
        add_node_mutation(genome, site_type=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="boundary must be a boolean"):
        add_node_mutation(genome, boundary=1)  # type: ignore[arg-type]
