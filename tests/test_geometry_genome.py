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
    GeometryValidationError,
    artificial_rule_graph,
    canonical_graph_hash,
    geometry_to_bytes,
    random_graph,
    tree,
)
from toposc_lab.search import (
    GEOMETRY_GENOME_SCHEMA_VERSION,
    GeometryGenome,
    GeometryGenomeValidationError,
    geometry_from_genome,
    geometry_to_genome,
    validate_geometry_genome,
)


def _rich_embedded_geometry() -> Geometry:
    return Geometry(
        n_sites=4,
        edges=(
            GeometryEdge(
                1,
                0,
                edge_type="oriented_bottom",
                boundary_crossing=True,
                displacement=(-1.0, -0.0),
                metadata={"weights": np.asarray((1, 2), dtype=np.int16)},
            ),
            GeometryEdge(1, 2, displacement=(0.0, 1.0)),
            GeometryEdge(2, 3, displacement=(-1.0, 0.0)),
            GeometryEdge(3, 0, displacement=(0.0, -1.0)),
        ),
        coordinates=np.asarray(
            ((-0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            dtype=np.float64,
        ),
        boundary_sites=frozenset({0, 1, 2, 3}),
        boundary_components=(GeometryBoundaryComponent("outer", 0, frozenset({0, 1, 2, 3})),),
        site_types=("A", "B", None, "edge"),
        dimension_records=(
            GeometryDimension(
                "lattice",
                2.0,
                "finite_geometry",
                "genome test",
                exact=True,
            ),
        ),
        metadata={
            "negative_zero": -0.0,
            "complex": complex(0.25, -0.0),
            "array": np.asarray((-0.0, 1.25), dtype=np.float32),
        },
        faces=(GeometryFace((0, 1, 2, 3), face_type="square"),),
    )


def test_rich_geometry_round_trip_preserves_exact_snapshot_and_fingerprint() -> None:
    geometry = _rich_embedded_geometry()

    genome = geometry_to_genome(geometry)
    restored = geometry_from_genome(genome)

    assert genome.schema_version == GEOMETRY_GENOME_SCHEMA_VERSION
    assert not genome.is_abstract
    assert geometry_to_bytes(restored) == geometry_to_bytes(geometry)
    assert exact_geometry_id(restored) == exact_geometry_id(geometry)
    assert canonical_graph_hash(restored) == canonical_graph_hash(geometry)
    assert tuple((edge.source, edge.target) for edge in restored.edges) == (
        (1, 0),
        (1, 2),
        (2, 3),
        (3, 0),
    )
    assert genome.coordinates is not None
    assert not genome.coordinates.flags.writeable


def test_abstract_random_graph_remains_abstract_without_invented_physics() -> None:
    geometry = random_graph(12, 0.3, seed=3)

    restored = GeometryGenome.from_geometry(geometry).to_geometry()

    assert restored.coordinates is None
    assert restored.embedding_dimension is None
    assert restored.boundary_sites == frozenset()
    assert restored.boundary_components == ()
    assert restored.dimension_records == ()
    assert geometry_to_bytes(restored) == geometry_to_bytes(geometry)


def test_artificial_higher_dimensional_geometry_preserves_declared_embedding() -> None:
    geometry = artificial_rule_graph(
        1,
        tuple(
            tuple(sign if component == axis else 0 for component in range(4))
            for axis in range(4)
            for sign in (-1, 1)
        ),
    )

    genome = geometry_to_genome(geometry)
    restored = geometry_from_genome(genome)

    assert genome.embedding_dimension == 4
    assert restored.coordinates is not None
    assert restored.coordinates.shape == (restored.n_sites, 4)
    assert all(len(edge.displacement or ()) == 4 for edge in restored.edges)
    assert restored.dimension_records == ()
    assert geometry_to_bytes(restored) == geometry_to_bytes(geometry)


def test_rooted_tree_structure_is_part_of_lossless_genome() -> None:
    geometry = tree((0, 0, 1, 1, 3))

    restored = geometry_from_genome(geometry_to_genome(geometry))

    assert restored.rooted_tree is not None
    assert restored.rooted_tree.parents == (None, 0, 0, 1, 1, 3)
    assert geometry_to_bytes(restored) == geometry_to_bytes(geometry)


def test_genome_validation_is_separate_from_geometry_validation() -> None:
    invalid_representation = replace(
        geometry_to_genome(Geometry(n_sites=2, edges=(GeometryEdge(0, 1),))),
        n_sites=1,
    )

    report = validate_geometry_genome(invalid_representation)

    assert not report.is_valid
    assert report.issues[0].code == "invalid_geometry_representation"
    with pytest.raises(GeometryGenomeValidationError):
        geometry_from_genome(invalid_representation)

    invalid_geometry_metadata = GeometryGenome(n_sites=1, metadata={"bad": object()})
    assert validate_geometry_genome(invalid_geometry_metadata).is_valid
    with pytest.raises(GeometryValidationError, match="invalid_metadata_type"):
        geometry_from_genome(invalid_geometry_metadata)


def test_genome_conversion_requires_public_domain_types() -> None:
    with pytest.raises(TypeError, match="Geometry instance"):
        geometry_to_genome(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="GeometryGenome instance"):
        validate_geometry_genome(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="GeometryGenome instance"):
        geometry_from_genome(object())  # type: ignore[arg-type]
