"""Contract tests for the physics-free bounded local rewire catalog."""

from __future__ import annotations

import json

import numpy as np
import pytest

from toposc_lab.cli import build_parser, main
from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.search.phase_10_local_catalog import (
    build_local_rewire_catalog,
    local_catalog_summary,
    validate_local_catalog_candidate,
)
from toposc_lab.search.phase_10_measurement_validation import (
    ARMS,
    OFFSETS,
    build_measurement_cell,
    measurement_reference,
)


def test_catalog_has_exact_deterministic_bounded_counts() -> None:
    catalog = build_local_rewire_catalog()
    summary = local_catalog_summary(catalog, include_candidates=True)
    assert summary["physics_evaluations"] == 0
    assert summary["total_valid_candidates"] == 256
    assert summary["total_known_measurement_variants"] == 12
    assert len(summary["candidates"]) == 256
    assert len({candidate.geometry_id for candidate in catalog.candidates}) == 256
    expected_classes = {16: (104, 52), 20: (120, 60)}
    for row in summary["sizes"]:
        rotation_classes, dihedral_classes = expected_classes[row["n"]]
        assert row == {
            "n": row["n"],
            "patch_origins": 6,
            "alternative_pairing_proposals": 2688,
            "proposal_rejections": {
                "added_edge_already_exists": 588,
                "added_edge_length_not_sqrt2": 1848,
            },
            "prefiltered_origin_operations": 252,
            "unique_prefiltered_operations": 212,
            "duplicate_prefiltered_origins": 40,
            "validation_rejections": {"straight_edge_crossing": 84},
            "valid_origin_memberships": 144,
            "valid_candidates": 128,
            "known_measurement_variants": 6,
            "rotation_classes": rotation_classes,
            "dihedral_classes": dihedral_classes,
        }
    assert summary["candidate_ids_sha256"] == (
        "13c9e2c750af250f20fd3d245d48b6abc9b2936176e5187dd76ba245ff1c6be0"
    )
    json.dumps(summary, allow_nan=False)


def test_every_candidate_round_trips_and_preserves_reference_resources() -> None:
    catalog = build_local_rewire_catalog()
    for candidate in catalog.candidates:
        geometry = candidate.genome.to_geometry()
        reference = measurement_reference(candidate.n)
        assert exact_geometry_id(geometry) == candidate.geometry_id
        assert validate_local_catalog_candidate(
            reference,
            geometry,
            candidate.removed_edges,
            candidate.added_edges,
            candidate.n,
        ) == ()
        assert geometry.n_sites == reference.n_sites
        assert geometry.n_edges == reference.n_edges
        assert geometry.coordinates is not None
        assert reference.coordinates is not None
        assert np.array_equal(geometry.coordinates, reference.coordinates)
        assert tuple(len(geometry.neighbors(i)) for i in range(geometry.n_sites)) == tuple(
            len(reference.neighbors(i)) for i in range(reference.n_sites)
        )


def test_known_flags_match_all_twelve_measurement_validation_geometries() -> None:
    catalog = build_local_rewire_catalog()
    known_ids = {
        candidate.geometry_id
        for candidate in catalog.candidates
        if candidate.known_measurement_variant
    }
    expected_ids = {
        build_measurement_cell(
            n,
            "topological",
            block="intervention",
            offset=offset,
            arm=arm,
        )["geometry_id"]
        for n in (16, 20)
        for offset in OFFSETS
        for arm in ARMS
    }
    assert known_ids == expected_ids


def test_delta_validator_rejects_an_unmodified_geometry() -> None:
    candidate = build_local_rewire_catalog().candidates[0]
    reference = measurement_reference(candidate.n)
    issues = validate_local_catalog_candidate(
        reference,
        reference,
        candidate.removed_edges,
        candidate.added_edges,
        candidate.n,
    )
    assert "edge_delta" in issues


def test_delta_validator_rejects_changed_edge_semantics() -> None:
    candidate = build_local_rewire_catalog().candidates[0]
    geometry = candidate.genome.to_geometry()
    first = geometry.edges[0]
    altered = Geometry(
        n_sites=geometry.n_sites,
        edges=(
            GeometryEdge(
                first.source,
                first.target,
                edge_type="wrong_semantics",
                displacement=first.displacement,
            ),
            *geometry.edges[1:],
        ),
        coordinates=geometry.coordinates,
        embedding_dimension=geometry.embedding_dimension,
        boundary_sites=geometry.boundary_sites,
        boundary_components=geometry.boundary_components,
        site_types=geometry.site_types,
        dimension_records=geometry.dimension_records,
        rooted_tree=geometry.rooted_tree,
        metadata=geometry.metadata,
        faces=geometry.faces,
    )
    assert "edge_semantics_changed" in validate_local_catalog_candidate(
        measurement_reference(candidate.n),
        altered,
        candidate.removed_edges,
        candidate.added_edges,
        candidate.n,
    )


def test_symmetry_signatures_are_diagnostic_and_do_not_deduplicate_ids() -> None:
    catalog = build_local_rewire_catalog()
    assert all(
        candidate.rotation_signature.startswith("c4:")
        and candidate.dihedral_signature.startswith("d4:")
        for candidate in catalog.candidates
    )
    assert any(
        len({candidate.geometry_id for candidate in catalog.candidates if candidate.n == n
             and candidate.rotation_signature == signature}) > 1
        for n in (16, 20)
        for signature in {
            item.rotation_signature for item in catalog.candidates if item.n == n
        }
    )


def test_cli_exposes_only_a_physics_free_catalog(capsys: pytest.CaptureFixture[str]) -> None:
    args = build_parser().parse_args(["phase-10-local-catalog"])
    assert args.details is False
    assert main(["phase-10-local-catalog"]) == 0
    output = capsys.readouterr().out
    assert "256 gültige Kandidaten" in output
    assert "Hamiltonians/Solveraufrufe: 0" in output


def test_candidate_genome_has_no_physics_fields() -> None:
    candidate = build_local_rewire_catalog().candidates[0]
    assert isinstance(candidate.genome.to_geometry(), Geometry)
    assert not hasattr(candidate.genome, "hamiltonian")
    assert not hasattr(candidate.genome, "fitness")
