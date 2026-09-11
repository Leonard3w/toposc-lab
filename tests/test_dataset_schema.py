from __future__ import annotations

import pytest

from toposc_lab.data.dataset_schema import (
    ArtifactReference,
    GeometryRecord,
    ModelParametersRecord,
    ObservableResultRecord,
    ReproducibilityMetadata,
    RobustnessFailureRecord,
    RobustnessResultRecord,
    SpectrumRecord,
    TopologyResultRecord,
    TopologyValidity,
    decode_scientific_value,
    encode_scientific_value,
)
from toposc_lab.geometry import (
    Geometry,
    GeometryBoundaryComponent,
    GeometryEdge,
    chain,
    geometry_to_bytes,
)


def test_schema_records_are_typed_and_geometry_round_trips() -> None:
    geometry = chain(4)
    stored = GeometryRecord.from_geometry(geometry, family_label="chain")

    assert geometry_to_bytes(stored.to_geometry()) == geometry_to_bytes(geometry)
    assert stored.family_label == "chain"
    assert stored.exact_id.startswith("geometry-archive-v1-sha256:")


def test_model_parameters_encode_complex_values_explicitly() -> None:
    model = ModelParametersRecord(
        model_name="kitaev_chain",
        model_version="1",
        parameters={"pairing": 1.5 - 0.25j, "nested": [1, 2.0]},
    )

    encoded = encode_scientific_value(model.parameters)
    assert encoded["pairing"] == {"__complex__": [1.5, -0.25]}
    decoded = decode_scientific_value(encoded)
    assert decoded == model.parameters


def test_spectrum_has_no_implicit_state_payload_and_checks_completeness() -> None:
    spectrum = SpectrumRecord(
        eigenvalues=(-1.0, 1.0),
        energy_unit="t",
        selection="closest_to_zero",
        basis_size=8,
    )

    assert spectrum.state_reference is None
    assert not spectrum.is_complete
    with pytest.raises(ValueError, match="complete spectrum"):
        SpectrumRecord(
            eigenvalues=(-1.0, 1.0),
            energy_unit="t",
            selection="all",
            basis_size=8,
            is_complete=True,
        )


def test_optional_artifact_reference_is_integrity_checked() -> None:
    reference = ArtifactReference(
        uri="states/sample.npz",
        media_type="application/x-npz",
        sha256="a" * 64,
        shape=(8, 2),
        dtype="complex128",
        compression="deflate",
    )
    assert reference.shape == (8, 2)


def test_observable_topology_robustness_and_provenance_contracts() -> None:
    observable = ObservableResultRecord(
        kind="spectral_gap",
        version="1",
        values={"gap": 0.2},
        units={"gap": "t"},
        conventions={"definition": "smallest_positive_energy"},
    )
    topology = TopologyResultRecord(
        method="pfaffian_1d",
        version="1",
        validity=TopologyValidity.VALID,
        invariant_value=-1,
        is_topological=True,
        parameters={"momenta": [0.0, 3.141592653589793]},
        tolerances={"phs": 1e-10},
    )
    robustness = RobustnessResultRecord(
        protocol="onsite_uniform",
        version="1",
        seeds=(2, 3),
        parameters={"width": 0.1},
        statistics={"success_fraction": 0.5, "sample_count": 2},
        uncertainty={"standard_error": 0.35},
        failures=(RobustnessFailureRecord(3, "evaluation", "LinAlgError", "failed"),),
    )
    provenance = ReproducibilityMetadata(
        seed=11,
        git_commit="dadd0da",
        git_dirty=True,
        package_version="0.1.0",
        solver_name="exact_diagonalization",
        solver_version="1",
        solver_settings={"driver": "eigh"},
        tolerances={"zero_energy": 1e-9},
        timestamp_utc="2026-09-11T12:00:00Z",
        runtime={"python": "3.11"},
    )

    assert observable.units["gap"] == "t"
    assert topology.is_topological is True
    assert robustness.failures[0].seed == 3
    assert provenance.schema_version == 1


def test_invalid_topology_cannot_make_a_scientific_claim() -> None:
    with pytest.raises(ValueError, match="cannot carry a topology claim"):
        TopologyResultRecord(
            method="bott_2d",
            version="1",
            validity=TopologyValidity.NOT_APPLICABLE,
            invariant_value=None,
            is_topological=False,
            parameters={},
            tolerances={},
            reason="geometry has no coordinates",
        )


def test_geometry_record_rejects_archive_identity_tampering() -> None:
    record = GeometryRecord.from_geometry(chain(3))
    with pytest.raises(ValueError, match="exact_id"):
        GeometryRecord(
            archive=record.archive,
            exact_id="geometry-archive-v1-sha256:" + "0" * 64,
            family_fingerprint=record.family_fingerprint,
        )


def test_family_fingerprint_is_relabeling_invariant() -> None:
    original = chain(4)
    relabeled = type(original)(
        n_sites=4,
        edges=(GeometryEdge(3, 1), GeometryEdge(1, 0), GeometryEdge(0, 2)),
        coordinates=original.coordinates[[3, 1, 0, 2]],
        embedding_dimension=1,
        boundary_sites=frozenset({2, 3}),
    )

    assert (
        GeometryRecord.from_geometry(original).family_fingerprint
        == GeometryRecord.from_geometry(relabeled).family_fingerprint
    )
    assert GeometryRecord.from_geometry(original).exact_id != GeometryRecord.from_geometry(relabeled).exact_id


def test_geometry_record_preserves_labels_boundaries_and_metadata_without_loss() -> None:
    geometry = Geometry(
        n_sites=3,
        edges=(
            GeometryEdge(0, 1, edge_type="strong", metadata={"weight": 2}),
            GeometryEdge(1, 2, edge_type="weak", boundary_crossing=True),
        ),
        coordinates=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        embedding_dimension=2,
        boundary_sites=frozenset({0, 2}),
        boundary_components=(
            GeometryBoundaryComponent("outer", 0, frozenset({0, 2})),
        ),
        site_types=("left", "bulk", "right"),
        metadata={"generator": "fixture", "labels": ["a", "b"]},
    )
    stored = GeometryRecord.from_geometry(geometry, family_label="fixture_family")

    assert geometry_to_bytes(stored.to_geometry()) == geometry_to_bytes(geometry)
