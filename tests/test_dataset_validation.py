from __future__ import annotations

from dataclasses import replace

from dataset_fixtures import representative_dataset_record

from toposc_lab.data.dataset_codec import create_dataset_record
from toposc_lab.data.dataset_schema import ModelParametersRecord, SpectrumRecord
from toposc_lab.data.dataset_storage import ExactPhysicsDataset, dataset_to_bytes
from toposc_lab.data.dataset_validation import (
    DatasetValidationSeverity,
    validate_dataset,
    validate_dataset_bytes,
    validate_dataset_record,
)


def test_representative_record_and_serialized_dataset_validate() -> None:
    dataset = ExactPhysicsDataset((representative_dataset_record(),))

    assert validate_dataset(dataset).is_valid
    assert validate_dataset_bytes(dataset_to_bytes(dataset)).is_valid


def test_validator_catches_inconsistent_basis_and_model_dimensions() -> None:
    original = representative_dataset_record()
    invalid = create_dataset_record(
        geometry=original.geometry,
        model=ModelParametersRecord(
            model_name=original.model.model_name,
            model_version=original.model.model_version,
            parameters={**original.model.parameters, "n_sites": 99, "embedding_dimension": 2},
        ),
        spectrum=SpectrumRecord(
            eigenvalues=(-1.0, 1.0),
            energy_unit="hopping",
            selection="closest_to_zero",
            basis_size=5,
        ),
        observables=original.observables,
        topology=original.topology,
        robustness=original.robustness,
        provenance=original.provenance,
    )

    codes = {issue.code for issue in validate_dataset_record(invalid).issues}
    assert codes == {
        "inconsistent_basis_dimension",
        "inconsistent_embedding_dimension",
        "inconsistent_model_site_count",
    }


def test_validator_catches_tampered_topology_claim_and_record_identity() -> None:
    record = representative_dataset_record()
    topology = replace(
        record.topology[0],
        validity=record.topology[0].validity.UNRESOLVED,
        invariant_value=None,
        is_topological=None,
        reason="gap closed",
    )
    object.__setattr__(topology, "is_topological", True)
    object.__setattr__(record, "topology", (topology,))

    report = validate_dataset_record(record)

    assert any(issue.code == "invalid_topology_claim" for issue in report.issues)
    assert any(issue.code == "record_identity_mismatch" for issue in report.issues)
    assert all(issue.severity is DatasetValidationSeverity.ERROR for issue in report.issues)


def test_repeated_geometry_is_visible_but_not_confused_with_duplicate_record() -> None:
    first = representative_dataset_record(seed=1)
    second = representative_dataset_record(seed=2)
    report = validate_dataset(ExactPhysicsDataset((first, second)))

    assert report.is_valid
    assert [issue.code for issue in report.issues] == ["repeated_exact_geometry"]
