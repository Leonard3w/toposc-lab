from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from toposc_lab.core.results import BasisLayout, SimulationResult
from toposc_lab.evaluation import (
    CandidateFailureStage,
    CandidateIssueCategory,
    CandidateIssueSeverity,
    CandidateStageFailure,
    CandidateValidityIssue,
    CandidateValidityReport,
    GeometryEvaluation,
    GeometryEvaluationRun,
    create_reproducibility_record,
)
from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.geometry import Geometry, GeometryEdge, geometry_to_bytes
from toposc_lab.robustness import exact_model_parameter_set_id
from toposc_lab.search import (
    BATCH_EVALUATION_EXECUTION_VERSION,
    EVALUATED_CANDIDATE_ARCHIVE_FORMAT,
    EVALUATED_CANDIDATE_ARCHIVE_SCHEMA_VERSION,
    BatchEvaluationCandidate,
    BatchEvaluationFailure,
    BatchEvaluationFailureStage,
    BatchEvaluationMember,
    BatchEvaluationRequest,
    BatchEvaluationResult,
    EvaluatedCandidateStorageError,
    StoredCandidateOutcome,
    load_evaluated_candidate_batch,
    save_evaluated_candidate_batch,
)


def _oriented_geometry() -> Geometry:
    return Geometry(
        n_sites=2,
        edges=(GeometryEdge(1, 0, edge_type="oriented"),),
        metadata={"abstract": True},
    )


def _valid_run(candidate: BatchEvaluationCandidate) -> GeometryEvaluationRun:
    result = SimulationResult(
        model_name="ledger_test_model",
        eigenvalues=np.asarray((-1.0, 1.0)),
        eigenvectors=np.eye(2, dtype=complex),
        basis_layout=BasisLayout(spatial_shape=(2,)),
        parameters=candidate.model_parameters,
    )
    evaluation = GeometryEvaluation(
        gap=2.0,
        low_energy_states={0: -1.0, 1: 1.0},
        zero_mode_count=0,
    )
    reproducibility = create_reproducibility_record(
        candidate.geometry,
        seed=candidate.evaluation_seed,
        model_name="ledger_test_model",
        model_parameters=candidate.model_parameters,
        solver_name="exact_diagonalization",
        solver_settings={"check_hermitian": True},
        evaluation_settings={"reference_energy": -0.0},
        code_version="phase-9.4-test",
    )
    return GeometryEvaluationRun(
        simulation_result=result,
        evaluation=evaluation,
        validity=CandidateValidityReport(()),
        reproducibility=reproducibility,
    )


def _invalid_run(*, with_stage_failure: bool = False) -> GeometryEvaluationRun:
    if with_stage_failure:
        failure = CandidateStageFailure(
            stage=CandidateFailureStage.SOLVER,
            error_type="RuntimeError",
            message="synthetic solver failure",
        )
        code = "solver_failure"
        category = CandidateIssueCategory.EXECUTION
    else:
        failure = None
        code = "disconnected_components"
        category = CandidateIssueCategory.GEOMETRY
    issue = CandidateValidityIssue(
        code=code,
        severity=CandidateIssueSeverity.ERROR,
        category=category,
        message="synthetic invalid candidate",
        path="geometry",
    )
    return GeometryEvaluationRun(
        simulation_result=None,
        evaluation=None,
        validity=CandidateValidityReport((issue,)),
        failure=failure,
    )


def _batch() -> BatchEvaluationResult:
    valid = BatchEvaluationCandidate(
        geometry=_oriented_geometry(),
        model_parameters={
            "integer": 2,
            "float": -0.0,
            "nested": {"values": (True, None, "x")},
        },
        evaluation_seed=41,
    )
    invalid = BatchEvaluationCandidate(
        geometry=Geometry(n_sites=2),
        model_parameters={"integer": 2**100},
        evaluation_seed=42,
    )
    failed = BatchEvaluationCandidate(
        geometry=_oriented_geometry(),
        model_parameters={"label": "callback"},
        evaluation_seed=43,
    )
    request = BatchEvaluationRequest((valid, invalid, failed))
    return BatchEvaluationResult(
        request=request,
        members=(
            BatchEvaluationMember(0, valid, _valid_run(valid), None),
            BatchEvaluationMember(1, invalid, _invalid_run(), None),
            BatchEvaluationMember(
                2,
                failed,
                None,
                BatchEvaluationFailure(
                    stage=BatchEvaluationFailureStage.EVALUATION_CALLBACK,
                    error_type="ValueError",
                    message="synthetic callback failure",
                ),
            ),
        ),
    )


def test_archive_round_trip_retains_every_candidate_and_terminal_outcome(
    tmp_path: Path,
) -> None:
    original = _batch()

    destination = save_evaluated_candidate_batch(tmp_path / "batch", original)
    stored = load_evaluated_candidate_batch(destination)

    assert destination.suffix == ".zip"
    assert stored.schema_version == EVALUATED_CANDIDATE_ARCHIVE_SCHEMA_VERSION
    assert stored.batch_execution_version == BATCH_EVALUATION_EXECUTION_VERSION
    assert tuple(record.candidate_index for record in stored.candidates) == (0, 1, 2)
    assert tuple(record.outcome for record in stored.candidates) == (
        StoredCandidateOutcome.VALID_EVALUATION,
        StoredCandidateOutcome.INVALID_EVALUATION,
        StoredCandidateOutcome.CALLBACK_FAILURE,
    )

    for member, record in zip(original.members, stored.candidates, strict=True):
        assert geometry_to_bytes(record.geometry) == geometry_to_bytes(member.candidate.geometry)
        assert record.geometry_id == exact_geometry_id(member.candidate.geometry)
        assert record.model_parameters == member.candidate.model_parameters
        assert record.model_parameter_set_id == exact_model_parameter_set_id(
            member.candidate.model_parameters
        )
        assert record.evaluation_seed == member.candidate.evaluation_seed

    valid, invalid, failed = stored.candidates
    assert valid.geometry.edges[0].source == 1
    assert valid.geometry.edges[0].target == 0
    assert valid.geometry.coordinates is None
    assert valid.reproducibility is not None
    assert valid.reproducibility.evaluation_settings["reference_energy"] == -0.0
    assert valid.simulation_result_available
    assert valid.evaluation_result_available
    assert invalid.validity is not None
    assert tuple(issue.code for issue in invalid.validity.errors) == ("disconnected_components",)
    assert not invalid.simulation_result_available
    assert not invalid.evaluation_result_available
    assert failed.callback_failure is not None
    assert failed.callback_failure.message == "synthetic callback failure"
    assert failed.validity is None


def test_archive_retains_pipeline_stage_failure(tmp_path: Path) -> None:
    candidate = BatchEvaluationCandidate(
        geometry=_oriented_geometry(),
        model_parameters={"onsite": 0.0},
    )
    result = BatchEvaluationResult(
        request=BatchEvaluationRequest((candidate,)),
        members=(BatchEvaluationMember(0, candidate, _invalid_run(with_stage_failure=True), None),),
    )

    stored = load_evaluated_candidate_batch(
        save_evaluated_candidate_batch(tmp_path / "stage_failure.zip", result)
    )

    record = stored.candidates[0]
    assert record.outcome is StoredCandidateOutcome.INVALID_EVALUATION
    assert record.stage_failure == CandidateStageFailure(
        stage=CandidateFailureStage.SOLVER,
        error_type="RuntimeError",
        message="synthetic solver failure",
    )


def test_archive_is_deterministic_and_contains_no_scientific_payload_files(
    tmp_path: Path,
) -> None:
    result = _batch()
    first = save_evaluated_candidate_batch(tmp_path / "first.zip", result)
    second = save_evaluated_candidate_batch(tmp_path / "second.zip", result)

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == [
            "manifest.json",
            "geometries/00000000.npz",
            "geometries/00000001.npz",
            "geometries/00000002.npz",
        ]
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["format"] == EVALUATED_CANDIDATE_ARCHIVE_FORMAT
    assert "eigenvalues" not in json.dumps(manifest)
    assert "eigenvectors" not in json.dumps(manifest)


def test_existing_archive_is_not_silently_overwritten(tmp_path: Path) -> None:
    destination = save_evaluated_candidate_batch(tmp_path / "batch.zip", _batch())
    original_bytes = destination.read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        save_evaluated_candidate_batch(destination, _batch())

    assert destination.read_bytes() == original_bytes


def test_loader_rejects_tampered_candidate_identity(tmp_path: Path) -> None:
    source = save_evaluated_candidate_batch(tmp_path / "source.zip", _batch())
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(members["manifest.json"])
    manifest["candidates"][0]["geometry_id"] = "tampered"
    members["manifest.json"] = json.dumps(manifest).encode("utf-8")
    with zipfile.ZipFile(tampered, mode="w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)

    with pytest.raises(EvaluatedCandidateStorageError, match="inconsistent"):
        load_evaluated_candidate_batch(tampered)


def test_storage_api_rejects_wrong_types_suffixes_and_limits(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="BatchEvaluationResult"):
        save_evaluated_candidate_batch(tmp_path / "batch.zip", object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=".zip suffix"):
        save_evaluated_candidate_batch(tmp_path / "batch.npz", _batch())

    source = save_evaluated_candidate_batch(tmp_path / "limited.zip", _batch())
    with pytest.raises(EvaluatedCandidateStorageError, match="max_archive_bytes"):
        load_evaluated_candidate_batch(source, max_archive_bytes=1)
    with pytest.raises(EvaluatedCandidateStorageError, match="max_members"):
        load_evaluated_candidate_batch(source, max_members=1)
    with pytest.raises(ValueError, match="positive"):
        load_evaluated_candidate_batch(source, max_uncompressed_bytes=0)


def test_loader_rejects_corrupt_archive(tmp_path: Path) -> None:
    source = tmp_path / "corrupt.zip"
    source.write_bytes(b"not a zip archive")

    with pytest.raises(EvaluatedCandidateStorageError, match="could not load"):
        load_evaluated_candidate_batch(source)
