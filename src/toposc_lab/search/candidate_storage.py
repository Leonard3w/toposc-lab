"""Lossless candidate-input and outcome ledger for Phase-9 batch evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from numbers import Integral
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any
import zipfile

from toposc_lab.evaluation.reproducibility import (
    GEOMETRY_ID_SCHEME,
    ReproducibilityRecord,
    exact_geometry_id,
)
from toposc_lab.evaluation.validity import (
    CandidateFailureStage,
    CandidateIssueCategory,
    CandidateIssueSeverity,
    CandidateStageFailure,
    CandidateValidityIssue,
    CandidateValidityReport,
)
from toposc_lab.geometry import Geometry, geometry_from_bytes, geometry_to_bytes
from toposc_lab.robustness.disorder import (
    MODEL_PARAMETER_SET_ID_SCHEME,
    DisorderParameterValue,
    ModelParameterSet,
    exact_model_parameter_set_id,
)
from toposc_lab.search.batch_evaluation import (
    BatchEvaluationFailure,
    BatchEvaluationFailureStage,
    BatchEvaluationMember,
    BatchEvaluationResult,
)
from toposc_lab.search.random_model_parameters import _freeze_parameter_mapping

EVALUATED_CANDIDATE_ARCHIVE_FORMAT = "toposc_lab_evaluated_candidates"
EVALUATED_CANDIDATE_ARCHIVE_SCHEMA_VERSION = 1
DEFAULT_CANDIDATE_ARCHIVE_MAX_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_CANDIDATE_ARCHIVE_MAX_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
DEFAULT_CANDIDATE_ARCHIVE_MAX_MEMBERS = 100_001

_MANIFEST_MEMBER = "manifest.json"
_SUPPORTED_COMPRESSION = frozenset((zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED))
_INTEGER_PATTERN = re.compile(r"^(?:0|-?[1-9][0-9]*)$")


class EvaluatedCandidateStorageError(ValueError):
    """An evaluated-candidate archive is malformed, unsupported, or unsafe."""


class StoredCandidateOutcome(str, Enum):
    """Terminal Phase-9.3 outcome retained for one candidate."""

    VALID_EVALUATION = "valid_evaluation"
    INVALID_EVALUATION = "invalid_evaluation"
    CALLBACK_FAILURE = "callback_failure"


@dataclass(frozen=True, slots=True)
class StoredEvaluatedCandidate:
    """Exact candidate input plus audit metadata for its terminal batch outcome.

    Numerical spectra and scientific result payloads deliberately remain outside
    this Phase-9 ledger. The availability flags say which payloads existed in the
    original in-memory run; Phase 11 owns their persistent schemas.
    """

    candidate_index: int
    geometry: Geometry
    geometry_id: str
    model_parameters: ModelParameterSet
    model_parameter_set_id: str
    evaluation_seed: int | None
    outcome: StoredCandidateOutcome
    validity: CandidateValidityReport | None
    reproducibility: ReproducibilityRecord | None
    stage_failure: CandidateStageFailure | None
    callback_failure: BatchEvaluationFailure | None
    simulation_result_available: bool
    evaluation_result_available: bool

    def __post_init__(self) -> None:
        if isinstance(self.candidate_index, bool) or not isinstance(self.candidate_index, Integral):
            raise TypeError("candidate_index must be an integer")
        index = int(self.candidate_index)
        if index < 0:
            raise ValueError("candidate_index must be non-negative")
        if not isinstance(self.geometry, Geometry):
            raise TypeError("geometry must be Geometry")
        parameters = _freeze_parameter_mapping(
            self.model_parameters,
            name="model_parameters",
        )
        if self.evaluation_seed is not None and (
            isinstance(self.evaluation_seed, bool) or not isinstance(self.evaluation_seed, Integral)
        ):
            raise TypeError("evaluation_seed must be an integer or None")
        seed = None if self.evaluation_seed is None else int(self.evaluation_seed)
        if seed is not None and seed < 0:
            raise ValueError("evaluation_seed must be non-negative")
        if not isinstance(self.outcome, StoredCandidateOutcome):
            raise TypeError("outcome must be StoredCandidateOutcome")
        if self.validity is not None and not isinstance(self.validity, CandidateValidityReport):
            raise TypeError("validity must be CandidateValidityReport or None")
        if self.reproducibility is not None and not isinstance(
            self.reproducibility, ReproducibilityRecord
        ):
            raise TypeError("reproducibility must be ReproducibilityRecord or None")
        if self.stage_failure is not None and not isinstance(
            self.stage_failure, CandidateStageFailure
        ):
            raise TypeError("stage_failure must be CandidateStageFailure or None")
        if self.callback_failure is not None and not isinstance(
            self.callback_failure, BatchEvaluationFailure
        ):
            raise TypeError("callback_failure must be BatchEvaluationFailure or None")
        if not isinstance(self.simulation_result_available, bool):
            raise TypeError("simulation_result_available must be a boolean")
        if not isinstance(self.evaluation_result_available, bool):
            raise TypeError("evaluation_result_available must be a boolean")

        expected_geometry_id = exact_geometry_id(self.geometry)
        if self.geometry_id != expected_geometry_id:
            raise ValueError("geometry_id does not match the exact geometry snapshot")
        expected_parameters_id = exact_model_parameter_set_id(parameters)
        if self.model_parameter_set_id != expected_parameters_id:
            raise ValueError("model_parameter_set_id does not match the parameter snapshot")

        is_callback_failure = self.outcome is StoredCandidateOutcome.CALLBACK_FAILURE
        if is_callback_failure:
            if self.callback_failure is None:
                raise ValueError("callback_failure outcome requires callback failure data")
            if any(
                value is not None
                for value in (self.validity, self.reproducibility, self.stage_failure)
            ):
                raise ValueError("callback_failure outcome must not contain evaluation-run data")
            if self.simulation_result_available or self.evaluation_result_available:
                raise ValueError("callback_failure outcome cannot expose evaluation payloads")
        else:
            if self.callback_failure is not None or self.validity is None:
                raise ValueError("evaluation outcomes require validity and no callback failure")
            expected_validity = self.outcome is StoredCandidateOutcome.VALID_EVALUATION
            if self.validity.is_valid is not expected_validity:
                raise ValueError("outcome does not match the validity report")
            if self.evaluation_result_available and (
                not self.simulation_result_available or self.reproducibility is None
            ):
                raise ValueError(
                    "an available evaluation result requires simulation and provenance"
                )
            if expected_validity and (
                not self.simulation_result_available
                or not self.evaluation_result_available
                or self.reproducibility is None
                or self.stage_failure is not None
            ):
                raise ValueError("a valid evaluation outcome is incomplete")
            if self.stage_failure is not None:
                failure_code = f"{self.stage_failure.stage.value}_failure"
                if failure_code not in {issue.code for issue in self.validity.errors}:
                    raise ValueError("validity report must contain the recorded stage failure")

        if self.reproducibility is not None:
            if self.reproducibility.geometry_id != self.geometry_id:
                raise ValueError("reproducibility geometry_id does not match the candidate")
            if self.reproducibility.seed != seed:
                raise ValueError("reproducibility seed does not match evaluation_seed")
            if (
                exact_model_parameter_set_id(self.reproducibility.model_parameters)
                != expected_parameters_id
            ):
                raise ValueError("reproducibility model parameters do not match the candidate")

        object.__setattr__(self, "candidate_index", index)
        object.__setattr__(self, "model_parameters", parameters)
        object.__setattr__(self, "evaluation_seed", seed)


@dataclass(frozen=True, slots=True)
class StoredEvaluatedCandidateBatch:
    """Ordered audit snapshot loaded from one Phase-9 candidate archive."""

    batch_execution_version: int
    candidates: tuple[StoredEvaluatedCandidate, ...]
    schema_version: int = field(
        default=EVALUATED_CANDIDATE_ARCHIVE_SCHEMA_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        if isinstance(self.batch_execution_version, bool) or not isinstance(
            self.batch_execution_version, Integral
        ):
            raise TypeError("batch_execution_version must be an integer")
        execution_version = int(self.batch_execution_version)
        if execution_version < 1:
            raise ValueError("batch_execution_version must be at least one")
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("candidates must contain at least one record")
        if not all(isinstance(item, StoredEvaluatedCandidate) for item in candidates):
            raise TypeError("candidates must contain StoredEvaluatedCandidate records")
        if tuple(item.candidate_index for item in candidates) != tuple(range(len(candidates))):
            raise ValueError("candidate indices must preserve exact batch order")
        object.__setattr__(self, "batch_execution_version", execution_version)
        object.__setattr__(self, "candidates", candidates)


def save_evaluated_candidate_batch(
    path: str | Path,
    result: BatchEvaluationResult,
) -> Path:
    """Atomically persist every ordered member of one completed batch.

    Existing targets are not overwritten so that an earlier search ledger cannot
    be silently replaced. The archive stores exact candidate inputs and terminal
    audit outcomes, not Phase-11 numerical-result datasets.
    """
    if not isinstance(result, BatchEvaluationResult):
        raise TypeError("result must be BatchEvaluationResult")
    destination = _normalize_archive_path(path)
    if destination.exists():
        raise FileExistsError(f"candidate archive already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    records = tuple(_record_from_member(member) for member in result.members)
    stored_batch = StoredEvaluatedCandidateBatch(
        batch_execution_version=result.execution_version,
        candidates=records,
    )
    geometry_payloads = tuple(
        geometry_to_bytes(record.geometry) for record in stored_batch.candidates
    )
    manifest = _encode_batch(stored_batch)
    manifest_bytes = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w+b") as handle:
            with zipfile.ZipFile(handle, mode="w") as archive:
                _write_member(
                    archive,
                    _MANIFEST_MEMBER,
                    manifest_bytes,
                    compression=zipfile.ZIP_DEFLATED,
                )
                for record, payload in zip(
                    stored_batch.candidates,
                    geometry_payloads,
                    strict=True,
                ):
                    _write_member(
                        archive,
                        _geometry_member(record.candidate_index),
                        payload,
                        compression=zipfile.ZIP_STORED,
                    )
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists():
            raise FileExistsError(f"candidate archive already exists: {destination}")
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def load_evaluated_candidate_batch(
    path: str | Path,
    *,
    max_archive_bytes: int = DEFAULT_CANDIDATE_ARCHIVE_MAX_BYTES,
    max_uncompressed_bytes: int = DEFAULT_CANDIDATE_ARCHIVE_MAX_UNCOMPRESSED_BYTES,
    max_members: int = DEFAULT_CANDIDATE_ARCHIVE_MAX_MEMBERS,
) -> StoredEvaluatedCandidateBatch:
    """Load and verify a Phase-9 candidate ledger without pickle support."""
    source = _normalize_archive_path(path)
    if not source.is_file():
        raise FileNotFoundError(f"candidate archive does not exist: {source}")
    archive_limit = _positive_integer(max_archive_bytes, name="max_archive_bytes")
    uncompressed_limit = _positive_integer(
        max_uncompressed_bytes,
        name="max_uncompressed_bytes",
    )
    member_limit = _positive_integer(max_members, name="max_members")
    if source.stat().st_size > archive_limit:
        raise EvaluatedCandidateStorageError(
            f"candidate archive exceeds max_archive_bytes={archive_limit}"
        )

    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            infos = archive.infolist()
            _validate_container(
                infos,
                max_uncompressed_bytes=uncompressed_limit,
                max_members=member_limit,
            )
            if _MANIFEST_MEMBER not in {info.filename for info in infos}:
                raise EvaluatedCandidateStorageError(
                    "candidate archive does not contain manifest.json"
                )
            manifest = _parse_manifest(archive.read(_MANIFEST_MEMBER))
            return _decode_batch(manifest, archive=archive, infos=infos)
    except EvaluatedCandidateStorageError:
        raise
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        raise EvaluatedCandidateStorageError(
            f"could not load candidate archive: {source}"
        ) from error


def _record_from_member(member: BatchEvaluationMember) -> StoredEvaluatedCandidate:
    candidate = member.candidate
    geometry_id = exact_geometry_id(candidate.geometry)
    parameters_id = exact_model_parameter_set_id(candidate.model_parameters)
    if member.failure is not None:
        return StoredEvaluatedCandidate(
            candidate_index=member.candidate_index,
            geometry=candidate.geometry,
            geometry_id=geometry_id,
            model_parameters=candidate.model_parameters,
            model_parameter_set_id=parameters_id,
            evaluation_seed=candidate.evaluation_seed,
            outcome=StoredCandidateOutcome.CALLBACK_FAILURE,
            validity=None,
            reproducibility=None,
            stage_failure=None,
            callback_failure=member.failure,
            simulation_result_available=False,
            evaluation_result_available=False,
        )
    evaluation = member.evaluation
    assert evaluation is not None
    return StoredEvaluatedCandidate(
        candidate_index=member.candidate_index,
        geometry=candidate.geometry,
        geometry_id=geometry_id,
        model_parameters=candidate.model_parameters,
        model_parameter_set_id=parameters_id,
        evaluation_seed=candidate.evaluation_seed,
        outcome=(
            StoredCandidateOutcome.VALID_EVALUATION
            if evaluation.is_valid
            else StoredCandidateOutcome.INVALID_EVALUATION
        ),
        validity=evaluation.validity,
        reproducibility=evaluation.reproducibility,
        stage_failure=evaluation.failure,
        callback_failure=None,
        simulation_result_available=evaluation.simulation_result is not None,
        evaluation_result_available=evaluation.evaluation is not None,
    )


def _encode_batch(batch: StoredEvaluatedCandidateBatch) -> dict[str, object]:
    return {
        "format": EVALUATED_CANDIDATE_ARCHIVE_FORMAT,
        "schema_version": batch.schema_version,
        "batch_execution_version": batch.batch_execution_version,
        "candidates": [_encode_candidate(record) for record in batch.candidates],
    }


def _encode_candidate(record: StoredEvaluatedCandidate) -> dict[str, object]:
    return {
        "candidate_index": record.candidate_index,
        "geometry_member": _geometry_member(record.candidate_index),
        "geometry_id": record.geometry_id,
        "geometry_id_scheme": GEOMETRY_ID_SCHEME,
        "model_parameters": _encode_parameter_value(record.model_parameters),
        "model_parameter_set_id": record.model_parameter_set_id,
        "model_parameter_set_id_scheme": MODEL_PARAMETER_SET_ID_SCHEME,
        "evaluation_seed": record.evaluation_seed,
        "outcome": record.outcome.value,
        "validity": _encode_validity(record.validity),
        "reproducibility": _encode_reproducibility(record.reproducibility),
        "stage_failure": _encode_stage_failure(record.stage_failure),
        "callback_failure": _encode_callback_failure(record.callback_failure),
        "simulation_result_available": record.simulation_result_available,
        "evaluation_result_available": record.evaluation_result_available,
    }


def _encode_validity(report: CandidateValidityReport | None) -> object:
    if report is None:
        return None
    return {
        "issues": [
            {
                "code": issue.code,
                "severity": issue.severity.value,
                "category": issue.category.value,
                "message": issue.message,
                "path": issue.path,
            }
            for issue in report.issues
        ]
    }


def _encode_reproducibility(record: ReproducibilityRecord | None) -> object:
    if record is None:
        return None
    return {
        "seed": record.seed,
        "model_name": record.model_name,
        "model_parameters": _encode_parameter_value(record.model_parameters),
        "geometry_id": record.geometry_id,
        "geometry_id_scheme": record.geometry_id_scheme,
        "solver_name": record.solver_name,
        "solver_settings": _encode_parameter_value(record.solver_settings),
        "evaluation_settings": _encode_parameter_value(record.evaluation_settings),
        "code_version": record.code_version,
        "code_version_source": record.code_version_source,
        "warnings": list(record.warnings),
    }


def _encode_stage_failure(failure: CandidateStageFailure | None) -> object:
    if failure is None:
        return None
    return {
        "stage": failure.stage.value,
        "error_type": failure.error_type,
        "message": failure.message,
    }


def _encode_callback_failure(failure: BatchEvaluationFailure | None) -> object:
    if failure is None:
        return None
    return {
        "stage": failure.stage.value,
        "error_type": failure.error_type,
        "message": failure.message,
    }


def _decode_batch(
    value: object,
    *,
    archive: zipfile.ZipFile,
    infos: list[zipfile.ZipInfo],
) -> StoredEvaluatedCandidateBatch:
    root = _mapping(value, path="manifest")
    _exact_keys(
        root,
        {"format", "schema_version", "batch_execution_version", "candidates"},
        path="manifest",
    )
    if root["format"] != EVALUATED_CANDIDATE_ARCHIVE_FORMAT:
        raise EvaluatedCandidateStorageError("unsupported candidate archive format")
    schema_version = _integer(root["schema_version"], path="schema_version")
    if schema_version != EVALUATED_CANDIDATE_ARCHIVE_SCHEMA_VERSION:
        raise EvaluatedCandidateStorageError(
            f"unsupported candidate archive schema version {schema_version}"
        )
    candidate_values = _list(root["candidates"], path="candidates")
    if not candidate_values:
        raise EvaluatedCandidateStorageError("candidates must not be empty")
    expected_members = {_MANIFEST_MEMBER}
    records: list[StoredEvaluatedCandidate] = []
    for expected_index, item in enumerate(candidate_values):
        record, geometry_member = _decode_candidate(
            item,
            expected_index=expected_index,
            archive=archive,
        )
        records.append(record)
        expected_members.add(geometry_member)
    actual_members = {info.filename for info in infos}
    if actual_members != expected_members:
        raise EvaluatedCandidateStorageError("candidate archive members do not match the manifest")
    return StoredEvaluatedCandidateBatch(
        batch_execution_version=_integer(
            root["batch_execution_version"],
            path="batch_execution_version",
        ),
        candidates=tuple(records),
    )


def _decode_candidate(
    value: object,
    *,
    expected_index: int,
    archive: zipfile.ZipFile,
) -> tuple[StoredEvaluatedCandidate, str]:
    path = f"candidates[{expected_index}]"
    record = _mapping(value, path=path)
    expected_keys = {
        "candidate_index",
        "geometry_member",
        "geometry_id",
        "geometry_id_scheme",
        "model_parameters",
        "model_parameter_set_id",
        "model_parameter_set_id_scheme",
        "evaluation_seed",
        "outcome",
        "validity",
        "reproducibility",
        "stage_failure",
        "callback_failure",
        "simulation_result_available",
        "evaluation_result_available",
    }
    _exact_keys(record, expected_keys, path=path)
    index = _integer(record["candidate_index"], path=f"{path}.candidate_index")
    if index != expected_index:
        raise EvaluatedCandidateStorageError("candidate indices must preserve exact batch order")
    geometry_member = _string(
        record["geometry_member"],
        path=f"{path}.geometry_member",
    )
    if geometry_member != _geometry_member(index):
        raise EvaluatedCandidateStorageError(
            f"{path}.geometry_member does not match candidate_index"
        )
    if record["geometry_id_scheme"] != GEOMETRY_ID_SCHEME:
        raise EvaluatedCandidateStorageError(f"{path}.geometry_id_scheme is unsupported")
    if record["model_parameter_set_id_scheme"] != MODEL_PARAMETER_SET_ID_SCHEME:
        raise EvaluatedCandidateStorageError(f"{path}.model_parameter_set_id_scheme is unsupported")
    try:
        geometry_payload = archive.read(geometry_member)
    except KeyError as error:
        raise EvaluatedCandidateStorageError(
            f"{path} references missing geometry member"
        ) from error
    try:
        geometry = geometry_from_bytes(geometry_payload)
    except (TypeError, ValueError) as error:
        raise EvaluatedCandidateStorageError(
            f"{path} contains an invalid geometry archive"
        ) from error
    parameters = _decode_model_parameter_mapping(
        record["model_parameters"],
        path=f"{path}.model_parameters",
    )
    try:
        outcome = StoredCandidateOutcome(_string(record["outcome"], path=f"{path}.outcome"))
    except ValueError as error:
        raise EvaluatedCandidateStorageError(f"{path}.outcome is unsupported") from error
    seed_value = record["evaluation_seed"]
    seed = None if seed_value is None else _integer(seed_value, path=f"{path}.evaluation_seed")
    try:
        return (
            StoredEvaluatedCandidate(
                candidate_index=index,
                geometry=geometry,
                geometry_id=_string(record["geometry_id"], path=f"{path}.geometry_id"),
                model_parameters=parameters,
                model_parameter_set_id=_string(
                    record["model_parameter_set_id"],
                    path=f"{path}.model_parameter_set_id",
                ),
                evaluation_seed=seed,
                outcome=outcome,
                validity=_decode_validity(record["validity"], path=f"{path}.validity"),
                reproducibility=_decode_reproducibility(
                    record["reproducibility"],
                    path=f"{path}.reproducibility",
                ),
                stage_failure=_decode_stage_failure(
                    record["stage_failure"],
                    path=f"{path}.stage_failure",
                ),
                callback_failure=_decode_callback_failure(
                    record["callback_failure"],
                    path=f"{path}.callback_failure",
                ),
                simulation_result_available=_boolean(
                    record["simulation_result_available"],
                    path=f"{path}.simulation_result_available",
                ),
                evaluation_result_available=_boolean(
                    record["evaluation_result_available"],
                    path=f"{path}.evaluation_result_available",
                ),
            ),
            geometry_member,
        )
    except (TypeError, ValueError) as error:
        raise EvaluatedCandidateStorageError(f"{path} is inconsistent") from error


def _decode_validity(value: object, *, path: str) -> CandidateValidityReport | None:
    if value is None:
        return None
    record = _mapping(value, path=path)
    _exact_keys(record, {"issues"}, path=path)
    issues: list[CandidateValidityIssue] = []
    for index, item in enumerate(_list(record["issues"], path=f"{path}.issues")):
        issue_path = f"{path}.issues[{index}]"
        issue = _mapping(item, path=issue_path)
        _exact_keys(
            issue,
            {"code", "severity", "category", "message", "path"},
            path=issue_path,
        )
        try:
            issues.append(
                CandidateValidityIssue(
                    code=_string(issue["code"], path=f"{issue_path}.code"),
                    severity=CandidateIssueSeverity(
                        _string(issue["severity"], path=f"{issue_path}.severity")
                    ),
                    category=CandidateIssueCategory(
                        _string(issue["category"], path=f"{issue_path}.category")
                    ),
                    message=_string(issue["message"], path=f"{issue_path}.message"),
                    path=_optional_string(issue["path"], path=f"{issue_path}.path"),
                )
            )
        except (TypeError, ValueError) as error:
            raise EvaluatedCandidateStorageError(f"{issue_path} is invalid") from error
    return CandidateValidityReport(tuple(issues))


def _decode_reproducibility(
    value: object,
    *,
    path: str,
) -> ReproducibilityRecord | None:
    if value is None:
        return None
    record = _mapping(value, path=path)
    expected_keys = {
        "seed",
        "model_name",
        "model_parameters",
        "geometry_id",
        "geometry_id_scheme",
        "solver_name",
        "solver_settings",
        "evaluation_settings",
        "code_version",
        "code_version_source",
        "warnings",
    }
    _exact_keys(record, expected_keys, path=path)
    seed_value = record["seed"]
    seed = None if seed_value is None else _integer(seed_value, path=f"{path}.seed")
    warnings = tuple(
        _string(item, path=f"{path}.warnings[{index}]")
        for index, item in enumerate(_list(record["warnings"], path=f"{path}.warnings"))
    )
    try:
        return ReproducibilityRecord(
            seed=seed,
            model_name=_string(record["model_name"], path=f"{path}.model_name"),
            model_parameters=_decode_parameter_mapping(
                record["model_parameters"], path=f"{path}.model_parameters"
            ),
            geometry_id=_string(record["geometry_id"], path=f"{path}.geometry_id"),
            geometry_id_scheme=_string(
                record["geometry_id_scheme"], path=f"{path}.geometry_id_scheme"
            ),
            solver_name=_string(record["solver_name"], path=f"{path}.solver_name"),
            solver_settings=_decode_parameter_mapping(
                record["solver_settings"], path=f"{path}.solver_settings"
            ),
            evaluation_settings=_decode_parameter_mapping(
                record["evaluation_settings"], path=f"{path}.evaluation_settings"
            ),
            code_version=_optional_string(record["code_version"], path=f"{path}.code_version"),
            code_version_source=_string(
                record["code_version_source"], path=f"{path}.code_version_source"
            ),  # type: ignore[arg-type]
            warnings=warnings,
        )
    except (TypeError, ValueError) as error:
        raise EvaluatedCandidateStorageError(f"{path} is invalid") from error


def _decode_stage_failure(
    value: object,
    *,
    path: str,
) -> CandidateStageFailure | None:
    if value is None:
        return None
    record = _failure_mapping(value, path=path)
    try:
        return CandidateStageFailure(
            stage=CandidateFailureStage(record["stage"]),
            error_type=record["error_type"],
            message=record["message"],
        )
    except (TypeError, ValueError) as error:
        raise EvaluatedCandidateStorageError(f"{path} is invalid") from error


def _decode_callback_failure(
    value: object,
    *,
    path: str,
) -> BatchEvaluationFailure | None:
    if value is None:
        return None
    record = _failure_mapping(value, path=path)
    try:
        return BatchEvaluationFailure(
            stage=BatchEvaluationFailureStage(record["stage"]),
            error_type=record["error_type"],
            message=record["message"],
        )
    except (TypeError, ValueError) as error:
        raise EvaluatedCandidateStorageError(f"{path} is invalid") from error


def _failure_mapping(value: object, *, path: str) -> dict[str, str]:
    record = _mapping(value, path=path)
    _exact_keys(record, {"stage", "error_type", "message"}, path=path)
    return {
        name: _string(record[name], path=f"{path}.{name}")
        for name in ("stage", "error_type", "message")
    }


def _encode_parameter_value(value: DisorderParameterValue) -> object:
    if value is None:
        return {"type": "none"}
    if isinstance(value, str):
        return {"type": "string", "value": value}
    if isinstance(value, bool):
        return {"type": "boolean", "value": value}
    if isinstance(value, Integral):
        return {"type": "integer", "value": str(int(value))}
    if isinstance(value, float):
        return {"type": "float", "value": value.hex()}
    if isinstance(value, Mapping):
        return {
            "type": "mapping",
            "items": [[key, _encode_parameter_value(value[key])] for key in sorted(value)],
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [_encode_parameter_value(item) for item in value],
        }
    raise TypeError(f"unsupported parameter type {type(value).__name__}")


def _decode_parameter_mapping(value: object, *, path: str) -> ModelParameterSet:
    decoded = _decode_parameter_value(value, path=path)
    if not isinstance(decoded, Mapping):
        raise EvaluatedCandidateStorageError(f"{path} must encode a mapping")
    return MappingProxyType(dict(decoded))


def _decode_model_parameter_mapping(value: object, *, path: str) -> ModelParameterSet:
    decoded = _decode_parameter_mapping(value, path=path)
    return _freeze_parameter_mapping(decoded, name=path)


def _decode_parameter_value(value: object, *, path: str) -> DisorderParameterValue:
    node = _mapping(value, path=path)
    node_type = _string(node.get("type"), path=f"{path}.type")
    if node_type == "none":
        _exact_keys(node, {"type"}, path=path)
        return None
    if node_type == "string":
        _exact_keys(node, {"type", "value"}, path=path)
        return _string(node["value"], path=f"{path}.value")
    if node_type == "boolean":
        _exact_keys(node, {"type", "value"}, path=path)
        return _boolean(node["value"], path=f"{path}.value")
    if node_type == "integer":
        _exact_keys(node, {"type", "value"}, path=path)
        encoded = _string(node["value"], path=f"{path}.value")
        if _INTEGER_PATTERN.fullmatch(encoded) is None:
            raise EvaluatedCandidateStorageError(f"{path}.value is invalid")
        try:
            return int(encoded)
        except ValueError as error:
            raise EvaluatedCandidateStorageError(f"{path}.value is invalid") from error
    if node_type == "float":
        _exact_keys(node, {"type", "value"}, path=path)
        encoded = _string(node["value"], path=f"{path}.value")
        try:
            result = float.fromhex(encoded)
        except ValueError as error:
            raise EvaluatedCandidateStorageError(f"{path}.value is invalid") from error
        if not math.isfinite(result):
            raise EvaluatedCandidateStorageError(f"{path}.value must be finite")
        return result
    if node_type == "tuple":
        _exact_keys(node, {"type", "items"}, path=path)
        return tuple(
            _decode_parameter_value(item, path=f"{path}.items[{index}]")
            for index, item in enumerate(_list(node["items"], path=f"{path}.items"))
        )
    if node_type == "mapping":
        _exact_keys(node, {"type", "items"}, path=path)
        result_mapping: dict[str, DisorderParameterValue] = {}
        for index, item in enumerate(_list(node["items"], path=f"{path}.items")):
            pair = _list(item, path=f"{path}.items[{index}]")
            if len(pair) != 2:
                raise EvaluatedCandidateStorageError(
                    f"{path}.items[{index}] must contain key and value"
                )
            key = _string(pair[0], path=f"{path}.items[{index}][0]")
            if key in result_mapping:
                raise EvaluatedCandidateStorageError(f"{path} has duplicate key {key!r}")
            result_mapping[key] = _decode_parameter_value(pair[1], path=f"{path}[{key!r}]")
        return MappingProxyType(result_mapping)
    raise EvaluatedCandidateStorageError(f"{path} has unknown type {node_type!r}")


def _normalize_archive_path(path: str | Path) -> Path:
    destination = Path(path)
    if destination.suffix == "":
        return destination.with_suffix(".zip")
    if destination.suffix.lower() != ".zip":
        raise ValueError("evaluated-candidate archives must use the .zip suffix")
    return destination


def _geometry_member(candidate_index: int) -> str:
    return f"geometries/{candidate_index:08d}.npz"


def _write_member(
    archive: zipfile.ZipFile,
    name: str,
    data: bytes,
    *,
    compression: int,
) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = 0o600 << 16
    archive.writestr(info, data)


def _validate_container(
    infos: list[zipfile.ZipInfo],
    *,
    max_uncompressed_bytes: int,
    max_members: int,
) -> None:
    if not infos:
        raise EvaluatedCandidateStorageError("candidate archive is empty")
    if len(infos) > max_members:
        raise EvaluatedCandidateStorageError(f"candidate archive exceeds max_members={max_members}")
    names = [info.filename for info in infos]
    if len(set(names)) != len(names):
        raise EvaluatedCandidateStorageError("candidate archive has duplicate members")
    total_size = 0
    for info in infos:
        if info.is_dir() or info.flag_bits & 0x1:
            raise EvaluatedCandidateStorageError(
                "candidate archive contains a directory or encrypted member"
            )
        if info.compress_type not in _SUPPORTED_COMPRESSION:
            raise EvaluatedCandidateStorageError("candidate archive uses unsupported compression")
        total_size += info.file_size
        if total_size > max_uncompressed_bytes:
            raise EvaluatedCandidateStorageError(
                f"candidate archive exceeds max_uncompressed_bytes={max_uncompressed_bytes}"
            )


def _parse_manifest(data: bytes) -> object:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise EvaluatedCandidateStorageError(
            "candidate manifest is not valid UTF-8 JSON"
        ) from error


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluatedCandidateStorageError(
                f"candidate manifest has duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise EvaluatedCandidateStorageError(
        f"candidate manifest contains non-finite constant {value!r}"
    )


def _mapping(value: object, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise EvaluatedCandidateStorageError(f"{path} must be a JSON object")
    return value


def _list(value: object, *, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise EvaluatedCandidateStorageError(f"{path} must be a JSON array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, path: str) -> None:
    if set(value) != expected:
        raise EvaluatedCandidateStorageError(f"{path} has invalid fields")


def _integer(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluatedCandidateStorageError(f"{path} must be an integer")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _boolean(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise EvaluatedCandidateStorageError(f"{path} must be a boolean")
    return value


def _string(value: object, *, path: str) -> str:
    if not isinstance(value, str):
        raise EvaluatedCandidateStorageError(f"{path} must be a string")
    return value


def _optional_string(value: object, *, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path=path)
