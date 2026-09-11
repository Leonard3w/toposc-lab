"""Strict canonical JSON codec for exact-physics dataset records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

from toposc_lab.data.dataset_schema import (
    DATASET_RECORD_ID_SCHEME,
    DATASET_SCHEMA_VERSION,
    ArtifactReference,
    DatasetRecord,
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
    geometry_archive_from_text,
    geometry_archive_to_text,
)


class DatasetCodecError(ValueError):
    """A dataset payload is malformed, unsupported, or internally inconsistent."""


def create_dataset_record(
    *,
    geometry: GeometryRecord,
    model: ModelParametersRecord,
    spectrum: SpectrumRecord,
    observables: tuple[ObservableResultRecord, ...],
    topology: tuple[TopologyResultRecord, ...],
    robustness: tuple[RobustnessResultRecord, ...],
    provenance: ReproducibilityMetadata,
) -> DatasetRecord:
    """Build a record whose identifier covers every scientific and provenance field."""
    provisional = DatasetRecord(
        record_id=f"{DATASET_RECORD_ID_SCHEME}:" + "0" * 64,
        geometry=geometry,
        model=model,
        spectrum=spectrum,
        observables=observables,
        topology=topology,
        robustness=robustness,
        provenance=provenance,
    )
    digest = hashlib.sha256(_canonical_json(_record_payload(provisional))).hexdigest()
    return replace(provisional, record_id=f"{DATASET_RECORD_ID_SCHEME}:{digest}")


def record_to_dict(record: DatasetRecord) -> dict[str, Any]:
    """Encode one validated record as a canonicalizable JSON object."""
    if not isinstance(record, DatasetRecord):
        raise TypeError("record must be DatasetRecord")
    payload = _record_payload(record)
    payload["record_id"] = record.record_id
    return payload


def record_from_dict(value: object) -> DatasetRecord:
    """Decode and integrity-check one record from strict JSON-compatible data."""
    try:
        payload = _mapping(value, "record")
        _exact_keys(
            payload,
            {
                "schema_version",
                "result_kind",
                "record_id",
                "geometry",
                "model",
                "spectrum",
                "observables",
                "topology",
                "robustness",
                "provenance",
            },
            "record",
        )
        if _integer(payload["schema_version"], "record.schema_version") != DATASET_SCHEMA_VERSION:
            raise DatasetCodecError("record schema version is incompatible")
        if _string(payload["result_kind"], "record.result_kind") != "exact":
            raise DatasetCodecError("only exact scientific records are supported")
        record = DatasetRecord(
            record_id=_string(payload["record_id"], "record.record_id"),
            geometry=_decode_geometry(payload["geometry"]),
            model=_decode_model(payload["model"]),
            spectrum=_decode_spectrum(payload["spectrum"]),
            observables=tuple(
                _decode_observable(item)
                for item in _list(payload["observables"], "record.observables")
            ),
            topology=tuple(
                _decode_topology(item)
                for item in _list(payload["topology"], "record.topology")
            ),
            robustness=tuple(
                _decode_robustness(item)
                for item in _list(payload["robustness"], "record.robustness")
            ),
            provenance=_decode_provenance(payload["provenance"]),
        )
        expected = create_dataset_record(
            geometry=record.geometry,
            model=record.model,
            spectrum=record.spectrum,
            observables=record.observables,
            topology=record.topology,
            robustness=record.robustness,
            provenance=record.provenance,
        )
        if record.record_id != expected.record_id:
            raise DatasetCodecError("record_id does not match record contents")
        return record
    except DatasetCodecError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise DatasetCodecError(str(error)) from error


def canonical_record_bytes(record: DatasetRecord) -> bytes:
    """Return deterministic UTF-8 JSON bytes for hashing and regression checks."""
    return _canonical_json(record_to_dict(record))


def _record_payload(record: DatasetRecord) -> dict[str, Any]:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "result_kind": "exact",
        "geometry": _encode_geometry(record.geometry),
        "model": _encode_model(record.model),
        "spectrum": _encode_spectrum(record.spectrum),
        "observables": [_encode_observable(item) for item in record.observables],
        "topology": [_encode_topology(item) for item in record.topology],
        "robustness": [_encode_robustness(item) for item in record.robustness],
        "provenance": _encode_provenance(record.provenance),
    }


def _encode_geometry(record: GeometryRecord) -> dict[str, Any]:
    return {
        "archive": geometry_archive_to_text(record.archive),
        "archive_schema_version": record.archive_schema_version,
        "exact_id": record.exact_id,
        "family_fingerprint": record.family_fingerprint,
        "family_fingerprint_scheme": record.family_fingerprint_scheme,
        "family_label": record.family_label,
    }


def _decode_geometry(value: object) -> GeometryRecord:
    data = _mapping(value, "geometry")
    keys = {
        "archive",
        "archive_schema_version",
        "exact_id",
        "family_fingerprint",
        "family_fingerprint_scheme",
        "family_label",
    }
    _exact_keys(data, keys, "geometry")
    return GeometryRecord(
        archive=geometry_archive_from_text(_string(data["archive"], "geometry.archive")),
        archive_schema_version=_integer(data["archive_schema_version"], "geometry.archive_schema_version"),
        exact_id=_string(data["exact_id"], "geometry.exact_id"),
        family_fingerprint=_string(data["family_fingerprint"], "geometry.family_fingerprint"),
        family_fingerprint_scheme=_string(
            data["family_fingerprint_scheme"], "geometry.family_fingerprint_scheme"
        ),
        family_label=_optional_string(data["family_label"], "geometry.family_label"),
    )


def _encode_model(record: ModelParametersRecord) -> dict[str, Any]:
    return {
        "model_name": record.model_name,
        "model_version": record.model_version,
        "parameters": encode_scientific_value(record.parameters),
    }


def _decode_model(value: object) -> ModelParametersRecord:
    data = _mapping(value, "model")
    _exact_keys(data, {"model_name", "model_version", "parameters"}, "model")
    return ModelParametersRecord(
        model_name=_string(data["model_name"], "model.model_name"),
        model_version=_string(data["model_version"], "model.model_version"),
        parameters=cast(
            Mapping[str, Any],
            decode_scientific_value(data["parameters"], path="model.parameters"),
        ),
    )


def _encode_artifact(reference: ArtifactReference | None) -> dict[str, Any] | None:
    if reference is None:
        return None
    return {
        "uri": reference.uri,
        "media_type": reference.media_type,
        "sha256": reference.sha256,
        "shape": None if reference.shape is None else list(reference.shape),
        "dtype": reference.dtype,
        "compression": reference.compression,
    }


def _decode_artifact(value: object) -> ArtifactReference | None:
    if value is None:
        return None
    data = _mapping(value, "state_reference")
    _exact_keys(data, {"uri", "media_type", "sha256", "shape", "dtype", "compression"}, "state_reference")
    shape_value = data["shape"]
    shape = None
    if shape_value is not None:
        shape = tuple(_integer(item, "state_reference.shape") for item in _list(shape_value, "state_reference.shape"))
    return ArtifactReference(
        uri=_string(data["uri"], "state_reference.uri"),
        media_type=_string(data["media_type"], "state_reference.media_type"),
        sha256=_string(data["sha256"], "state_reference.sha256"),
        shape=shape,
        dtype=_optional_string(data["dtype"], "state_reference.dtype"),
        compression=_optional_string(data["compression"], "state_reference.compression"),
    )


def _encode_spectrum(record: SpectrumRecord) -> dict[str, Any]:
    return {
        "eigenvalues": list(record.eigenvalues),
        "energy_unit": record.energy_unit,
        "selection": record.selection,
        "basis_size": record.basis_size,
        "is_complete": record.is_complete,
        "state_reference": _encode_artifact(record.state_reference),
    }


def _decode_spectrum(value: object) -> SpectrumRecord:
    data = _mapping(value, "spectrum")
    keys = {"eigenvalues", "energy_unit", "selection", "basis_size", "is_complete", "state_reference"}
    _exact_keys(data, keys, "spectrum")
    return SpectrumRecord(
        eigenvalues=tuple(
            _real(item, "spectrum.eigenvalues")
            for item in _list(data["eigenvalues"], "spectrum.eigenvalues")
        ),
        energy_unit=_string(data["energy_unit"], "spectrum.energy_unit"),
        selection=_string(data["selection"], "spectrum.selection"),
        basis_size=_integer(data["basis_size"], "spectrum.basis_size"),
        is_complete=_boolean(data["is_complete"], "spectrum.is_complete"),
        state_reference=_decode_artifact(data["state_reference"]),
    )


def _encode_observable(record: ObservableResultRecord) -> dict[str, Any]:
    return {
        "kind": record.kind,
        "version": record.version,
        "values": encode_scientific_value(record.values),
        "units": dict(record.units),
        "conventions": encode_scientific_value(record.conventions),
    }


def _decode_observable(value: object) -> ObservableResultRecord:
    data = _mapping(value, "observable")
    _exact_keys(data, {"kind", "version", "values", "units", "conventions"}, "observable")
    units_data = _mapping(data["units"], "observable.units")
    return ObservableResultRecord(
        kind=_string(data["kind"], "observable.kind"),
        version=_string(data["version"], "observable.version"),
        values=cast(Mapping[str, Any], decode_scientific_value(data["values"], path="observable.values")),
        units={key: _string(item, f"observable.units.{key}") for key, item in units_data.items()},
        conventions=cast(
            Mapping[str, Any],
            decode_scientific_value(data["conventions"], path="observable.conventions"),
        ),
    )


def _encode_topology(record: TopologyResultRecord) -> dict[str, Any]:
    return {
        "method": record.method,
        "version": record.version,
        "validity": record.validity.value,
        "invariant_value": record.invariant_value,
        "is_topological": record.is_topological,
        "parameters": encode_scientific_value(record.parameters),
        "tolerances": dict(record.tolerances),
        "reason": record.reason,
        "warnings": list(record.warnings),
    }


def _decode_topology(value: object) -> TopologyResultRecord:
    data = _mapping(value, "topology")
    keys = {"method", "version", "validity", "invariant_value", "is_topological", "parameters", "tolerances", "reason", "warnings"}
    _exact_keys(data, keys, "topology")
    try:
        validity = TopologyValidity(_string(data["validity"], "topology.validity"))
    except ValueError as error:
        raise DatasetCodecError("topology.validity is unsupported") from error
    invariant = data["invariant_value"]
    if invariant is not None:
        invariant = _real(invariant, "topology.invariant_value")
        if isinstance(data["invariant_value"], int):
            invariant = int(invariant)
    tolerances = _mapping(data["tolerances"], "topology.tolerances")
    return TopologyResultRecord(
        method=_string(data["method"], "topology.method"),
        version=_string(data["version"], "topology.version"),
        validity=validity,
        invariant_value=cast(int | float | None, invariant),
        is_topological=_optional_boolean(data["is_topological"], "topology.is_topological"),
        parameters=cast(Mapping[str, Any], decode_scientific_value(data["parameters"], path="topology.parameters")),
        tolerances={key: _real(item, f"topology.tolerances.{key}") for key, item in tolerances.items()},
        reason=_optional_string(data["reason"], "topology.reason"),
        warnings=tuple(_string(item, "topology.warning") for item in _list(data["warnings"], "topology.warnings")),
    )


def _encode_failure(record: RobustnessFailureRecord) -> dict[str, Any]:
    return {
        "seed": record.seed,
        "stage": record.stage,
        "error_type": record.error_type,
        "message": record.message,
    }


def _decode_failure(value: object) -> RobustnessFailureRecord:
    data = _mapping(value, "robustness.failure")
    _exact_keys(data, {"seed", "stage", "error_type", "message"}, "robustness.failure")
    return RobustnessFailureRecord(
        seed=_integer(data["seed"], "robustness.failure.seed"),
        stage=_string(data["stage"], "robustness.failure.stage"),
        error_type=_string(data["error_type"], "robustness.failure.error_type"),
        message=_string(data["message"], "robustness.failure.message"),
    )


def _encode_robustness(record: RobustnessResultRecord) -> dict[str, Any]:
    return {
        "protocol": record.protocol,
        "version": record.version,
        "seeds": list(record.seeds),
        "parameters": encode_scientific_value(record.parameters),
        "statistics": dict(record.statistics),
        "uncertainty": dict(record.uncertainty),
        "failures": [_encode_failure(item) for item in record.failures],
    }


def _decode_robustness(value: object) -> RobustnessResultRecord:
    data = _mapping(value, "robustness")
    keys = {"protocol", "version", "seeds", "parameters", "statistics", "uncertainty", "failures"}
    _exact_keys(data, keys, "robustness")
    statistics = _mapping(data["statistics"], "robustness.statistics")
    uncertainty = _mapping(data["uncertainty"], "robustness.uncertainty")
    decoded_statistics: dict[str, float | int] = {}
    for key, item in statistics.items():
        number = _real(item, f"robustness.statistics.{key}")
        decoded_statistics[key] = int(number) if isinstance(item, int) else number
    return RobustnessResultRecord(
        protocol=_string(data["protocol"], "robustness.protocol"),
        version=_string(data["version"], "robustness.version"),
        seeds=tuple(_integer(item, "robustness.seed") for item in _list(data["seeds"], "robustness.seeds")),
        parameters=cast(Mapping[str, Any], decode_scientific_value(data["parameters"], path="robustness.parameters")),
        statistics=decoded_statistics,
        uncertainty={key: _real(item, f"robustness.uncertainty.{key}") for key, item in uncertainty.items()},
        failures=tuple(_decode_failure(item) for item in _list(data["failures"], "robustness.failures")),
    )


def _encode_provenance(record: ReproducibilityMetadata) -> dict[str, Any]:
    return {
        "schema_version": record.schema_version,
        "seed": record.seed,
        "git_commit": record.git_commit,
        "git_dirty": record.git_dirty,
        "package_version": record.package_version,
        "solver_name": record.solver_name,
        "solver_version": record.solver_version,
        "solver_settings": encode_scientific_value(record.solver_settings),
        "tolerances": dict(record.tolerances),
        "timestamp_utc": record.timestamp_utc,
        "runtime": encode_scientific_value(record.runtime),
    }


def _decode_provenance(value: object) -> ReproducibilityMetadata:
    data = _mapping(value, "provenance")
    keys = {"schema_version", "seed", "git_commit", "git_dirty", "package_version", "solver_name", "solver_version", "solver_settings", "tolerances", "timestamp_utc", "runtime"}
    _exact_keys(data, keys, "provenance")
    if _integer(data["schema_version"], "provenance.schema_version") != DATASET_SCHEMA_VERSION:
        raise DatasetCodecError("provenance schema version is incompatible")
    tolerances = _mapping(data["tolerances"], "provenance.tolerances")
    return ReproducibilityMetadata(
        seed=_integer(data["seed"], "provenance.seed"),
        git_commit=_string(data["git_commit"], "provenance.git_commit"),
        git_dirty=_boolean(data["git_dirty"], "provenance.git_dirty"),
        package_version=_string(data["package_version"], "provenance.package_version"),
        solver_name=_string(data["solver_name"], "provenance.solver_name"),
        solver_version=_string(data["solver_version"], "provenance.solver_version"),
        solver_settings=cast(Mapping[str, Any], decode_scientific_value(data["solver_settings"], path="provenance.solver_settings")),
        tolerances={key: _real(item, f"provenance.tolerances.{key}") for key, item in tolerances.items()},
        timestamp_utc=_string(data["timestamp_utc"], "provenance.timestamp_utc"),
        runtime=cast(Mapping[str, Any], decode_scientific_value(data["runtime"], path="provenance.runtime")),
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise DatasetCodecError(f"{path} must be a JSON object")
    return cast(dict[str, Any], value)


def _list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise DatasetCodecError(f"{path} must be a JSON array")
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], path: str) -> None:
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise DatasetCodecError(f"{path} fields mismatch; missing={missing}, extra={extra}")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise DatasetCodecError(f"{path} must be a string")
    return value


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DatasetCodecError(f"{path} must be an integer")
    return value


def _real(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetCodecError(f"{path} must be a real number")
    return float(value)


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise DatasetCodecError(f"{path} must be a boolean")
    return value


def _optional_boolean(value: object, path: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, path)
