"""Versioned, typed records for exact-physics datasets.

The schema deliberately stores scientific results without recomputing or
reinterpreting them.  Large state arrays are represented by explicit artifact
references; they are never included by a default constructor.
"""

from __future__ import annotations

import base64
import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, TypeAlias

from toposc_lab.geometry import (
    CANONICAL_GRAPH_HASH_ALGORITHM,
    GEOMETRY_ARCHIVE_SCHEMA_VERSION,
    Geometry,
    canonical_graph_hash,
    geometry_from_bytes,
    geometry_to_bytes,
)

DATASET_FORMAT = "toposc_lab_exact_physics_dataset"
DATASET_SCHEMA_VERSION = 1
EXACT_GEOMETRY_ID_SCHEME = "geometry-archive-v1-sha256"
DATASET_RECORD_ID_SCHEME = "dataset-record-v1-sha256"

ScientificScalar: TypeAlias = None | bool | int | float | str | complex
ScientificValue: TypeAlias = (
    ScientificScalar
    | tuple["ScientificValue", ...]
    | Mapping[str, "ScientificValue"]
)

_IDENTIFIER = re.compile(r"^[a-zA-Z][a-zA-Z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TopologyValidity(str, Enum):
    """Applicability and numerical status of a stored topology result."""

    VALID = "valid"
    INVALID = "invalid"
    NOT_APPLICABLE = "not_applicable"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class GeometryRecord:
    """Lossless geometry snapshot plus two deliberately distinct identities.

    ``exact_id`` identifies the full serialized snapshot and is sensitive to
    labels, coordinates, metadata, and edge orientation. ``family_fingerprint``
    is relabeling-invariant and intentionally excludes coordinates and metadata;
    it is a candidate-group key, not an isomorphism proof.
    """

    archive: bytes = field(repr=False)
    exact_id: str
    family_fingerprint: str
    archive_schema_version: int = GEOMETRY_ARCHIVE_SCHEMA_VERSION
    family_fingerprint_scheme: str = CANONICAL_GRAPH_HASH_ALGORITHM
    family_label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.archive, bytes) or not self.archive:
            raise ValueError("geometry archive must be non-empty bytes")
        geometry = geometry_from_bytes(self.archive)
        expected_exact_id = exact_geometry_id_from_bytes(self.archive)
        if self.exact_id != expected_exact_id:
            raise ValueError("geometry exact_id does not match the archive")
        expected_family = canonical_graph_hash(geometry)
        if self.family_fingerprint != expected_family:
            raise ValueError("geometry family_fingerprint does not match the archive")
        if self.archive_schema_version != GEOMETRY_ARCHIVE_SCHEMA_VERSION:
            raise ValueError("unsupported geometry archive schema version")
        if self.family_fingerprint_scheme != CANONICAL_GRAPH_HASH_ALGORITHM:
            raise ValueError("unsupported geometry family fingerprint scheme")
        object.__setattr__(self, "family_label", _optional_text(self.family_label, "family_label"))

    @classmethod
    def from_geometry(
        cls,
        geometry: Geometry,
        *,
        family_label: str | None = None,
    ) -> GeometryRecord:
        """Create both identities from one immutable geometry snapshot."""
        if not isinstance(geometry, Geometry):
            raise TypeError("geometry must be Geometry")
        archive = geometry_to_bytes(geometry)
        return cls(
            archive=archive,
            exact_id=exact_geometry_id_from_bytes(archive),
            family_fingerprint=canonical_graph_hash(geometry),
            family_label=family_label,
        )

    def to_geometry(self) -> Geometry:
        """Reconstruct the exact stored geometry."""
        return geometry_from_bytes(self.archive)


@dataclass(frozen=True, slots=True)
class ModelParametersRecord:
    """Model identity and typed parameters used by the exact calculation.

    Complex parameters remain Python ``complex`` values in memory and are
    encoded by an explicit tagged representation in the dataset codec.
    """

    model_name: str
    model_version: str
    parameters: Mapping[str, ScientificValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_name", _technical_name(self.model_name, "model_name"))
        object.__setattr__(self, "model_version", _text(self.model_version, "model_version"))
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters, "parameters"))


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Integrity-checked external reference to optional large numerical data."""

    uri: str
    media_type: str
    sha256: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    compression: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "uri", _text(self.uri, "uri"))
        object.__setattr__(self, "media_type", _text(self.media_type, "media_type"))
        if not isinstance(self.sha256, str) or _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("sha256 must contain 64 lowercase hexadecimal digits")
        if self.shape is not None:
            shape = tuple(_positive_integer(value, "artifact shape") for value in self.shape)
            object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", _optional_text(self.dtype, "dtype"))
        object.__setattr__(self, "compression", _optional_text(self.compression, "compression"))


@dataclass(frozen=True, slots=True)
class SpectrumRecord:
    """Stored eigenvalues and an optional reference to selected/full states."""

    eigenvalues: tuple[float, ...]
    energy_unit: str
    selection: str
    basis_size: int
    is_complete: bool = False
    state_reference: ArtifactReference | None = None

    def __post_init__(self) -> None:
        values = tuple(_finite_real(value, "eigenvalue") for value in self.eigenvalues)
        if not values:
            raise ValueError("eigenvalues must not be empty")
        if tuple(sorted(values)) != values:
            raise ValueError("eigenvalues must be sorted in ascending order")
        basis_size = _positive_integer(self.basis_size, "basis_size")
        if len(values) > basis_size:
            raise ValueError("stored eigenvalue count cannot exceed basis_size")
        if self.is_complete and len(values) != basis_size:
            raise ValueError("a complete spectrum must contain basis_size eigenvalues")
        if not isinstance(self.is_complete, bool):
            raise TypeError("is_complete must be a boolean")
        if self.state_reference is not None and not isinstance(
            self.state_reference, ArtifactReference
        ):
            raise TypeError("state_reference must be ArtifactReference or None")
        object.__setattr__(self, "eigenvalues", values)
        object.__setattr__(self, "energy_unit", _text(self.energy_unit, "energy_unit"))
        object.__setattr__(self, "selection", _text(self.selection, "selection"))
        object.__setattr__(self, "basis_size", basis_size)


@dataclass(frozen=True, slots=True)
class ObservableResultRecord:
    """Versioned serializable output of one observable calculation."""

    kind: str
    version: str
    values: Mapping[str, ScientificValue]
    units: Mapping[str, str] = field(default_factory=dict)
    conventions: Mapping[str, ScientificValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _technical_name(self.kind, "observable kind"))
        object.__setattr__(self, "version", _text(self.version, "observable version"))
        values = _freeze_mapping(self.values, "observable values")
        units = {
            _technical_name(name, "observable unit key"): _text(unit, "observable unit")
            for name, unit in self.units.items()
        }
        unknown_units = set(units).difference(values)
        if unknown_units:
            raise ValueError(f"units reference unknown observable values: {sorted(unknown_units)}")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "units", MappingProxyType(dict(sorted(units.items()))))
        object.__setattr__(self, "conventions", _freeze_mapping(self.conventions, "conventions"))


@dataclass(frozen=True, slots=True)
class TopologyResultRecord:
    """Topology value with explicit applicability and numerical provenance."""

    method: str
    version: str
    validity: TopologyValidity
    invariant_value: int | float | None
    is_topological: bool | None
    parameters: Mapping[str, ScientificValue]
    tolerances: Mapping[str, float]
    reason: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", _technical_name(self.method, "topology method"))
        object.__setattr__(self, "version", _text(self.version, "topology version"))
        if not isinstance(self.validity, TopologyValidity):
            raise TypeError("validity must be TopologyValidity")
        value = self.invariant_value
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("invariant_value must be a real number or None")
            numeric = _finite_real(value, "invariant_value")
            value = int(value) if isinstance(value, Integral) else numeric
        if self.is_topological is not None and not isinstance(self.is_topological, bool):
            raise TypeError("is_topological must be a boolean or None")
        reason = _optional_text(self.reason, "topology reason")
        if self.validity is TopologyValidity.VALID:
            if value is None or self.is_topological is None:
                raise ValueError("valid topology requires a value and classification")
        else:
            if reason is None:
                raise ValueError("non-valid topology requires an explicit reason")
            if self.is_topological is not None:
                raise ValueError("non-valid topology cannot carry a topology claim")
        tolerances = {
            _technical_name(name, "tolerance name"): _nonnegative_real(number, "tolerance")
            for name, number in self.tolerances.items()
        }
        object.__setattr__(self, "invariant_value", value)
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters, "topology parameters"))
        object.__setattr__(self, "tolerances", MappingProxyType(dict(sorted(tolerances.items()))))
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "warnings", _messages(self.warnings, "topology warnings"))


@dataclass(frozen=True, slots=True)
class RobustnessFailureRecord:
    """Stable failure summary for one explicitly seeded perturbation."""

    seed: int
    stage: str
    error_type: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "failure seed"))
        object.__setattr__(self, "stage", _technical_name(self.stage, "failure stage"))
        object.__setattr__(self, "error_type", _technical_name(self.error_type, "error type"))
        object.__setattr__(self, "message", _text(self.message, "failure message"))


@dataclass(frozen=True, slots=True)
class RobustnessResultRecord:
    """Perturbation protocol, ensemble statistics, uncertainty, and failures."""

    protocol: str
    version: str
    seeds: tuple[int, ...]
    parameters: Mapping[str, ScientificValue]
    statistics: Mapping[str, float | int]
    uncertainty: Mapping[str, float]
    failures: tuple[RobustnessFailureRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol", _technical_name(self.protocol, "robustness protocol"))
        object.__setattr__(self, "version", _text(self.version, "robustness version"))
        seeds = tuple(_nonnegative_integer(seed, "robustness seed") for seed in self.seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("robustness seeds must be non-empty and unique")
        statistics: dict[str, float | int] = {}
        for name, value in self.statistics.items():
            key = _technical_name(name, "statistic name")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("robustness statistics must be real numbers")
            statistics[key] = int(value) if isinstance(value, Integral) else _finite_real(value, key)
        uncertainty = {
            _technical_name(name, "uncertainty name"): _nonnegative_real(value, name)
            for name, value in self.uncertainty.items()
        }
        failures = tuple(self.failures)
        if any(not isinstance(item, RobustnessFailureRecord) for item in failures):
            raise TypeError("failures must contain only RobustnessFailureRecord values")
        if any(item.seed not in seeds for item in failures):
            raise ValueError("failure seeds must belong to the perturbation ensemble")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters, "robustness parameters"))
        object.__setattr__(self, "statistics", MappingProxyType(dict(sorted(statistics.items()))))
        object.__setattr__(self, "uncertainty", MappingProxyType(dict(sorted(uncertainty.items()))))
        object.__setattr__(self, "failures", failures)


@dataclass(frozen=True, slots=True)
class ReproducibilityMetadata:
    """Minimum provenance needed to rerun one exact numerical record."""

    seed: int
    git_commit: str
    git_dirty: bool
    package_version: str
    solver_name: str
    solver_version: str
    solver_settings: Mapping[str, ScientificValue]
    tolerances: Mapping[str, float]
    timestamp_utc: str
    runtime: Mapping[str, ScientificValue] = field(default_factory=dict)
    schema_version: int = field(default=DATASET_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, "seed"))
        object.__setattr__(self, "git_commit", _text(self.git_commit, "git_commit"))
        if not isinstance(self.git_dirty, bool):
            raise TypeError("git_dirty must be a boolean")
        object.__setattr__(self, "package_version", _text(self.package_version, "package_version"))
        object.__setattr__(self, "solver_name", _technical_name(self.solver_name, "solver_name"))
        object.__setattr__(self, "solver_version", _text(self.solver_version, "solver_version"))
        object.__setattr__(self, "solver_settings", _freeze_mapping(self.solver_settings, "solver_settings"))
        tolerances = {
            _technical_name(name, "tolerance name"): _nonnegative_real(value, name)
            for name, value in self.tolerances.items()
        }
        object.__setattr__(self, "tolerances", MappingProxyType(dict(sorted(tolerances.items()))))
        timestamp = _text(self.timestamp_utc, "timestamp_utc")
        try:
            parsed = datetime.fromisoformat(timestamp)
        except ValueError as error:
            raise ValueError("timestamp_utc must be an ISO-8601 timestamp") from error
        utc_offset = parsed.utcoffset()
        if parsed.tzinfo is None or utc_offset is None:
            raise ValueError("timestamp_utc must include a UTC offset")
        if utc_offset.total_seconds() != 0:
            raise ValueError("timestamp_utc must be expressed in UTC")
        object.__setattr__(self, "timestamp_utc", timestamp)
        object.__setattr__(self, "runtime", _freeze_mapping(self.runtime, "runtime"))


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """One exact physics label and all data required to interpret it."""

    record_id: str
    geometry: GeometryRecord
    model: ModelParametersRecord
    spectrum: SpectrumRecord
    observables: tuple[ObservableResultRecord, ...]
    topology: tuple[TopologyResultRecord, ...]
    robustness: tuple[RobustnessResultRecord, ...]
    provenance: ReproducibilityMetadata
    result_kind: str = field(default="exact", init=False)
    schema_version: int = field(default=DATASET_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id.startswith(
            f"{DATASET_RECORD_ID_SCHEME}:"
        ):
            raise ValueError("record_id uses an unsupported scheme")
        digest = self.record_id.rsplit(":", 1)[-1]
        if _SHA256.fullmatch(digest) is None:
            raise ValueError("record_id must end in a SHA-256 digest")
        if not isinstance(self.geometry, GeometryRecord):
            raise TypeError("geometry must be GeometryRecord")
        if not isinstance(self.model, ModelParametersRecord):
            raise TypeError("model must be ModelParametersRecord")
        if not isinstance(self.spectrum, SpectrumRecord):
            raise TypeError("spectrum must be SpectrumRecord")
        if not isinstance(self.provenance, ReproducibilityMetadata):
            raise TypeError("provenance must be ReproducibilityMetadata")
        observables = tuple(self.observables)
        topology = tuple(self.topology)
        robustness = tuple(self.robustness)
        if any(not isinstance(value, ObservableResultRecord) for value in observables):
            raise TypeError("observables must contain only ObservableResultRecord values")
        if any(not isinstance(value, TopologyResultRecord) for value in topology):
            raise TypeError("topology must contain only TopologyResultRecord values")
        if any(not isinstance(value, RobustnessResultRecord) for value in robustness):
            raise TypeError("robustness must contain only RobustnessResultRecord values")
        object.__setattr__(self, "observables", observables)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "robustness", robustness)
        observable_keys = tuple((item.kind, item.version) for item in self.observables)
        topology_keys = tuple((item.method, item.version) for item in self.topology)
        robustness_keys = tuple((item.protocol, item.version) for item in self.robustness)
        if len(set(observable_keys)) != len(observable_keys):
            raise ValueError("observable kind/version pairs must be unique")
        if len(set(topology_keys)) != len(topology_keys):
            raise ValueError("topology method/version pairs must be unique")
        if len(set(robustness_keys)) != len(robustness_keys):
            raise ValueError("robustness protocol/version pairs must be unique")


def exact_geometry_id(geometry: Geometry) -> str:
    """Return the deterministic identity of an exact serialized snapshot."""
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be Geometry")
    return exact_geometry_id_from_bytes(geometry_to_bytes(geometry))


def exact_geometry_id_from_bytes(payload: bytes) -> str:
    """Hash already serialized geometry bytes under the declared scheme."""
    digest = hashlib.sha256(payload).hexdigest()
    return f"{EXACT_GEOMETRY_ID_SCHEME}:{digest}"


def encode_scientific_value(value: ScientificValue) -> Any:
    """Encode a scientific value into strict JSON-compatible tagged data."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return _finite_real(value, "scientific value")
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError("complex scientific values must be finite")
        return {"__complex__": [float(value.real), float(value.imag)]}
    if isinstance(value, Mapping):
        return {key: encode_scientific_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [encode_scientific_value(item) for item in value]
    raise TypeError(f"scientific value has unsupported type {type(value).__name__}")


def decode_scientific_value(value: Any, *, path: str = "value") -> ScientificValue:
    """Decode strict JSON data, rejecting ambiguous complex encodings."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _finite_real(value, path)
    if isinstance(value, list):
        return tuple(
            decode_scientific_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, dict):
        if set(value) == {"__complex__"}:
            parts = value["__complex__"]
            if not isinstance(parts, list) or len(parts) != 2:
                raise ValueError(f"{path} has malformed complex encoding")
            return complex(
                _finite_real(parts[0], f"{path}.real"),
                _finite_real(parts[1], f"{path}.imag"),
            )
        if "__complex__" in value:
            raise ValueError(f"{path} has ambiguous complex encoding")
        if any(not isinstance(key, str) or not key for key in value):
            raise ValueError(f"{path} mapping keys must be non-empty strings")
        return MappingProxyType(
            {
                key: decode_scientific_value(item, path=f"{path}.{key}")
                for key, item in sorted(value.items())
            }
        )
    raise TypeError(f"{path} has unsupported JSON type {type(value).__name__}")


def geometry_archive_to_text(payload: bytes) -> str:
    """Return the canonical base64 representation used by the JSON codec."""
    return base64.b64encode(payload).decode("ascii")


def geometry_archive_from_text(value: str) -> bytes:
    """Decode canonical base64 while rejecting malformed input."""
    if not isinstance(value, str) or not value:
        raise ValueError("geometry archive text must be non-empty")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as error:
        raise ValueError("geometry archive is not valid base64") from error


def _freeze_mapping(values: Mapping[str, Any], name: str) -> Mapping[str, ScientificValue]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    prepared: dict[str, ScientificValue] = {}
    for key in sorted(values):
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        prepared[key] = _freeze_value(values[key], f"{name}[{key!r}]")
    return MappingProxyType(prepared)


def _freeze_value(value: Any, name: str) -> ScientificValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return _finite_real(value, name)
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError(f"{name} must be finite")
        return complex(value)
    if isinstance(value, Mapping):
        return _freeze_mapping(value, name)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_value(item, f"{name}[{index}]") for index, item in enumerate(value))
    raise TypeError(f"{name} has unsupported type {type(value).__name__}")


def _technical_name(value: object, name: str) -> str:
    result = _text(value, name)
    if _IDENTIFIER.fullmatch(result) is None:
        raise ValueError(f"{name} must contain only letters, digits, '.', '_', or '-'")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_real(value: object, name: str) -> float:
    result = _finite_real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_integer(value: object, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _messages(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of strings")
    result = tuple(_text(value, name) for value in values)
    return result
