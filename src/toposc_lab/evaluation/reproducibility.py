"""Immutable provenance records for geometry-evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Literal, TypeAlias

from toposc_lab.geometry.base import Geometry
from toposc_lab.geometry.serialization import geometry_to_bytes

ReproducibilityScalar: TypeAlias = None | bool | int | float | str
ReproducibilityValue: TypeAlias = (
    ReproducibilityScalar
    | tuple["ReproducibilityValue", ...]
    | Mapping[str, "ReproducibilityValue"]
)
CodeVersionSource: TypeAlias = Literal[
    "explicit",
    "package_metadata",
    "unavailable",
]

GEOMETRY_ID_SCHEME = "toposc-geometry-archive-v1-sha256"
_PACKAGE_NAME = "toposc-lab"
_GEOMETRY_ID_WARNING = (
    "The geometry ID hashes the exact version-1 serialized snapshot, including "
    "edge orientation and geometric data. It is representation-sensitive and is "
    "not a canonical isomorphism or physical-deduplication identifier."
)


@dataclass(frozen=True, slots=True)
class ReproducibilityRecord:
    """Provenance needed to repeat one completed numerical evaluation."""

    seed: int | None
    model_name: str
    model_parameters: Mapping[str, ReproducibilityValue]
    geometry_id: str
    geometry_id_scheme: str
    solver_name: str
    solver_settings: Mapping[str, ReproducibilityValue]
    evaluation_settings: Mapping[str, ReproducibilityValue]
    code_version: str | None
    code_version_source: CodeVersionSource
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _optional_seed(self.seed))
        for name in ("model_name", "geometry_id", "geometry_id_scheme", "solver_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(
            self,
            "model_parameters",
            _freeze_mapping(self.model_parameters, name="model_parameters"),
        )
        object.__setattr__(
            self,
            "solver_settings",
            _freeze_mapping(self.solver_settings, name="solver_settings"),
        )
        object.__setattr__(
            self,
            "evaluation_settings",
            _freeze_mapping(self.evaluation_settings, name="evaluation_settings"),
        )
        if self.code_version is not None and (
            not isinstance(self.code_version, str) or not self.code_version.strip()
        ):
            raise ValueError("code_version must be a non-empty string or None")
        if self.code_version_source not in {
            "explicit",
            "package_metadata",
            "unavailable",
        }:
            raise ValueError("code_version_source is not supported")
        if (self.code_version is None) != (
            self.code_version_source == "unavailable"
        ):
            raise ValueError("code_version and code_version_source are inconsistent")
        if isinstance(self.warnings, str) or not isinstance(self.warnings, Sequence):
            raise TypeError("warnings must be a sequence of strings")
        warnings = tuple(self.warnings)
        if any(not isinstance(item, str) or not item.strip() for item in warnings):
            raise ValueError("warnings must contain only non-empty strings")
        object.__setattr__(self, "warnings", warnings)


def exact_geometry_id(geometry: Geometry) -> str:
    """Hash the exact serialized geometry snapshot used by an evaluation."""
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be Geometry")
    digest = hashlib.sha256(geometry_to_bytes(geometry)).hexdigest()
    return f"{GEOMETRY_ID_SCHEME}:{digest}"


def create_reproducibility_record(
    geometry: Geometry,
    *,
    seed: int | None,
    model_name: str,
    model_parameters: Mapping[str, object],
    solver_name: str,
    solver_settings: Mapping[str, object],
    evaluation_settings: Mapping[str, object],
    code_version: str | None = None,
) -> ReproducibilityRecord:
    """Build one validated record, resolving package version when necessary."""
    resolved_version, version_source = _resolve_code_version(code_version)
    warnings = [_GEOMETRY_ID_WARNING]
    if resolved_version is None:
        warnings.append(
            "The code version was unavailable; supply code_version explicitly when "
            "running from an uninstalled source tree."
        )
    return ReproducibilityRecord(
        seed=seed,
        model_name=model_name,
        model_parameters=_freeze_mapping(
            model_parameters,
            name="model_parameters",
        ),
        geometry_id=exact_geometry_id(geometry),
        geometry_id_scheme=GEOMETRY_ID_SCHEME,
        solver_name=solver_name,
        solver_settings=_freeze_mapping(
            solver_settings,
            name="solver_settings",
        ),
        evaluation_settings=_freeze_mapping(
            evaluation_settings,
            name="evaluation_settings",
        ),
        code_version=resolved_version,
        code_version_source=version_source,
        warnings=tuple(warnings),
    )


def _resolve_code_version(
    code_version: str | None,
) -> tuple[str | None, CodeVersionSource]:
    if code_version is not None:
        if not isinstance(code_version, str):
            raise TypeError("code_version must be a string or None")
        if not code_version.strip():
            raise ValueError("code_version must be a non-empty string or None")
        return code_version, "explicit"
    try:
        return version(_PACKAGE_NAME), "package_metadata"
    except PackageNotFoundError:
        return None, "unavailable"


def _optional_seed(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("seed must be an integer or None")
    result = int(value)
    if result < 0:
        raise ValueError("seed must be non-negative")
    return result


def _freeze_mapping(
    values: Mapping[str, object],
    *,
    name: str,
) -> Mapping[str, ReproducibilityValue]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) or not key for key in values):
        raise ValueError(f"{name} keys must be non-empty strings")
    prepared: dict[str, ReproducibilityValue] = {}
    for key in sorted(values):
        prepared[key] = _freeze_value(values[key], name=f"{name}[{key!r}]")
    return MappingProxyType(prepared)


def _freeze_value(value: object, *, name: str) -> ReproducibilityValue:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result
    if isinstance(value, Mapping):
        return _freeze_mapping(value, name=name)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{name} has unsupported type {type(value).__name__}")
