"""Neutral, reproducible execution contract for disorder transformations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from numbers import Integral, Real
import re
from types import MappingProxyType
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from toposc_lab.evaluation.reproducibility import (
    GEOMETRY_ID_SCHEME,
    exact_geometry_id,
)
from toposc_lab.geometry import Geometry

DisorderParameterScalar: TypeAlias = None | bool | int | float | str
DisorderParameterValue: TypeAlias = (
    DisorderParameterScalar
    | tuple["DisorderParameterValue", ...]
    | Mapping[str, "DisorderParameterValue"]
)
HamiltonianArray: TypeAlias = NDArray[np.generic]
ModelParameterSet: TypeAlias = Mapping[str, DisorderParameterValue]
DisorderState: TypeAlias = Geometry | HamiltonianArray | ModelParameterSet

DISORDER_RNG_ALGORITHM = "numpy.random.PCG64"
HAMILTONIAN_ID_SCHEME = "toposc-hamiltonian-array-v1-sha256"
MODEL_PARAMETER_SET_ID_SCHEME = "toposc-model-parameter-set-v1-sha256"
_DISORDER_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class DisorderTarget(str, Enum):
    """Kind of scientific object transformed by one disorder definition."""

    GEOMETRY = "geometry"
    HAMILTONIAN = "hamiltonian"
    MODEL_PARAMETERS = "model_parameters"


@dataclass(frozen=True, slots=True)
class DisorderRequest:
    """Parameters and mandatory seed for one disorder realization."""

    seed: int
    parameters: Mapping[str, DisorderParameterValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _validated_seed(self.seed))
        object.__setattr__(
            self,
            "parameters",
            _freeze_parameter_mapping(self.parameters, name="parameters"),
        )


@runtime_checkable
class DisorderTransform(Protocol):
    """Structural interface for a versioned, target-specific perturbation."""

    @property
    def key(self) -> str:
        """Stable technical key naming the disorder algorithm."""
        ...

    @property
    def version(self) -> int:
        """Positive version of the disorder algorithm."""
        ...

    @property
    def target(self) -> DisorderTarget:
        """Scientific state kind accepted and returned by the transform."""
        ...

    def apply(
        self,
        source: DisorderState,
        *,
        parameters: Mapping[str, DisorderParameterValue],
        rng: np.random.Generator,
    ) -> DisorderState:
        """Return a new state using only the supplied random generator."""
        ...


DisorderFunction: TypeAlias = Callable[
    [
        DisorderState,
        Mapping[str, DisorderParameterValue],
        np.random.Generator,
    ],
    DisorderState,
]


@dataclass(frozen=True, slots=True)
class FunctionDisorderTransform:
    """Adapt a plain transformation function to ``DisorderTransform``."""

    key: str
    target: DisorderTarget
    function: DisorderFunction = field(repr=False, compare=False)
    version: int = 1

    def __post_init__(self) -> None:
        _validate_transform_metadata(self)
        if not callable(self.function):
            raise TypeError("function must be callable")

    def apply(
        self,
        source: DisorderState,
        *,
        parameters: Mapping[str, DisorderParameterValue],
        rng: np.random.Generator,
    ) -> DisorderState:
        """Call the adapted function without creating another RNG."""
        return self.function(source, parameters, rng)


@dataclass(frozen=True, slots=True)
class DisorderSnapshot:
    """Exact, representation-sensitive identity of one disorder state."""

    target: DisorderTarget
    identifier: str
    scheme: str

    def __post_init__(self) -> None:
        if not isinstance(self.target, DisorderTarget):
            raise TypeError("target must be DisorderTarget")
        for name in ("identifier", "scheme"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not self.identifier.startswith(f"{self.scheme}:"):
            raise ValueError("identifier must use the declared snapshot scheme")


@dataclass(frozen=True, slots=True)
class DisorderProvenance:
    """Auditable relationship between a clean source and one realization."""

    disorder_key: str
    disorder_version: int
    parameters: Mapping[str, DisorderParameterValue]
    seed: int
    rng_algorithm: str
    source: DisorderSnapshot
    result: DisorderSnapshot

    def __post_init__(self) -> None:
        _validate_key_and_version(self.disorder_key, self.disorder_version)
        object.__setattr__(
            self,
            "parameters",
            _freeze_parameter_mapping(self.parameters, name="parameters"),
        )
        object.__setattr__(self, "seed", _validated_seed(self.seed))
        if self.rng_algorithm != DISORDER_RNG_ALGORITHM:
            raise ValueError(
                f"rng_algorithm must be {DISORDER_RNG_ALGORITHM!r}"
            )
        if not isinstance(self.source, DisorderSnapshot):
            raise TypeError("source must be DisorderSnapshot")
        if not isinstance(self.result, DisorderSnapshot):
            raise TypeError("result must be DisorderSnapshot")
        if self.source.target is not self.result.target:
            raise ValueError("source and result snapshot targets must match")


@dataclass(frozen=True, slots=True)
class DisorderRealization:
    """A transformed state kept separate from its immutable provenance."""

    state: DisorderState
    provenance: DisorderProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, DisorderProvenance):
            raise TypeError("provenance must be DisorderProvenance")
        prepared = _prepare_state(self.state, target=self.provenance.result.target)
        snapshot = _snapshot(prepared, target=self.provenance.result.target)
        if snapshot != self.provenance.result:
            raise ValueError("state does not match the recorded result snapshot")
        object.__setattr__(self, "state", prepared)


def realize_disorder(
    source: DisorderState,
    *,
    transform: DisorderTransform,
    request: DisorderRequest,
) -> DisorderRealization:
    """Apply one transform with an explicit PCG64 stream and record both states."""
    if not isinstance(transform, DisorderTransform):
        raise TypeError("transform must implement DisorderTransform")
    if not isinstance(request, DisorderRequest):
        raise TypeError("request must be DisorderRequest")
    _validate_transform_metadata(transform)

    prepared_source = _prepare_state(source, target=transform.target)
    source_snapshot = _snapshot(prepared_source, target=transform.target)
    generator = np.random.Generator(np.random.PCG64(request.seed))
    transformed = transform.apply(
        prepared_source,
        parameters=request.parameters,
        rng=generator,
    )
    prepared_result = _prepare_state(transformed, target=transform.target)
    if transform.target is DisorderTarget.HAMILTONIAN:
        assert isinstance(prepared_source, np.ndarray)
        assert isinstance(prepared_result, np.ndarray)
        if prepared_result.shape != prepared_source.shape:
            raise ValueError("Hamiltonian disorder must preserve the matrix shape")
    elif transform.target is DisorderTarget.MODEL_PARAMETERS:
        assert isinstance(prepared_source, Mapping)
        assert isinstance(prepared_result, Mapping)
        if tuple(prepared_result) != tuple(prepared_source):
            raise ValueError(
                "model-parameter disorder must preserve the parameter keys"
            )
    result_snapshot = _snapshot(prepared_result, target=transform.target)
    provenance = DisorderProvenance(
        disorder_key=transform.key,
        disorder_version=transform.version,
        parameters=request.parameters,
        seed=request.seed,
        rng_algorithm=DISORDER_RNG_ALGORITHM,
        source=source_snapshot,
        result=result_snapshot,
    )
    return DisorderRealization(state=prepared_result, provenance=provenance)


def exact_hamiltonian_id(hamiltonian: HamiltonianArray) -> str:
    """Hash the exact dtype, shape, and C-order values of a finite matrix."""
    matrix = _prepare_hamiltonian(hamiltonian)
    header = json.dumps(
        {
            "dtype": matrix.dtype.str,
            "shape": list(matrix.shape),
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, byteorder="big", signed=False))
    digest.update(header)
    digest.update(matrix.tobytes(order="C"))
    return f"{HAMILTONIAN_ID_SCHEME}:{digest.hexdigest()}"


def exact_model_parameter_set_id(parameters: ModelParameterSet) -> str:
    """Hash a deeply normalized parameter mapping with explicit scalar types."""
    prepared = _prepare_model_parameter_set(parameters)
    payload = json.dumps(
        _encode_parameter_value(prepared),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return f"{MODEL_PARAMETER_SET_ID_SCHEME}:{digest}"


def _prepare_state(state: DisorderState, *, target: DisorderTarget) -> DisorderState:
    if not isinstance(target, DisorderTarget):
        raise TypeError("transform target must be DisorderTarget")
    if target is DisorderTarget.GEOMETRY:
        if not isinstance(state, Geometry):
            raise TypeError("geometry disorder requires and returns Geometry")
        return state
    if target is DisorderTarget.HAMILTONIAN:
        if not isinstance(state, np.ndarray):
            raise TypeError("Hamiltonian disorder requires and returns numpy.ndarray")
        return _prepare_hamiltonian(state)
    if not isinstance(state, Mapping):
        raise TypeError(
            "model-parameter disorder requires and returns a parameter mapping"
        )
    return _prepare_model_parameter_set(state)


def _prepare_hamiltonian(hamiltonian: HamiltonianArray) -> HamiltonianArray:
    if not isinstance(hamiltonian, np.ndarray):
        raise TypeError("hamiltonian must be numpy.ndarray")
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("hamiltonian must be a square two-dimensional array")
    if not np.issubdtype(hamiltonian.dtype, np.number):
        raise TypeError("hamiltonian must have a numerical dtype")
    if not np.all(np.isfinite(hamiltonian)):
        raise ValueError("hamiltonian must contain only finite values")
    contiguous = np.ascontiguousarray(hamiltonian)
    immutable_buffer = contiguous.tobytes(order="C")
    return np.frombuffer(immutable_buffer, dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _prepare_model_parameter_set(
    parameters: ModelParameterSet,
) -> ModelParameterSet:
    return _freeze_parameter_mapping(parameters, name="model_parameters")


def _snapshot(state: DisorderState, *, target: DisorderTarget) -> DisorderSnapshot:
    if target is DisorderTarget.GEOMETRY:
        assert isinstance(state, Geometry)
        return DisorderSnapshot(
            target=target,
            identifier=exact_geometry_id(state),
            scheme=GEOMETRY_ID_SCHEME,
        )
    if target is DisorderTarget.HAMILTONIAN:
        assert isinstance(state, np.ndarray)
        return DisorderSnapshot(
            target=target,
            identifier=exact_hamiltonian_id(state),
            scheme=HAMILTONIAN_ID_SCHEME,
        )
    assert isinstance(state, Mapping)
    return DisorderSnapshot(
        target=target,
        identifier=exact_model_parameter_set_id(state),
        scheme=MODEL_PARAMETER_SET_ID_SCHEME,
    )


def _validate_transform_metadata(transform: DisorderTransform) -> None:
    _validate_key_and_version(transform.key, transform.version)
    if not isinstance(transform.target, DisorderTarget):
        raise TypeError("transform target must be DisorderTarget")


def _validate_key_and_version(key: str, version: int) -> None:
    if not isinstance(key, str) or _DISORDER_KEY_PATTERN.fullmatch(key) is None:
        raise ValueError(
            "disorder key must start with a lowercase letter and contain only "
            "lowercase letters, digits, and underscores"
        )
    if isinstance(version, bool) or not isinstance(version, Integral):
        raise TypeError("disorder version must be an integer")
    if int(version) < 1:
        raise ValueError("disorder version must be at least one")


def _validated_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    result = int(seed)
    if result < 0:
        raise ValueError("seed must be non-negative")
    return result


def _freeze_parameter_mapping(
    values: Mapping[str, DisorderParameterValue],
    *,
    name: str,
) -> Mapping[str, DisorderParameterValue]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) or not key.isidentifier() for key in values):
        raise ValueError(
            f"{name} keys must be non-empty Python-style identifiers"
        )
    prepared: dict[str, DisorderParameterValue] = {}
    for key in sorted(values):
        prepared[key] = _freeze_parameter_value(
            values[key],
            name=f"{name}[{key!r}]",
        )
    return MappingProxyType(prepared)


def _freeze_parameter_value(
    value: DisorderParameterValue,
    *,
    name: str,
) -> DisorderParameterValue:
    if value is None or isinstance(value, str) or isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result
    if isinstance(value, Mapping):
        return _freeze_parameter_mapping(value, name=name)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_parameter_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{name} has unsupported type {type(value).__name__}")


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
            "items": [
                [key, _encode_parameter_value(value[key])]
                for key in sorted(value)
            ],
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [_encode_parameter_value(item) for item in value],
        }
    raise TypeError(f"unsupported prepared parameter type {type(value).__name__}")
