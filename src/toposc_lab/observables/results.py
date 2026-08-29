"""Standard numerical records shared by observable result dataclasses."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np

ObservableScalar: TypeAlias = bool | int | float | None


@runtime_checkable
class StandardizedObservable(Protocol):
    """Structural interface implemented by specialized observable results."""

    def to_observable_record(self) -> ObservableRecord:
        """Return a validated numerical record."""
        ...


@dataclass(frozen=True, slots=True)
class ObservableRecord:
    """Validated, dataset-ready representation of one observable result.

    Arrays are defensive, read-only copies without object dtype. Scalars are
    numeric values (or ``None`` for a missing measurement), while descriptive
    labels and conventions belong in JSON-compatible ``metadata``.
    """

    kind: str
    scalars: Mapping[str, ObservableScalar] = field(default_factory=dict)
    arrays: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_name(self.kind, name="kind")

        prepared_scalars: dict[str, ObservableScalar] = {}
        for name, value in self.scalars.items():
            _validate_name(name, name="scalar name")
            prepared_scalars[name] = _validated_scalar(value, name=name)

        prepared_arrays: dict[str, np.ndarray] = {}
        for name, values in self.arrays.items():
            _validate_name(name, name="array name")
            if name in prepared_scalars:
                raise ValueError(f"observable field {name!r} is duplicated")
            array = np.asarray(values).copy()
            if array.dtype.hasobject:
                raise ValueError(f"array {name!r} must not use object dtype")
            if not (
                np.issubdtype(array.dtype, np.number)
                or np.issubdtype(array.dtype, np.bool_)
            ):
                raise TypeError(f"array {name!r} must have a numeric or boolean dtype")
            if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
                raise ValueError(f"array {name!r} must contain only finite values")
            array.setflags(write=False)
            prepared_arrays[name] = array

        try:
            prepared_metadata = json.loads(
                json.dumps(dict(self.metadata), allow_nan=False)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("metadata must be JSON compatible") from error
        for name in prepared_metadata:
            _validate_name(name, name="metadata name")
        object.__setattr__(self, "scalars", MappingProxyType(prepared_scalars))
        object.__setattr__(self, "arrays", MappingProxyType(prepared_arrays))
        object.__setattr__(self, "metadata", MappingProxyType(prepared_metadata))

    def numerical_arrays(self, *, prefix: str | None = None) -> dict[str, np.ndarray]:
        """Return NPZ-safe arrays, encoding missing scalar values as ``NaN``."""
        field_prefix = self.kind if prefix is None else prefix
        _validate_name(field_prefix, name="prefix")

        payload = {
            f"{field_prefix}_{name}": np.asarray(
                np.nan if value is None else value,
            )
            for name, value in self.scalars.items()
        }
        payload.update(
            {
                f"{field_prefix}_{name}": values.copy()
                for name, values in self.arrays.items()
            }
        )
        return payload

    def scalar_features(self) -> tuple[tuple[str, ...], np.ndarray]:
        """Return deterministic scalar feature names and a float ML vector."""
        names = tuple(sorted(self.scalars))
        values = np.asarray(
            [
                np.nan if self.scalars[name] is None else self.scalars[name]
                for name in names
            ],
            dtype=float,
        )
        return names, values


def stack_observable_records(
    records: Iterable[ObservableRecord],
    *,
    prefix: str | None = None,
) -> dict[str, np.ndarray]:
    """Stack compatible records along a leading dataset-sample axis."""
    items = tuple(records)
    if not items:
        raise ValueError("records must contain at least one ObservableRecord")
    if not all(isinstance(record, ObservableRecord) for record in items):
        raise TypeError("records must contain only ObservableRecord instances")

    reference = items[0]
    scalar_names = tuple(reference.scalars)
    array_names = tuple(reference.arrays)
    for record in items[1:]:
        if record.kind != reference.kind:
            raise ValueError("all observable records must have the same kind")
        if tuple(record.scalars) != scalar_names or tuple(record.arrays) != array_names:
            raise ValueError("all observable records must have the same field schema")
        if record.metadata != reference.metadata:
            raise ValueError("all observable records must have the same metadata")

    field_prefix = reference.kind if prefix is None else prefix
    _validate_name(field_prefix, name="prefix")
    payload = {
        f"{field_prefix}_{name}": np.asarray(
            [
                np.nan if record.scalars[name] is None else record.scalars[name]
                for record in items
            ]
        )
        for name in scalar_names
    }
    for name in array_names:
        try:
            payload[f"{field_prefix}_{name}"] = np.stack(
                [record.arrays[name] for record in items],
                axis=0,
            )
        except ValueError as error:
            raise ValueError(
                f"array field {name!r} has incompatible sample shapes"
            ) from error
    return payload


def _validate_name(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"{name} must be a non-empty Python-style identifier")


def _validated_scalar(value: object, *, name: str) -> ObservableScalar:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"scalar {name!r} must be finite or None")
        return result
    raise TypeError(f"scalar {name!r} must be numeric, boolean, or None")
