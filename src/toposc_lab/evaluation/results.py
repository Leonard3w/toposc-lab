"""Typed result records for automated geometry evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import TypeAlias, TypeVar

import numpy as np

from toposc_lab.observables.localization import LocalizationProfile
from toposc_lab.observables.majorana import MajoranaDiagnostics
from toposc_lab.topology.results import TopologyResult

DescriptorScalar: TypeAlias = bool | int | float | None
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class GeometryEvaluation:
    """Validated aggregate of scientific quantities for one geometry.

    The record stores results but does not compute them. State-dependent fields
    use eigensolver column indices as keys, and ``low_energy_states`` supplies
    the corresponding eigenenergies. Missing analyses remain explicit through
    ``None`` or empty collections.

    ``gap`` deliberately carries no implicit gap convention. The evaluation
    routine that eventually populates it must document the selected convention.
    Likewise, warnings record scientific limitations rather than silently
    converting them into a scalar score.
    """

    gap: float | None = None
    low_energy_states: Mapping[int, float] = field(default_factory=dict)
    zero_mode_count: int | None = None
    ipr: Mapping[int, float] = field(default_factory=dict)
    localization: Mapping[int, LocalizationProfile] = field(default_factory=dict)
    majorana_metrics: Mapping[int, MajoranaDiagnostics] = field(default_factory=dict)
    topology: tuple[TopologyResult, ...] = ()
    geometry_descriptors: Mapping[str, DescriptorScalar] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        gap = _optional_nonnegative_real(self.gap, name="gap")
        low_energy_states = _state_real_mapping(
            self.low_energy_states,
            name="low_energy_states",
            nonnegative=False,
        )
        zero_mode_count = _optional_nonnegative_integer(
            self.zero_mode_count,
            name="zero_mode_count",
        )
        if zero_mode_count is not None and zero_mode_count > len(low_energy_states):
            raise ValueError(
                "zero_mode_count must not exceed the number of low-energy states"
            )

        ipr = _state_real_mapping(self.ipr, name="ipr", nonnegative=True)
        localization = _state_result_mapping(
            self.localization,
            expected_type=LocalizationProfile,
            name="localization",
        )
        majorana_metrics = _state_result_mapping(
            self.majorana_metrics,
            expected_type=MajoranaDiagnostics,
            name="majorana_metrics",
        )
        known_states = set(low_energy_states)
        for field_name, state_indices in (
            ("ipr", ipr),
            ("localization", localization),
            ("majorana_metrics", majorana_metrics),
        ):
            unknown_states = set(state_indices).difference(known_states)
            if unknown_states:
                raise ValueError(
                    f"{field_name} contains states absent from low_energy_states: "
                    f"{sorted(unknown_states)}"
                )

        topology = tuple(self.topology)
        if any(not isinstance(result, TopologyResult) for result in topology):
            raise TypeError("topology must contain only TopologyResult objects")
        methods = tuple(result.method for result in topology)
        if len(set(methods)) != len(methods):
            raise ValueError("topology must contain at most one result per method")

        geometry_descriptors = _descriptor_mapping(self.geometry_descriptors)
        warnings = _messages(self.warnings)

        object.__setattr__(self, "gap", gap)
        object.__setattr__(self, "low_energy_states", MappingProxyType(low_energy_states))
        object.__setattr__(self, "zero_mode_count", zero_mode_count)
        object.__setattr__(self, "ipr", MappingProxyType(ipr))
        object.__setattr__(self, "localization", MappingProxyType(localization))
        object.__setattr__(
            self,
            "majorana_metrics",
            MappingProxyType(majorana_metrics),
        )
        object.__setattr__(self, "topology", topology)
        object.__setattr__(
            self,
            "geometry_descriptors",
            MappingProxyType(geometry_descriptors),
        )
        object.__setattr__(self, "warnings", warnings)


def _state_real_mapping(
    values: Mapping[int, float],
    *,
    name: str,
    nonnegative: bool,
) -> dict[int, float]:
    prepared: dict[int, float] = {}
    for state_index, value in values.items():
        index = _nonnegative_integer(state_index, name=f"{name} state index")
        if index in prepared:
            raise ValueError(f"{name} contains duplicate normalized state index {index}")
        prepared[index] = (
            _nonnegative_real(value, name=f"{name}[{index}]")
            if nonnegative
            else _finite_real(value, name=f"{name}[{index}]")
        )
    return prepared


def _state_result_mapping(
    values: Mapping[int, ResultT],
    *,
    expected_type: type[ResultT],
    name: str,
) -> dict[int, ResultT]:
    prepared: dict[int, ResultT] = {}
    for state_index, value in values.items():
        index = _nonnegative_integer(state_index, name=f"{name} state index")
        if index in prepared:
            raise ValueError(f"{name} contains duplicate normalized state index {index}")
        if not isinstance(value, expected_type):
            raise TypeError(
                f"{name}[{index}] must be {expected_type.__name__}"
            )
        prepared[index] = value
    return prepared


def _descriptor_mapping(
    values: Mapping[str, DescriptorScalar],
) -> dict[str, DescriptorScalar]:
    prepared: dict[str, DescriptorScalar] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("geometry descriptor names must be Python-style identifiers")
        if value is None or isinstance(value, bool):
            prepared[name] = value
        elif isinstance(value, Integral):
            prepared[name] = int(value)
        elif isinstance(value, Real):
            numeric_value = float(value)
            if not np.isfinite(numeric_value):
                raise ValueError(f"geometry descriptor {name!r} must be finite")
            prepared[name] = numeric_value
        else:
            raise TypeError(
                f"geometry descriptor {name!r} must be a scalar or None"
            )
    return prepared


def _messages(values: tuple[str, ...]) -> tuple[str, ...]:
    prepared = tuple(values)
    if any(not isinstance(message, str) for message in prepared):
        raise TypeError("warnings must contain only strings")
    if any(not message.strip() for message in prepared):
        raise ValueError("warnings must not contain empty messages")
    return prepared


def _finite_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


def _nonnegative_real(value: float, *, name: str) -> float:
    numeric_value = _finite_real(value, name=name)
    if numeric_value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric_value


def _optional_nonnegative_real(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    return _nonnegative_real(value, name=name)


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    integer_value = int(value)
    if integer_value < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer_value


def _optional_nonnegative_integer(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, name=name)
