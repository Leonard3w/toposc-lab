"""Multi-objective views that preserve separate scientific quantities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import TypeAlias

from toposc_lab.evaluation.results import GeometryEvaluation
from toposc_lab.topology.results import TopologyMethod

ObjectiveScalar: TypeAlias = bool | int | float


class ObjectiveDirection(str, Enum):
    """Explicit optimization preference without transforming the raw value."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ObjectiveQuantity(str, Enum):
    """Scientific quantities available to the Phase 7.8 objective view."""

    GAP = "gap"
    ZERO_MODE_COUNT = "zero_mode_count"
    STATE_IPR = "state_ipr"
    STATE_BOUNDARY_WEIGHT = "state_boundary_weight"
    STATE_MAJORANA_SELF_CONJUGACY = "state_majorana_self_conjugacy"
    TOPOLOGY_CLASSIFICATION = "topology_classification"
    GEOMETRY_DESCRIPTOR = "geometry_descriptor"


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    """Identify one raw objective and its caller-selected preference direction."""

    name: str
    quantity: ObjectiveQuantity
    direction: ObjectiveDirection
    state_index: int | None = None
    topology_method: TopologyMethod | None = None
    descriptor_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("objective name must be a Python-style identifier")
        if not isinstance(self.quantity, ObjectiveQuantity):
            raise TypeError("quantity must be ObjectiveQuantity")
        if not isinstance(self.direction, ObjectiveDirection):
            raise TypeError("direction must be ObjectiveDirection")

        state_index = _optional_nonnegative_integer(
            self.state_index,
            name="state_index",
        )
        if self.topology_method is not None and not isinstance(
            self.topology_method,
            TopologyMethod,
        ):
            raise TypeError("topology_method must be TopologyMethod or None")
        if self.descriptor_name is not None and (
            not isinstance(self.descriptor_name, str)
            or not self.descriptor_name.isidentifier()
        ):
            raise ValueError(
                "descriptor_name must be a Python-style identifier or None"
            )

        state_quantities = {
            ObjectiveQuantity.STATE_IPR,
            ObjectiveQuantity.STATE_BOUNDARY_WEIGHT,
            ObjectiveQuantity.STATE_MAJORANA_SELF_CONJUGACY,
        }
        if self.quantity in state_quantities:
            if state_index is None:
                raise ValueError(f"{self.quantity.value} requires state_index")
            if self.topology_method is not None or self.descriptor_name is not None:
                raise ValueError(
                    f"{self.quantity.value} accepts only the state_index selector"
                )
        elif self.quantity is ObjectiveQuantity.TOPOLOGY_CLASSIFICATION:
            if self.topology_method is None:
                raise ValueError("topology_classification requires topology_method")
            if state_index is not None or self.descriptor_name is not None:
                raise ValueError(
                    "topology_classification accepts only the topology_method selector"
                )
        elif self.quantity is ObjectiveQuantity.GEOMETRY_DESCRIPTOR:
            if self.descriptor_name is None:
                raise ValueError("geometry_descriptor requires descriptor_name")
            if state_index is not None or self.topology_method is not None:
                raise ValueError(
                    "geometry_descriptor accepts only the descriptor_name selector"
                )
        elif any(
            selector is not None
            for selector in (state_index, self.topology_method, self.descriptor_name)
        ):
            raise ValueError(f"{self.quantity.value} does not accept a selector")

        object.__setattr__(self, "state_index", state_index)


@dataclass(frozen=True, slots=True)
class ObjectiveValue:
    """One validated raw scientific value paired with its explicit specification."""

    spec: ObjectiveSpec
    value: ObjectiveScalar

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ObjectiveSpec):
            raise TypeError("spec must be ObjectiveSpec")
        object.__setattr__(self, "value", _objective_scalar(self.value, name="value"))


@dataclass(frozen=True, slots=True)
class MultiObjectiveEvaluation:
    """Immutable named objective vector with no scalar aggregation or ranking."""

    objectives: Mapping[str, ObjectiveValue]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.objectives, Mapping):
            raise TypeError("objectives must be a mapping")
        prepared: dict[str, ObjectiveValue] = {}
        for name, objective in self.objectives.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("objective keys must be Python-style identifiers")
            if not isinstance(objective, ObjectiveValue):
                raise TypeError("objectives must contain only ObjectiveValue objects")
            if name != objective.spec.name:
                raise ValueError("each objective key must equal its specification name")
            prepared[name] = objective
        if not prepared:
            raise ValueError("objectives must not be empty")

        object.__setattr__(self, "objectives", MappingProxyType(prepared))
        object.__setattr__(self, "warnings", _messages(self.warnings))


def evaluate_multi_objectives(
    evaluation: GeometryEvaluation,
    *,
    objectives: Iterable[ObjectiveSpec],
) -> MultiObjectiveEvaluation:
    """Extract an explicit vector without normalization or scalar reduction.

    Every requested source must be present in ``evaluation``. State-resolved
    quantities stay tied to their eigensolver column index, topology remains
    tied to its diagnostic method, and geometry descriptors retain their raw
    scalar type. ``direction`` records preference only; it does not modify the
    returned value.
    """
    if not isinstance(evaluation, GeometryEvaluation):
        raise TypeError("evaluation must be GeometryEvaluation")
    if isinstance(objectives, (str, bytes)) or not isinstance(objectives, Iterable):
        raise TypeError("objectives must be an iterable of ObjectiveSpec objects")
    specs = tuple(objectives)
    if not specs:
        raise ValueError("objectives must not be empty")
    if any(not isinstance(spec, ObjectiveSpec) for spec in specs):
        raise TypeError("objectives must contain only ObjectiveSpec objects")
    names = tuple(spec.name for spec in specs)
    if len(set(names)) != len(names):
        raise ValueError("objective names must be unique")

    values = {
        spec.name: ObjectiveValue(
            spec=spec,
            value=_extract_objective(evaluation, spec=spec),
        )
        for spec in specs
    }
    return MultiObjectiveEvaluation(
        objectives=values,
        warnings=_objective_warnings(evaluation, specs=specs),
    )


def _extract_objective(
    evaluation: GeometryEvaluation,
    *,
    spec: ObjectiveSpec,
) -> ObjectiveScalar:
    if spec.quantity is ObjectiveQuantity.GAP:
        if evaluation.gap is None:
            raise ValueError(f"objective {spec.name!r} requires evaluation.gap")
        return evaluation.gap

    if spec.quantity is ObjectiveQuantity.ZERO_MODE_COUNT:
        if evaluation.zero_mode_count is None:
            raise ValueError(
                f"objective {spec.name!r} requires evaluation.zero_mode_count"
            )
        return evaluation.zero_mode_count

    if spec.quantity is ObjectiveQuantity.STATE_IPR:
        assert spec.state_index is not None
        try:
            value = evaluation.ipr[spec.state_index]
        except KeyError as error:
            raise ValueError(
                f"objective {spec.name!r} requires IPR for state {spec.state_index}"
            ) from error
        return _unit_interval_real(value, name=f"objective {spec.name!r} IPR")

    if spec.quantity is ObjectiveQuantity.STATE_BOUNDARY_WEIGHT:
        assert spec.state_index is not None
        try:
            profile = evaluation.localization[spec.state_index]
        except KeyError as error:
            raise ValueError(
                f"objective {spec.name!r} requires localization for state "
                f"{spec.state_index}"
            ) from error
        return _unit_interval_real(
            profile.edge_weight,
            name=f"objective {spec.name!r} boundary weight",
        )

    if spec.quantity is ObjectiveQuantity.STATE_MAJORANA_SELF_CONJUGACY:
        assert spec.state_index is not None
        try:
            diagnostics = evaluation.majorana_metrics[spec.state_index]
        except KeyError as error:
            raise ValueError(
                f"objective {spec.name!r} requires Majorana diagnostics for state "
                f"{spec.state_index}"
            ) from error
        return _unit_interval_real(
            diagnostics.self_conjugacy,
            name=f"objective {spec.name!r} Majorana self-conjugacy",
        )

    if spec.quantity is ObjectiveQuantity.TOPOLOGY_CLASSIFICATION:
        assert spec.topology_method is not None
        matching = tuple(
            result
            for result in evaluation.topology
            if result.method is spec.topology_method
        )
        if not matching:
            raise ValueError(
                f"objective {spec.name!r} requires topology method "
                f"{spec.topology_method.value}"
            )
        classification = matching[0].is_topological
        if classification is None:
            raise ValueError(
                f"objective {spec.name!r} requires a resolved topology classification"
            )
        return classification

    if spec.quantity is ObjectiveQuantity.GEOMETRY_DESCRIPTOR:
        assert spec.descriptor_name is not None
        if spec.descriptor_name not in evaluation.geometry_descriptors:
            raise ValueError(
                f"objective {spec.name!r} requires geometry descriptor "
                f"{spec.descriptor_name!r}"
            )
        descriptor_value = evaluation.geometry_descriptors[spec.descriptor_name]
        if descriptor_value is None:
            raise ValueError(
                f"objective {spec.name!r} requires a defined geometry descriptor "
                f"{spec.descriptor_name!r}"
            )
        return _objective_scalar(
            descriptor_value,
            name=f"geometry descriptor {spec.descriptor_name!r}",
        )

    raise AssertionError(f"unsupported objective quantity: {spec.quantity!r}")


def _objective_warnings(
    evaluation: GeometryEvaluation,
    *,
    specs: tuple[ObjectiveSpec, ...],
) -> tuple[str, ...]:
    warnings = [
        *evaluation.warnings,
        "Objectives remain separate; this result defines neither a scalar score nor a candidate ranking.",
        "Objective directions are caller-selected preferences, not claims about physical significance.",
        "Objective vectors are directly comparable only when their specifications and source conventions agree.",
    ]
    quantities = {spec.quantity for spec in specs}
    if ObjectiveQuantity.GAP in quantities:
        warnings.append(
            "The gap retains the evaluator's finite-system convention and is not automatically a thermodynamic bulk gap."
        )
    if ObjectiveQuantity.ZERO_MODE_COUNT in quantities:
        warnings.append(
            "Zero-mode count depends on the numerical tolerance and does not establish Majorana character or topology."
        )
    if quantities.intersection(
        {
            ObjectiveQuantity.STATE_IPR,
            ObjectiveQuantity.STATE_BOUNDARY_WEIGHT,
            ObjectiveQuantity.STATE_MAJORANA_SELF_CONJUGACY,
        }
    ):
        warnings.append(
            "State objectives depend on the selected eigensolver state and can be basis-dependent within degenerate subspaces."
        )
    if ObjectiveQuantity.TOPOLOGY_CLASSIFICATION in quantities:
        warnings.append(
            "Topology classifications must be interpreted with each method's assumptions, confidence, and warnings."
        )
    if ObjectiveQuantity.GEOMETRY_DESCRIPTOR in quantities:
        warnings.append(
            "Geometry descriptors are structural quantities and do not by themselves imply physical performance."
        )
    return tuple(dict.fromkeys(warnings))


def _objective_scalar(value: object, *, name: str) -> ObjectiveScalar:
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            return numeric_value
    raise ValueError(f"{name} must be a finite scalar")


def _unit_interval_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return numeric_value


def _optional_nonnegative_integer(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer or None")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _messages(values: tuple[str, ...]) -> tuple[str, ...]:
    prepared = tuple(values)
    if not prepared:
        raise ValueError("warnings must not be empty")
    if any(not isinstance(message, str) for message in prepared):
        raise TypeError("warnings must contain only strings")
    if any(not message.strip() for message in prepared):
        raise ValueError("warnings must not contain empty messages")
    if len(set(prepared)) != len(prepared):
        raise ValueError("warnings must not contain duplicates")
    return prepared
