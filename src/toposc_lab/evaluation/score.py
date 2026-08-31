"""Transparent scalar engineering score for completed geometry evaluations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from types import MappingProxyType

from toposc_lab.evaluation.results import GeometryEvaluation


class BasicScoreComponent(str, Enum):
    """Stable identifiers for the normalized terms in the basic score."""

    NORMALIZED_GAP = "normalized_gap"
    ZERO_MODE_PRESENCE = "zero_mode_presence"
    MAXIMUM_IPR = "maximum_ipr"
    MAXIMUM_BOUNDARY_WEIGHT = "maximum_boundary_weight"
    MAXIMUM_MAJORANA_SELF_CONJUGACY = "maximum_majorana_self_conjugacy"
    TOPOLOGICAL_METHOD_FRACTION = "topological_method_fraction"


@dataclass(frozen=True, slots=True)
class BasicScalarScore:
    """Auditable weighted average used only for engineering ranking.

    ``components`` contains the normalized scientific inputs, while
    ``normalized_weights`` and ``contributions`` make the construction of
    ``value`` explicit. This record is not a topological invariant or a
    replacement for the separate quantities in :class:`GeometryEvaluation`.
    """

    value: float
    components: Mapping[BasicScoreComponent, float]
    normalized_weights: Mapping[BasicScoreComponent, float]
    contributions: Mapping[BasicScoreComponent, float]
    gap_scale: float
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        value = _unit_interval_real(self.value, name="value")
        components = _unit_interval_mapping(self.components, name="components")
        normalized_weights = _unit_interval_mapping(
            self.normalized_weights,
            name="normalized_weights",
        )
        contributions = _unit_interval_mapping(
            self.contributions,
            name="contributions",
        )
        gap_scale = _positive_finite_real(self.gap_scale, name="gap_scale")
        warnings = _warnings(self.warnings)

        component_keys = set(components)
        if not component_keys:
            raise ValueError("components must not be empty")
        if set(normalized_weights) != component_keys:
            raise ValueError("normalized_weights must use the same keys as components")
        if set(contributions) != component_keys:
            raise ValueError("contributions must use the same keys as components")
        if not math.isclose(
            sum(normalized_weights.values()),
            1.0,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("normalized_weights must sum to one")
        for component in component_keys:
            expected = components[component] * normalized_weights[component]
            if not math.isclose(
                contributions[component],
                expected,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "each contribution must equal component times normalized weight"
                )
        if not math.isclose(
            sum(contributions.values()),
            value,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("value must equal the sum of contributions")

        object.__setattr__(self, "value", value)
        object.__setattr__(self, "components", MappingProxyType(components))
        object.__setattr__(
            self,
            "normalized_weights",
            MappingProxyType(normalized_weights),
        )
        object.__setattr__(
            self,
            "contributions",
            MappingProxyType(contributions),
        )
        object.__setattr__(self, "gap_scale", gap_scale)
        object.__setattr__(self, "warnings", warnings)


def compute_basic_scalar_score(
    evaluation: GeometryEvaluation,
    *,
    weights: Mapping[BasicScoreComponent, float],
    gap_scale: float = 1.0,
) -> BasicScalarScore:
    """Return an explicit weighted engineering score between zero and one.

    Only components named in ``weights`` are evaluated. Every supplied weight
    must be finite and strictly positive; omit a component instead of assigning
    it zero weight. Requested but unavailable quantities raise an error so that
    candidates are never compared after silent per-candidate renormalization.

    The gap term uses ``gap / (gap + gap_scale)``. Other terms are already
    dimensionless: zero-mode presence is binary, state summaries use their
    maximum value, and the topology term is the fraction of resolved methods
    reporting a topological result.
    """
    if not isinstance(evaluation, GeometryEvaluation):
        raise TypeError("evaluation must be GeometryEvaluation")
    prepared_weights = _positive_weight_mapping(weights)
    prepared_gap_scale = _positive_finite_real(gap_scale, name="gap_scale")

    components = {
        component: _component_value(
            evaluation,
            component=component,
            gap_scale=prepared_gap_scale,
        )
        for component in prepared_weights
    }
    total_weight = sum(prepared_weights.values())
    normalized_weights = {
        component: weight / total_weight
        for component, weight in prepared_weights.items()
    }
    contributions = {
        component: components[component] * normalized_weights[component]
        for component in components
    }

    return BasicScalarScore(
        value=sum(contributions.values()),
        components=components,
        normalized_weights=normalized_weights,
        contributions=contributions,
        gap_scale=prepared_gap_scale,
        warnings=_score_warnings(tuple(components)),
    )


def _component_value(
    evaluation: GeometryEvaluation,
    *,
    component: BasicScoreComponent,
    gap_scale: float,
) -> float:
    if component is BasicScoreComponent.NORMALIZED_GAP:
        if evaluation.gap is None:
            raise ValueError("normalized_gap requires evaluation.gap")
        return evaluation.gap / (evaluation.gap + gap_scale)

    if component is BasicScoreComponent.ZERO_MODE_PRESENCE:
        if evaluation.zero_mode_count is None:
            raise ValueError("zero_mode_presence requires evaluation.zero_mode_count")
        return float(evaluation.zero_mode_count > 0)

    if component is BasicScoreComponent.MAXIMUM_IPR:
        if not evaluation.ipr:
            raise ValueError("maximum_ipr requires at least one evaluation.ipr value")
        return _maximum_unit_interval(evaluation.ipr.values(), name="maximum_ipr")

    if component is BasicScoreComponent.MAXIMUM_BOUNDARY_WEIGHT:
        if not evaluation.localization:
            raise ValueError(
                "maximum_boundary_weight requires at least one localization profile"
            )
        return _maximum_unit_interval(
            (profile.edge_weight for profile in evaluation.localization.values()),
            name="maximum_boundary_weight",
        )

    if component is BasicScoreComponent.MAXIMUM_MAJORANA_SELF_CONJUGACY:
        if not evaluation.majorana_metrics:
            raise ValueError(
                "maximum_majorana_self_conjugacy requires at least one Majorana diagnostic"
            )
        return _maximum_unit_interval(
            (
                diagnostics.self_conjugacy
                for diagnostics in evaluation.majorana_metrics.values()
            ),
            name="maximum_majorana_self_conjugacy",
        )

    if component is BasicScoreComponent.TOPOLOGICAL_METHOD_FRACTION:
        if not evaluation.topology:
            raise ValueError(
                "topological_method_fraction requires at least one topology result"
            )
        if any(result.is_topological is None for result in evaluation.topology):
            raise ValueError(
                "topological_method_fraction requires all topology results to be resolved"
            )
        topological_count = sum(
            result.is_topological is True for result in evaluation.topology
        )
        return topological_count / len(evaluation.topology)

    raise AssertionError(f"unsupported basic score component: {component!r}")


def _positive_weight_mapping(
    weights: Mapping[BasicScoreComponent, float],
) -> dict[BasicScoreComponent, float]:
    if not isinstance(weights, Mapping):
        raise TypeError("weights must be a mapping")
    if not weights:
        raise ValueError("weights must not be empty")

    prepared: dict[BasicScoreComponent, float] = {}
    for component, weight in weights.items():
        if not isinstance(component, BasicScoreComponent):
            raise TypeError("weight keys must be BasicScoreComponent members")
        prepared[component] = _positive_finite_real(
            weight,
            name=f"weight for {component.value}",
        )
    return prepared


def _unit_interval_mapping(
    values: Mapping[BasicScoreComponent, float],
    *,
    name: str,
) -> dict[BasicScoreComponent, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    prepared: dict[BasicScoreComponent, float] = {}
    for component, value in values.items():
        if not isinstance(component, BasicScoreComponent):
            raise TypeError(f"{name} keys must be BasicScoreComponent members")
        prepared[component] = _unit_interval_real(
            value,
            name=f"{name}[{component.value}]",
        )
    return prepared


def _finite_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


def _positive_finite_real(value: float, *, name: str) -> float:
    numeric_value = _finite_real(value, name=name)
    if numeric_value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric_value


def _unit_interval_real(value: float, *, name: str) -> float:
    numeric_value = _finite_real(value, name=name)
    if not 0.0 <= numeric_value <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return numeric_value


def _maximum_unit_interval(values: Iterable[float], *, name: str) -> float:
    prepared = tuple(
        _unit_interval_real(value, name=f"{name} input") for value in values
    )
    if not prepared:
        raise ValueError(f"{name} inputs must not be empty")
    return max(prepared)


def _warnings(values: tuple[str, ...]) -> tuple[str, ...]:
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


def _score_warnings(
    components: tuple[BasicScoreComponent, ...],
) -> tuple[str, ...]:
    warnings = [
        "This scalar score is an engineering convenience, not a physical observable or topological invariant.",
        "Scores are comparable only with identical components, weights, gap_scale, and input completeness.",
    ]
    selected = set(components)
    if BasicScoreComponent.ZERO_MODE_PRESENCE in selected:
        warnings.append(
            "Zero-mode presence alone does not establish Majorana character or non-trivial topology."
        )
    if selected.intersection(
        {
            BasicScoreComponent.MAXIMUM_IPR,
            BasicScoreComponent.MAXIMUM_BOUNDARY_WEIGHT,
            BasicScoreComponent.MAXIMUM_MAJORANA_SELF_CONJUGACY,
        }
    ):
        warnings.append(
            "Maximum state summaries can hide distributions, degeneracy-basis dependence, and differing maximizing states."
        )
    if BasicScoreComponent.MAXIMUM_IPR in selected:
        warnings.append(
            "Maximum IPR rewards concentration without identifying its physical location."
        )
    if BasicScoreComponent.TOPOLOGICAL_METHOD_FRACTION in selected:
        warnings.append(
            "The topology fraction does not replace the separate method results and assumptions."
        )
    return tuple(warnings)
