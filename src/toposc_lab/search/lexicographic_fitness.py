"""Explicit ordered fitness from separately derived scientific quantities.

Derivations belong to a versioned evaluator, never to geometry metadata. Values
remain raw; the direction-adjusted tuple is used only for ordering.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from math import isfinite
from numbers import Real
from types import MappingProxyType

from toposc_lab.evaluation import GeometryEvaluationRun, ObjectiveDirection


@dataclass(frozen=True, slots=True, kw_only=True)
class DerivedEvaluationRun(GeometryEvaluationRun):
    """A complete pipeline result plus named, versioned scientific derivations."""

    derivation_identifier: str
    derived_quantities: Mapping[str, float]

    def __post_init__(self) -> None:
        GeometryEvaluationRun.__post_init__(self)
        _identifier(self.derivation_identifier)
        if not isinstance(self.derived_quantities, Mapping):
            raise TypeError("derived_quantities must be a mapping")
        values: dict[str, float] = {}
        for name, value in self.derived_quantities.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("derived quantity names must be identifiers")
            values[name] = _finite_real(value)
        if values and not self.is_valid:
            raise ValueError("invalid evaluations cannot supply derived fitness quantities")
        object.__setattr__(self, "derived_quantities", MappingProxyType(values))

    @classmethod
    def from_run(
        cls, run: GeometryEvaluationRun, *, identifier: str, quantities: Mapping[str, float]
    ) -> DerivedEvaluationRun:
        """Retain the exact original evaluation objects without editing observables."""
        return cls(
            **{item.name: getattr(run, item.name) for item in fields(GeometryEvaluationRun)},
            derivation_identifier=identifier,
            derived_quantities=quantities,
        )


@dataclass(frozen=True, slots=True)
class LexicographicFitnessDefinition:
    """Highest-priority-first raw quantity names, directions and derivation identity."""

    quantities: tuple[str, ...]
    directions: tuple[ObjectiveDirection, ...]
    derivation_identifier: str
    version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        quantities, directions = tuple(self.quantities), tuple(self.directions)
        if not quantities or any(
            not isinstance(n, str) or not n.isidentifier() for n in quantities
        ):
            raise ValueError("quantities must contain nonempty identifier names")
        if len(set(quantities)) != len(quantities):
            raise ValueError("quantities must be distinct")
        if len(directions) != len(quantities) or any(
            not isinstance(d, ObjectiveDirection) for d in directions
        ):
            raise ValueError("one ObjectiveDirection is required per quantity")
        _identifier(self.derivation_identifier)
        object.__setattr__(self, "quantities", quantities)
        object.__setattr__(self, "directions", directions)


@dataclass(frozen=True, slots=True)
class LexicographicFitness:
    """Finite raw values bound to their exact ordered definition; no scalarization."""

    definition: LexicographicFitnessDefinition
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.definition, LexicographicFitnessDefinition):
            raise TypeError("definition must be LexicographicFitnessDefinition")
        values = tuple(_finite_real(value) for value in self.values)
        if len(values) != len(self.definition.quantities):
            raise ValueError("values must match the ordered definition")
        object.__setattr__(self, "values", values)

    @property
    def ordering_key(self) -> tuple[float, ...]:
        """Lexicographic key with larger preferred; raw values stay unchanged."""
        return tuple(
            value if direction is ObjectiveDirection.MAXIMIZE else -value
            for value, direction in zip(self.values, self.definition.directions, strict=True)
        )


def construct_lexicographic_fitness(
    run: GeometryEvaluationRun, definition: LexicographicFitnessDefinition
) -> LexicographicFitness:
    """Require a truthful matching derivation and every requested finite quantity."""
    if not isinstance(run, DerivedEvaluationRun) or not run.is_valid:
        raise ValueError("lexicographic fitness requires a valid DerivedEvaluationRun")
    if run.derivation_identifier != definition.derivation_identifier:
        raise ValueError("scientific derivation identifier differs from fitness definition")
    return LexicographicFitness(
        definition, tuple(run.derived_quantities[name] for name in definition.quantities)
    )


def _finite_real(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("derived fitness values must be explicit real numbers, not booleans")
    if not isfinite(value):
        raise ValueError("derived fitness values must be finite")
    return float(value)


def _identifier(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("derivation_identifier must be a nonempty versioned label")
