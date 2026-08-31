"""Explicit success-fraction metrics for completed disorder ensembles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Integral
from typing import TypeAlias

from toposc_lab.robustness.ensemble import (
    DisorderEnsembleMember,
    DisorderEnsembleRequest,
    DisorderEnsembleResult,
)

RobustnessSuccessPredicate: TypeAlias = Callable[[DisorderEnsembleMember], bool]


@dataclass(frozen=True, slots=True)
class RobustnessSuccessCriterion:
    """Explicit scientific definition of success for one ensemble member."""

    key: str
    description: str
    predicate: RobustnessSuccessPredicate = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _criterion_key(self.key))
        object.__setattr__(
            self,
            "description",
            _criterion_description(self.description),
        )
        if not callable(self.predicate):
            raise TypeError("predicate must be callable")

    def evaluate(self, member: DisorderEnsembleMember) -> bool:
        """Evaluate the criterion and require an actual boolean decision."""
        if not isinstance(member, DisorderEnsembleMember):
            raise TypeError("member must be DisorderEnsembleMember")
        result = self.predicate(member)
        if not isinstance(result, bool):
            raise TypeError("robustness success predicate must return a boolean")
        return result


@dataclass(frozen=True, slots=True)
class RobustnessFractionMetric:
    """Auditable per-seed decisions for ``N_successful / N_total``."""

    criterion_key: str
    criterion_description: str
    request: DisorderEnsembleRequest
    successes: tuple[bool, ...]
    execution_failure_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        criterion_key = _criterion_key(self.criterion_key)
        criterion_description = _criterion_description(
            self.criterion_description
        )
        if not isinstance(self.request, DisorderEnsembleRequest):
            raise TypeError("request must be DisorderEnsembleRequest")
        successes = tuple(self.successes)
        if any(not isinstance(success, bool) for success in successes):
            raise TypeError("successes must contain only booleans")
        if len(successes) != len(self.request.seeds):
            raise ValueError("successes must contain one decision per requested seed")

        failure_indices = tuple(
            _nonnegative_integer(index, name="execution failure index")
            for index in self.execution_failure_indices
        )
        if failure_indices != tuple(sorted(set(failure_indices))):
            raise ValueError(
                "execution_failure_indices must be unique and ascending"
            )
        if any(index >= len(successes) for index in failure_indices):
            raise ValueError("execution failure index is outside the ensemble")
        if any(successes[index] for index in failure_indices):
            raise ValueError("an execution failure cannot be marked successful")

        object.__setattr__(self, "criterion_key", criterion_key)
        object.__setattr__(
            self,
            "criterion_description",
            criterion_description,
        )
        object.__setattr__(self, "successes", successes)
        object.__setattr__(
            self,
            "execution_failure_indices",
            failure_indices,
        )

    @property
    def successful_count(self) -> int:
        """Number of members satisfying the explicit criterion."""
        return sum(self.successes)

    @property
    def total_count(self) -> int:
        """Number of requested disorder realizations in the denominator."""
        return len(self.successes)

    @property
    def value(self) -> float:
        """Return the exact empirical success fraction."""
        return self.successful_count / self.total_count

    @property
    def successful_seeds(self) -> tuple[int, ...]:
        """Seeds whose completed members satisfied the criterion."""
        return tuple(
            seed
            for seed, success in zip(
                self.request.seeds,
                self.successes,
                strict=True,
            )
            if success
        )

    @property
    def unsuccessful_seeds(self) -> tuple[int, ...]:
        """Seeds that failed execution or did not satisfy the criterion."""
        return tuple(
            seed
            for seed, success in zip(
                self.request.seeds,
                self.successes,
                strict=True,
            )
            if not success
        )

    @property
    def execution_failure_seeds(self) -> tuple[int, ...]:
        """Operationally failed seeds, retained as a subset of unsuccessful seeds."""
        return tuple(
            self.request.seeds[index]
            for index in self.execution_failure_indices
        )


def compute_robustness_fraction(
    ensemble: DisorderEnsembleResult,
    *,
    criterion: RobustnessSuccessCriterion,
) -> RobustnessFractionMetric:
    """Compute an explicit empirical success fraction over all requested seeds."""
    if not isinstance(ensemble, DisorderEnsembleResult):
        raise TypeError("ensemble must be DisorderEnsembleResult")
    if not isinstance(criterion, RobustnessSuccessCriterion):
        raise TypeError("criterion must be RobustnessSuccessCriterion")

    successes: list[bool] = []
    execution_failure_indices: list[int] = []
    for member in ensemble.members:
        if member.failure is not None:
            successes.append(False)
            execution_failure_indices.append(member.sample_index)
            continue
        successes.append(criterion.evaluate(member))

    return RobustnessFractionMetric(
        criterion_key=criterion.key,
        criterion_description=criterion.description,
        request=ensemble.request,
        successes=tuple(successes),
        execution_failure_indices=tuple(execution_failure_indices),
    )


def _criterion_key(value: str) -> str:
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError("criterion key must be a Python-style identifier")
    return value


def _criterion_description(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("criterion description must be a non-empty string")
    return value.strip()


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result
