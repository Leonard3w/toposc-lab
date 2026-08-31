"""Deterministic execution of explicitly seeded disorder ensembles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from typing import Protocol, TypeAlias, runtime_checkable

from toposc_lab.evaluation import GeometryEvaluationRun
from toposc_lab.robustness.disorder import (
    DisorderProvenance,
    DisorderRealization,
    DisorderState,
)

DISORDER_ENSEMBLE_EXECUTION_VERSION = 1


@runtime_checkable
class DisorderOutcome(Protocol):
    """Read-only common surface of ordinary and specialized realizations."""

    @property
    def state(self) -> DisorderState:
        """Return the transformed scientific state."""
        ...

    @property
    def provenance(self) -> DisorderProvenance:
        """Return the immutable disorder provenance."""
        ...


DisorderRealizationFactory: TypeAlias = Callable[[int], DisorderOutcome]
DisorderEvaluationFactory: TypeAlias = Callable[
    [DisorderOutcome, int],
    GeometryEvaluationRun,
]


@dataclass(frozen=True, slots=True)
class DisorderEnsembleRequest:
    """Ordered explicit seeds defining one reproducible ensemble execution."""

    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.seeds, (str, bytes, bytearray)):
            raise TypeError("seeds must be an iterable of integers")
        try:
            seeds = tuple(
                _nonnegative_integer(seed, name="ensemble seed")
                for seed in self.seeds
            )
        except TypeError as error:
            if "not iterable" in str(error):
                raise TypeError("seeds must be an iterable of integers") from error
            raise
        if not seeds:
            raise ValueError("an ensemble must contain at least one seed")
        if len(set(seeds)) != len(seeds):
            raise ValueError("ensemble seeds must be unique")
        object.__setattr__(self, "seeds", seeds)


class DisorderEnsembleFailureStage(str, Enum):
    """Ensemble-owned callback stage that raised an ordinary exception."""

    DISORDER_REALIZATION = "disorder_realization"
    EVALUATION_CALLBACK = "evaluation_callback"


@dataclass(frozen=True, slots=True)
class DisorderEnsembleFailure:
    """Serializable callback failure without an exception or traceback."""

    stage: DisorderEnsembleFailureStage
    error_type: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.stage, DisorderEnsembleFailureStage):
            raise TypeError("stage must be DisorderEnsembleFailureStage")
        if not isinstance(self.error_type, str) or not self.error_type.isidentifier():
            raise ValueError("error_type must be a Python-style identifier")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("failure message must be a non-empty string")
        object.__setattr__(self, "message", self.message.strip())

    @classmethod
    def from_exception(
        cls,
        stage: DisorderEnsembleFailureStage,
        error: Exception,
    ) -> DisorderEnsembleFailure:
        """Create a stable record from a caught ordinary exception."""
        if not isinstance(stage, DisorderEnsembleFailureStage):
            raise TypeError("stage must be DisorderEnsembleFailureStage")
        if not isinstance(error, Exception):
            raise TypeError("error must be an Exception")
        message = str(error).strip() or "The callback raised without a message."
        return cls(
            stage=stage,
            error_type=type(error).__name__,
            message=message,
        )


@dataclass(frozen=True, slots=True)
class DisorderEnsembleMember:
    """Separated disorder, evaluation, and execution-failure data for one seed."""

    sample_index: int
    seed: int
    disorder: DisorderOutcome | None
    evaluation: GeometryEvaluationRun | None
    failure: DisorderEnsembleFailure | None

    def __post_init__(self) -> None:
        sample_index = _nonnegative_integer(
            self.sample_index,
            name="sample_index",
        )
        seed = _nonnegative_integer(self.seed, name="seed")
        if self.disorder is not None:
            _validate_disorder_outcome(self.disorder, expected_seed=seed)
        if self.evaluation is not None and not isinstance(
            self.evaluation,
            GeometryEvaluationRun,
        ):
            raise TypeError("evaluation must be GeometryEvaluationRun or None")
        if self.failure is not None and not isinstance(
            self.failure,
            DisorderEnsembleFailure,
        ):
            raise TypeError("failure must be DisorderEnsembleFailure or None")

        if self.disorder is None:
            if self.evaluation is not None:
                raise ValueError("evaluation requires a disorder outcome")
            if self.failure is None or self.failure.stage is not (
                DisorderEnsembleFailureStage.DISORDER_REALIZATION
            ):
                raise ValueError(
                    "missing disorder requires a realization-stage failure"
                )
        elif self.failure is not None:
            if self.failure.stage is not (
                DisorderEnsembleFailureStage.EVALUATION_CALLBACK
            ):
                raise ValueError(
                    "a retained disorder outcome only permits evaluation failure"
                )
            if self.evaluation is not None:
                raise ValueError("a failed evaluation callback cannot have an evaluation")

        object.__setattr__(self, "sample_index", sample_index)
        object.__setattr__(self, "seed", seed)


@dataclass(frozen=True, slots=True)
class DisorderEnsembleResult:
    """Ordered execution records without robustness or statistical aggregation."""

    request: DisorderEnsembleRequest
    members: tuple[DisorderEnsembleMember, ...]
    evaluations_requested: bool
    execution_version: int = field(
        default=DISORDER_ENSEMBLE_EXECUTION_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.request, DisorderEnsembleRequest):
            raise TypeError("request must be DisorderEnsembleRequest")
        members = tuple(self.members)
        if not all(isinstance(member, DisorderEnsembleMember) for member in members):
            raise TypeError("members must contain only DisorderEnsembleMember values")
        if not isinstance(self.evaluations_requested, bool):
            raise TypeError("evaluations_requested must be a boolean")
        if len(members) != len(self.request.seeds):
            raise ValueError("members must contain one record per requested seed")

        for sample_index, (member, seed) in enumerate(
            zip(members, self.request.seeds, strict=True)
        ):
            if member.sample_index != sample_index or member.seed != seed:
                raise ValueError(
                    "member indices and seeds must match request order exactly"
                )
            if self.evaluations_requested:
                if member.failure is None and member.evaluation is None:
                    raise ValueError(
                        "completed members require an evaluation when requested"
                    )
            elif member.evaluation is not None or (
                member.failure is not None
                and member.failure.stage is (
                    DisorderEnsembleFailureStage.EVALUATION_CALLBACK
                )
            ):
                raise ValueError(
                    "evaluation data is not allowed when evaluations were not requested"
                )

        _validate_common_disorder_definition(members)
        object.__setattr__(self, "members", members)


def execute_disorder_ensemble(
    request: DisorderEnsembleRequest,
    *,
    realization_factory: DisorderRealizationFactory,
    evaluation_factory: DisorderEvaluationFactory | None = None,
) -> DisorderEnsembleResult:
    """Execute one disorder definition sequentially in explicit seed order."""
    if not isinstance(request, DisorderEnsembleRequest):
        raise TypeError("request must be DisorderEnsembleRequest")
    if not callable(realization_factory):
        raise TypeError("realization_factory must be callable")
    if evaluation_factory is not None and not callable(evaluation_factory):
        raise TypeError("evaluation_factory must be callable or None")

    members: list[DisorderEnsembleMember] = []
    for sample_index, seed in enumerate(request.seeds):
        try:
            disorder = realization_factory(seed)
        except Exception as error:
            members.append(
                DisorderEnsembleMember(
                    sample_index=sample_index,
                    seed=seed,
                    disorder=None,
                    evaluation=None,
                    failure=DisorderEnsembleFailure.from_exception(
                        DisorderEnsembleFailureStage.DISORDER_REALIZATION,
                        error,
                    ),
                )
            )
            continue

        _validate_disorder_outcome(disorder, expected_seed=seed)
        _validate_outcome_against_previous(disorder, members=members)
        if evaluation_factory is None:
            members.append(
                DisorderEnsembleMember(
                    sample_index=sample_index,
                    seed=seed,
                    disorder=disorder,
                    evaluation=None,
                    failure=None,
                )
            )
            continue

        try:
            evaluation = evaluation_factory(disorder, seed)
        except Exception as error:
            members.append(
                DisorderEnsembleMember(
                    sample_index=sample_index,
                    seed=seed,
                    disorder=disorder,
                    evaluation=None,
                    failure=DisorderEnsembleFailure.from_exception(
                        DisorderEnsembleFailureStage.EVALUATION_CALLBACK,
                        error,
                    ),
                )
            )
            continue
        if not isinstance(evaluation, GeometryEvaluationRun):
            raise TypeError("evaluation_factory must return GeometryEvaluationRun")
        members.append(
            DisorderEnsembleMember(
                sample_index=sample_index,
                seed=seed,
                disorder=disorder,
                evaluation=evaluation,
                failure=None,
            )
        )

    return DisorderEnsembleResult(
        request=request,
        members=tuple(members),
        evaluations_requested=evaluation_factory is not None,
    )


def _validate_disorder_outcome(
    disorder: object,
    *,
    expected_seed: int,
) -> None:
    if not isinstance(disorder, DisorderOutcome):
        raise TypeError(
            "realization_factory must return an object implementing DisorderOutcome"
        )
    if disorder.provenance.seed != expected_seed:
        raise ValueError(
            "disorder provenance seed must match the requested ensemble seed"
        )
    DisorderRealization(
        state=disorder.state,
        provenance=disorder.provenance,
    )


def _validate_outcome_against_previous(
    disorder: DisorderOutcome,
    *,
    members: list[DisorderEnsembleMember],
) -> None:
    for member in members:
        if member.disorder is not None:
            if _disorder_definition(disorder.provenance) != _disorder_definition(
                member.disorder.provenance
            ):
                raise ValueError(
                    "all ensemble outcomes must share one source and disorder definition"
                )
            return


def _validate_common_disorder_definition(
    members: tuple[DisorderEnsembleMember, ...],
) -> None:
    reference: tuple[object, ...] | None = None
    for member in members:
        if member.disorder is None:
            continue
        definition = _disorder_definition(member.disorder.provenance)
        if reference is None:
            reference = definition
        elif definition != reference:
            raise ValueError(
                "all ensemble outcomes must share one source and disorder definition"
            )


def _disorder_definition(provenance: DisorderProvenance) -> tuple[object, ...]:
    return (
        provenance.disorder_key,
        provenance.disorder_version,
        provenance.parameters,
        provenance.rng_algorithm,
        provenance.source,
    )


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result
