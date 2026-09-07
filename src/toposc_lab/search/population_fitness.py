"""Failure-aware fitness evaluation for ordered geometry populations."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import TypeAlias

from toposc_lab.evaluation import (
    BasicScalarScore,
    BasicScoreComponent,
    GeometryEvaluationRun,
    MultiObjectiveEvaluation,
    ObjectiveDirection,
    ObjectiveSpec,
    compute_basic_scalar_score,
    evaluate_multi_objectives,
)
from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.search.initial_population import (
    InitialPopulation,
    InitialPopulationMember,
)

POPULATION_FITNESS_VERSION = 1

_POPULATION_FITNESS_WARNINGS = (
    (
        "Fitness values are engineering search signals, not physical observables, "
        "topological invariants, or scientific evidence by themselves."
    ),
    (
        "Fitness values are directly comparable only under the same definition and "
        "evaluation protocol."
    ),
    (
        "Invalid evaluations and execution failures remain without fitness; they are "
        "not silently replaced by zero or a worst-case value."
    ),
    "This result defines neither ranking, selection, elitism, nor reproduction.",
)


@dataclass(frozen=True, slots=True)
class ScalarFitnessDefinition:
    """One predeclared weighted engineering-score construction."""

    weights: Mapping[BasicScoreComponent, float]
    direction: ObjectiveDirection
    gap_scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.weights, Mapping):
            raise TypeError("weights must be a mapping")
        if not self.weights:
            raise ValueError("weights must not be empty")
        weights: dict[BasicScoreComponent, float] = {}
        for component, weight in self.weights.items():
            if not isinstance(component, BasicScoreComponent):
                raise TypeError("weight keys must be BasicScoreComponent members")
            weights[component] = _positive_finite_real(
                weight,
                name=f"weight for {component.value}",
            )
        if not isinstance(self.direction, ObjectiveDirection):
            raise TypeError("direction must be ObjectiveDirection")
        object.__setattr__(self, "weights", MappingProxyType(weights))
        object.__setattr__(
            self,
            "gap_scale",
            _positive_finite_real(self.gap_scale, name="gap_scale"),
        )


@dataclass(frozen=True, slots=True)
class MultiObjectiveFitnessDefinition:
    """One predeclared ordered vector of separate raw objectives."""

    objectives: tuple[ObjectiveSpec, ...]

    def __post_init__(self) -> None:
        if isinstance(self.objectives, (str, bytes, bytearray)) or not isinstance(
            self.objectives,
            Iterable,
        ):
            raise TypeError("objectives must be an iterable of ObjectiveSpec values")
        objectives = tuple(self.objectives)
        if not objectives:
            raise ValueError("objectives must not be empty")
        if not all(isinstance(objective, ObjectiveSpec) for objective in objectives):
            raise TypeError("objectives must contain only ObjectiveSpec values")
        names = tuple(objective.name for objective in objectives)
        if len(set(names)) != len(names):
            raise ValueError("objective names must be unique")
        object.__setattr__(self, "objectives", objectives)


PopulationFitnessDefinition: TypeAlias = (
    ScalarFitnessDefinition | MultiObjectiveFitnessDefinition
)
PopulationFitness: TypeAlias = BasicScalarScore | MultiObjectiveEvaluation


class PopulationFitnessStatus(str, Enum):
    """Availability state for one population member's fitness."""

    AVAILABLE = "available"
    INVALID_EVALUATION = "invalid_evaluation"
    EVALUATION_FAILURE = "evaluation_failure"
    FITNESS_CONSTRUCTION_FAILURE = "fitness_construction_failure"


class PopulationFitnessFailureStage(str, Enum):
    """Callback or construction stage that raised an ordinary exception."""

    EVALUATION_CALLBACK = "evaluation_callback"
    FITNESS_CONSTRUCTION = "fitness_construction"


@dataclass(frozen=True, slots=True)
class PopulationFitnessFailure:
    """Serializable failure information without an exception or traceback."""

    stage: PopulationFitnessFailureStage
    error_type: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.stage, PopulationFitnessFailureStage):
            raise TypeError("stage must be PopulationFitnessFailureStage")
        if not isinstance(self.error_type, str) or not self.error_type.isidentifier():
            raise ValueError("error_type must be a Python identifier")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("failure message must be non-empty")
        object.__setattr__(self, "message", self.message.strip())

    @classmethod
    def from_exception(
        cls,
        stage: PopulationFitnessFailureStage,
        error: Exception,
    ) -> PopulationFitnessFailure:
        """Create a stable failure record from an ordinary exception."""
        if not isinstance(stage, PopulationFitnessFailureStage):
            raise TypeError("stage must be PopulationFitnessFailureStage")
        if not isinstance(error, Exception):
            raise TypeError("error must be an Exception")
        message = str(error).strip() or "The stage raised without a message."
        return cls(
            stage=stage,
            error_type=type(error).__name__,
            message=message,
        )


@dataclass(frozen=True, slots=True)
class PopulationFitnessMember:
    """One source population member and its retained fitness outcome."""

    population_member: InitialPopulationMember
    status: PopulationFitnessStatus
    evaluation: GeometryEvaluationRun | None
    fitness: PopulationFitness | None
    failure: PopulationFitnessFailure | None

    def __post_init__(self) -> None:
        if not isinstance(self.population_member, InitialPopulationMember):
            raise TypeError("population_member must be InitialPopulationMember")
        if not isinstance(self.status, PopulationFitnessStatus):
            raise TypeError("status must be PopulationFitnessStatus")
        if self.evaluation is not None and not isinstance(
            self.evaluation,
            GeometryEvaluationRun,
        ):
            raise TypeError("evaluation must be GeometryEvaluationRun or None")
        if self.fitness is not None and not isinstance(
            self.fitness,
            (BasicScalarScore, MultiObjectiveEvaluation),
        ):
            raise TypeError("fitness has an unsupported result type")
        if self.failure is not None and not isinstance(
            self.failure,
            PopulationFitnessFailure,
        ):
            raise TypeError("failure must be PopulationFitnessFailure or None")
        _validate_member_state(self)

    @property
    def member_index(self) -> int:
        """Original generation-zero index retained without renumbering."""
        return self.population_member.member_index

    @property
    def is_available(self) -> bool:
        """Whether this member has a completed fitness payload."""
        return self.status is PopulationFitnessStatus.AVAILABLE


@dataclass(frozen=True, slots=True)
class PopulationFitnessResult:
    """Complete population-order ledger without ranking or selection."""

    population: InitialPopulation
    definition: PopulationFitnessDefinition
    members: tuple[PopulationFitnessMember, ...]
    version: int = field(default=POPULATION_FITNESS_VERSION, init=False)
    warnings: tuple[str, ...] = field(
        default=_POPULATION_FITNESS_WARNINGS,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.population, InitialPopulation):
            raise TypeError("population must be InitialPopulation")
        _require_fitness_definition(self.definition)
        if isinstance(self.members, (str, bytes, bytearray)) or not isinstance(
            self.members,
            Iterable,
        ):
            raise TypeError("members must be an iterable of PopulationFitnessMember")
        members = tuple(self.members)
        if len(members) != self.population.population_size:
            raise ValueError("members must retain every initial population member")
        if not all(isinstance(member, PopulationFitnessMember) for member in members):
            raise TypeError("members must contain only PopulationFitnessMember values")
        for member, source_member in zip(
            members,
            self.population.members,
            strict=True,
        ):
            if member.population_member is not source_member:
                raise ValueError("fitness members must preserve population order and identity")
            if member.evaluation is not None:
                _validate_evaluation_correspondence(
                    member.evaluation,
                    population_member=source_member,
                )
            if member.fitness is not None:
                _validate_fitness_against_definition(
                    member.fitness,
                    definition=self.definition,
                )
        object.__setattr__(self, "members", members)

    @property
    def available_members(self) -> tuple[PopulationFitnessMember, ...]:
        """Fitness-bearing members in original population order."""
        return tuple(member for member in self.members if member.is_available)


PopulationEvaluator: TypeAlias = Callable[
    [InitialPopulationMember],
    GeometryEvaluationRun,
]


def evaluate_population_fitness(
    population: InitialPopulation,
    *,
    definition: PopulationFitnessDefinition,
    evaluator: PopulationEvaluator,
) -> PopulationFitnessResult:
    """Evaluate and construct fitness once per member in population order.

    Ordinary evaluator and fitness-construction exceptions are retained as
    member failures and later members continue. Wrong callback return types,
    mismatched reproducibility identities, and malformed definitions are API
    contract errors and raise without producing a partial result.
    """
    if not isinstance(population, InitialPopulation):
        raise TypeError("population must be InitialPopulation")
    _require_fitness_definition(definition)
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")

    members: list[PopulationFitnessMember] = []
    for population_member in population.members:
        try:
            evaluation = evaluator(population_member)
        except Exception as error:  # noqa: BLE001 - retain per-member callback failures
            members.append(
                PopulationFitnessMember(
                    population_member=population_member,
                    status=PopulationFitnessStatus.EVALUATION_FAILURE,
                    evaluation=None,
                    fitness=None,
                    failure=PopulationFitnessFailure.from_exception(
                        PopulationFitnessFailureStage.EVALUATION_CALLBACK,
                        error,
                    ),
                )
            )
            continue
        if not isinstance(evaluation, GeometryEvaluationRun):
            raise TypeError("evaluator must return GeometryEvaluationRun")
        _validate_evaluation_correspondence(
            evaluation,
            population_member=population_member,
        )
        if not evaluation.is_valid:
            members.append(
                PopulationFitnessMember(
                    population_member=population_member,
                    status=PopulationFitnessStatus.INVALID_EVALUATION,
                    evaluation=evaluation,
                    fitness=None,
                    failure=None,
                )
            )
            continue

        try:
            fitness = _construct_fitness(evaluation, definition=definition)
        except Exception as error:  # noqa: BLE001 - retain per-member fitness failures
            members.append(
                PopulationFitnessMember(
                    population_member=population_member,
                    status=PopulationFitnessStatus.FITNESS_CONSTRUCTION_FAILURE,
                    evaluation=evaluation,
                    fitness=None,
                    failure=PopulationFitnessFailure.from_exception(
                        PopulationFitnessFailureStage.FITNESS_CONSTRUCTION,
                        error,
                    ),
                )
            )
            continue
        members.append(
            PopulationFitnessMember(
                population_member=population_member,
                status=PopulationFitnessStatus.AVAILABLE,
                evaluation=evaluation,
                fitness=fitness,
                failure=None,
            )
        )

    return PopulationFitnessResult(
        population=population,
        definition=definition,
        members=tuple(members),
    )


def _construct_fitness(
    evaluation: GeometryEvaluationRun,
    *,
    definition: PopulationFitnessDefinition,
) -> PopulationFitness:
    scientific_evaluation = evaluation.evaluation
    if scientific_evaluation is None:
        raise ValueError("a valid evaluation run must contain a scientific evaluation")
    if isinstance(definition, ScalarFitnessDefinition):
        return compute_basic_scalar_score(
            scientific_evaluation,
            weights=definition.weights,
            gap_scale=definition.gap_scale,
        )
    return evaluate_multi_objectives(
        scientific_evaluation,
        objectives=definition.objectives,
    )


def _require_fitness_definition(value: object) -> None:
    if not isinstance(
        value,
        (ScalarFitnessDefinition, MultiObjectiveFitnessDefinition),
    ):
        raise TypeError(
            "definition must be ScalarFitnessDefinition or "
            "MultiObjectiveFitnessDefinition"
        )


def _validate_member_state(member: PopulationFitnessMember) -> None:
    if member.status is PopulationFitnessStatus.AVAILABLE:
        if (
            member.evaluation is None
            or not member.evaluation.is_valid
            or member.fitness is None
            or member.failure is not None
        ):
            raise ValueError("available fitness requires a valid evaluation and fitness")
        return
    if member.status is PopulationFitnessStatus.INVALID_EVALUATION:
        if (
            member.evaluation is None
            or member.evaluation.is_valid
            or member.fitness is not None
            or member.failure is not None
        ):
            raise ValueError("invalid-evaluation status requires only an invalid run")
        return
    if member.status is PopulationFitnessStatus.EVALUATION_FAILURE:
        if (
            member.evaluation is not None
            or member.fitness is not None
            or member.failure is None
            or member.failure.stage
            is not PopulationFitnessFailureStage.EVALUATION_CALLBACK
        ):
            raise ValueError("evaluation-failure status has inconsistent payloads")
        return
    if (
        member.evaluation is None
        or not member.evaluation.is_valid
        or member.fitness is not None
        or member.failure is None
        or member.failure.stage
        is not PopulationFitnessFailureStage.FITNESS_CONSTRUCTION
    ):
        raise ValueError("fitness-construction-failure status has inconsistent payloads")


def _validate_evaluation_correspondence(
    evaluation: GeometryEvaluationRun,
    *,
    population_member: InitialPopulationMember,
) -> None:
    reproducibility = evaluation.reproducibility
    if reproducibility is None:
        return
    expected_geometry_id = exact_geometry_id(population_member.genome.to_geometry())
    if reproducibility.geometry_id != expected_geometry_id:
        raise ValueError(
            "evaluation reproducibility geometry_id does not match the population member"
        )


def _validate_fitness_against_definition(
    fitness: PopulationFitness,
    *,
    definition: PopulationFitnessDefinition,
) -> None:
    if isinstance(definition, ScalarFitnessDefinition):
        if not isinstance(fitness, BasicScalarScore):
            raise TypeError("scalar fitness definition requires BasicScalarScore results")
        if tuple(fitness.components) != tuple(definition.weights):
            raise ValueError("scalar fitness components do not match the definition")
        total_weight = sum(definition.weights.values())
        expected_weights = {
            component: weight / total_weight
            for component, weight in definition.weights.items()
        }
        if any(
            not math.isclose(
                fitness.normalized_weights[component],
                expected_weight,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            for component, expected_weight in expected_weights.items()
        ) or not math.isclose(
            fitness.gap_scale,
            definition.gap_scale,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("scalar fitness normalization does not match the definition")
        return

    if not isinstance(fitness, MultiObjectiveEvaluation):
        raise TypeError(
            "multi-objective fitness definition requires MultiObjectiveEvaluation results"
        )
    actual_specs = tuple(objective.spec for objective in fitness.objectives.values())
    if actual_specs != definition.objectives:
        raise ValueError("multi-objective fitness specifications do not match the definition")


def _positive_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result
