"""Seed-explicit tournament selection from evaluated geometry populations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from toposc_lab.evaluation import (
    BasicScalarScore,
    MultiObjectiveEvaluation,
    ObjectiveDirection,
    ObjectiveSpec,
)
from toposc_lab.search.population_fitness import (
    MultiObjectiveFitnessDefinition,
    PopulationFitnessMember,
    PopulationFitnessResult,
    ScalarFitnessDefinition,
)

TOURNAMENT_SELECTION_VERSION = 1
TOURNAMENT_SELECTION_RNG_ALGORITHM = "numpy.random.PCG64"

_SELECTION_WARNINGS = (
    (
        "Selection is an engineering search decision, not scientific evidence "
        "or a discovery claim."
    ),
    (
        "Invalid and failed fitness members remain in the source ledger but are not "
        "eligible tournament contestants."
    ),
    (
        "Multi-objective tournaments use Pareto dominance without scalarization; "
        "incomparable nondominated contestants are resolved by the explicit RNG."
    ),
    "Tournament selection does not guarantee that any best member survives.",
    "This phase performs neither elitism, mutation, crossover, nor reproduction.",
)


@dataclass(frozen=True, slots=True)
class TournamentSelectionConfig:
    """Requested winner count and contestants per independent tournament."""

    selection_count: int
    tournament_size: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selection_count",
            _positive_integer(self.selection_count, name="selection_count"),
        )
        object.__setattr__(
            self,
            "tournament_size",
            _positive_integer(self.tournament_size, name="tournament_size"),
        )


@dataclass(frozen=True, slots=True)
class TournamentSelectionRecord:
    """One auditable tournament draw and its selected source member."""

    selection_index: int
    contestants: tuple[PopulationFitnessMember, ...]
    winner: PopulationFitnessMember

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selection_index",
            _nonnegative_integer(self.selection_index, name="selection_index"),
        )
        if isinstance(self.contestants, (str, bytes, bytearray)) or not isinstance(
            self.contestants,
            Iterable,
        ):
            raise TypeError("contestants must be an iterable of fitness members")
        contestants = tuple(self.contestants)
        if not contestants:
            raise ValueError("a tournament must contain at least one contestant")
        if not all(isinstance(item, PopulationFitnessMember) for item in contestants):
            raise TypeError("contestants must contain only PopulationFitnessMember values")
        if any(not item.is_available for item in contestants):
            raise ValueError("tournament contestants must have available fitness")
        if len({id(item) for item in contestants}) != len(contestants):
            raise ValueError("a tournament must not contain duplicate contestants")
        if not isinstance(self.winner, PopulationFitnessMember):
            raise TypeError("winner must be PopulationFitnessMember")
        if all(self.winner is not contestant for contestant in contestants):
            raise ValueError("winner must be one of the exact tournament contestants")
        object.__setattr__(self, "contestants", contestants)

    @property
    def contestant_member_indices(self) -> tuple[int, ...]:
        """Original population indices in their sampled tournament order."""
        return tuple(contestant.member_index for contestant in self.contestants)

    @property
    def winner_member_index(self) -> int:
        """Original population index of the selected member."""
        return self.winner.member_index


@dataclass(frozen=True, slots=True)
class PopulationSelectionResult:
    """Complete ordered tournament ledger without reproduction or elitism."""

    source: PopulationFitnessResult
    config: TournamentSelectionConfig
    seed: int
    records: tuple[TournamentSelectionRecord, ...]
    rng_algorithm: str = field(
        default=TOURNAMENT_SELECTION_RNG_ALGORITHM,
        init=False,
    )
    version: int = field(default=TOURNAMENT_SELECTION_VERSION, init=False)
    warnings: tuple[str, ...] = field(default=_SELECTION_WARNINGS, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, PopulationFitnessResult):
            raise TypeError("source must be PopulationFitnessResult")
        if not isinstance(self.config, TournamentSelectionConfig):
            raise TypeError("config must be TournamentSelectionConfig")
        seed = _nonnegative_integer(self.seed, name="seed")
        if isinstance(self.records, (str, bytes, bytearray)) or not isinstance(
            self.records,
            Iterable,
        ):
            raise TypeError("records must be an iterable of TournamentSelectionRecord")
        records = tuple(self.records)
        if len(records) != self.config.selection_count:
            raise ValueError("records must contain exactly config.selection_count values")
        if not all(isinstance(item, TournamentSelectionRecord) for item in records):
            raise TypeError("records must contain only TournamentSelectionRecord values")

        expected = _perform_tournaments(
            self.source,
            config=self.config,
            seed=seed,
        )
        for selection_index, (record, expected_record) in enumerate(
            zip(records, expected, strict=True)
        ):
            if record.selection_index != selection_index:
                raise ValueError("selection indices must match record order exactly")
            if tuple(map(id, record.contestants)) != tuple(
                map(id, expected_record.contestants)
            ):
                raise ValueError("record contestants do not match the seeded draw")
            if record.winner is not expected_record.winner:
                raise ValueError("record winner does not match the selection rule")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "records", records)

    @property
    def selected_members(self) -> tuple[PopulationFitnessMember, ...]:
        """Selected source-member references in selection-output order."""
        return tuple(record.winner for record in self.records)


def select_population_members(
    source: PopulationFitnessResult,
    *,
    config: TournamentSelectionConfig,
    seed: int,
) -> PopulationSelectionResult:
    """Run independent seeded tournaments over available fitness members.

    Contestants are sampled without replacement within each tournament.
    Winners are selected with replacement across tournaments. Scalar ties and
    multiple Pareto-nondominated contestants are resolved uniformly through
    the one local PCG64 stream.
    """
    if not isinstance(source, PopulationFitnessResult):
        raise TypeError("source must be PopulationFitnessResult")
    if not isinstance(config, TournamentSelectionConfig):
        raise TypeError("config must be TournamentSelectionConfig")
    prepared_seed = _nonnegative_integer(seed, name="seed")
    records = _perform_tournaments(
        source,
        config=config,
        seed=prepared_seed,
    )
    return PopulationSelectionResult(
        source=source,
        config=config,
        seed=prepared_seed,
        records=records,
    )


def _perform_tournaments(
    source: PopulationFitnessResult,
    *,
    config: TournamentSelectionConfig,
    seed: int,
) -> tuple[TournamentSelectionRecord, ...]:
    available = source.available_members
    if not available:
        raise ValueError("selection requires at least one available fitness member")
    if config.tournament_size > len(available):
        raise ValueError(
            "tournament_size must not exceed the number of available fitness members"
        )

    random_number_generator = np.random.Generator(np.random.PCG64(seed))
    records: list[TournamentSelectionRecord] = []
    for selection_index in range(config.selection_count):
        sampled_positions = random_number_generator.choice(
            len(available),
            size=config.tournament_size,
            replace=False,
        )
        contestants = tuple(available[int(position)] for position in sampled_positions)
        best = _best_contestants(contestants, source=source)
        winner_position = int(random_number_generator.integers(len(best)))
        records.append(
            TournamentSelectionRecord(
                selection_index=selection_index,
                contestants=contestants,
                winner=best[winner_position],
            )
        )
    return tuple(records)


def _best_contestants(
    contestants: tuple[PopulationFitnessMember, ...],
    *,
    source: PopulationFitnessResult,
) -> tuple[PopulationFitnessMember, ...]:
    definition = source.definition
    if isinstance(definition, ScalarFitnessDefinition):
        values = tuple(_scalar_fitness_value(member) for member in contestants)
        target = (
            max(values)
            if definition.direction is ObjectiveDirection.MAXIMIZE
            else min(values)
        )
        return tuple(
            member
            for member, value in zip(contestants, values, strict=True)
            if value == target
        )

    assert isinstance(definition, MultiObjectiveFitnessDefinition)
    return tuple(
        candidate
        for candidate in contestants
        if not any(
            other is not candidate
            and _pareto_dominates(
                other,
                candidate,
                objectives=definition.objectives,
            )
            for other in contestants
        )
    )


def _scalar_fitness_value(member: PopulationFitnessMember) -> float:
    fitness = member.fitness
    if not isinstance(fitness, BasicScalarScore):
        raise TypeError("scalar selection requires BasicScalarScore fitness")
    return fitness.value


def _pareto_dominates(
    first: PopulationFitnessMember,
    second: PopulationFitnessMember,
    *,
    objectives: tuple[ObjectiveSpec, ...],
) -> bool:
    first_fitness = first.fitness
    second_fitness = second.fitness
    if not isinstance(first_fitness, MultiObjectiveEvaluation) or not isinstance(
        second_fitness,
        MultiObjectiveEvaluation,
    ):
        raise TypeError("multi-objective selection requires MultiObjectiveEvaluation fitness")

    no_worse = True
    strictly_better = False
    for objective in objectives:
        first_value = first_fitness.objectives[objective.name].value
        second_value = second_fitness.objectives[objective.name].value
        if objective.direction is ObjectiveDirection.MAXIMIZE:
            no_worse = no_worse and first_value >= second_value
            strictly_better = strictly_better or first_value > second_value
        else:
            no_worse = no_worse and first_value <= second_value
            strictly_better = strictly_better or first_value < second_value
    return no_worse and strictly_better


def _positive_integer(value: object, *, name: str) -> int:
    result = _nonnegative_integer(value, name=name)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result
