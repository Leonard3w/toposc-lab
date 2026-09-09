"""Deterministic, tie-preserving elitism for evaluated geometry populations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Integral

from toposc_lab.evaluation import (
    BasicScalarScore,
    MultiObjectiveEvaluation,
    ObjectiveDirection,
    ObjectiveSpec,
)
from toposc_lab.search.lexicographic_fitness import (
    LexicographicFitness,
    LexicographicFitnessDefinition,
)
from toposc_lab.search.population_fitness import (
    MultiObjectiveFitnessDefinition,
    PopulationFitnessMember,
    PopulationFitnessResult,
    ScalarFitnessDefinition,
)

POPULATION_ELITISM_VERSION = 1

_ELITISM_WARNINGS = (
    (
        "Elite retention is an engineering search decision, not scientific evidence "
        "or a discovery claim."
    ),
    (
        "Invalid and failed fitness members remain in the source ledger but cannot "
        "enter an elite tier."
    ),
    (
        "Complete scalar tie tiers and Pareto fronts are retained at the cutoff, so "
        "the actual elite count may exceed the configured minimum."
    ),
    (
        "Multi-objective tiers use Pareto dominance without scalarization, weighting, "
        "or lexicographic priority."
    ),
    (
        "This phase records a retention contract but performs neither population "
        "construction, crossover, mutation, nor a generation loop."
    ),
)


@dataclass(frozen=True, slots=True)
class ElitismConfig:
    """Minimum retained count before completing the cutoff fitness tier."""

    minimum_elite_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_elite_count",
            _positive_integer(
                self.minimum_elite_count,
                name="minimum_elite_count",
            ),
        )


@dataclass(frozen=True, slots=True)
class EliteTier:
    """One complete scalar-equality tier or Pareto front in source order."""

    tier_index: int
    members: tuple[PopulationFitnessMember, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tier_index",
            _nonnegative_integer(self.tier_index, name="tier_index"),
        )
        if isinstance(self.members, (str, bytes, bytearray)) or not isinstance(
            self.members,
            Iterable,
        ):
            raise TypeError("members must be an iterable of fitness members")
        members = tuple(self.members)
        if not members:
            raise ValueError("an elite tier must contain at least one member")
        if not all(isinstance(member, PopulationFitnessMember) for member in members):
            raise TypeError("members must contain only PopulationFitnessMember values")
        if any(not member.is_available for member in members):
            raise ValueError("elite tier members must have available fitness")
        if len({id(member) for member in members}) != len(members):
            raise ValueError("an elite tier must not contain duplicate members")
        object.__setattr__(self, "members", members)

    @property
    def member_indices(self) -> tuple[int, ...]:
        """Original population indices in stable source order."""
        return tuple(member.member_index for member in self.members)


@dataclass(frozen=True, slots=True)
class PopulationElitismResult:
    """Auditable elite-retention contract without next-generation assembly."""

    source: PopulationFitnessResult
    config: ElitismConfig
    tiers: tuple[EliteTier, ...]
    version: int = field(default=POPULATION_ELITISM_VERSION, init=False)
    warnings: tuple[str, ...] = field(default=_ELITISM_WARNINGS, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, PopulationFitnessResult):
            raise TypeError("source must be PopulationFitnessResult")
        if not isinstance(self.config, ElitismConfig):
            raise TypeError("config must be ElitismConfig")
        if isinstance(self.tiers, (str, bytes, bytearray)) or not isinstance(
            self.tiers,
            Iterable,
        ):
            raise TypeError("tiers must be an iterable of EliteTier values")
        tiers = tuple(self.tiers)
        if not tiers:
            raise ValueError("tiers must contain at least one EliteTier")
        if not all(isinstance(tier, EliteTier) for tier in tiers):
            raise TypeError("tiers must contain only EliteTier values")
        expected = _select_elite_tiers(self.source, config=self.config)
        if len(tiers) != len(expected):
            raise ValueError("tiers must contain every selected fitness tier")
        for tier_index, (tier, expected_tier) in enumerate(
            zip(tiers, expected, strict=True)
        ):
            if tier.tier_index != tier_index:
                raise ValueError("tier indices must match tier order exactly")
            if tuple(map(id, tier.members)) != tuple(map(id, expected_tier.members)):
                raise ValueError("elite tier members do not match the fitness contract")
        object.__setattr__(self, "tiers", tiers)

    @property
    def elite_members(self) -> tuple[PopulationFitnessMember, ...]:
        """Distinct elite source-member references ordered by tier and source."""
        return tuple(member for tier in self.tiers for member in tier.members)

    @property
    def elite_count(self) -> int:
        """Actual count after retaining the complete cutoff tier."""
        return sum(len(tier.members) for tier in self.tiers)

    @property
    def cutoff_was_expanded(self) -> bool:
        """Whether complete-tier retention exceeded the configured minimum."""
        return self.elite_count > self.config.minimum_elite_count


def identify_population_elites(
    source: PopulationFitnessResult,
    *,
    config: ElitismConfig,
) -> PopulationElitismResult:
    """Identify complete leading fitness tiers without constructing survivors.

    Scalar fitness produces exact-equality tiers ordered by the definition's
    direction. Multi-objective fitness produces successive Pareto fronts. Full
    tiers are included until the configured minimum is reached, so no arbitrary
    order or random rule splits an equal or incomparable cutoff tier.
    """
    if not isinstance(source, PopulationFitnessResult):
        raise TypeError("source must be PopulationFitnessResult")
    if not isinstance(config, ElitismConfig):
        raise TypeError("config must be ElitismConfig")
    tiers = _select_elite_tiers(source, config=config)
    return PopulationElitismResult(source=source, config=config, tiers=tiers)


def _select_elite_tiers(
    source: PopulationFitnessResult,
    *,
    config: ElitismConfig,
) -> tuple[EliteTier, ...]:
    available = source.available_members
    if not available:
        raise ValueError("elitism requires at least one available fitness member")
    if config.minimum_elite_count > len(available):
        raise ValueError(
            "minimum_elite_count must not exceed the number of available fitness "
            "members"
        )

    all_tiers = _fitness_tiers(available, source=source)
    selected: list[EliteTier] = []
    selected_count = 0
    for tier in all_tiers:
        selected.append(tier)
        selected_count += len(tier.members)
        if selected_count >= config.minimum_elite_count:
            break
    return tuple(selected)


def _fitness_tiers(
    members: tuple[PopulationFitnessMember, ...],
    *,
    source: PopulationFitnessResult,
) -> tuple[EliteTier, ...]:
    definition = source.definition
    if isinstance(definition, LexicographicFitnessDefinition):
        groups: dict[tuple[float, ...], list[PopulationFitnessMember]] = {}
        for member in members:
            if not isinstance(member.fitness, LexicographicFitness):
                raise TypeError("lexicographic elitism requires ordered raw fitness")
            groups.setdefault(member.fitness.ordering_key, []).append(member)
        return tuple(
            EliteTier(index, tuple(groups[key]))
            for index, key in enumerate(sorted(groups, reverse=True))
        )
    if isinstance(definition, ScalarFitnessDefinition):
        ordered = sorted(
            members,
            key=_scalar_fitness_value,
            reverse=definition.direction is ObjectiveDirection.MAXIMIZE,
        )
        tiers: list[EliteTier] = []
        for member in ordered:
            value = _scalar_fitness_value(member)
            if not tiers or _scalar_fitness_value(tiers[-1].members[0]) != value:
                tiers.append(EliteTier(tier_index=len(tiers), members=(member,)))
                continue
            previous = tiers[-1]
            tiers[-1] = EliteTier(
                tier_index=previous.tier_index,
                members=(*previous.members, member),
            )
        return tuple(tiers)

    assert isinstance(definition, MultiObjectiveFitnessDefinition)
    remaining = list(members)
    fronts: list[EliteTier] = []
    while remaining:
        front_members = tuple(
            candidate
            for candidate in remaining
            if not any(
                other is not candidate
                and _pareto_dominates(
                    other,
                    candidate,
                    objectives=definition.objectives,
                )
                for other in remaining
            )
        )
        fronts.append(EliteTier(tier_index=len(fronts), members=front_members))
        front_ids = {id(member) for member in front_members}
        remaining = [member for member in remaining if id(member) not in front_ids]
    return tuple(fronts)


def _scalar_fitness_value(member: PopulationFitnessMember) -> float:
    fitness = member.fitness
    if not isinstance(fitness, BasicScalarScore):
        raise TypeError("scalar elitism requires BasicScalarScore fitness")
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
        raise TypeError("multi-objective elitism requires MultiObjectiveEvaluation fitness")

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
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
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
