"""Explicit k-nearest-neighbor novelty scores for geometry populations."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from typing import TypeAlias

from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.search.generation_population import (
    GenerationPopulation,
    GenerationPopulationMember,
)
from toposc_lab.search.geometry_genome import GeometryGenome
from toposc_lab.search.initial_population import (
    InitialPopulation,
    InitialPopulationMember,
)

NOVELTY_SCORE_VERSION = 1

_NOVELTY_WARNINGS = (
    (
        "Novelty is measured only under the caller-supplied, versioned distance "
        "contract; the library defines no universal geometry or physical distance."
    ),
    (
        "Exact geometry identifiers locate snapshots for audit only; they do not "
        "determine distance, equivalence, or novelty."
    ),
    (
        "The comparison cohort is immutable input to this evaluation, not a "
        "persistent or automatically updated novelty archive."
    ),
    (
        "Novelty scores do not modify fitness, Pareto dominance, selection, "
        "elitism, diversity-family occupancy, or generation composition."
    ),
)

NoveltyPopulation: TypeAlias = InitialPopulation | GenerationPopulation
NoveltyPopulationMember: TypeAlias = (
    InitialPopulationMember | GenerationPopulationMember
)
NoveltyDistance: TypeAlias = Callable[[GeometryGenome, GeometryGenome], float]


@dataclass(frozen=True, slots=True)
class NoveltyScoreConfig:
    """Fixed number of nearest comparisons used in every novelty score."""

    neighbor_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "neighbor_count",
            _positive_integer(self.neighbor_count, name="neighbor_count"),
        )


@dataclass(frozen=True, slots=True)
class NoveltyReferenceCohort:
    """Explicit ordered in-memory comparison genomes, without archive behavior."""

    identifier: str
    genomes: tuple[GeometryGenome, ...]

    def __post_init__(self) -> None:
        identifier = _nonempty_string(self.identifier, name="identifier")
        if isinstance(self.genomes, (str, bytes, bytearray)) or not isinstance(
            self.genomes,
            Iterable,
        ):
            raise TypeError("genomes must be an iterable of GeometryGenome values")
        genomes = tuple(self.genomes)
        if not genomes:
            raise ValueError("a novelty reference cohort must not be empty")
        if not all(isinstance(genome, GeometryGenome) for genome in genomes):
            raise TypeError("genomes must contain only GeometryGenome values")
        object.__setattr__(self, "identifier", identifier)
        object.__setattr__(self, "genomes", genomes)


class NoveltyComparisonSource(str, Enum):
    """Origin of one candidate comparison."""

    POPULATION = "population"
    REFERENCE_COHORT = "reference_cohort"


@dataclass(frozen=True, slots=True)
class NoveltyComparison:
    """One evaluated distance from a candidate to an exact comparison genome."""

    source: NoveltyComparisonSource
    source_index: int
    genome: GeometryGenome
    distance: float
    geometry_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, NoveltyComparisonSource):
            raise TypeError("source must be NoveltyComparisonSource")
        object.__setattr__(
            self,
            "source_index",
            _nonnegative_integer(self.source_index, name="source_index"),
        )
        if not isinstance(self.genome, GeometryGenome):
            raise TypeError("genome must be a GeometryGenome")
        object.__setattr__(
            self,
            "distance",
            _nonnegative_finite_real(self.distance, name="distance"),
        )
        object.__setattr__(self, "geometry_id", exact_geometry_id(self.genome.to_geometry()))


@dataclass(frozen=True, slots=True)
class PopulationNoveltyMember:
    """Complete comparison ledger and derived novelty for one population member."""

    population_member: NoveltyPopulationMember
    comparisons: tuple[NoveltyComparison, ...]
    nearest_neighbors: tuple[NoveltyComparison, ...]
    score: float
    geometry_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.population_member,
            (InitialPopulationMember, GenerationPopulationMember),
        ):
            raise TypeError("population_member must be a supported population member")
        comparisons = _comparison_tuple(self.comparisons, name="comparisons")
        nearest_neighbors = _comparison_tuple(
            self.nearest_neighbors,
            name="nearest_neighbors",
        )
        object.__setattr__(
            self,
            "score",
            _nonnegative_finite_real(self.score, name="score"),
        )
        object.__setattr__(self, "comparisons", comparisons)
        object.__setattr__(self, "nearest_neighbors", nearest_neighbors)
        object.__setattr__(
            self,
            "geometry_id",
            exact_geometry_id(self.population_member.genome.to_geometry()),
        )

    @property
    def generation_index(self) -> int:
        """Generation containing this candidate."""
        return self.population_member.generation_index

    @property
    def member_index(self) -> int:
        """Candidate index retained from the source population."""
        return self.population_member.member_index


@dataclass(frozen=True, slots=True)
class PopulationNoveltyReport:
    """Auditable novelty scores for one exact ordered population."""

    population: NoveltyPopulation
    config: NoveltyScoreConfig
    distance_identifier: str
    reference_cohort: NoveltyReferenceCohort | None
    members: tuple[PopulationNoveltyMember, ...]
    version: int = field(default=NOVELTY_SCORE_VERSION, init=False)
    warnings: tuple[str, ...] = field(default=_NOVELTY_WARNINGS, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.population, (InitialPopulation, GenerationPopulation)):
            raise TypeError("population must be a supported novelty population")
        if not isinstance(self.config, NoveltyScoreConfig):
            raise TypeError("config must be a NoveltyScoreConfig")
        distance_identifier = _nonempty_string(
            self.distance_identifier,
            name="distance_identifier",
        )
        if self.reference_cohort is not None and not isinstance(
            self.reference_cohort,
            NoveltyReferenceCohort,
        ):
            raise TypeError("reference_cohort must be NoveltyReferenceCohort or None")
        if isinstance(self.members, (str, bytes, bytearray)) or not isinstance(
            self.members,
            Iterable,
        ):
            raise TypeError("members must be an iterable of PopulationNoveltyMember")
        members = tuple(self.members)
        if len(members) != self.population.population_size:
            raise ValueError("members must score every population member")
        if not all(isinstance(member, PopulationNoveltyMember) for member in members):
            raise TypeError("members must contain only PopulationNoveltyMember values")
        for member, population_member in zip(
            members,
            self.population.members,
            strict=True,
        ):
            if member.population_member is not population_member:
                raise ValueError("novelty members must preserve population order and identity")
            self._validate_member(member)
        object.__setattr__(self, "distance_identifier", distance_identifier)
        object.__setattr__(self, "members", members)

    @property
    def scores(self) -> tuple[float, ...]:
        """Novelty scores in exact source-population order."""
        return tuple(member.score for member in self.members)

    def score_for_member(self, member_index: int) -> float:
        """Return the novelty score at one exact population member index."""
        prepared_index = _nonnegative_integer(member_index, name="member_index")
        if prepared_index >= len(self.members):
            raise IndexError("member_index is outside the population")
        return self.members[prepared_index].score

    def _validate_member(self, member: PopulationNoveltyMember) -> None:
        expected_sources: list[tuple[NoveltyComparisonSource, int, GeometryGenome]] = [
            (NoveltyComparisonSource.POPULATION, source.member_index, source.genome)
            for source in self.population.members
            if source is not member.population_member
        ]
        if self.reference_cohort is not None:
            expected_sources.extend(
                (
                    NoveltyComparisonSource.REFERENCE_COHORT,
                    source_index,
                    genome,
                )
                for source_index, genome in enumerate(self.reference_cohort.genomes)
            )
        if len(member.comparisons) != len(expected_sources):
            raise ValueError("comparisons must retain every eligible comparison exactly once")
        if len(member.comparisons) < self.config.neighbor_count:
            raise ValueError("neighbor_count exceeds the retained comparisons")
        for comparison, (source, source_index, genome) in zip(
            member.comparisons,
            expected_sources,
            strict=True,
        ):
            if (
                comparison.source is not source
                or comparison.source_index != source_index
                or comparison.genome is not genome
            ):
                raise ValueError(
                    "comparisons must preserve source order, indices, and genome identity"
                )
        expected_nearest = tuple(
            sorted(member.comparisons, key=_comparison_sort_key)[
                : self.config.neighbor_count
            ]
        )
        if member.nearest_neighbors != expected_nearest:
            raise ValueError("nearest_neighbors do not match the configured k-nearest set")
        expected_score = math.fsum(
            comparison.distance for comparison in expected_nearest
        ) / self.config.neighbor_count
        if member.score != expected_score:
            raise ValueError("score does not match the mean k-nearest distance")


class NoveltyDistanceEvaluationError(RuntimeError):
    """Raised when the caller-supplied distance callback fails."""

    def __init__(
        self,
        *,
        distance_identifier: str,
        generation_index: int,
        member_index: int,
        source: NoveltyComparisonSource,
        source_index: int,
        error: Exception,
    ) -> None:
        self.distance_identifier = _nonempty_string(
            distance_identifier,
            name="distance_identifier",
        )
        self.generation_index = _nonnegative_integer(
            generation_index,
            name="generation_index",
        )
        self.member_index = _nonnegative_integer(member_index, name="member_index")
        if not isinstance(source, NoveltyComparisonSource):
            raise TypeError("source must be NoveltyComparisonSource")
        self.source = source
        self.source_index = _nonnegative_integer(source_index, name="source_index")
        if not isinstance(error, Exception):
            raise TypeError("error must be an Exception")
        self.error_type = type(error).__name__
        self.error_message = str(error).strip() or "The distance callback raised without a message."
        super().__init__(
            f"novelty distance {self.distance_identifier!r} failed for generation "
            f"{self.generation_index}, member {self.member_index}, "
            f"{self.source.value} {self.source_index}: "
            f"{self.error_type}: {self.error_message}"
        )


class InvalidNoveltyDistanceError(ValueError):
    """Raised when one callback result is not a finite nonnegative real value."""

    def __init__(
        self,
        *,
        distance_identifier: str,
        generation_index: int,
        member_index: int,
        source: NoveltyComparisonSource,
        source_index: int,
        value: object,
    ) -> None:
        self.distance_identifier = _nonempty_string(
            distance_identifier,
            name="distance_identifier",
        )
        self.generation_index = _nonnegative_integer(
            generation_index,
            name="generation_index",
        )
        self.member_index = _nonnegative_integer(member_index, name="member_index")
        if not isinstance(source, NoveltyComparisonSource):
            raise TypeError("source must be NoveltyComparisonSource")
        self.source = source
        self.source_index = _nonnegative_integer(source_index, name="source_index")
        self.value = value
        super().__init__(
            f"novelty distance {self.distance_identifier!r} returned {value!r} for "
            f"generation {self.generation_index}, member {self.member_index}, "
            f"{self.source.value} {self.source_index}; a finite nonnegative real is required"
        )


def evaluate_population_novelty(
    population: NoveltyPopulation,
    *,
    config: NoveltyScoreConfig,
    distance: NoveltyDistance,
    distance_identifier: str,
    reference_cohort: NoveltyReferenceCohort | None = None,
) -> PopulationNoveltyReport:
    """Evaluate mean k-nearest distance without changing search decisions."""
    if not isinstance(population, (InitialPopulation, GenerationPopulation)):
        raise TypeError("population must be a supported novelty population")
    if not isinstance(config, NoveltyScoreConfig):
        raise TypeError("config must be a NoveltyScoreConfig")
    if not callable(distance):
        raise TypeError("distance must be callable")
    prepared_identifier = _nonempty_string(
        distance_identifier,
        name="distance_identifier",
    )
    if reference_cohort is not None and not isinstance(
        reference_cohort,
        NoveltyReferenceCohort,
    ):
        raise TypeError("reference_cohort must be NoveltyReferenceCohort or None")
    comparison_count = population.population_size - 1 + (
        0 if reference_cohort is None else len(reference_cohort.genomes)
    )
    if config.neighbor_count > comparison_count:
        raise ValueError(
            "neighbor_count exceeds the available population and reference comparisons"
        )

    scored_members: list[PopulationNoveltyMember] = []
    for candidate in population.members:
        comparisons: list[NoveltyComparison] = []
        for source_member in population.members:
            if source_member is candidate:
                continue
            comparisons.append(
                _evaluate_comparison(
                    candidate=candidate,
                    source=NoveltyComparisonSource.POPULATION,
                    source_index=source_member.member_index,
                    source_genome=source_member.genome,
                    distance=distance,
                    distance_identifier=prepared_identifier,
                )
            )
        if reference_cohort is not None:
            for source_index, source_genome in enumerate(reference_cohort.genomes):
                comparisons.append(
                    _evaluate_comparison(
                        candidate=candidate,
                        source=NoveltyComparisonSource.REFERENCE_COHORT,
                        source_index=source_index,
                        source_genome=source_genome,
                        distance=distance,
                        distance_identifier=prepared_identifier,
                    )
                )
        prepared_comparisons = tuple(comparisons)
        nearest = tuple(
            sorted(prepared_comparisons, key=_comparison_sort_key)[
                : config.neighbor_count
            ]
        )
        score = math.fsum(item.distance for item in nearest) / config.neighbor_count
        scored_members.append(
            PopulationNoveltyMember(
                population_member=candidate,
                comparisons=prepared_comparisons,
                nearest_neighbors=nearest,
                score=score,
            )
        )
    return PopulationNoveltyReport(
        population=population,
        config=config,
        distance_identifier=prepared_identifier,
        reference_cohort=reference_cohort,
        members=tuple(scored_members),
    )


def _evaluate_comparison(
    *,
    candidate: NoveltyPopulationMember,
    source: NoveltyComparisonSource,
    source_index: int,
    source_genome: GeometryGenome,
    distance: NoveltyDistance,
    distance_identifier: str,
) -> NoveltyComparison:
    try:
        value = distance(candidate.genome, source_genome)
    except Exception as error:
        raise NoveltyDistanceEvaluationError(
            distance_identifier=distance_identifier,
            generation_index=candidate.generation_index,
            member_index=candidate.member_index,
            source=source,
            source_index=source_index,
            error=error,
        ) from error
    try:
        prepared_distance = _nonnegative_finite_real(value, name="distance")
    except (TypeError, ValueError) as error:
        raise InvalidNoveltyDistanceError(
            distance_identifier=distance_identifier,
            generation_index=candidate.generation_index,
            member_index=candidate.member_index,
            source=source,
            source_index=source_index,
            value=value,
        ) from error
    return NoveltyComparison(
        source=source,
        source_index=source_index,
        genome=source_genome,
        distance=prepared_distance,
    )


def _comparison_tuple(
    values: object,
    *,
    name: str,
) -> tuple[NoveltyComparison, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Iterable):
        raise TypeError(f"{name} must be an iterable of NoveltyComparison values")
    prepared = tuple(values)
    if not all(isinstance(value, NoveltyComparison) for value in prepared):
        raise TypeError(f"{name} must contain only NoveltyComparison values")
    return prepared


def _comparison_sort_key(
    comparison: NoveltyComparison,
) -> tuple[float, int, int]:
    source_order = (
        0 if comparison.source is NoveltyComparisonSource.POPULATION else 1
    )
    return (comparison.distance, source_order, comparison.source_index)


def _nonnegative_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


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


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()
