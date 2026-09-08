"""Paired, equal-attempt-budget evolution versus independent random sampling.

This is an engineering benchmark, not a physical discovery or significance test.
Callbacks must be pure functions of their explicit inputs and implement the frozen
candidate-space/evaluation protocol. No historical experiment is loaded or rerun.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np

from toposc_lab.evaluation import BasicScalarScore, GeometryEvaluationRun, ObjectiveDirection
from toposc_lab.evaluation.reproducibility import ReproducibilityRecord
from toposc_lab.search.diversity_preservation import DiversityFamilyClassifier
from toposc_lab.search.generation_loop import (
    GenerationLoopConfig,
    GenerationLoopResult,
    OffspringProducer,
    run_generation_loop,
)
from toposc_lab.search.generation_population import GenerationPopulation, GenerationPopulationMember
from toposc_lab.search.geometry_genome import GeometryGenome
from toposc_lab.search.initial_population import InitialPopulationError, create_initial_population
from toposc_lab.search.mutation_validity import (
    MutationValidityPolicy,
    MutationValidityReport,
    validate_geometry_mutation,
)
from toposc_lab.search.population_fitness import (
    FitnessPopulationMember,
    MultiObjectiveFitnessDefinition,
    PopulationEvaluator,
    PopulationFitnessDefinition,
    PopulationFitnessMember,
    PopulationFitnessResult,
    ScalarFitnessDefinition,
    evaluate_population_fitness,
)

SEARCH_BENCHMARK_VERSION = 1
BenchmarkSampler: TypeAlias = Callable[[int], GeometryGenome]
BenchmarkEvaluator: TypeAlias = Callable[[GeometryGenome, int], GeometryEvaluationRun]
BenchmarkSuccessPredicate: TypeAlias = Callable[[PopulationFitnessMember], bool]


class InvalidBenchmarkSampleError(ValueError):
    """A rejected random draw, retained without repair or replacement sampling."""

    def __init__(
        self,
        *,
        trial_seed: int,
        attempt_index: int,
        sampling_seed: int,
        genome: GeometryGenome,
        report: MutationValidityReport,
    ) -> None:
        self.trial_seed = _integer(trial_seed, "trial_seed")
        self.attempt_index = _integer(attempt_index, "attempt_index")
        self.sampling_seed = _integer(sampling_seed, "sampling_seed")
        if not isinstance(genome, GeometryGenome):
            raise TypeError("genome must be GeometryGenome")
        if not isinstance(report, MutationValidityReport) or report.is_valid:
            raise ValueError("report must be an invalid MutationValidityReport")
        self.genome = genome
        self.report = report
        super().__init__(
            f"invalid benchmark sample in trial {trial_seed} at attempt index {attempt_index}: "
            + "; ".join(issue.message for issue in report.issues)
        )


@dataclass(frozen=True, slots=True)
class BenchmarkSuccessCriterion:
    """Frozen caller-declared hit definition; only available members are tested."""

    identifier: str
    description: str
    predicate: BenchmarkSuccessPredicate = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _labels(self, ("identifier", "description"))
        if not callable(self.predicate):
            raise TypeError("predicate must be callable")


@dataclass(frozen=True, slots=True)
class SearchBenchmarkProtocol:
    """Predeclared experiment, fixed physical evaluation, and paired run seeds.

    The candidate-space identifier covers resource controls not expressible in
    MutationValidityPolicy. Identifiers declare policies; they cannot attest to
    arbitrary callback code. No settings are fitted to benchmark outcomes.
    """

    identifier: str
    candidate_space_identifier: str
    sampler_identifier: str
    evaluator_identifier: str
    producer_identifier: str
    code_version: str
    population_size: int
    generation_config: GenerationLoopConfig
    trial_seeds: tuple[int, ...]
    definition: PopulationFitnessDefinition
    validity_policy: MutationValidityPolicy
    criterion: BenchmarkSuccessCriterion
    family_classifier_identifier: str | None = None
    version: int = field(default=SEARCH_BENCHMARK_VERSION, init=False)

    def __post_init__(self) -> None:
        _labels(
            self,
            (
                "identifier",
                "candidate_space_identifier",
                "sampler_identifier",
                "evaluator_identifier",
                "producer_identifier",
                "code_version",
            ),
        )
        size = _integer(self.population_size, "population_size", minimum=1)
        if not isinstance(self.generation_config, GenerationLoopConfig):
            raise TypeError("generation_config must be GenerationLoopConfig")
        if self.generation_config.elitism.minimum_elite_count > size:
            raise ValueError("minimum elite count exceeds population_size")
        if self.generation_config.selection.tournament_size > size:
            raise ValueError("tournament size exceeds population_size")
        seeds = tuple(_integer(seed, "trial seed") for seed in self.trial_seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("trial_seeds must be nonempty and distinct")
        if not isinstance(
            self.definition, (ScalarFitnessDefinition, MultiObjectiveFitnessDefinition)
        ):
            raise TypeError("definition must be a population fitness definition")
        if not isinstance(self.validity_policy, MutationValidityPolicy):
            raise TypeError("validity_policy must be MutationValidityPolicy")
        if not isinstance(self.criterion, BenchmarkSuccessCriterion):
            raise TypeError("criterion must be BenchmarkSuccessCriterion")
        if self.generation_config.diversity is None:
            if self.family_classifier_identifier is not None:
                raise ValueError("family classifier identifier requires diversity")
        else:
            _labels(self, ("family_classifier_identifier",))
        object.__setattr__(self, "population_size", size)
        object.__setattr__(self, "trial_seeds", seeds)

    @property
    def attempts_per_arm(self) -> int:
        """Every population slot, including generation zero and reevaluated elites."""
        return self.population_size * (self.generation_config.generation_count + 1)


@dataclass(frozen=True, slots=True)
class BenchmarkProgress:
    """One inclusive evaluation-attempt prefix; None never stands for zero."""

    attempt_count: int
    available_count: int
    successful_count: int
    first_success_attempt: int | None
    best_scalar_score: float | None

    def __post_init__(self) -> None:
        for name in ("attempt_count", "available_count", "successful_count"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if not 0 <= self.successful_count <= self.available_count <= self.attempt_count:
            raise ValueError("progress counts must satisfy successful <= available <= attempts")
        if (self.first_success_attempt is None) != (self.successful_count == 0):
            raise ValueError("first_success_attempt must match success availability")
        if self.first_success_attempt is not None:
            first = _integer(self.first_success_attempt, "first_success_attempt", minimum=1)
            if first + self.successful_count - 1 > self.attempt_count:
                raise ValueError("first_success_attempt is inconsistent with prefix counts")
            object.__setattr__(self, "first_success_attempt", first)
        score = self.best_scalar_score
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, Real):
                raise TypeError("best_scalar_score must be a real number or None")
            if not isfinite(score) or not 0 <= score <= 1 or self.available_count == 0:
                raise ValueError(
                    "best_scalar_score must be an available finite unit-interval score"
                )
            object.__setattr__(self, "best_scalar_score", float(score))


@dataclass(frozen=True, slots=True)
class BenchmarkArmResult:
    """Complete ordered ledgers and strict hit decisions, including failures."""

    fitness_history: tuple[PopulationFitnessResult, ...]
    successes: tuple[bool, ...]

    def __post_init__(self) -> None:
        history = tuple(self.fitness_history)
        successes = tuple(self.successes)
        if not history or not all(isinstance(item, PopulationFitnessResult) for item in history):
            raise TypeError("fitness_history must contain population fitness results")
        first = history[0]
        for index, item in enumerate(history):
            if item.population.generation_index != index:
                raise ValueError("fitness history must contain contiguous generations from zero")
            if item.definition is not first.definition:
                raise ValueError("fitness history must share the exact definition")
            if item.population.validity_policy is not first.population.validity_policy:
                raise ValueError("fitness history must share the exact validity policy")
            if item.population.population_size != first.population.population_size:
                raise ValueError("fitness history must have constant population size")
            if item.population.embedding_dimension != first.population.embedding_dimension:
                raise ValueError("fitness history must preserve embedding dimension")
        members = tuple(member for item in history for member in item.members)
        if len(successes) != len(members) or any(type(value) is not bool for value in successes):
            raise ValueError("successes must retain one strict boolean per attempted member")
        if any(
            hit and not member.is_available for hit, member in zip(successes, members, strict=True)
        ):
            raise ValueError("unavailable fitness cannot be a success")
        object.__setattr__(self, "fitness_history", history)
        object.__setattr__(self, "successes", successes)

    @property
    def progress(self) -> tuple[BenchmarkProgress, ...]:
        """Best-so-far and first hit at every attempt, not only generation boundaries.

        Scalar direction is respected. Multi-objective ledgers retain their raw
        vectors but have no invented scalar quality ordering.
        """
        definition = self.fitness_history[0].definition
        best: float | None = None
        first_hit: int | None = None
        available = successful = 0
        steps: list[BenchmarkProgress] = []
        members = (member for item in self.fitness_history for member in item.members)
        for attempt, (member, hit) in enumerate(zip(members, self.successes, strict=True), 1):
            available += member.is_available
            successful += hit
            if hit and first_hit is None:
                first_hit = attempt
            if isinstance(member.fitness, BasicScalarScore):
                assert isinstance(definition, ScalarFitnessDefinition)
                score = member.fitness.value
                if best is None:
                    best = score
                elif definition.direction is ObjectiveDirection.MAXIMIZE:
                    best = max(best, score)
                else:
                    best = min(best, score)
            steps.append(BenchmarkProgress(attempt, available, successful, first_hit, best))
        return tuple(steps)

    @property
    def final_progress(self) -> BenchmarkProgress:
        return self.progress[-1]


@dataclass(frozen=True, slots=True)
class SearchBenchmarkTrial:
    """One paired experiment; the independent replication unit is this whole run."""

    seed: int
    evolution: GenerationLoopResult
    evolution_arm: BenchmarkArmResult
    random_arm: BenchmarkArmResult

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        if not isinstance(self.evolution, GenerationLoopResult):
            raise TypeError("evolution must be GenerationLoopResult")
        if not isinstance(self.evolution_arm, BenchmarkArmResult) or not isinstance(
            self.random_arm,
            BenchmarkArmResult,
        ):
            raise TypeError("arms must be BenchmarkArmResult")
        history = self.evolution.fitness_history
        if len(history) != len(self.evolution_arm.fitness_history) or any(
            left is not right
            for left, right in zip(
                history,
                self.evolution_arm.fitness_history,
                strict=True,
            )
        ):
            raise ValueError("evolution arm must retain the exact loop fitness history")
        if len(history) != len(self.random_arm.fitness_history):
            raise ValueError("arms must have equal generation budgets")
        for left, right in zip(history, self.random_arm.fitness_history, strict=True):
            if left.definition is not right.definition:
                raise ValueError("arms must share the exact fitness definition")
            if left.population.validity_policy is not right.population.validity_policy:
                raise ValueError("arms must share the exact validity policy")
            if left.population.population_size != right.population.population_size:
                raise ValueError("arms must have equal attempt budgets")
            if left.population.embedding_dimension != right.population.embedding_dimension:
                raise ValueError("arms must share embedding dimension")
        if history[0].population is not self.random_arm.fitness_history[0].population:
            raise ValueError("arms must start from the exact same initial population")
        if history[0] is self.random_arm.fitness_history[0]:
            raise ValueError("initial population must be evaluated separately for each arm")

    @property
    def scalar_quality_difference(self) -> float | None:
        """Direction-adjusted final best-score difference; positive favors evolution."""
        left = self.evolution_arm.final_progress.best_scalar_score
        right = self.random_arm.final_progress.best_scalar_score
        if left is None or right is None:
            return None
        definition = self.evolution.definition
        assert isinstance(definition, ScalarFitnessDefinition)
        return left - right if definition.direction is ObjectiveDirection.MAXIMIZE else right - left


@dataclass(frozen=True, slots=True)
class SearchBenchmarkResult:
    """All predeclared paired trials; no hidden pooling, winner claim, or file I/O."""

    protocol: SearchBenchmarkProtocol
    trials: tuple[SearchBenchmarkTrial, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, SearchBenchmarkProtocol):
            raise TypeError("protocol must be SearchBenchmarkProtocol")
        trials = tuple(self.trials)
        if not all(isinstance(trial, SearchBenchmarkTrial) for trial in trials):
            raise TypeError("trials must contain SearchBenchmarkTrial")
        if tuple(trial.seed for trial in trials) != self.protocol.trial_seeds:
            raise ValueError("trials must retain all protocol seeds in order")
        reference: ReproducibilityRecord | None = None
        for trial in trials:
            evolution = trial.evolution
            if evolution.config != self.protocol.generation_config:
                raise ValueError("trial generation config differs from protocol")
            if evolution.definition is not self.protocol.definition:
                raise ValueError("trial definition differs from protocol")
            if evolution.initial_population.validity_policy is not self.protocol.validity_policy:
                raise ValueError("trial validity policy differs from protocol")
            if evolution.initial_population.population_size != self.protocol.population_size:
                raise ValueError("trial size differs from protocol")
            if evolution.producer_identifier != self.protocol.producer_identifier:
                raise ValueError("trial producer differs from protocol")
            if evolution.family_classifier_identifier != self.protocol.family_classifier_identifier:
                raise ValueError("trial classifier differs from protocol")
            evolution_seed, _, evaluation_seeds = _seed_schedule(trial.seed, self.protocol)
            if evolution.seed != evolution_seed:
                raise ValueError("trial evolution seed differs from protocol schedule")
            reference = _audit_evaluations(trial, self.protocol, evaluation_seeds, reference)
        object.__setattr__(self, "trials", trials)

    @property
    def paired_success_differences(self) -> tuple[int, ...]:
        """Per-run hit difference (-1, 0, +1), not per-candidate independent samples."""
        return tuple(
            int(any(trial.evolution_arm.successes)) - int(any(trial.random_arm.successes))
            for trial in self.trials
        )


def run_search_benchmark(
    protocol: SearchBenchmarkProtocol,
    *,
    sampler: BenchmarkSampler,
    evaluator: BenchmarkEvaluator,
    offspring_producer: OffspringProducer,
    family_classifier: DiversityFamilyClassifier | None = None,
) -> SearchBenchmarkResult:
    """Run both arms with N*(G+1) evaluator attempts each, per declared seed.

    ``sampler(seed)`` draws one valid independent candidate from the declared
    distribution. It receives no fitness/history. ``evaluator(genome, seed)`` is
    identical for both arms and receives paired per-attempt seeds. Generation zero
    is sampled once but evaluated separately twice. No deduplication or caching.

    Ordinary evaluator failures are retained by Phase 10.11. Sampling, offspring,
    selection, diversity, provenance, or criterion contract errors abort without
    retries or a partial benchmark statistic. They must not be discarded from a
    campaign and replaced by favorable seeds. No benchmark persistence/resume API
    is introduced; individual evolution results use existing checkpoint APIs.
    """
    if not isinstance(protocol, SearchBenchmarkProtocol):
        raise TypeError("protocol must be SearchBenchmarkProtocol")
    if not all(callable(callback) for callback in (sampler, evaluator, offspring_producer)):
        raise TypeError("sampler, evaluator, and offspring_producer must be callable")
    if protocol.generation_config.diversity is None:
        if family_classifier is not None:
            raise ValueError("family_classifier requires a diversity policy")
    elif not callable(family_classifier):
        raise TypeError("diversity requires a callable family_classifier")
    trials: list[SearchBenchmarkTrial] = []
    for seed in protocol.trial_seeds:
        evolution_seed, sampling_seeds, evaluation_seeds = _seed_schedule(seed, protocol)
        size = protocol.population_size
        initial_genomes = tuple(sampler(draw_seed) for draw_seed in sampling_seeds[:size])
        try:
            initial = create_initial_population(
                initial_genomes,
                validity_policy=protocol.validity_policy,
            )
        except InitialPopulationError as error:
            raise InvalidBenchmarkSampleError(
                trial_seed=seed,
                attempt_index=error.candidate_index,
                sampling_seed=sampling_seeds[error.candidate_index],
                genome=initial_genomes[error.candidate_index],
                report=error.report,
            ) from error

        evaluate = _paired_evaluator(evaluator, evaluation_seeds, size)
        evolution = run_generation_loop(
            initial,
            definition=protocol.definition,
            evaluator=evaluate,
            config=protocol.generation_config,
            seed=evolution_seed,
            offspring_producer=offspring_producer,
            producer_identifier=protocol.producer_identifier,
            family_classifier=family_classifier,
            family_classifier_identifier=protocol.family_classifier_identifier,
        )
        random_history = [
            evaluate_population_fitness(
                initial,
                definition=protocol.definition,
                evaluator=evaluate,
            )
        ]
        for generation in range(1, protocol.generation_config.generation_count + 1):
            members: list[GenerationPopulationMember] = []
            for index in range(size):
                genome = sampler(sampling_seeds[generation * size + index])
                source = initial.genomes[0]
                report = validate_geometry_mutation(source, genome, policy=protocol.validity_policy)
                if not report.is_valid:
                    raise InvalidBenchmarkSampleError(
                        trial_seed=seed,
                        attempt_index=generation * size + index,
                        sampling_seed=sampling_seeds[generation * size + index],
                        genome=genome,
                        report=report,
                    )
                members.append(
                    GenerationPopulationMember(generation, index, genome, source, report)
                )
            population = GenerationPopulation(
                generation,
                tuple(members),
                protocol.validity_policy,
                initial.embedding_dimension,
            )
            random_history.append(
                evaluate_population_fitness(
                    population,
                    definition=protocol.definition,
                    evaluator=evaluate,
                )
            )
        trial = SearchBenchmarkTrial(
            seed,
            evolution,
            _assess_arm(evolution.fitness_history, protocol.criterion),
            _assess_arm(tuple(random_history), protocol.criterion),
        )
        _audit_evaluations(trial, protocol, evaluation_seeds)
        trials.append(trial)
    return SearchBenchmarkResult(protocol, tuple(trials))


def _assess_arm(
    history: tuple[PopulationFitnessResult, ...],
    criterion: BenchmarkSuccessCriterion,
) -> BenchmarkArmResult:
    decisions: list[bool] = []
    for result in history:
        for member in result.members:
            decision = criterion.predicate(member) if member.is_available else False
            if type(decision) is not bool:
                raise TypeError("benchmark success predicate must return a strict boolean")
            decisions.append(decision)
    return BenchmarkArmResult(history, tuple(decisions))


def _paired_evaluator(
    evaluator: BenchmarkEvaluator,
    seeds: tuple[int, ...],
    size: int,
) -> PopulationEvaluator:
    def evaluate(member: FitnessPopulationMember) -> GeometryEvaluationRun:
        index = member.generation_index * size + member.member_index
        return evaluator(member.genome, seeds[index])

    return evaluate


def _seed_schedule(
    seed: int,
    protocol: SearchBenchmarkProtocol,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    # Version 1: one evolution word, B sampling words, B evaluation words.
    # Separate local generator; no global NumPy state, redraws, or outcome dependence.
    rng = np.random.PCG64(seed)
    evolution_seed = int(rng.random_raw())
    budget = protocol.attempts_per_arm
    sampling_seeds = tuple(int(value) for value in rng.random_raw(budget))
    evaluation_seeds = tuple(int(value) for value in rng.random_raw(budget))
    return evolution_seed, sampling_seeds, evaluation_seeds


def _audit_evaluations(
    trial: SearchBenchmarkTrial,
    protocol: SearchBenchmarkProtocol,
    seeds: tuple[int, ...],
    reference: ReproducibilityRecord | None = None,
) -> ReproducibilityRecord | None:
    for arm in (trial.evolution_arm, trial.random_arm):
        members = (member for result in arm.fitness_history for member in result.members)
        for seed, member in zip(seeds, members, strict=True):
            if member.evaluation is None or member.evaluation.reproducibility is None:
                continue  # Pre-provenance failures still consume an attempted evaluation.
            record = member.evaluation.reproducibility
            if record.seed != seed or record.code_version != protocol.code_version:
                raise ValueError("evaluation seed/code version differs from benchmark protocol")
            if reference is None:
                reference = record
            for name in (
                "model_name",
                "model_parameters",
                "solver_name",
                "solver_settings",
                "evaluation_settings",
            ):
                if getattr(record, name) != getattr(reference, name):
                    raise ValueError(f"benchmark requires fixed evaluation provenance: {name}")
    return reference


def _labels(instance: object, names: tuple[str, ...]) -> None:
    for name in names:
        value = getattr(instance, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
        object.__setattr__(instance, name, value.strip())


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result
