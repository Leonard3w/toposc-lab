"""Seed-explicit composition and evaluation of geometry generations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import TypeAlias

import numpy as np

from toposc_lab.search.diversity_preservation import (
    DiversityFamilyClassifier,
    DiversityPreservationPolicy,
    PopulationDiversityReport,
    assess_population_diversity,
    validate_diversity_history,
)
from toposc_lab.search.generation_population import (
    GenerationPopulation,
    GenerationPopulationMember,
)
from toposc_lab.search.geometry_genome import GeometryGenome
from toposc_lab.search.initial_population import InitialPopulation
from toposc_lab.search.mutation_validity import (
    MutationValidityPolicy,
    MutationValidityReport,
    validate_geometry_mutation,
)
from toposc_lab.search.population_elitism import (
    ElitismConfig,
    PopulationElitismResult,
    identify_population_elites,
)
from toposc_lab.search.population_fitness import (
    PopulationEvaluator,
    PopulationFitnessDefinition,
    PopulationFitnessMember,
    PopulationFitnessResult,
    evaluate_population_fitness,
)
from toposc_lab.search.population_selection import (
    PopulationSelectionResult,
    TournamentSelectionConfig,
    select_population_members,
)

GENERATION_LOOP_VERSION = 2
GENERATION_LOOP_RNG_ALGORITHM = "numpy.random.PCG64"

_GENERATION_LOOP_WARNINGS = (
    (
        "The generation loop is an engineering search process, not scientific "
        "evidence or a discovery claim."
    ),
    (
        "Offspring construction is caller-defined; the loop does not infer parent "
        "pairing, mutation or crossover probabilities, or a repair strategy."
    ),
    (
        "Invalid offspring are retained in InvalidGenerationOffspringError and are "
        "not filtered, repaired, replaced, or retried."
    ),
    (
        "Duplicate genomes are retained; an optional explicit diversity family-"
        "occupancy gate does not deduplicate candidates or define novelty."
    ),
    "Checkpointing, resume behavior, and persistence remain separate contracts.",
)


@dataclass(frozen=True, slots=True)
class GenerationLoopConfig:
    """Transition count and existing selection/elitism contracts."""

    generation_count: int
    selection: TournamentSelectionConfig
    elitism: ElitismConfig
    diversity: DiversityPreservationPolicy | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "generation_count",
            _nonnegative_integer(self.generation_count, name="generation_count"),
        )
        if not isinstance(self.selection, TournamentSelectionConfig):
            raise TypeError("selection must be a TournamentSelectionConfig")
        if not isinstance(self.elitism, ElitismConfig):
            raise TypeError("elitism must be an ElitismConfig")
        if self.diversity is not None and not isinstance(
            self.diversity,
            DiversityPreservationPolicy,
        ):
            raise TypeError(
                "diversity must be a DiversityPreservationPolicy or None"
            )


@dataclass(frozen=True, slots=True)
class OffspringProposal:
    """One caller-produced genome with explicit selected-parent provenance."""

    genome: GeometryGenome
    parent_selection_indices: tuple[int, ...]
    validation_source_selection_index: int
    operator_identifier: str
    operator_seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.genome, GeometryGenome):
            raise TypeError("genome must be a GeometryGenome")
        if isinstance(
            self.parent_selection_indices,
            (str, bytes, bytearray),
        ) or not isinstance(self.parent_selection_indices, Iterable):
            raise TypeError("parent_selection_indices must be an iterable of integers")
        parent_indices = tuple(
            _nonnegative_integer(index, name="parent selection index")
            for index in self.parent_selection_indices
        )
        if not parent_indices:
            raise ValueError("parent_selection_indices must not be empty")
        validation_source_index = _nonnegative_integer(
            self.validation_source_selection_index,
            name="validation_source_selection_index",
        )
        if validation_source_index not in parent_indices:
            raise ValueError(
                "validation_source_selection_index must reference a declared parent"
            )
        operator_identifier = _nonempty_string(
            self.operator_identifier,
            name="operator_identifier",
        )
        operator_seed = self.operator_seed
        if operator_seed is not None:
            operator_seed = _nonnegative_integer(operator_seed, name="operator_seed")
        object.__setattr__(self, "parent_selection_indices", parent_indices)
        object.__setattr__(
            self,
            "validation_source_selection_index",
            validation_source_index,
        )
        object.__setattr__(self, "operator_identifier", operator_identifier)
        object.__setattr__(self, "operator_seed", operator_seed)


@dataclass(frozen=True, slots=True)
class GenerationReproductionRequest:
    """Exact inputs and capacity supplied to one offspring producer call."""

    source: PopulationFitnessResult
    selection: PopulationSelectionResult
    target_generation_index: int
    required_offspring_count: int
    seed: int
    validity_policy: MutationValidityPolicy
    source_diversity: PopulationDiversityReport | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, PopulationFitnessResult):
            raise TypeError("source must be a PopulationFitnessResult")
        if not isinstance(self.selection, PopulationSelectionResult):
            raise TypeError("selection must be a PopulationSelectionResult")
        if self.selection.source is not self.source:
            raise ValueError("selection must refer to the exact source fitness result")
        target_generation_index = _positive_integer(
            self.target_generation_index,
            name="target_generation_index",
        )
        if target_generation_index != self.source.population.generation_index + 1:
            raise ValueError("target_generation_index must follow the source generation")
        object.__setattr__(
            self,
            "required_offspring_count",
            _positive_integer(
                self.required_offspring_count,
                name="required_offspring_count",
            ),
        )
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, name="seed"))
        if not isinstance(self.validity_policy, MutationValidityPolicy):
            raise TypeError("validity_policy must be a MutationValidityPolicy")
        if self.validity_policy is not self.source.population.validity_policy:
            raise ValueError("validity_policy must be the source population policy")
        if self.source_diversity is not None:
            if not isinstance(self.source_diversity, PopulationDiversityReport):
                raise TypeError(
                    "source_diversity must be a PopulationDiversityReport or None"
                )
            if self.source_diversity.population is not self.source.population:
                raise ValueError(
                    "source_diversity must classify the exact source population"
                )
            if not self.source_diversity.satisfies_policy:
                raise ValueError("source_diversity must satisfy its occupancy policy")
        object.__setattr__(
            self,
            "target_generation_index",
            target_generation_index,
        )

    @property
    def selected_members(self) -> tuple[PopulationFitnessMember, ...]:
        """Selected fitness-member references in producer-visible slot order."""
        return self.selection.selected_members


@dataclass(frozen=True, slots=True)
class GenerationOffspringRecord:
    """One offspring proposal and its independently recomputed validity."""

    offspring_index: int
    proposal: OffspringProposal
    validity: MutationValidityReport

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "offspring_index",
            _nonnegative_integer(self.offspring_index, name="offspring_index"),
        )
        if not isinstance(self.proposal, OffspringProposal):
            raise TypeError("proposal must be an OffspringProposal")
        if not isinstance(self.validity, MutationValidityReport):
            raise TypeError("validity must be a MutationValidityReport")


class InvalidGenerationOffspringError(ValueError):
    """Raised with the complete proposed batch when any offspring is invalid."""

    def __init__(
        self,
        *,
        request: GenerationReproductionRequest,
        records: tuple[GenerationOffspringRecord, ...],
    ) -> None:
        if not isinstance(request, GenerationReproductionRequest):
            raise TypeError("request must be a GenerationReproductionRequest")
        if not records or not all(
            isinstance(record, GenerationOffspringRecord) for record in records
        ):
            raise TypeError("records must contain GenerationOffspringRecord values")
        invalid_records = tuple(record for record in records if not record.validity.is_valid)
        if not invalid_records:
            raise ValueError("an invalid-offspring error requires an invalid record")
        self.request = request
        self.records = records
        self.invalid_records = invalid_records
        details = "; ".join(
            f"offspring {record.offspring_index}: "
            + ", ".join(issue.code for issue in record.validity.issues)
            for record in invalid_records
        )
        super().__init__(
            f"generation {request.target_generation_index} contains invalid "
            f"offspring: {details}"
        )


@dataclass(frozen=True, slots=True)
class GenerationTransition:
    """Complete audit ledger for one evaluated population transition."""

    source_fitness: PopulationFitnessResult
    elitism: PopulationElitismResult
    selection: PopulationSelectionResult | None
    selection_seed: int
    reproduction_seed: int
    producer_identifier: str
    offspring: tuple[GenerationOffspringRecord, ...]
    population: GenerationPopulation
    fitness: PopulationFitnessResult
    diversity: PopulationDiversityReport | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_fitness, PopulationFitnessResult):
            raise TypeError("source_fitness must be a PopulationFitnessResult")
        if not isinstance(self.elitism, PopulationElitismResult):
            raise TypeError("elitism must be a PopulationElitismResult")
        if self.elitism.source is not self.source_fitness:
            raise ValueError("elitism must refer to the exact source fitness result")
        selection_seed = _nonnegative_integer(
            self.selection_seed,
            name="selection_seed",
        )
        reproduction_seed = _nonnegative_integer(
            self.reproduction_seed,
            name="reproduction_seed",
        )
        producer_identifier = _nonempty_string(
            self.producer_identifier,
            name="producer_identifier",
        )
        if isinstance(self.offspring, (str, bytes, bytearray)) or not isinstance(
            self.offspring,
            Iterable,
        ):
            raise TypeError("offspring must be an iterable of offspring records")
        offspring = tuple(self.offspring)
        if not all(isinstance(record, GenerationOffspringRecord) for record in offspring):
            raise TypeError("offspring must contain only GenerationOffspringRecord values")
        if not isinstance(self.population, GenerationPopulation):
            raise TypeError("population must be a GenerationPopulation")
        if not isinstance(self.fitness, PopulationFitnessResult):
            raise TypeError("fitness must be a PopulationFitnessResult")
        if self.diversity is not None:
            if not isinstance(self.diversity, PopulationDiversityReport):
                raise TypeError(
                    "diversity must be a PopulationDiversityReport or None"
                )
            if self.diversity.population is not self.population:
                raise ValueError("diversity must classify the exact target population")
            if not self.diversity.satisfies_policy:
                raise ValueError("target diversity must satisfy its occupancy policy")

        source_population = self.source_fitness.population
        expected_offspring_count = source_population.population_size - self.elitism.elite_count
        if expected_offspring_count < 0:
            raise ValueError("elite count exceeds source population capacity")
        if len(offspring) != expected_offspring_count:
            raise ValueError("offspring count must fill every non-elite population slot")
        if expected_offspring_count:
            if not isinstance(self.selection, PopulationSelectionResult):
                raise TypeError("a transition with offspring requires selection")
            if self.selection.source is not self.source_fitness:
                raise ValueError("selection must refer to the exact source fitness result")
            if self.selection.seed != selection_seed:
                raise ValueError("selection_seed must match the selection result")
        elif self.selection is not None:
            raise ValueError("a fully elite transition must not run unused selection")

        if self.population.generation_index != source_population.generation_index + 1:
            raise ValueError("target population must immediately follow the source")
        if self.population.population_size != source_population.population_size:
            raise ValueError("generation transitions must preserve population size")
        if self.population.validity_policy is not source_population.validity_policy:
            raise ValueError("generation transitions must preserve validity policy identity")
        if self.population.embedding_dimension != source_population.embedding_dimension:
            raise ValueError("generation transitions must preserve embedding dimension")
        if self.fitness.population is not self.population:
            raise ValueError("fitness must evaluate the exact target population")
        if self.fitness.definition is not self.source_fitness.definition:
            raise ValueError("generation transitions must preserve fitness definition identity")

        _validate_transition_members(
            source_fitness=self.source_fitness,
            elitism=self.elitism,
            selection=self.selection,
            offspring=offspring,
            population=self.population,
        )
        object.__setattr__(self, "selection_seed", selection_seed)
        object.__setattr__(self, "reproduction_seed", reproduction_seed)
        object.__setattr__(self, "producer_identifier", producer_identifier)
        object.__setattr__(self, "offspring", offspring)

    @property
    def target_generation_index(self) -> int:
        """Index of the newly composed and evaluated generation."""
        return self.population.generation_index


@dataclass(frozen=True, slots=True)
class GenerationLoopResult:
    """Initial fitness plus every ordered transition in one in-memory run."""

    initial_population: InitialPopulation
    config: GenerationLoopConfig
    seed: int
    definition: PopulationFitnessDefinition
    producer_identifier: str
    initial_fitness: PopulationFitnessResult
    transitions: tuple[GenerationTransition, ...]
    initial_diversity: PopulationDiversityReport | None = None
    family_classifier_identifier: str | None = None
    rng_algorithm: str = field(default=GENERATION_LOOP_RNG_ALGORITHM, init=False)
    version: int = field(default=GENERATION_LOOP_VERSION, init=False)
    warnings: tuple[str, ...] = field(default=_GENERATION_LOOP_WARNINGS, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.initial_population, InitialPopulation):
            raise TypeError("initial_population must be an InitialPopulation")
        if not isinstance(self.config, GenerationLoopConfig):
            raise TypeError("config must be a GenerationLoopConfig")
        seed = _nonnegative_integer(self.seed, name="seed")
        producer_identifier = _nonempty_string(
            self.producer_identifier,
            name="producer_identifier",
        )
        if not isinstance(self.initial_fitness, PopulationFitnessResult):
            raise TypeError("initial_fitness must be a PopulationFitnessResult")
        if self.initial_fitness.population is not self.initial_population:
            raise ValueError("initial_fitness must evaluate the exact initial population")
        if self.initial_fitness.definition is not self.definition:
            raise ValueError("initial_fitness must use the exact stored definition")
        classifier_identifier = self.family_classifier_identifier
        if self.config.diversity is None:
            if self.initial_diversity is not None or classifier_identifier is not None:
                raise ValueError(
                    "diversity records and classifier identifier require a policy"
                )
        else:
            if not isinstance(self.initial_diversity, PopulationDiversityReport):
                raise TypeError(
                    "a configured diversity policy requires initial_diversity"
                )
            classifier_identifier = _nonempty_string(
                classifier_identifier,
                name="family_classifier_identifier",
            )
            if self.initial_diversity.population is not self.initial_population:
                raise ValueError(
                    "initial_diversity must classify the exact initial population"
                )
            if self.initial_diversity.policy is not self.config.diversity:
                raise ValueError("initial_diversity must use the configured policy")
            if (
                self.initial_diversity.classifier_identifier
                != classifier_identifier
            ):
                raise ValueError(
                    "initial_diversity must use the stored classifier identifier"
                )
            if not self.initial_diversity.satisfies_policy:
                raise ValueError("initial_diversity must satisfy the configured policy")
        if isinstance(self.transitions, (str, bytes, bytearray)) or not isinstance(
            self.transitions,
            Iterable,
        ):
            raise TypeError("transitions must be an iterable of GenerationTransition")
        transitions = tuple(self.transitions)
        if len(transitions) != self.config.generation_count:
            raise ValueError("transitions must match config.generation_count")
        if not all(isinstance(item, GenerationTransition) for item in transitions):
            raise TypeError("transitions must contain only GenerationTransition values")

        expected_seeds = _derive_transition_seeds(seed, len(transitions))
        previous_fitness = self.initial_fitness
        diversity_reports: list[PopulationDiversityReport] = []
        if self.initial_diversity is not None:
            diversity_reports.append(self.initial_diversity)
        for transition_index, (transition, expected) in enumerate(
            zip(transitions, expected_seeds, strict=True)
        ):
            if transition.source_fitness is not previous_fitness:
                raise ValueError("transitions must form one exact fitness-result chain")
            if transition.target_generation_index != transition_index + 1:
                raise ValueError("transition generations must start at one and be consecutive")
            if (
                transition.selection_seed,
                transition.reproduction_seed,
            ) != expected:
                raise ValueError("transition seeds do not match the root PCG64 schedule")
            if transition.producer_identifier != producer_identifier:
                raise ValueError("every transition must use the stored producer identifier")
            if transition.elitism.config is not self.config.elitism:
                raise ValueError("every transition must use the stored elitism config")
            if transition.selection is not None and (
                transition.selection.config is not self.config.selection
            ):
                raise ValueError("every selection must use the stored selection config")
            if self.config.diversity is None:
                if transition.diversity is not None:
                    raise ValueError(
                        "transition diversity records require a configured policy"
                    )
            else:
                if not isinstance(transition.diversity, PopulationDiversityReport):
                    raise TypeError(
                        "every transition requires a target diversity report"
                    )
                if transition.diversity.policy is not self.config.diversity:
                    raise ValueError(
                        "every diversity report must use the configured policy"
                    )
                if (
                    transition.diversity.classifier_identifier
                    != classifier_identifier
                ):
                    raise ValueError(
                        "every diversity report must use the stored classifier"
                    )
                diversity_reports.append(transition.diversity)
            previous_fitness = transition.fitness
        validate_diversity_history(diversity_reports)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "producer_identifier", producer_identifier)
        object.__setattr__(
            self,
            "family_classifier_identifier",
            classifier_identifier,
        )
        object.__setattr__(self, "transitions", transitions)

    @property
    def populations(self) -> tuple[InitialPopulation | GenerationPopulation, ...]:
        """All populations from generation zero through the final generation."""
        return (
            self.initial_population,
            *(transition.population for transition in self.transitions),
        )

    @property
    def fitness_history(self) -> tuple[PopulationFitnessResult, ...]:
        """Fitness ledgers corresponding exactly to ``populations``."""
        return (
            self.initial_fitness,
            *(transition.fitness for transition in self.transitions),
        )

    @property
    def final_fitness(self) -> PopulationFitnessResult:
        """Fitness ledger for the final requested generation."""
        return self.fitness_history[-1]


OffspringProducer: TypeAlias = Callable[
    [GenerationReproductionRequest], Iterable[OffspringProposal]
]
GenerationCheckpointCallback: TypeAlias = Callable[[GenerationLoopResult], None]


def run_generation_loop(
    initial_population: InitialPopulation,
    *,
    definition: PopulationFitnessDefinition,
    evaluator: PopulationEvaluator,
    config: GenerationLoopConfig,
    seed: int,
    offspring_producer: OffspringProducer,
    producer_identifier: str,
    family_classifier: DiversityFamilyClassifier | None = None,
    family_classifier_identifier: str | None = None,
    checkpoint_callback: GenerationCheckpointCallback | None = None,
) -> GenerationLoopResult:
    """Compose fixed-size generations without inventing a variation policy.

    Each transition draws exactly two raw PCG64 words in order: one selection
    seed and one reproduction seed. Complete elite tiers occupy the first
    target slots. The caller's producer fills every remaining slot exactly
    once; invalid proposals raise with the complete batch and are never
    repaired, replaced, filtered, or retried.

    When supplied, ``checkpoint_callback`` observes the complete result prefix
    after generation zero and after every completed target fitness ledger.
    Its prefix config counts completed transitions; callers retain the planned
    target separately. Callback exceptions propagate before another transition.
    """
    if not isinstance(initial_population, InitialPopulation):
        raise TypeError("initial_population must be an InitialPopulation")
    if not isinstance(config, GenerationLoopConfig):
        raise TypeError("config must be a GenerationLoopConfig")
    prepared_seed = _nonnegative_integer(seed, name="seed")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    if not callable(offspring_producer):
        raise TypeError("offspring_producer must be callable")
    if checkpoint_callback is not None and not callable(checkpoint_callback):
        raise TypeError("checkpoint_callback must be callable or None")
    prepared_identifier = _nonempty_string(
        producer_identifier,
        name="producer_identifier",
    )
    diversity_policy = config.diversity
    prepared_classifier_identifier: str | None = None
    if diversity_policy is None:
        if family_classifier is not None or family_classifier_identifier is not None:
            raise ValueError(
                "family classifier inputs require config.diversity"
            )
    else:
        if not callable(family_classifier):
            raise TypeError(
                "a configured diversity policy requires a callable family_classifier"
            )
        prepared_classifier_identifier = _nonempty_string(
            family_classifier_identifier,
            name="family_classifier_identifier",
        )

    initial_diversity: PopulationDiversityReport | None = None
    if diversity_policy is not None:
        assert family_classifier is not None
        assert prepared_classifier_identifier is not None
        initial_diversity = assess_population_diversity(
            initial_population,
            policy=diversity_policy,
            classifier=family_classifier,
            classifier_identifier=prepared_classifier_identifier,
        )
        validate_diversity_history((initial_diversity,))
        initial_diversity.raise_for_violations()

    initial_fitness = evaluate_population_fitness(
        initial_population,
        definition=definition,
        evaluator=evaluator,
    )
    prefix = GenerationLoopResult(
        initial_population=initial_population,
        config=replace(config, generation_count=0),
        seed=prepared_seed,
        definition=definition,
        producer_identifier=prepared_identifier,
        initial_fitness=initial_fitness,
        transitions=(),
        initial_diversity=initial_diversity,
        family_classifier_identifier=prepared_classifier_identifier,
    )
    if checkpoint_callback is not None:
        checkpoint_callback(prefix)
    return _continue_generation_loop(
        prefix,
        config=config,
        evaluator=evaluator,
        offspring_producer=offspring_producer,
        family_classifier=family_classifier,
        checkpoint_callback=checkpoint_callback,
    )


def _continue_generation_loop(
    prefix: GenerationLoopResult,
    *,
    config: GenerationLoopConfig,
    evaluator: PopulationEvaluator,
    offspring_producer: OffspringProducer,
    family_classifier: DiversityFamilyClassifier | None,
    checkpoint_callback: GenerationCheckpointCallback | None,
) -> GenerationLoopResult:
    """Shared transition engine for a fresh run and a checked resume prefix.

    Callers validate external policies first. Only new generations trigger
    evaluation, classification, offspring production, and checkpoint callbacks.
    """
    initial_population = prefix.initial_population
    initial_fitness = prefix.initial_fitness
    initial_diversity = prefix.initial_diversity
    definition = prefix.definition
    prepared_seed = prefix.seed
    prepared_identifier = prefix.producer_identifier
    prepared_classifier_identifier = prefix.family_classifier_identifier
    diversity_policy = config.diversity
    diversity_history = [
        report for report in (
            initial_diversity, *(transition.diversity for transition in prefix.transitions),
        ) if report is not None
    ]
    current_fitness = prefix.final_fitness
    current_diversity = diversity_history[-1] if diversity_history else None
    transition_seeds = _derive_transition_seeds(
        prepared_seed,
        config.generation_count,
    )[len(prefix.transitions):]
    transitions = list(prefix.transitions)

    def completed_prefix() -> GenerationLoopResult:
        return GenerationLoopResult(
            initial_population=initial_population,
            config=replace(config, generation_count=len(transitions)),
            seed=prepared_seed,
            definition=definition,
            producer_identifier=prepared_identifier,
            initial_fitness=initial_fitness,
            transitions=tuple(transitions),
            initial_diversity=initial_diversity,
            family_classifier_identifier=prepared_classifier_identifier,
        )

    for selection_seed, reproduction_seed in transition_seeds:
        source_population = current_fitness.population
        target_generation_index = source_population.generation_index + 1
        elitism = identify_population_elites(
            current_fitness,
            config=config.elitism,
        )
        required_offspring_count = (
            source_population.population_size - elitism.elite_count
        )
        selection: PopulationSelectionResult | None = None
        offspring_records: tuple[GenerationOffspringRecord, ...] = ()

        if required_offspring_count:
            selection = select_population_members(
                current_fitness,
                config=config.selection,
                seed=selection_seed,
            )
            request = GenerationReproductionRequest(
                source=current_fitness,
                selection=selection,
                target_generation_index=target_generation_index,
                required_offspring_count=required_offspring_count,
                seed=reproduction_seed,
                validity_policy=source_population.validity_policy,
                source_diversity=current_diversity,
            )
            proposals = _collect_proposals(
                offspring_producer(request),
                required_count=required_offspring_count,
            )
            offspring_records = _validate_offspring(
                request,
                proposals=proposals,
            )
            if any(not record.validity.is_valid for record in offspring_records):
                raise InvalidGenerationOffspringError(
                    request=request,
                    records=offspring_records,
                )

        population = _compose_population(
            source_fitness=current_fitness,
            elitism=elitism,
            selection=selection,
            offspring=offspring_records,
            target_generation_index=target_generation_index,
        )
        target_diversity: PopulationDiversityReport | None = None
        if diversity_policy is not None:
            assert family_classifier is not None
            assert prepared_classifier_identifier is not None
            target_diversity = assess_population_diversity(
                population,
                policy=diversity_policy,
                classifier=family_classifier,
                classifier_identifier=prepared_classifier_identifier,
            )
            validate_diversity_history((*diversity_history, target_diversity))
            target_diversity.raise_for_violations()
        target_fitness = evaluate_population_fitness(
            population,
            definition=definition,
            evaluator=evaluator,
        )
        transition = GenerationTransition(
            source_fitness=current_fitness,
            elitism=elitism,
            selection=selection,
            selection_seed=selection_seed,
            reproduction_seed=reproduction_seed,
            producer_identifier=prepared_identifier,
            offspring=offspring_records,
            population=population,
            fitness=target_fitness,
            diversity=target_diversity,
        )
        transitions.append(transition)
        current_fitness = target_fitness
        current_diversity = target_diversity
        if target_diversity is not None:
            diversity_history.append(target_diversity)
        if checkpoint_callback is not None:
            checkpoint_callback(completed_prefix())

    return GenerationLoopResult(
        initial_population=initial_population,
        config=config,
        seed=prepared_seed,
        definition=definition,
        producer_identifier=prepared_identifier,
        initial_fitness=initial_fitness,
        transitions=tuple(transitions),
        initial_diversity=initial_diversity,
        family_classifier_identifier=prepared_classifier_identifier,
    )


def _collect_proposals(
    values: Iterable[OffspringProposal],
    *,
    required_count: int,
) -> tuple[OffspringProposal, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Iterable):
        raise TypeError("offspring_producer must return an iterable of proposals")
    proposals = tuple(values)
    if len(proposals) != required_count:
        raise ValueError(
            "offspring_producer must return exactly required_offspring_count proposals"
        )
    if not all(isinstance(proposal, OffspringProposal) for proposal in proposals):
        raise TypeError("offspring_producer must return only OffspringProposal values")
    return proposals


def _validate_offspring(
    request: GenerationReproductionRequest,
    *,
    proposals: tuple[OffspringProposal, ...],
) -> tuple[GenerationOffspringRecord, ...]:
    selected = request.selection.selected_members
    records: list[GenerationOffspringRecord] = []
    for offspring_index, proposal in enumerate(proposals):
        if any(index >= len(selected) for index in proposal.parent_selection_indices):
            raise ValueError("offspring parent selection index is outside selection output")
        source_member = selected[proposal.validation_source_selection_index]
        validity = validate_geometry_mutation(
            source_member.population_member.genome,
            proposal.genome,
            policy=request.validity_policy,
        )
        records.append(
            GenerationOffspringRecord(
                offspring_index=offspring_index,
                proposal=proposal,
                validity=validity,
            )
        )
    return tuple(records)


def _compose_population(
    *,
    source_fitness: PopulationFitnessResult,
    elitism: PopulationElitismResult,
    selection: PopulationSelectionResult | None,
    offspring: tuple[GenerationOffspringRecord, ...],
    target_generation_index: int,
) -> GenerationPopulation:
    policy = source_fitness.population.validity_policy
    member_inputs: list[
        tuple[GeometryGenome, GeometryGenome, MutationValidityReport]
    ] = []
    for elite in elitism.elite_members:
        genome = elite.population_member.genome
        member_inputs.append(
            (genome, genome, validate_geometry_mutation(genome, genome, policy=policy))
        )
    if offspring:
        assert selection is not None
        selected = selection.selected_members
        for record in offspring:
            source = selected[
                record.proposal.validation_source_selection_index
            ].population_member.genome
            member_inputs.append((record.proposal.genome, source, record.validity))

    members = tuple(
        GenerationPopulationMember(
            generation_index=target_generation_index,
            member_index=member_index,
            genome=genome,
            validation_source=validation_source,
            validity=validity,
        )
        for member_index, (genome, validation_source, validity) in enumerate(
            member_inputs
        )
    )
    return GenerationPopulation(
        generation_index=target_generation_index,
        members=members,
        validity_policy=policy,
        embedding_dimension=source_fitness.population.embedding_dimension,
    )


def _validate_transition_members(
    *,
    source_fitness: PopulationFitnessResult,
    elitism: PopulationElitismResult,
    selection: PopulationSelectionResult | None,
    offspring: tuple[GenerationOffspringRecord, ...],
    population: GenerationPopulation,
) -> None:
    elite_count = elitism.elite_count
    for target_member, elite in zip(
        population.members[:elite_count],
        elitism.elite_members,
        strict=True,
    ):
        source_genome = elite.population_member.genome
        if target_member.genome is not source_genome:
            raise ValueError("elite members must preserve exact genome identity")
        if target_member.validation_source is not source_genome:
            raise ValueError("elite validation source must be the retained genome")

    if not offspring:
        return
    assert selection is not None
    selected = selection.selected_members
    target_offspring = population.members[elite_count:]
    for offspring_index, (target_member, record) in enumerate(
        zip(target_offspring, offspring, strict=True)
    ):
        if record.offspring_index != offspring_index:
            raise ValueError("offspring indices must match offspring order exactly")
        proposal = record.proposal
        if any(index >= len(selected) for index in proposal.parent_selection_indices):
            raise ValueError("offspring parent selection index is outside selection output")
        validation_source = selected[
            proposal.validation_source_selection_index
        ].population_member.genome
        expected_validity = validate_geometry_mutation(
            validation_source,
            proposal.genome,
            policy=source_fitness.population.validity_policy,
        )
        if not record.validity.is_valid or record.validity != expected_validity:
            raise ValueError("offspring record must contain its exact valid policy report")
        if target_member.genome is not proposal.genome:
            raise ValueError("target offspring must preserve proposal genome identity")
        if target_member.validation_source is not validation_source:
            raise ValueError("target offspring must preserve its validation source")
        if target_member.validity != record.validity:
            raise ValueError("target offspring must preserve its validity report")


def _derive_transition_seeds(
    seed: int,
    generation_count: int,
) -> tuple[tuple[int, int], ...]:
    random_number_generator = np.random.Generator(np.random.PCG64(seed))
    return tuple(
        (
            int(random_number_generator.bit_generator.random_raw()),
            int(random_number_generator.bit_generator.random_raw()),
        )
        for _ in range(generation_count)
    )


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


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
