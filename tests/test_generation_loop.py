from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import BasisLayout
from toposc_lab.evaluation import (
    BasicScoreComponent,
    GeometryEvaluationRun,
    GeometryModelAdapter,
    ObjectiveDirection,
    evaluate_geometry,
)
from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.search import (
    GENERATION_LOOP_RNG_ALGORITHM,
    GENERATION_LOOP_VERSION,
    GENERATION_POPULATION_VERSION,
    ElitismConfig,
    GenerationLoopConfig,
    GenerationPopulation,
    GenerationPopulationMember,
    GenerationReproductionRequest,
    GeometryGenome,
    InitialPopulation,
    InitialPopulationMember,
    InvalidGenerationOffspringError,
    MutationValidityPolicy,
    OffspringProposal,
    ScalarFitnessDefinition,
    TournamentSelectionConfig,
    create_initial_population,
    geometry_to_genome,
    remove_edge_mutation,
    run_generation_loop,
    validate_geometry_mutation,
)


class _LoopModel(BaseModel):
    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry

    @property
    def parameters(self) -> dict[str, Any]:
        return {"energy_scale": float(self.geometry.n_sites)}

    @property
    def basis_layout(self) -> BasisLayout:
        return BasisLayout(
            spatial_shape=(self.geometry.n_sites,),
            components_per_site=1,
            ordering="site_major",
            component_labels=("orbital",),
        )

    def hamiltonian(self) -> np.ndarray:
        negative_count = self.geometry.n_sites // 2
        positive_count = self.geometry.n_sites - negative_count
        scale = float(self.geometry.n_sites)
        eigenvalues = np.concatenate(
            (
                -scale * np.arange(1, negative_count + 1, dtype=float),
                scale * np.arange(1, positive_count + 1, dtype=float),
            )
        )
        return np.diag(eigenvalues).astype(complex)


PopulationMember = InitialPopulationMember | GenerationPopulationMember


def _genome(n_sites: int, *, edge_count: int | None = None) -> GeometryGenome:
    if edge_count is None:
        edge_count = n_sites - 1
    possible_edges = tuple(
        GeometryEdge(source, target)
        for source in range(n_sites)
        for target in range(source + 1, n_sites)
    )
    return geometry_to_genome(
        Geometry(n_sites=n_sites, edges=possible_edges[:edge_count])
    )


def _population(
    *,
    duplicate: bool = False,
    policy: MutationValidityPolicy | None = None,
) -> InitialPopulation:
    genomes = (_genome(3),) * 4 if duplicate else tuple(_genome(size) for size in range(2, 6))
    return create_initial_population(genomes, validity_policy=policy)


def _definition() -> ScalarFitnessDefinition:
    return ScalarFitnessDefinition(
        weights={BasicScoreComponent.NORMALIZED_GAP: 1.0},
        direction=ObjectiveDirection.MAXIMIZE,
    )


def _evaluate(member: PopulationMember) -> GeometryEvaluationRun:
    return evaluate_geometry(
        member.genome.to_geometry(),
        adapter=GeometryModelAdapter(model_factory=_LoopModel),
        seed=100 * member.generation_index + member.member_index,
        code_version="phase-10.15-test",
    )


def _config(
    *,
    generation_count: int = 2,
    minimum_elites: int = 1,
) -> GenerationLoopConfig:
    return GenerationLoopConfig(
        generation_count=generation_count,
        selection=TournamentSelectionConfig(selection_count=4, tournament_size=2),
        elitism=ElitismConfig(minimum_elite_count=minimum_elites),
    )


def _identity_producer(
    request: GenerationReproductionRequest,
) -> tuple[OffspringProposal, ...]:
    return tuple(
        OffspringProposal(
            genome=request.selected_members[index % len(request.selected_members)]
            .population_member.genome,
            parent_selection_indices=(index % len(request.selected_members),),
            validation_source_selection_index=index % len(request.selected_members),
            operator_identifier="test.identity.v1",
            operator_seed=request.seed,
        )
        for index in range(request.required_offspring_count)
    )


def test_loop_composes_and_evaluates_consecutive_fixed_size_generations() -> None:
    initial = _population()
    result = run_generation_loop(
        initial,
        definition=_definition(),
        evaluator=_evaluate,
        config=_config(),
        seed=123,
        offspring_producer=_identity_producer,
        producer_identifier="tests.identity-producer.v1",
    )

    assert result.initial_population is initial
    assert tuple(population.generation_index for population in result.populations) == (
        0,
        1,
        2,
    )
    assert tuple(population.population_size for population in result.populations) == (
        4,
        4,
        4,
    )
    assert tuple(
        tuple(member.generation_index for member in population.members)
        for population in result.populations
    ) == ((0, 0, 0, 0), (1, 1, 1, 1), (2, 2, 2, 2))
    assert tuple(
        tuple(member.member_index for member in population.members)
        for population in result.populations
    ) == ((0, 1, 2, 3),) * 3
    assert tuple(fitness.population for fitness in result.fitness_history) == (
        result.populations
    )
    assert result.final_fitness is result.transitions[-1].fitness
    assert result.version == GENERATION_LOOP_VERSION
    assert result.rng_algorithm == GENERATION_LOOP_RNG_ALGORITHM


def test_root_seed_has_a_fixed_two_word_schedule_per_transition() -> None:
    result = run_generation_loop(
        _population(),
        definition=_definition(),
        evaluator=_evaluate,
        config=_config(),
        seed=123,
        offspring_producer=_identity_producer,
        producer_identifier="tests.identity-producer.v1",
    )

    assert tuple(
        (transition.selection_seed, transition.reproduction_seed)
        for transition in result.transitions
    ) == (
        (12587170189557361101, 992822559630912803),
        (4064922177151560342, 3401059606364785669),
    )


def test_complete_elite_tier_is_retained_first_with_exact_genome_identity() -> None:
    result = run_generation_loop(
        _population(),
        definition=_definition(),
        evaluator=_evaluate,
        config=_config(generation_count=1),
        seed=8,
        offspring_producer=_identity_producer,
        producer_identifier="tests.identity-producer.v1",
    )
    transition = result.transitions[0]

    for target_member, elite in zip(
        transition.population.members,
        transition.elitism.elite_members,
    ):
        assert target_member.genome is elite.population_member.genome
        assert target_member.validation_source is elite.population_member.genome


def test_cutoff_expansion_can_fill_capacity_without_unused_selection_or_producer() -> None:
    producer_calls = 0

    def producer(
        request: GenerationReproductionRequest,
    ) -> tuple[OffspringProposal, ...]:
        nonlocal producer_calls
        del request
        producer_calls += 1
        return ()

    initial = _population(duplicate=True)
    result = run_generation_loop(
        initial,
        definition=_definition(),
        evaluator=_evaluate,
        config=_config(generation_count=1),
        seed=9,
        offspring_producer=producer,
        producer_identifier="tests.must-not-run.v1",
    )
    transition = result.transitions[0]

    assert transition.elitism.cutoff_was_expanded
    assert transition.elitism.elite_count == initial.population_size
    assert transition.selection is None
    assert transition.offspring == ()
    assert producer_calls == 0
    assert all(
        target is source
        for target, source in zip(
            transition.population.genomes,
            initial.genomes,
            strict=True,
        )
    )


def test_zero_transitions_evaluates_only_generation_zero() -> None:
    calls: list[tuple[int, int]] = []

    def evaluator(member: PopulationMember) -> GeometryEvaluationRun:
        calls.append((member.generation_index, member.member_index))
        return _evaluate(member)

    result = run_generation_loop(
        _population(),
        definition=_definition(),
        evaluator=evaluator,
        config=_config(generation_count=0),
        seed=1,
        offspring_producer=_identity_producer,
        producer_identifier="tests.identity-producer.v1",
    )

    assert result.transitions == ()
    assert len(result.populations) == len(result.fitness_history) == 1
    assert calls == [(0, 0), (0, 1), (0, 2), (0, 3)]


def test_producer_receives_exact_selection_capacity_policy_and_seed() -> None:
    seen: list[GenerationReproductionRequest] = []

    def producer(
        request: GenerationReproductionRequest,
    ) -> tuple[OffspringProposal, ...]:
        seen.append(request)
        return _identity_producer(request)

    initial = _population()
    result = run_generation_loop(
        initial,
        definition=_definition(),
        evaluator=_evaluate,
        config=_config(generation_count=1),
        seed=21,
        offspring_producer=producer,
        producer_identifier="tests.inspect.v1",
    )
    transition = result.transitions[0]
    request = seen[0]

    assert request.source is result.initial_fitness
    assert request.selection is transition.selection
    assert request.required_offspring_count == (
        initial.population_size - transition.elitism.elite_count
    )
    assert request.target_generation_index == 1
    assert request.seed == transition.reproduction_seed
    assert request.validity_policy is initial.validity_policy


def test_invalid_batch_is_retained_without_filter_repair_or_retry() -> None:
    policy = MutationValidityPolicy(require_connected=True)
    calls = 0

    def producer(
        request: GenerationReproductionRequest,
    ) -> tuple[OffspringProposal, ...]:
        nonlocal calls
        calls += 1
        proposals = list(_identity_producer(request))
        proposals[1] = replace(
            proposals[1],
            genome=_genome(3, edge_count=0),
            operator_identifier="test.invalid-disconnected.v1",
        )
        return tuple(proposals)

    with pytest.raises(
        InvalidGenerationOffspringError,
        match="generation 1 contains invalid offspring",
    ) as caught:
        run_generation_loop(
            _population(policy=policy),
            definition=_definition(),
            evaluator=_evaluate,
            config=_config(generation_count=1),
            seed=13,
            offspring_producer=producer,
            producer_identifier="tests.invalid.v1",
        )

    assert calls == 1
    assert len(caught.value.records) == caught.value.request.required_offspring_count
    assert tuple(record.offspring_index for record in caught.value.records) == (0, 1, 2)
    assert tuple(record.offspring_index for record in caught.value.invalid_records) == (1,)
    assert tuple(
        issue.code for issue in caught.value.invalid_records[0].validity.issues
    ) == ("disconnected_candidate",)


def test_duplicate_offspring_are_retained_without_diversity_filtering() -> None:
    def duplicate_producer(
        request: GenerationReproductionRequest,
    ) -> tuple[OffspringProposal, ...]:
        source = request.selected_members[0].population_member.genome
        return tuple(
            OffspringProposal(
                genome=source,
                parent_selection_indices=(0,),
                validation_source_selection_index=0,
                operator_identifier="test.duplicate.v1",
            )
            for _ in range(request.required_offspring_count)
        )

    transition = run_generation_loop(
        _population(),
        definition=_definition(),
        evaluator=_evaluate,
        config=_config(generation_count=1),
        seed=4,
        offspring_producer=duplicate_producer,
        producer_identifier="tests.duplicate.v1",
    ).transitions[0]

    offspring_genomes = transition.population.genomes[transition.elitism.elite_count :]
    assert len(offspring_genomes) == 3
    assert all(genome is offspring_genomes[0] for genome in offspring_genomes)


def test_producer_can_compose_an_existing_deterministic_mutation_primitive() -> None:
    def mutation_producer(
        request: GenerationReproductionRequest,
    ) -> tuple[OffspringProposal, ...]:
        proposals: list[OffspringProposal] = []
        for offspring_index in range(request.required_offspring_count):
            parent_slot = offspring_index % len(request.selected_members)
            parent = request.selected_members[parent_slot].population_member.genome
            proposals.append(
                OffspringProposal(
                    genome=remove_edge_mutation(parent, len(parent.edges) - 1),
                    parent_selection_indices=(parent_slot,),
                    validation_source_selection_index=parent_slot,
                    operator_identifier="remove_edge_mutation.v1",
                )
            )
        return tuple(proposals)

    transition = run_generation_loop(
        _population(),
        definition=_definition(),
        evaluator=_evaluate,
        config=_config(generation_count=1),
        seed=28,
        offspring_producer=mutation_producer,
        producer_identifier="tests.remove-edge-producer.v1",
    ).transitions[0]

    assert transition.selection is not None
    for record in transition.offspring:
        parent_slot = record.proposal.validation_source_selection_index
        parent = transition.selection.selected_members[parent_slot]
        assert len(record.proposal.genome.edges) == len(parent.population_member.genome.edges) - 1
        assert record.validity.is_valid


@pytest.mark.parametrize("returned_count", (0, 2, 4))
def test_producer_must_fill_exactly_the_non_elite_capacity(returned_count: int) -> None:
    def wrong_count(
        request: GenerationReproductionRequest,
    ) -> tuple[OffspringProposal, ...]:
        proposals = _identity_producer(request)
        return proposals[:returned_count] if returned_count < 3 else (*proposals, proposals[0])

    with pytest.raises(ValueError, match="exactly required_offspring_count"):
        run_generation_loop(
            _population(),
            definition=_definition(),
            evaluator=_evaluate,
            config=_config(generation_count=1),
            seed=2,
            offspring_producer=wrong_count,
            producer_identifier="tests.wrong-count.v1",
        )


def test_parent_provenance_must_reference_the_selection_output() -> None:
    def invalid_parent(
        request: GenerationReproductionRequest,
    ) -> tuple[OffspringProposal, ...]:
        proposals = list(_identity_producer(request))
        proposals[0] = replace(
            proposals[0],
            parent_selection_indices=(99,),
            validation_source_selection_index=99,
        )
        return tuple(proposals)

    with pytest.raises(ValueError, match="outside selection output"):
        run_generation_loop(
            _population(),
            definition=_definition(),
            evaluator=_evaluate,
            config=_config(generation_count=1),
            seed=2,
            offspring_producer=invalid_parent,
            producer_identifier="tests.invalid-parent.v1",
        )


def test_ordinary_target_evaluation_failure_remains_in_fitness_ledger() -> None:
    def evaluator(member: PopulationMember) -> GeometryEvaluationRun:
        if member.generation_index == 1 and member.member_index == 2:
            raise RuntimeError("declared target failure")
        return _evaluate(member)

    transition = run_generation_loop(
        _population(),
        definition=_definition(),
        evaluator=evaluator,
        config=_config(generation_count=1),
        seed=15,
        offspring_producer=_identity_producer,
        producer_identifier="tests.identity-producer.v1",
    ).transitions[0]

    failed = transition.fitness.members[2]
    assert failed.failure is not None
    assert failed.failure.error_type == "RuntimeError"
    assert failed.failure.message == "declared target failure"
    assert len(transition.fitness.members) == transition.population.population_size


def test_generation_population_constructor_rechecks_transition_validity() -> None:
    genome = _genome(3)
    source = geometry_to_genome(
        Geometry(
            n_sites=3,
            edges=(GeometryEdge(0, 1), GeometryEdge(1, 2)),
            coordinates=np.arange(3, dtype=float).reshape(3, 1),
        )
    )
    policy = MutationValidityPolicy()
    wrong_report = validate_geometry_mutation(genome, genome, policy=policy)
    member = GenerationPopulationMember(1, 0, genome, source, wrong_report)

    with pytest.raises(ValueError, match="validation source"):
        GenerationPopulation(
            generation_index=1,
            members=(member,),
            validity_policy=policy,
            embedding_dimension=None,
        )


def test_result_constructor_rejects_seed_schedule_tampering() -> None:
    result = run_generation_loop(
        _population(),
        definition=_definition(),
        evaluator=_evaluate,
        config=_config(generation_count=1),
        seed=44,
        offspring_producer=_identity_producer,
        producer_identifier="tests.identity-producer.v1",
    )

    with pytest.raises(ValueError, match="root PCG64 schedule"):
        replace(result, seed=45)


def test_versions_and_scientific_boundaries_are_explicit() -> None:
    result = run_generation_loop(
        _population(),
        definition=_definition(),
        evaluator=_evaluate,
        config=_config(generation_count=1),
        seed=5,
        offspring_producer=_identity_producer,
        producer_identifier="tests.identity-producer.v1",
    )

    assert result.transitions[0].population.population_version == (
        GENERATION_POPULATION_VERSION
    )
    warning_text = " ".join(result.warnings).lower()
    assert "diversity" in warning_text
    assert "novelty" in warning_text
    assert "checkpoint" in warning_text
    assert "scientific evidence" in warning_text
    assert not hasattr(result, "diversity_score")
    assert not hasattr(result, "checkpoint")


@pytest.mark.parametrize(
    ("generation_count", "error_type"),
    ((-1, ValueError), (True, TypeError), (1.5, TypeError)),
)
def test_config_rejects_invalid_generation_count(
    generation_count: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        _config(generation_count=generation_count)  # type: ignore[arg-type]


def test_proposal_requires_declared_validation_parent_and_operator() -> None:
    genome = _genome(3)

    with pytest.raises(ValueError, match="declared parent"):
        OffspringProposal(genome, (0,), 1, "test.operator")
    with pytest.raises(ValueError, match="non-empty"):
        OffspringProposal(genome, (0,), 0, "  ")
