from __future__ import annotations

from dataclasses import replace
from pathlib import Path
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
    ObjectiveQuantity,
    ObjectiveSpec,
    evaluate_geometry,
)
from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.search import (
    DiversityPreservationPolicy,
    ElitismConfig,
    GenerationLoopConfig,
    GenerationLoopResult,
    GenerationReproductionRequest,
    GeometryGenome,
    GeometrySearchFamily,
    IncompatibleSearchCheckpointError,
    InconsistentGeometryFamilyError,
    InvalidGenerationOffspringError,
    MultiObjectiveFitnessDefinition,
    NoveltyScoreConfig,
    OffspringProposal,
    ScalarFitnessDefinition,
    SearchCheckpoint,
    SearchCheckpointError,
    TournamentSelectionConfig,
    add_edge_mutation,
    create_initial_population,
    create_search_checkpoint,
    evaluate_population_novelty,
    geometry_to_genome,
    load_search_checkpoint,
    remove_edge_mutation,
    resume_search,
    run_generation_loop,
    save_search_checkpoint,
)
from toposc_lab.search.population_fitness import FitnessPopulationMember


class _ResumeModel(BaseModel):
    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry

    @property
    def parameters(self) -> dict[str, Any]:
        return {"hopping": 0.1}

    @property
    def basis_layout(self) -> BasisLayout:
        return BasisLayout((self.geometry.n_sites,))

    def hamiltonian(self) -> np.ndarray:
        size = self.geometry.n_sites
        matrix = np.diag(np.linspace(-size, size, size)).astype(complex)
        for edge in self.geometry.edges:
            matrix[edge.source, edge.target] = matrix[edge.target, edge.source] = 0.1
        return matrix


def _evaluate(member: FitnessPopulationMember) -> GeometryEvaluationRun:
    return evaluate_geometry(
        member.genome.to_geometry(),
        adapter=GeometryModelAdapter(_ResumeModel),
        seed=member.generation_index * 10 + member.member_index,
        code_version="tests.resume.v1",
    )


def _classify(genome: GeometryGenome) -> GeometrySearchFamily:
    return GeometrySearchFamily("declared_fixture", 1)


def _produce(request: GenerationReproductionRequest) -> tuple[OffspringProposal, ...]:
    rng = np.random.Generator(np.random.PCG64(request.seed))
    offspring: list[OffspringProposal] = []
    for _ in range(request.required_offspring_count):
        parent_slot = int(rng.integers(len(request.selected_members)))
        genome = request.selected_members[parent_slot].population_member.genome
        endpoints = tuple(sorted(int(x) for x in rng.choice(genome.n_sites, 2, replace=False)))
        existing = next(
            (
                index
                for index, edge in enumerate(genome.edges)
                if tuple(sorted((edge.source, edge.target))) == endpoints
            ),
            None,
        )
        mutated = (
            add_edge_mutation(genome, GeometryEdge(*endpoints))
            if existing is None
            else remove_edge_mutation(genome, existing)
        )
        offspring.append(
            OffspringProposal(
                genome=mutated,
                parent_selection_indices=(parent_slot,),
                validation_source_selection_index=parent_slot,
                operator_identifier="tests.seeded-edge-toggle.v1",
                operator_seed=request.seed,
            )
        )
    return tuple(offspring)


def _run(**overrides: Any) -> GenerationLoopResult:
    population = create_initial_population(
        tuple(
            geometry_to_genome(
                Geometry(
                    n_sites=size,
                    edges=tuple(GeometryEdge(i, i + 1) for i in range(size - 1)),
                )
            )
            for size in (3, 4, 5, 6)
        )
    )
    options: dict[str, Any] = {
        "initial_population": population,
        "definition": ScalarFitnessDefinition(
            weights={BasicScoreComponent.NORMALIZED_GAP: 1.0},
            direction=ObjectiveDirection.MAXIMIZE,
        ),
        "evaluator": _evaluate,
        "config": GenerationLoopConfig(
            generation_count=4,
            selection=TournamentSelectionConfig(4, 2),
            elitism=ElitismConfig(1),
            diversity=DiversityPreservationPolicy(minimum_distinct_families=1),
        ),
        "seed": 1649,
        "offspring_producer": _produce,
        "producer_identifier": "tests.seeded-producer.v1",
        "family_classifier": _classify,
        "family_classifier_identifier": "tests.classifier.v1",
    }
    options.update(overrides)
    return run_generation_loop(**options)


def _checkpoint(prefix: GenerationLoopResult) -> SearchCheckpoint:
    return create_search_checkpoint(
        prefix,
        requested_generation_count=4,
        evaluator_identifier="tests.evaluator.v1",
        code_version="tests.resume.v1",
    )


def _resume(checkpoint: SearchCheckpoint, **overrides: Any) -> SearchCheckpoint:
    options: dict[str, Any] = {
        "evaluator": _evaluate,
        "evaluator_identifier": "tests.evaluator.v1",
        "offspring_producer": _produce,
        "producer_identifier": "tests.seeded-producer.v1",
        "code_version": "tests.resume.v1",
        "family_classifier": _classify,
        "family_classifier_identifier": "tests.classifier.v1",
    }
    options.update(overrides)
    return resume_search(checkpoint, **options)


class _Interrupted(RuntimeError):
    pass


def _interrupt_at(path: Path, completed: int, **options: Any) -> SearchCheckpoint:
    def save_and_interrupt(prefix: GenerationLoopResult) -> None:
        if prefix.config.generation_count == completed:
            save_search_checkpoint(path, _checkpoint(prefix))
            raise _Interrupted

    with pytest.raises(_Interrupted):
        _run(checkpoint_callback=save_and_interrupt, **options)
    return load_search_checkpoint(path)


@pytest.mark.parametrize("completed", (0, 1, 3))
def test_disk_resume_matches_uninterrupted_search_byte_for_byte(
    tmp_path: Path, completed: int
) -> None:
    full = _checkpoint(_run())
    stored = _interrupt_at(tmp_path / "prefix.zip", completed)
    resumed = _resume(stored)
    first = save_search_checkpoint(tmp_path / "full.zip", full)
    second = save_search_checkpoint(tmp_path / "resumed.zip", resumed)
    assert first.read_bytes() == second.read_bytes()
    assert resumed.result.initial_population is stored.result.initial_population
    assert all(
        first is second
        for first, second in zip(
            stored.result.transitions,
            resumed.result.transitions,
        )
    )


def test_only_new_generations_invoke_evaluation_reproduction_and_classifier(tmp_path: Path) -> None:
    stored = _interrupt_at(tmp_path / "prefix.zip", 2)
    evaluated: list[tuple[int, int]] = []
    produced: list[int] = []
    classified: list[GeometryGenome] = []
    observed: list[SearchCheckpoint] = []

    def evaluator(member: FitnessPopulationMember) -> GeometryEvaluationRun:
        evaluated.append((member.generation_index, member.member_index))
        return _evaluate(member)

    def producer(request: GenerationReproductionRequest) -> tuple[OffspringProposal, ...]:
        produced.append(request.target_generation_index)
        return _produce(request)

    def classifier(genome: GeometryGenome) -> GeometrySearchFamily:
        classified.append(genome)
        return _classify(genome)

    resumed = _resume(
        stored,
        evaluator=evaluator,
        offspring_producer=producer,
        family_classifier=classifier,
        checkpoint_callback=observed.append,
    )
    assert evaluated == [(generation, member) for generation in (3, 4) for member in range(4)]
    assert all(generation in (3, 4) for generation in produced)
    assert len(classified) == 8
    assert [item.completed_generation_index for item in observed] == [3, 4]
    assert all(item.requested_generation_count == 4 for item in observed)
    assert resumed.result.initial_fitness is stored.result.initial_fitness
    assert resumed.result.transitions[2].source_fitness is stored.result.final_fitness


def test_repeated_disk_interruptions_keep_original_target_and_seed_schedule(tmp_path: Path) -> None:
    path = tmp_path / "latest.zip"
    stored = _interrupt_at(path, 1)

    def save_and_stop(checkpoint: SearchCheckpoint) -> None:
        save_search_checkpoint(path, checkpoint)
        raise _Interrupted

    with pytest.raises(_Interrupted):
        _resume(stored, checkpoint_callback=save_and_stop)
    second = load_search_checkpoint(path)
    assert second.completed_generation_index == 2
    assert second.requested_generation_count == 4
    final = _resume(second)
    full_path = save_search_checkpoint(tmp_path / "full.zip", _checkpoint(_run()))
    final_path = save_search_checkpoint(tmp_path / "final.zip", final)
    assert full_path.read_bytes() == final_path.read_bytes()


def test_completed_checkpoint_is_an_identity_preserving_noop() -> None:
    stored = _checkpoint(_run())

    def unexpected(*args: Any) -> Any:
        pytest.fail("completed search must not invoke a callback")

    assert (
        _resume(
            stored,
            evaluator=unexpected,
            offspring_producer=unexpected,
            family_classifier=unexpected,
            checkpoint_callback=unexpected,
        )
        is stored
    )


@pytest.mark.parametrize(
    "field",
    (
        "evaluator_identifier",
        "producer_identifier",
        "code_version",
        "family_classifier_identifier",
    ),
)
def test_changed_policy_is_rejected_before_any_search_callback(tmp_path: Path, field: str) -> None:
    stored = _interrupt_at(tmp_path / "prefix.zip", 1)

    def unexpected(*args: Any) -> Any:
        pytest.fail("incompatible resume invoked a callback")

    with pytest.raises(IncompatibleSearchCheckpointError, match=field):
        _resume(
            stored,
            evaluator=unexpected,
            offspring_producer=unexpected,
            family_classifier=unexpected,
            checkpoint_callback=unexpected,
            **{field: "changed.v2"},
        )


@pytest.mark.parametrize("runtime", ("python", "numpy", "scipy", "toposc-lab", "extra"))
def test_runtime_version_mismatch_is_rejected(tmp_path: Path, runtime: str) -> None:
    stored = _interrupt_at(tmp_path / "prefix.zip", 0)
    stored = replace(stored, runtime_versions={**stored.runtime_versions, runtime: "different"})
    with pytest.raises(IncompatibleSearchCheckpointError, match="runtime version"):
        _resume(stored)


def test_missing_runtime_inventory_is_rejected(tmp_path: Path) -> None:
    stored = _interrupt_at(tmp_path / "prefix.zip", 0)
    stored = replace(stored, runtime_versions={"python": stored.runtime_versions["python"]})
    with pytest.raises(IncompatibleSearchCheckpointError, match="runtime version"):
        _resume(stored)


def test_in_memory_tampered_prefix_is_validated_before_callbacks(tmp_path: Path) -> None:
    stored = _interrupt_at(tmp_path / "prefix.zip", 1)
    # Simulate tampering that bypassed the normal frozen-record constructors.
    object.__setattr__(stored.result, "seed", 999)
    with pytest.raises(SearchCheckpointError, match="seed"):
        _resume(stored)


def test_novelty_reports_are_retained_without_implicit_new_evaluation(tmp_path: Path) -> None:
    stored = _interrupt_at(tmp_path / "prefix.zip", 1)
    novelty = evaluate_population_novelty(
        stored.result.populations[-1],
        config=NoveltyScoreConfig(1),
        distance=lambda first, second: float(abs(first.n_sites - second.n_sites)),
        distance_identifier="tests.site-count.v1",
    )
    stored = replace(stored, novelty_reports=(novelty,))
    observed: list[SearchCheckpoint] = []
    resumed = _resume(stored, checkpoint_callback=observed.append)
    assert resumed.novelty_reports == (novelty,)
    assert resumed.novelty_reports[0] is novelty
    assert all(item.novelty_reports[0] is novelty for item in observed)
    loaded = load_search_checkpoint(save_search_checkpoint(tmp_path / "final.zip", resumed))
    assert loaded.novelty_reports[0].population is loaded.result.populations[1]


def test_existing_evaluator_failures_survive_without_retry(tmp_path: Path) -> None:
    calls: list[tuple[int, int]] = []

    def evaluator(member: FitnessPopulationMember) -> GeometryEvaluationRun:
        calls.append((member.generation_index, member.member_index))
        if member.member_index == 3:
            raise RuntimeError("deterministic fixture failure")
        return _evaluate(member)

    stored = _interrupt_at(tmp_path / "prefix.zip", 1, evaluator=evaluator)
    calls.clear()
    resumed = _resume(stored, evaluator=evaluator)
    assert all(generation >= 2 for generation, _ in calls)
    for ledger in resumed.result.fitness_history:
        assert ledger.members[3].failure is not None
        assert ledger.members[3].failure.message == "deterministic fixture failure"
    full = _checkpoint(_run(evaluator=evaluator))
    assert save_search_checkpoint(tmp_path / "full.zip", full).read_bytes() == (
        save_search_checkpoint(tmp_path / "resumed.zip", resumed).read_bytes()
    )


def test_resume_without_diversity_and_multiobjective_full_elitism(tmp_path: Path) -> None:
    genome = geometry_to_genome(Geometry(n_sites=4, edges=(GeometryEdge(0, 1),)))
    options = {
        "initial_population": create_initial_population((genome,) * 4),
        "definition": MultiObjectiveFitnessDefinition(
            (ObjectiveSpec("gap", ObjectiveQuantity.GAP, ObjectiveDirection.MAXIMIZE),)
        ),
        "config": GenerationLoopConfig(4, TournamentSelectionConfig(4, 2), ElitismConfig(1)),
        "family_classifier": None,
        "family_classifier_identifier": None,
    }
    stored = _interrupt_at(tmp_path / "prefix.zip", 1, **options)

    def unexpected(request: GenerationReproductionRequest) -> tuple[OffspringProposal, ...]:
        pytest.fail("complete elite tier must fill all slots")

    with pytest.raises(IncompatibleSearchCheckpointError, match="diversity policy"):
        _resume(stored)
    resumed = _resume(
        stored,
        family_classifier=None,
        family_classifier_identifier=None,
        offspring_producer=unexpected,
    )
    assert all(item.selection is None for item in resumed.result.transitions)
    assert save_search_checkpoint(
        tmp_path / "full.zip", _checkpoint(_run(**options))
    ).read_bytes() == (save_search_checkpoint(tmp_path / "resumed.zip", resumed).read_bytes())


@pytest.mark.parametrize(
    "field", ("evaluator", "offspring_producer", "family_classifier", "checkpoint_callback")
)
def test_noncallable_resume_inputs_fail_before_work(tmp_path: Path, field: str) -> None:
    stored = _interrupt_at(tmp_path / "prefix.zip", 1)
    with pytest.raises(TypeError, match="callable"):
        _resume(stored, **{field: 17})


def test_family_consistency_uses_history_across_the_checkpoint_boundary(tmp_path: Path) -> None:
    stored = _interrupt_at(tmp_path / "prefix.zip", 1)

    def unexpected(member: FitnessPopulationMember) -> GeometryEvaluationRun:
        pytest.fail("inconsistent target families must stop before target evaluation")

    with pytest.raises(InconsistentGeometryFamilyError):
        _resume(
            stored,
            evaluator=unexpected,
            family_classifier=lambda genome: GeometrySearchFamily("changed_family", 1),
        )


def test_invalid_resumed_offspring_is_retained_without_retry(tmp_path: Path) -> None:
    path = tmp_path / "prefix.zip"
    stored = _interrupt_at(path, 0)
    previous_bytes = path.read_bytes()
    calls: list[int] = []

    def producer(request: GenerationReproductionRequest) -> tuple[OffspringProposal, ...]:
        calls.append(request.target_generation_index)
        return tuple(
            replace(proposal, genome=GeometryGenome(n_sites=0)) for proposal in _produce(request)
        )

    def unexpected(*args: Any) -> Any:
        pytest.fail("invalid offspring must stop before target evaluation or checkpoint emission")

    with pytest.raises(InvalidGenerationOffspringError) as caught:
        _resume(
            stored,
            offspring_producer=producer,
            evaluator=unexpected,
            checkpoint_callback=unexpected,
        )
    assert calls == [1]
    assert caught.value.request.source is stored.result.final_fitness
    assert path.read_bytes() == previous_bytes


def test_multiobjective_partial_front_resume_matches_full_run(tmp_path: Path) -> None:
    definition = MultiObjectiveFitnessDefinition(
        (
            ObjectiveSpec("gap", ObjectiveQuantity.GAP, ObjectiveDirection.MAXIMIZE),
            ObjectiveSpec(
                "size",
                ObjectiveQuantity.GEOMETRY_DESCRIPTOR,
                ObjectiveDirection.MINIMIZE,
                descriptor_name="site_count",
            ),
        )
    )
    stored = _interrupt_at(tmp_path / "prefix.zip", 1, definition=definition)
    resumed = _resume(stored)
    assert any(transition.selection is not None for transition in resumed.result.transitions)
    full_path = save_search_checkpoint(
        tmp_path / "full.zip", _checkpoint(_run(definition=definition))
    )
    resumed_path = save_search_checkpoint(tmp_path / "resumed.zip", resumed)
    assert full_path.read_bytes() == resumed_path.read_bytes()
