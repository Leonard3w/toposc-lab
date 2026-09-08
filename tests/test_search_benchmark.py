from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import BasisLayout
from toposc_lab.evaluation import (
    BasicScalarScore,
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
    BenchmarkArmResult,
    BenchmarkProgress,
    BenchmarkSuccessCriterion,
    DiversityPreservationPolicy,
    ElitismConfig,
    GenerationLoopConfig,
    GenerationReproductionRequest,
    GeometryGenome,
    GeometrySearchFamily,
    InvalidBenchmarkSampleError,
    MultiObjectiveFitnessDefinition,
    MutationValidityPolicy,
    OffspringProposal,
    PopulationFitnessMember,
    PopulationFitnessStatus,
    ScalarFitnessDefinition,
    SearchBenchmarkProtocol,
    SearchBenchmarkResult,
    TournamentSelectionConfig,
    geometry_to_genome,
    run_search_benchmark,
)
from toposc_lab.search.search_benchmark import _seed_schedule


class _BenchmarkModel(BaseModel):
    """Controlled synthetic score fixture, not a physical resource comparison."""

    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry

    @property
    def parameters(self) -> dict[str, Any]:
        return {"fixture": "edge_count"}

    @property
    def basis_layout(self) -> BasisLayout:
        return BasisLayout((3,))

    def hamiltonian(self) -> np.ndarray:
        value = 1.0 + len(self.geometry.edges) / 2
        return np.diag([-value, value, 5.0]).astype(complex)


def _genome(edge_count: int = 0) -> GeometryGenome:
    return geometry_to_genome(
        Geometry(
            n_sites=3,
            edges=(GeometryEdge(0, 1), GeometryEdge(1, 2))[:edge_count],
        )
    )


def _sample(seed: int) -> GeometryGenome:
    rng = np.random.Generator(np.random.PCG64(seed))
    return _genome(int(rng.integers(3)))


def _evaluate(genome: GeometryGenome, seed: int) -> GeometryEvaluationRun:
    return evaluate_geometry(
        genome.to_geometry(),
        adapter=GeometryModelAdapter(_BenchmarkModel),
        seed=seed,
        code_version="tests.benchmark.v1",
    )


def _produce(request: GenerationReproductionRequest) -> tuple[OffspringProposal, ...]:
    return tuple(
        OffspringProposal(_genome(2), (0,), 0, "test.connected.v1")
        for _ in range(request.required_offspring_count)
    )


def _hit(member: PopulationFitnessMember) -> bool:
    assert isinstance(member.fitness, BasicScalarScore)
    return member.fitness.value > 0.77


def _protocol(**changes: Any) -> SearchBenchmarkProtocol:
    return replace(
        SearchBenchmarkProtocol(
            identifier="tests.benchmark.v1",
            candidate_space_identifier="three-sites.v1",
            sampler_identifier="test.bernoulli.v1",
            evaluator_identifier="test.evaluator.v1",
            producer_identifier="test.connected.v1",
            code_version="tests.benchmark.v1",
            population_size=3,
            generation_config=GenerationLoopConfig(
                2, TournamentSelectionConfig(3, 1), ElitismConfig(1)
            ),
            trial_seeds=(1701, 1702),
            definition=ScalarFitnessDefinition(
                {BasicScoreComponent.NORMALIZED_GAP: 1.0},
                ObjectiveDirection.MAXIMIZE,
            ),
            validity_policy=MutationValidityPolicy(minimum_site_count=3, maximum_site_count=3),
            criterion=BenchmarkSuccessCriterion("test.hit.v1", "score > 0.77", _hit),
        ),
        **changes,
    )


def _run(protocol: SearchBenchmarkProtocol | None = None, **changes: Any) -> SearchBenchmarkResult:
    options = {"sampler": _sample, "evaluator": _evaluate, "offspring_producer": _produce}
    options.update(changes)
    return run_search_benchmark(protocol or _protocol(), **options)


def test_equal_budget_shared_start_seed_pairing_and_independent_draws() -> None:
    protocol = _protocol()
    sampled: list[int] = []
    evaluated: list[int] = []

    def sample(seed: int) -> GeometryGenome:
        sampled.append(seed)
        return _sample(seed)

    def evaluate(genome: GeometryGenome, seed: int) -> GeometryEvaluationRun:
        evaluated.append(seed)
        return _evaluate(genome, seed)

    result = _run(protocol, sampler=sample, evaluator=evaluate)
    assert protocol.attempts_per_arm == 9
    expected_samples: list[int] = []
    expected_evaluations: list[int] = []
    for trial in result.trials:
        evolution_seed, sampling_seeds, evaluation_seeds = _seed_schedule(trial.seed, protocol)
        expected_samples.extend(sampling_seeds)
        expected_evaluations.extend((*evaluation_seeds, *evaluation_seeds))
        assert trial.evolution.seed == evolution_seed
        assert trial.evolution.initial_population is trial.random_arm.fitness_history[0].population
        assert trial.evolution.initial_fitness is not trial.random_arm.fitness_history[0]
        assert trial.evolution_arm.final_progress.attempt_count == 9
        assert trial.random_arm.final_progress.attempt_count == 9
    assert sampled == expected_samples
    assert evaluated == expected_evaluations


def test_seed_schedule_matches_frozen_pcg64_word_order() -> None:
    protocol = _protocol()
    words = tuple(int(x) for x in np.random.PCG64(1701).random_raw(19))
    assert _seed_schedule(1701, protocol) == (words[0], words[1:10], words[10:])


def test_reproducible_full_ledgers_and_global_rng_unchanged() -> None:
    before = np.random.get_state()
    protocol = _protocol()
    first, second = _run(protocol), _run(protocol)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    for left, right in zip(first.trials, second.trials, strict=True):
        for arm_name in ("evolution_arm", "random_arm"):
            a, b = getattr(left, arm_name), getattr(right, arm_name)
            assert a.progress == b.progress
            for h1, h2 in zip(a.fitness_history, b.fitness_history, strict=True):
                assert tuple(m.evaluation.reproducibility for m in h1.members) == tuple(
                    m.evaluation.reproducibility for m in h2.members
                )


@pytest.mark.parametrize("direction", list(ObjectiveDirection))
def test_known_best_so_far_direction_first_hit_and_quality_difference(
    direction: ObjectiveDirection,
) -> None:
    protocol = _protocol(
        trial_seeds=(1701,),
        definition=ScalarFitnessDefinition(
            {BasicScoreComponent.NORMALIZED_GAP: 1.0},
            direction,
        ),
    )
    _, seeds, _ = _seed_schedule(1701, protocol)
    result = _run(protocol, sampler=lambda seed: _genome(1 if seed == seeds[0] else 0))
    for trial in result.trials:
        left, right = trial.evolution_arm.progress, trial.random_arm.progress
        assert left[0].best_scalar_score == pytest.approx(3 / 4)
        assert left[2].first_success_attempt is None
        expected_first = 5 if direction is ObjectiveDirection.MAXIMIZE else 6
        assert left[-1].first_success_attempt == expected_first
        assert right[-1].first_success_attempt is None
        if direction is ObjectiveDirection.MAXIMIZE:
            assert left[-1].best_scalar_score == pytest.approx(4 / 5)
            assert trial.scalar_quality_difference == pytest.approx(4 / 5 - 3 / 4)
        else:
            assert left[-1].best_scalar_score == pytest.approx(2 / 3)
            assert trial.scalar_quality_difference == 0
    assert result.paired_success_differences == (1,)


def test_random_can_win_and_no_winner_is_hard_coded() -> None:
    def produce(request: GenerationReproductionRequest) -> tuple[OffspringProposal, ...]:
        return tuple(
            OffspringProposal(_genome(), (0,), 0, "empty.v1")
            for _ in range(request.required_offspring_count)
        )

    protocol = _protocol(trial_seeds=(1701,))
    _, seeds, _ = _seed_schedule(1701, protocol)
    result = _run(
        protocol,
        sampler=lambda seed: _genome(2 if seed not in seeds[:3] else 0),
        offspring_producer=produce,
    )
    assert result.trials[0].scalar_quality_difference < 0
    assert result.paired_success_differences == (-1,)


def test_elites_and_duplicates_are_reevaluated_and_charged() -> None:
    protocol = _protocol(
        generation_config=GenerationLoopConfig(
            2,
            TournamentSelectionConfig(3, 1),
            ElitismConfig(1),
        )
    )
    count = 0

    def evaluate(genome: GeometryGenome, seed: int) -> GeometryEvaluationRun:
        nonlocal count
        count += 1
        return _evaluate(genome, seed)

    def forbidden(request: GenerationReproductionRequest) -> tuple[OffspringProposal, ...]:
        pytest.fail("all tied members are complete elites")

    result = _run(
        protocol, sampler=lambda seed: _genome(), evaluator=evaluate, offspring_producer=forbidden
    )
    assert count == 36
    assert all(trial.scalar_quality_difference == 0 for trial in result.trials)


@pytest.mark.parametrize("failure_kind", ["callback", "invalid", "fitness"])
def test_failures_keep_budget_missing_quality_and_never_reach_criterion(failure_kind: str) -> None:
    def forbidden(member: PopulationFitnessMember) -> bool:
        pytest.fail("unavailable member must not reach criterion")

    protocol = _protocol(
        generation_config=GenerationLoopConfig(
            0, TournamentSelectionConfig(1, 1), ElitismConfig(1)
        ),
        criterion=BenchmarkSuccessCriterion("never.v1", "never called", forbidden),
    )

    def evaluate(genome: GeometryGenome, seed: int) -> GeometryEvaluationRun:
        if failure_kind == "callback":
            raise RuntimeError("record this failure")
        if failure_kind == "invalid":

            def bad_factory(geometry: Geometry) -> BaseModel:
                raise ValueError("model construction fails before provenance")

            return evaluate_geometry(
                genome.to_geometry(),
                adapter=GeometryModelAdapter(bad_factory),
                seed=seed,
                code_version="tests.benchmark.v1",
            )
        return _evaluate(genome, seed)

    if failure_kind == "fitness":
        protocol = replace(
            protocol,
            definition=ScalarFitnessDefinition(
                {BasicScoreComponent.MAXIMUM_MAJORANA_SELF_CONJUGACY: 1.0},
                ObjectiveDirection.MAXIMIZE,
            ),
        )
    result = _run(protocol, evaluator=evaluate)
    for trial in result.trials:
        for arm in (trial.evolution_arm, trial.random_arm):
            assert arm.final_progress.attempt_count == 3
            assert arm.final_progress.available_count == 0
            assert arm.final_progress.best_scalar_score is None
            assert arm.final_progress.first_success_attempt is None
            assert arm.successes == (False,) * 3
        assert trial.scalar_quality_difference is None


def test_partial_callback_failure_keeps_index_and_later_attempts() -> None:
    protocol = _protocol(trial_seeds=(1701,))
    _, _, seeds = _seed_schedule(1701, protocol)

    def evaluate(genome: GeometryGenome, seed: int) -> GeometryEvaluationRun:
        if seed == seeds[0]:
            raise RuntimeError("first slot failed")
        return _evaluate(genome, seed)

    result = _run(protocol, evaluator=evaluate)
    for arm in (result.trials[0].evolution_arm, result.trials[0].random_arm):
        assert (
            arm.fitness_history[0].members[0].status is PopulationFitnessStatus.EVALUATION_FAILURE
        )
        assert arm.progress[0].best_scalar_score is None
        assert arm.progress[1].available_count == 1
        assert arm.final_progress.available_count == 8


def test_multi_objective_hit_analysis_without_hidden_scalarization() -> None:
    definition = MultiObjectiveFitnessDefinition(
        (
            ObjectiveSpec("gap", ObjectiveQuantity.GAP, ObjectiveDirection.MAXIMIZE),
            ObjectiveSpec(
                "zero_modes", ObjectiveQuantity.ZERO_MODE_COUNT, ObjectiveDirection.MINIMIZE
            ),
        )
    )
    criterion = BenchmarkSuccessCriterion(
        "raw-gap.v1",
        "raw gap >= 4",
        lambda m: bool(
            m.evaluation.evaluation.gap >= 4,
        ),
    )
    result = _run(_protocol(definition=definition, criterion=criterion))
    for trial in result.trials:
        assert trial.scalar_quality_difference is None
        assert trial.evolution_arm.final_progress.best_scalar_score is None
        assert trial.evolution_arm.final_progress.successful_count > 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("identifier", " "),
        ("candidate_space_identifier", ""),
        ("code_version", None),
        ("population_size", 0),
        ("population_size", True),
        ("population_size", 2.5),
        ("trial_seeds", ()),
        ("trial_seeds", (1, 1)),
        ("trial_seeds", (True,)),
        ("trial_seeds", (-1,)),
        ("definition", None),
        ("validity_policy", None),
        ("criterion", None),
        ("generation_config", None),
        ("family_classifier_identifier", "x"),
    ],
)
def test_protocol_rejects_invalid_inputs(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _protocol(**{field: value})


@pytest.mark.parametrize("field", ["seed", "code_version", "model_parameters", "solver_settings"])
def test_provenance_drift_is_rejected(field: str) -> None:
    protocol = _protocol(trial_seeds=(1701,))
    _, _, seeds = _seed_schedule(1701, protocol)

    def evaluate(genome: GeometryGenome, seed: int) -> GeometryEvaluationRun:
        run = _evaluate(genome, seed)
        if seed == seeds[-1]:
            changed = {
                "seed": 0,
                "code_version": "wrong",
                "model_parameters": {"t": 7.0},
                "solver_settings": {"changed": True},
            }[field]
            return replace(run, reproducibility=replace(run.reproducibility, **{field: changed}))
        return run

    with pytest.raises(ValueError, match="provenance|protocol"):
        _run(protocol, evaluator=evaluate)


def test_cross_trial_provenance_drift_is_rejected() -> None:
    protocol = _protocol()
    second_seeds = _seed_schedule(1702, protocol)[2]

    def evaluate(genome: GeometryGenome, seed: int) -> GeometryEvaluationRun:
        run = _evaluate(genome, seed)
        if seed in second_seeds:
            return replace(
                run, reproducibility=replace(run.reproducibility, model_parameters={"x": 1})
            )
        return run

    with pytest.raises(ValueError, match="provenance"):
        _run(protocol, evaluator=evaluate)


@pytest.mark.parametrize("value", [1, None, np.bool_(True)])
def test_nonboolean_success_criterion_aborts(value: object) -> None:
    protocol = _protocol(criterion=BenchmarkSuccessCriterion("bad.v1", "bad", lambda member: value))
    with pytest.raises(TypeError, match="strict boolean"):
        _run(protocol)


def test_invalid_sampler_never_retries() -> None:
    calls: list[int] = []

    def sample(seed: int) -> GeometryGenome:
        calls.append(seed)
        return geometry_to_genome(Geometry(n_sites=4, edges=()))

    with pytest.raises(InvalidBenchmarkSampleError, match="invalid") as caught:
        _run(sampler=sample)
    assert len(calls) == 3  # one complete initial batch, no replacement sampling
    assert caught.value.genome.n_sites == 4
    assert caught.value.attempt_index == 0
    assert caught.value.sampling_seed == calls[0]
    assert caught.value.trial_seed == 1701
    assert not caught.value.report.is_valid


def test_random_dimension_change_is_rejected() -> None:
    protocol = _protocol(trial_seeds=(1701,))
    _, seeds, _ = _seed_schedule(1701, protocol)

    def sample(seed: int) -> GeometryGenome:
        if seed in seeds[:3]:
            return _genome()
        return geometry_to_genome(Geometry(n_sites=3, edges=(), coordinates=np.zeros((3, 2))))

    with pytest.raises(InvalidBenchmarkSampleError) as caught:
        _run(protocol, sampler=sample)
    assert caught.value.attempt_index == 3
    assert caught.value.sampling_seed == seeds[3]
    assert caught.value.genome.embedding_dimension == 2


def test_optional_diversity_classifier_applies_to_evolution_only() -> None:
    protocol = _protocol(
        generation_config=GenerationLoopConfig(
            1,
            TournamentSelectionConfig(3, 1),
            ElitismConfig(1),
            DiversityPreservationPolicy(minimum_distinct_families=1),
        ),
        family_classifier_identifier="test.family.v1",
    )
    with pytest.raises(TypeError, match="diversity"):
        _run(protocol)
    result = _run(protocol, family_classifier=lambda genome: GeometrySearchFamily("fixture", 1))
    assert all(trial.evolution.initial_diversity is not None for trial in result.trials)


def test_result_invariants_and_immutability() -> None:
    result = _run()
    trial = result.trials[0]
    with pytest.raises(FrozenInstanceError):
        result.protocol.identifier = "changed"
    with pytest.raises(ValueError, match="all protocol seeds"):
        replace(result, trials=result.trials[:1])
    with pytest.raises(ValueError, match="contiguous"):
        replace(trial.random_arm, fitness_history=trial.random_arm.fitness_history[1:])
    with pytest.raises(ValueError, match="strict boolean"):
        replace(trial.random_arm, successes=(True,))
    with pytest.raises(ValueError, match="separately"):
        replace(trial, random_arm=trial.evolution_arm)
    with pytest.raises(ValueError, match="seed"):
        replace(
            result,
            trials=(replace(trial, evolution=replace(trial.evolution, seed=0)), *result.trials[1:]),
        )


def test_failed_members_cannot_be_forged_as_successful() -> None:
    protocol = _protocol(
        generation_config=GenerationLoopConfig(
            0,
            TournamentSelectionConfig(1, 1),
            ElitismConfig(1),
        )
    )

    def fail(genome: GeometryGenome, seed: int) -> GeometryEvaluationRun:
        raise RuntimeError("failed")

    arm = _run(protocol, evaluator=fail).trials[0].random_arm
    with pytest.raises(ValueError, match="unavailable"):
        BenchmarkArmResult(arm.fitness_history, (True, False, False))


def test_minimization_improvement_has_positive_difference() -> None:
    protocol = _protocol(
        trial_seeds=(1701,),
        definition=ScalarFitnessDefinition(
            {BasicScoreComponent.NORMALIZED_GAP: 1.0},
            ObjectiveDirection.MINIMIZE,
        ),
    )
    _, seeds, _ = _seed_schedule(1701, protocol)

    def produce(request: GenerationReproductionRequest) -> tuple[OffspringProposal, ...]:
        return tuple(
            OffspringProposal(_genome(), (0,), 0, "empty.v1")
            for _ in range(request.required_offspring_count)
        )

    trial = _run(
        protocol,
        sampler=lambda seed: _genome(1 if seed == seeds[0] else 2),
        offspring_producer=produce,
    ).trials[0]
    assert trial.evolution_arm.final_progress.best_scalar_score == pytest.approx(2 / 3)
    assert trial.random_arm.final_progress.best_scalar_score == pytest.approx(3 / 4)
    assert trial.scalar_quality_difference == pytest.approx(3 / 4 - 2 / 3)


def test_incomplete_evolution_does_not_become_a_complete_benchmark() -> None:
    count = 0

    def fail(genome: GeometryGenome, seed: int) -> GeometryEvaluationRun:
        nonlocal count
        count += 1
        raise RuntimeError("no fitness for selection")

    with pytest.raises(ValueError):
        _run(evaluator=fail)
    assert count == 3


def test_criterion_exceptions_and_interrupts_propagate() -> None:
    def bad_criterion(member: PopulationFitnessMember) -> bool:
        raise RuntimeError("invalid criterion")

    protocol = _protocol(criterion=BenchmarkSuccessCriterion("bad.v1", "raises", bad_criterion))
    with pytest.raises(RuntimeError, match="invalid criterion"):
        _run(protocol)

    def interrupt(genome: GeometryGenome, seed: int) -> GeometryEvaluationRun:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run(evaluator=interrupt)


@pytest.mark.parametrize(
    "values",
    [
        (1, 2, 0, None, None),
        (1, 0, 1, 1, None),
        (1, 1, 0, 1, 0.5),
        (1, 1, 1, 2, 0.5),
        (1, 1, 1, None, 0.5),
        (1, 0, 0, None, 0.5),
        (1, 1, 0, None, float("nan")),
        (True, 0, 0, None, None),
    ],
)
def test_progress_rejects_inconsistent_counts_and_scores(values: tuple[object, ...]) -> None:
    with pytest.raises((TypeError, ValueError)):
        BenchmarkProgress(*values)


@pytest.mark.parametrize("name", ["sampler", "evaluator", "offspring_producer"])
def test_callback_preflight_before_any_sampling(name: str) -> None:
    def forbidden(seed: int) -> GeometryGenome:
        pytest.fail("preflight must run before sampling")

    callbacks: dict[str, Any] = {"sampler": forbidden, name: None}
    with pytest.raises(TypeError, match="callable"):
        _run(**callbacks)
