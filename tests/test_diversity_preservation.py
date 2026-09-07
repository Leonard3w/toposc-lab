from __future__ import annotations

import operator
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
    DIVERSITY_PRESERVATION_VERSION,
    DiversityClassificationError,
    DiversityPolicyViolation,
    DiversityPreservationError,
    DiversityPreservationPolicy,
    ElitismConfig,
    GenerationLoopConfig,
    GenerationPopulationMember,
    GenerationReproductionRequest,
    GeometryGenome,
    GeometrySearchFamily,
    InconsistentGeometryFamilyError,
    InitialPopulation,
    InitialPopulationMember,
    OffspringProposal,
    PopulationDiversityReport,
    ScalarFitnessDefinition,
    TournamentSelectionConfig,
    assess_population_diversity,
    create_initial_population,
    geometry_to_genome,
    run_generation_loop,
    validate_diversity_history,
)

_EVEN_FAMILY = GeometrySearchFamily("even_sites", 1)
_ODD_FAMILY = GeometrySearchFamily("odd_sites", 1)


class _DiversityModel(BaseModel):
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


def _genome(n_sites: int) -> GeometryGenome:
    return geometry_to_genome(
        Geometry(
            n_sites=n_sites,
            edges=tuple(GeometryEdge(index, index + 1) for index in range(n_sites - 1)),
        )
    )


def _population(
    sizes: tuple[int, ...] = (2, 3, 4, 5),
) -> InitialPopulation:
    return create_initial_population(tuple(_genome(size) for size in sizes))


def _parity_classifier(genome: GeometryGenome) -> GeometrySearchFamily:
    return _EVEN_FAMILY if genome.n_sites % 2 == 0 else _ODD_FAMILY


def _policy() -> DiversityPreservationPolicy:
    return DiversityPreservationPolicy(
        minimum_distinct_families=2,
        maximum_members_per_family=2,
    )


def _definition() -> ScalarFitnessDefinition:
    return ScalarFitnessDefinition(
        weights={BasicScoreComponent.NORMALIZED_GAP: 1.0},
        direction=ObjectiveDirection.MAXIMIZE,
    )


def _evaluate(member: PopulationMember) -> GeometryEvaluationRun:
    return evaluate_geometry(
        member.genome.to_geometry(),
        adapter=GeometryModelAdapter(model_factory=_DiversityModel),
        seed=100 * member.generation_index + member.member_index,
        code_version="phase-10.16-test",
    )


def _loop_config(*, generation_count: int = 1) -> GenerationLoopConfig:
    return GenerationLoopConfig(
        generation_count=generation_count,
        selection=TournamentSelectionConfig(selection_count=4, tournament_size=2),
        elitism=ElitismConfig(minimum_elite_count=1),
        diversity=_policy(),
    )


def _balanced_producer(
    request: GenerationReproductionRequest,
) -> tuple[OffspringProposal, ...]:
    sizes = (2, 3, 4)
    assert request.required_offspring_count == len(sizes)
    return tuple(
        OffspringProposal(
            genome=_genome(size),
            parent_selection_indices=(0,),
            validation_source_selection_index=0,
            operator_identifier="tests.parity-builder.v1",
            operator_seed=request.seed,
        )
        for size in sizes
    )


def test_report_preserves_population_order_and_explicit_family_counts() -> None:
    population = _population()
    calls: list[int] = []

    def classifier(genome: GeometryGenome) -> GeometrySearchFamily:
        calls.append(genome.n_sites)
        return _parity_classifier(genome)

    report = assess_population_diversity(
        population,
        policy=_policy(),
        classifier=classifier,
        classifier_identifier="tests.parity-classifier.v1",
    )

    assert report.population is population
    assert calls == [2, 3, 4, 5]
    assert tuple(member.population_member for member in report.members) == (
        population.members
    )
    assert tuple(member.family for member in report.members) == (
        _EVEN_FAMILY,
        _ODD_FAMILY,
        _EVEN_FAMILY,
        _ODD_FAMILY,
    )
    assert tuple(report.family_counts.items()) == (
        (_EVEN_FAMILY, 2),
        (_ODD_FAMILY, 2),
    )
    with pytest.raises(TypeError):
        report.family_counts[_EVEN_FAMILY] = 3  # type: ignore[index]
    assert report.distinct_family_count == 2
    assert report.satisfies_policy
    assert report.version == DIVERSITY_PRESERVATION_VERSION


def test_family_lookup_is_by_exact_population_member_index() -> None:
    report = assess_population_diversity(
        _population(),
        policy=_policy(),
        classifier=_parity_classifier,
        classifier_identifier="tests.parity-classifier.v1",
    )

    assert report.family_for_member(0) is _EVEN_FAMILY
    assert report.family_for_member(1) is _ODD_FAMILY
    with pytest.raises(IndexError, match="outside"):
        report.family_for_member(4)


def test_minimum_and_capacity_violations_are_complete_and_stably_ordered() -> None:
    one_family = GeometrySearchFamily("one_family", 3)
    report = assess_population_diversity(
        _population(),
        policy=_policy(),
        classifier=lambda genome: one_family,
        classifier_identifier="tests.one-family.v1",
    )

    assert not report.satisfies_policy
    assert tuple(violation.code for violation in report.violations) == (
        "minimum_distinct_families_not_met",
        "family_capacity_exceeded",
    )
    assert report.violations[0].family is None
    assert report.violations[1].family is one_family
    with pytest.raises(DiversityPreservationError) as caught:
        report.raise_for_violations()
    assert caught.value.report is report


def test_capacity_violations_follow_family_first_occurrence_order() -> None:
    first = GeometrySearchFamily("first", 1)
    second = GeometrySearchFamily("second", 1)
    assignments = iter((first, second, first, second))
    report = assess_population_diversity(
        _population(),
        policy=DiversityPreservationPolicy(maximum_members_per_family=1),
        classifier=lambda genome: next(assignments),
        classifier_identifier="tests.ordered-labels.v1",
    )

    assert tuple(violation.family for violation in report.violations) == (
        first,
        second,
    )


def test_duplicate_exact_snapshots_require_consistent_family_labels() -> None:
    genome = _genome(3)
    population = create_initial_population((genome, genome))
    assignments = iter((_EVEN_FAMILY, _ODD_FAMILY))

    with pytest.raises(
        InconsistentGeometryFamilyError,
        match="inconsistent search-family labels",
    ):
        assess_population_diversity(
            population,
            policy=DiversityPreservationPolicy(minimum_distinct_families=1),
            classifier=lambda candidate: next(assignments),
            classifier_identifier="tests.inconsistent.v1",
        )


def test_history_requires_stable_labels_for_a_repeated_exact_snapshot() -> None:
    genome = _genome(3)
    first = assess_population_diversity(
        create_initial_population((genome,)),
        policy=DiversityPreservationPolicy(minimum_distinct_families=1),
        classifier=lambda candidate: _EVEN_FAMILY,
        classifier_identifier="tests.history.v1",
    )
    second = assess_population_diversity(
        create_initial_population((genome,)),
        policy=DiversityPreservationPolicy(minimum_distinct_families=1),
        classifier=lambda candidate: _ODD_FAMILY,
        classifier_identifier="tests.history.v1",
    )

    with pytest.raises(InconsistentGeometryFamilyError):
        validate_diversity_history((first, second))


def test_classifier_exception_is_bound_to_exact_member_without_retry() -> None:
    calls: list[int] = []

    def classifier(genome: GeometryGenome) -> GeometrySearchFamily:
        calls.append(genome.n_sites)
        if genome.n_sites == 4:
            raise RuntimeError("declared classifier failure")
        return _parity_classifier(genome)

    with pytest.raises(DiversityClassificationError) as caught:
        assess_population_diversity(
            _population(),
            policy=_policy(),
            classifier=classifier,
            classifier_identifier="tests.failing-classifier.v1",
        )

    assert calls == [2, 3, 4]
    assert caught.value.generation_index == 0
    assert caught.value.member_index == 2
    assert caught.value.error_type == "RuntimeError"
    assert caught.value.error_message == "declared classifier failure"


def test_classifier_keyboard_interrupt_is_not_converted_to_a_failure_record() -> None:
    def interrupted(genome: GeometryGenome) -> GeometrySearchFamily:
        del genome
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        assess_population_diversity(
            _population(),
            policy=_policy(),
            classifier=interrupted,
            classifier_identifier="tests.interrupted.v1",
        )


def test_classifier_must_return_an_explicit_versioned_family() -> None:
    with pytest.raises(TypeError, match="GeometrySearchFamily"):
        assess_population_diversity(
            _population(),
            policy=_policy(),
            classifier=lambda genome: "even",  # type: ignore[return-value]
            classifier_identifier="tests.wrong-return.v1",
        )


def test_report_constructor_recomputes_policy_violations() -> None:
    report = assess_population_diversity(
        _population(),
        policy=_policy(),
        classifier=lambda genome: GeometrySearchFamily("single", 1),
        classifier_identifier="tests.single.v1",
    )

    with pytest.raises(ValueError, match="occupancy policy"):
        replace(report, violations=())
    forged = DiversityPolicyViolation("forged", "not recomputed")
    with pytest.raises(ValueError, match="occupancy policy"):
        replace(report, violations=(forged,))


def test_loop_records_valid_diversity_before_each_fitness_ledger() -> None:
    seen_source_reports: list[PopulationDiversityReport] = []

    def producer(
        request: GenerationReproductionRequest,
    ) -> tuple[OffspringProposal, ...]:
        assert request.source_diversity is not None
        seen_source_reports.append(request.source_diversity)
        return _balanced_producer(request)

    result = run_generation_loop(
        _population(),
        definition=_definition(),
        evaluator=_evaluate,
        config=_loop_config(),
        seed=31,
        offspring_producer=producer,
        producer_identifier="tests.balanced-producer.v1",
        family_classifier=_parity_classifier,
        family_classifier_identifier="tests.parity-classifier.v1",
    )

    assert result.initial_diversity is seen_source_reports[0]
    assert result.initial_diversity is not None
    assert result.initial_diversity.population is result.initial_population
    assert result.transitions[0].diversity is not None
    assert result.transitions[0].diversity.population is result.transitions[0].population
    assert tuple(result.transitions[0].diversity.family_counts.values()) == (2, 2)
    assert result.family_classifier_identifier == "tests.parity-classifier.v1"


def test_initial_diversity_violation_stops_before_fitness_or_reproduction() -> None:
    evaluation_calls = 0
    producer_calls = 0

    def evaluator(member: PopulationMember) -> GeometryEvaluationRun:
        nonlocal evaluation_calls
        evaluation_calls += 1
        return _evaluate(member)

    def producer(
        request: GenerationReproductionRequest,
    ) -> tuple[OffspringProposal, ...]:
        nonlocal producer_calls
        producer_calls += 1
        return _balanced_producer(request)

    with pytest.raises(DiversityPreservationError) as caught:
        run_generation_loop(
            _population((2, 4, 6, 8)),
            definition=_definition(),
            evaluator=evaluator,
            config=_loop_config(),
            seed=32,
            offspring_producer=producer,
            producer_identifier="tests.must-not-run.v1",
            family_classifier=_parity_classifier,
            family_classifier_identifier="tests.parity-classifier.v1",
        )

    assert caught.value.report.population.generation_index == 0
    assert evaluation_calls == producer_calls == 0


def test_target_diversity_violation_stops_before_target_fitness_without_retry() -> None:
    evaluation_calls: list[tuple[int, int]] = []
    producer_calls = 0

    def evaluator(member: PopulationMember) -> GeometryEvaluationRun:
        evaluation_calls.append((member.generation_index, member.member_index))
        return _evaluate(member)

    def collapsing_producer(
        request: GenerationReproductionRequest,
    ) -> tuple[OffspringProposal, ...]:
        nonlocal producer_calls
        producer_calls += 1
        return tuple(
            OffspringProposal(
                genome=_genome(3),
                parent_selection_indices=(0,),
                validation_source_selection_index=0,
                operator_identifier="tests.collapse.v1",
            )
            for _ in range(request.required_offspring_count)
        )

    with pytest.raises(DiversityPreservationError) as caught:
        run_generation_loop(
            _population(),
            definition=_definition(),
            evaluator=evaluator,
            config=_loop_config(),
            seed=33,
            offspring_producer=collapsing_producer,
            producer_identifier="tests.collapsing-producer.v1",
            family_classifier=_parity_classifier,
            family_classifier_identifier="tests.parity-classifier.v1",
        )

    assert producer_calls == 1
    assert evaluation_calls == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert caught.value.report.population.generation_index == 1
    assert len(caught.value.report.members) == 4
    assert tuple(caught.value.report.family_counts.values()) == (4,)


def test_zero_transition_loop_still_audits_initial_diversity() -> None:
    result = run_generation_loop(
        _population(),
        definition=_definition(),
        evaluator=_evaluate,
        config=_loop_config(generation_count=0),
        seed=34,
        offspring_producer=_balanced_producer,
        producer_identifier="tests.unused-producer.v1",
        family_classifier=_parity_classifier,
        family_classifier_identifier="tests.parity-classifier.v1",
    )

    assert result.transitions == ()
    assert result.initial_diversity is not None
    assert result.initial_diversity.satisfies_policy


def test_loop_rejects_classifier_inputs_without_a_policy_and_missing_inputs_with_one() -> None:
    config_without_diversity = GenerationLoopConfig(
        generation_count=0,
        selection=TournamentSelectionConfig(selection_count=1, tournament_size=1),
        elitism=ElitismConfig(minimum_elite_count=1),
    )
    with pytest.raises(ValueError, match="require config.diversity"):
        run_generation_loop(
            _population(),
            definition=_definition(),
            evaluator=_evaluate,
            config=config_without_diversity,
            seed=1,
            offspring_producer=_balanced_producer,
            producer_identifier="tests.unused.v1",
            family_classifier=_parity_classifier,
            family_classifier_identifier="tests.parity-classifier.v1",
        )
    with pytest.raises(TypeError, match="requires a callable"):
        run_generation_loop(
            _population(),
            definition=_definition(),
            evaluator=_evaluate,
            config=_loop_config(generation_count=0),
            seed=1,
            offspring_producer=_balanced_producer,
            producer_identifier="tests.unused.v1",
        )


@pytest.mark.parametrize(
    ("minimum", "maximum", "error_type"),
    (
        (None, None, ValueError),
        (0, None, ValueError),
        (True, None, TypeError),
        (None, 0, ValueError),
        (None, 1.5, TypeError),
    ),
)
def test_policy_requires_positive_explicit_constraints(
    minimum: object,
    maximum: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        DiversityPreservationPolicy(
            minimum_distinct_families=minimum,  # type: ignore[arg-type]
            maximum_members_per_family=maximum,  # type: ignore[arg-type]
        )


def test_family_keys_are_explicit_versioned_technical_labels() -> None:
    family = GeometrySearchFamily("amorphous_reference", 2)
    assert family.family_key == "amorphous_reference"
    assert family.family_version == 2
    with pytest.raises(TypeError):
        operator.lt(family, GeometrySearchFamily("quasiperiodic_reference", 1))
    with pytest.raises(ValueError, match="lowercase"):
        GeometrySearchFamily("GraphHash:abc", 1)
    with pytest.raises(ValueError, match="positive"):
        GeometrySearchFamily("valid_key", 0)


def test_diversity_contract_defines_no_novelty_or_distance_score() -> None:
    report = assess_population_diversity(
        _population(),
        policy=_policy(),
        classifier=_parity_classifier,
        classifier_identifier="tests.parity-classifier.v1",
    )

    warning_text = " ".join(report.warnings).lower()
    assert "not inferred" in warning_text
    assert "novelty" in warning_text
    assert "distance" in warning_text
    assert not hasattr(report, "diversity_score")
    assert not hasattr(report, "novelty_score")
    assert not hasattr(report, "pairwise_distance")
