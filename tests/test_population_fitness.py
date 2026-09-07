from __future__ import annotations

from collections.abc import Mapping
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
    ModelGeometryRequirements,
    MultiObjectiveEvaluation,
    ObjectiveDirection,
    ObjectiveQuantity,
    ObjectiveSpec,
    evaluate_geometry,
)
from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.search import (
    POPULATION_FITNESS_VERSION,
    GeometryGenome,
    InitialPopulation,
    InitialPopulationMember,
    MultiObjectiveFitnessDefinition,
    PopulationFitnessFailure,
    PopulationFitnessFailureStage,
    PopulationFitnessMember,
    PopulationFitnessResult,
    PopulationFitnessStatus,
    ScalarFitnessDefinition,
    create_initial_population,
    evaluate_population_fitness,
    geometry_to_genome,
)
from toposc_lab.topology import TopologyMethod


class _FitnessModel(BaseModel):
    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry

    @property
    def parameters(self) -> dict[str, Any]:
        return {"onsite": 0.25, "hopping": 1.0}

    @property
    def basis_layout(self) -> BasisLayout:
        return BasisLayout(
            spatial_shape=(self.geometry.n_sites,),
            components_per_site=1,
            ordering="site_major",
            component_labels=("orbital",),
        )

    def hamiltonian(self) -> np.ndarray:
        matrix = np.eye(self.geometry.n_sites, dtype=complex) * 0.25
        for edge in self.geometry.edges:
            matrix[edge.source, edge.target] = 1.0
            matrix[edge.target, edge.source] = 1.0
        return matrix


def _genome(n_sites: int, *, connected: bool = True) -> GeometryGenome:
    edges = (
        tuple(GeometryEdge(site, site + 1) for site in range(n_sites - 1))
        if connected
        else ()
    )
    return geometry_to_genome(Geometry(n_sites=n_sites, edges=edges))


def _population() -> InitialPopulation:
    return create_initial_population((_genome(3), _genome(4), _genome(5)))


def _evaluate_member(member: InitialPopulationMember) -> GeometryEvaluationRun:
    return evaluate_geometry(
        member.genome.to_geometry(),
        adapter=GeometryModelAdapter(
            model_factory=_FitnessModel,
            requirements=ModelGeometryRequirements(
                require_connected=True,
                require_edges=True,
            ),
        ),
        seed=member.member_index,
        code_version="phase-10.11-test",
    )


def _scalar_definition(*, gap_scale: float = 1.0) -> ScalarFitnessDefinition:
    return ScalarFitnessDefinition(
        weights={BasicScoreComponent.NORMALIZED_GAP: 1.0},
        direction=ObjectiveDirection.MAXIMIZE,
        gap_scale=gap_scale,
    )


def test_scalar_fitness_is_evaluated_once_in_population_order() -> None:
    population = _population()
    calls: list[int] = []

    def evaluator(member: InitialPopulationMember) -> GeometryEvaluationRun:
        calls.append(member.member_index)
        return _evaluate_member(member)

    result = evaluate_population_fitness(
        population,
        definition=_scalar_definition(),
        evaluator=evaluator,
    )

    assert calls == [0, 1, 2]
    assert result.population is population
    assert result.version == POPULATION_FITNESS_VERSION
    assert tuple(member.member_index for member in result.members) == (0, 1, 2)
    assert all(member.status is PopulationFitnessStatus.AVAILABLE for member in result.members)
    assert all(isinstance(member.fitness, BasicScalarScore) for member in result.members)
    assert result.available_members == result.members


def test_multi_objective_fitness_retains_raw_values_and_directions() -> None:
    population = create_initial_population((_genome(3),))
    definition = MultiObjectiveFitnessDefinition(
        objectives=(
            ObjectiveSpec(
                "spectral_gap",
                ObjectiveQuantity.GAP,
                ObjectiveDirection.MAXIMIZE,
            ),
            ObjectiveSpec(
                "site_count",
                ObjectiveQuantity.GEOMETRY_DESCRIPTOR,
                ObjectiveDirection.MINIMIZE,
                descriptor_name="site_count",
            ),
        )
    )

    result = evaluate_population_fitness(
        population,
        definition=definition,
        evaluator=_evaluate_member,
    )

    fitness = result.members[0].fitness
    assert isinstance(fitness, MultiObjectiveEvaluation)
    assert fitness.objectives["spectral_gap"].spec.direction is ObjectiveDirection.MAXIMIZE
    assert fitness.objectives["site_count"].value == 3
    assert not hasattr(result.members[0], "rank")


def test_invalid_evaluation_remains_present_without_fitness() -> None:
    population = create_initial_population((_genome(3), _genome(2, connected=False)))

    result = evaluate_population_fitness(
        population,
        definition=_scalar_definition(),
        evaluator=_evaluate_member,
    )

    invalid = result.members[1]
    assert invalid.status is PopulationFitnessStatus.INVALID_EVALUATION
    assert invalid.evaluation is not None
    assert not invalid.evaluation.is_valid
    assert invalid.fitness is None
    assert invalid.failure is None
    assert result.available_members == (result.members[0],)


def test_evaluator_failure_is_retained_and_later_members_continue() -> None:
    population = _population()
    calls: list[int] = []

    def evaluator(member: InitialPopulationMember) -> GeometryEvaluationRun:
        calls.append(member.member_index)
        if member.member_index == 1:
            raise RuntimeError("synthetic evaluation failure")
        return _evaluate_member(member)

    result = evaluate_population_fitness(
        population,
        definition=_scalar_definition(),
        evaluator=evaluator,
    )

    assert calls == [0, 1, 2]
    failed = result.members[1]
    assert failed.status is PopulationFitnessStatus.EVALUATION_FAILURE
    assert failed.evaluation is None
    assert failed.fitness is None
    assert failed.failure == PopulationFitnessFailure(
        stage=PopulationFitnessFailureStage.EVALUATION_CALLBACK,
        error_type="RuntimeError",
        message="synthetic evaluation failure",
    )
    assert tuple(member.member_index for member in result.available_members) == (0, 2)


def test_missing_requested_fitness_quantity_is_a_separate_failure() -> None:
    population = create_initial_population((_genome(3),))
    definition = MultiObjectiveFitnessDefinition(
        objectives=(
            ObjectiveSpec(
                "topology",
                ObjectiveQuantity.TOPOLOGY_CLASSIFICATION,
                ObjectiveDirection.MAXIMIZE,
                topology_method=TopologyMethod.PFAFFIAN_1D,
            ),
        )
    )

    result = evaluate_population_fitness(
        population,
        definition=definition,
        evaluator=_evaluate_member,
    )

    member = result.members[0]
    assert member.status is PopulationFitnessStatus.FITNESS_CONSTRUCTION_FAILURE
    assert member.evaluation is not None and member.evaluation.is_valid
    assert member.fitness is None
    assert member.failure is not None
    assert member.failure.stage is PopulationFitnessFailureStage.FITNESS_CONSTRUCTION
    assert member.failure.error_type == "ValueError"
    assert "topology method" in member.failure.message


def test_all_failures_can_be_reported_without_inventing_fitness() -> None:
    population = _population()

    def evaluator(member: InitialPopulationMember) -> GeometryEvaluationRun:
        raise ValueError(f"failed member {member.member_index}")

    result = evaluate_population_fitness(
        population,
        definition=_scalar_definition(),
        evaluator=evaluator,
    )

    assert result.available_members == ()
    assert all(
        member.status is PopulationFitnessStatus.EVALUATION_FAILURE
        for member in result.members
    )


def test_reproducibility_geometry_mismatch_is_an_api_error() -> None:
    population = create_initial_population((_genome(3),))
    other_population = create_initial_population((_genome(4),))

    def mismatched(member: InitialPopulationMember) -> GeometryEvaluationRun:
        del member
        return _evaluate_member(other_population.members[0])

    with pytest.raises(ValueError, match="geometry_id"):
        evaluate_population_fitness(
            population,
            definition=_scalar_definition(),
            evaluator=mismatched,
        )


def test_wrong_evaluator_return_type_is_an_api_error() -> None:
    calls = 0

    def evaluator(member: InitialPopulationMember) -> str:
        nonlocal calls
        del member
        calls += 1
        return "not an evaluation"

    with pytest.raises(TypeError, match="must return GeometryEvaluationRun"):
        evaluate_population_fitness(
            _population(),
            definition=_scalar_definition(),
            evaluator=evaluator,  # type: ignore[arg-type]
        )
    assert calls == 1


def test_keyboard_interrupt_is_not_converted_to_member_failure() -> None:
    def evaluator(member: InitialPopulationMember) -> GeometryEvaluationRun:
        del member
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        evaluate_population_fitness(
            create_initial_population((_genome(3),)),
            definition=_scalar_definition(),
            evaluator=evaluator,
        )


@pytest.mark.parametrize(
    ("weights", "gap_scale", "error_type"),
    (
        ({}, 1.0, ValueError),
        ({"gap": 1.0}, 1.0, TypeError),
        ({BasicScoreComponent.NORMALIZED_GAP: 0.0}, 1.0, ValueError),
        ({BasicScoreComponent.NORMALIZED_GAP: np.inf}, 1.0, ValueError),
        ({BasicScoreComponent.NORMALIZED_GAP: 1.0}, 0.0, ValueError),
    ),
)
def test_scalar_definition_rejects_malformed_configuration(
    weights: Mapping[object, float],
    gap_scale: float,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        ScalarFitnessDefinition(
            weights=weights,  # type: ignore[arg-type]
            direction=ObjectiveDirection.MAXIMIZE,
            gap_scale=gap_scale,
        )


def test_scalar_definition_freezes_caller_mapping() -> None:
    weights = {BasicScoreComponent.NORMALIZED_GAP: 1.0}

    definition = ScalarFitnessDefinition(weights, ObjectiveDirection.MAXIMIZE)
    weights[BasicScoreComponent.NORMALIZED_GAP] = 2.0

    assert definition.weights[BasicScoreComponent.NORMALIZED_GAP] == 1.0
    with pytest.raises(TypeError):
        definition.weights[BasicScoreComponent.NORMALIZED_GAP] = 2.0  # type: ignore[index]

    with pytest.raises(TypeError, match="ObjectiveDirection"):
        ScalarFitnessDefinition(
            {BasicScoreComponent.NORMALIZED_GAP: 1.0},
            "maximize",  # type: ignore[arg-type]
        )


def test_multi_objective_definition_rejects_empty_invalid_or_duplicate_specs() -> None:
    spec = ObjectiveSpec("gap", ObjectiveQuantity.GAP, ObjectiveDirection.MAXIMIZE)

    with pytest.raises(ValueError, match="must not be empty"):
        MultiObjectiveFitnessDefinition(())
    with pytest.raises(TypeError, match="ObjectiveSpec"):
        MultiObjectiveFitnessDefinition((object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="names must be unique"):
        MultiObjectiveFitnessDefinition((spec, spec))


def test_failure_record_validates_and_normalizes_fields() -> None:
    failure = PopulationFitnessFailure(
        PopulationFitnessFailureStage.EVALUATION_CALLBACK,
        "ValueError",
        "  failed  ",
    )

    assert failure.message == "failed"
    with pytest.raises(ValueError, match="identifier"):
        PopulationFitnessFailure(
            PopulationFitnessFailureStage.EVALUATION_CALLBACK,
            "not valid",
            "failed",
        )


def test_population_fitness_member_rejects_inconsistent_status_payloads() -> None:
    population = create_initial_population((_genome(3),))
    source_member = population.members[0]
    evaluation = _evaluate_member(source_member)
    failure = PopulationFitnessFailure(
        PopulationFitnessFailureStage.EVALUATION_CALLBACK,
        "ValueError",
        "failed",
    )

    with pytest.raises(ValueError, match="available fitness"):
        PopulationFitnessMember(
            source_member,
            PopulationFitnessStatus.AVAILABLE,
            evaluation,
            None,
            None,
        )
    with pytest.raises(ValueError, match="evaluation-failure"):
        PopulationFitnessMember(
            source_member,
            PopulationFitnessStatus.EVALUATION_FAILURE,
            evaluation,
            None,
            failure,
        )


def test_result_requires_population_order_identity_and_matching_definition() -> None:
    population = _population()
    result = evaluate_population_fitness(
        population,
        definition=_scalar_definition(),
        evaluator=_evaluate_member,
    )

    with pytest.raises(ValueError, match="order and identity"):
        PopulationFitnessResult(
            population,
            result.definition,
            (result.members[1], result.members[0], result.members[2]),
        )
    with pytest.raises(ValueError, match="normalization"):
        PopulationFitnessResult(
            population,
            _scalar_definition(gap_scale=2.0),
            result.members,
        )


def test_api_requires_population_definition_and_evaluator() -> None:
    population = _population()

    with pytest.raises(TypeError, match="population must be"):
        evaluate_population_fitness(
            object(),  # type: ignore[arg-type]
            definition=_scalar_definition(),
            evaluator=_evaluate_member,
        )
    with pytest.raises(TypeError, match="definition must be"):
        evaluate_population_fitness(
            population,
            definition=object(),  # type: ignore[arg-type]
            evaluator=_evaluate_member,
        )
    with pytest.raises(TypeError, match="evaluator must be callable"):
        evaluate_population_fitness(
            population,
            definition=_scalar_definition(),
            evaluator=None,  # type: ignore[arg-type]
        )
