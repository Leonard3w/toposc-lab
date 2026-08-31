from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import BasisLayout
from toposc_lab.evaluation import (
    GeometryEvaluationRun,
    GeometryModelAdapter,
    ModelGeometryRequirements,
    evaluate_geometry,
)
from toposc_lab.geometry import Geometry, chain
from toposc_lab.robustness import (
    DisorderEnsembleMember,
    DisorderEnsembleRequest,
    DisorderEnsembleResult,
    DisorderOutcome,
    RobustnessFractionMetric,
    RobustnessSuccessCriterion,
    apply_uniform_coordinate_perturbation,
    apply_uniform_parameter_perturbation,
    compute_robustness_fraction,
    execute_disorder_ensemble,
)


class _GeometryModel(BaseModel):
    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry

    @property
    def basis_layout(self) -> BasisLayout:
        return BasisLayout(spatial_shape=(self.geometry.n_sites,))

    def hamiltonian(self) -> np.ndarray:
        return np.diag(np.linspace(-1.0, 1.0, self.geometry.n_sites))


def _model_factory(geometry: Geometry) -> BaseModel:
    return _GeometryModel(geometry)


def _parameter_ensemble(*, seeds: tuple[int, ...]) -> DisorderEnsembleResult:
    return execute_disorder_ensemble(
        DisorderEnsembleRequest(seeds=seeds),
        realization_factory=lambda seed: apply_uniform_parameter_perturbation(
            {"mass": 0.5},
            widths={"mass": 0.2},
            seed=seed,
        ),
    )


def test_success_fraction_uses_every_requested_seed_in_order() -> None:
    ensemble = _parameter_ensemble(seeds=(7, 2, 9, 4))
    criterion = RobustnessSuccessCriterion(
        key="odd_seed_fixture",
        description="The explicit test criterion accepts odd seeds.",
        predicate=lambda member: member.seed % 2 == 1,
    )

    metric = compute_robustness_fraction(ensemble, criterion=criterion)

    assert isinstance(metric, RobustnessFractionMetric)
    assert metric.criterion_key == "odd_seed_fixture"
    assert metric.successes == (True, False, True, False)
    assert metric.successful_count == 2
    assert metric.total_count == 4
    assert metric.value == pytest.approx(0.5)
    assert metric.successful_seeds == (7, 9)
    assert metric.unsuccessful_seeds == (2, 4)
    assert metric.execution_failure_seeds == ()


def test_execution_failures_stay_in_denominator_and_skip_predicate() -> None:
    called_seeds: list[int] = []

    def realize(seed: int) -> DisorderOutcome:
        if seed == 2:
            raise RuntimeError("fixture realization failure")
        return apply_uniform_parameter_perturbation(
            {"mass": 0.5},
            widths={"mass": 0.2},
            seed=seed,
        )

    ensemble = execute_disorder_ensemble(
        DisorderEnsembleRequest(seeds=(1, 2, 3)),
        realization_factory=realize,
    )

    def successful(member: DisorderEnsembleMember) -> bool:
        called_seeds.append(member.seed)
        return True

    metric = compute_robustness_fraction(
        ensemble,
        criterion=RobustnessSuccessCriterion(
            key="completed_member",
            description="Every completed member is accepted by this fixture.",
            predicate=successful,
        ),
    )

    assert called_seeds == [1, 3]
    assert metric.successes == (True, False, True)
    assert metric.successful_count == 2
    assert metric.total_count == 3
    assert metric.value == pytest.approx(2.0 / 3.0)
    assert metric.execution_failure_indices == (1,)
    assert metric.execution_failure_seeds == (2,)
    assert set(metric.execution_failure_seeds) <= set(metric.unsuccessful_seeds)


def test_evaluation_callback_failure_is_also_an_unsuccessful_member() -> None:
    geometry = chain(3)

    def evaluate(disorder: DisorderOutcome, seed: int) -> GeometryEvaluationRun:
        del disorder
        if seed == 5:
            raise ValueError("fixture evaluation failure")
        return evaluate_geometry(
            geometry,
            adapter=GeometryModelAdapter(model_factory=_model_factory),
            seed=seed,
            code_version="phase-8.10-test",
        )

    ensemble = execute_disorder_ensemble(
        DisorderEnsembleRequest(seeds=(3, 5, 7)),
        realization_factory=lambda seed: apply_uniform_coordinate_perturbation(
            geometry,
            width=0.1,
            seed=seed,
        ),
        evaluation_factory=evaluate,
    )
    metric = compute_robustness_fraction(
        ensemble,
        criterion=RobustnessSuccessCriterion(
            key="valid_evaluation",
            description="Require a completed valid Phase-7 evaluation run.",
            predicate=lambda member: (
                member.evaluation is not None and member.evaluation.is_valid
            ),
        ),
    )

    assert metric.successes == (True, False, True)
    assert metric.execution_failure_seeds == (5,)
    assert metric.value == pytest.approx(2.0 / 3.0)


def test_invalid_phase_7_run_is_decided_only_by_explicit_criterion() -> None:
    geometry = chain(3)
    valid_adapter = GeometryModelAdapter(model_factory=_model_factory)
    invalid_adapter = GeometryModelAdapter(
        model_factory=_model_factory,
        requirements=ModelGeometryRequirements(required_spatial_axes=(1,)),
    )

    def evaluate(disorder: DisorderOutcome, seed: int) -> GeometryEvaluationRun:
        assert isinstance(disorder.state, Geometry)
        return evaluate_geometry(
            disorder.state,
            adapter=valid_adapter if seed == 1 else invalid_adapter,
            seed=seed,
            code_version="phase-8.10-test",
        )

    ensemble = execute_disorder_ensemble(
        DisorderEnsembleRequest(seeds=(1, 2)),
        realization_factory=lambda seed: apply_uniform_coordinate_perturbation(
            geometry,
            width=0.1,
            seed=seed,
        ),
        evaluation_factory=evaluate,
    )
    assert all(member.failure is None for member in ensemble.members)

    validity_metric = compute_robustness_fraction(
        ensemble,
        criterion=RobustnessSuccessCriterion(
            key="phase_7_validity",
            description="Use the explicit Phase-7 validity decision.",
            predicate=lambda member: (
                member.evaluation is not None and member.evaluation.is_valid
            ),
        ),
    )
    completion_metric = compute_robustness_fraction(
        ensemble,
        criterion=RobustnessSuccessCriterion(
            key="evaluation_present",
            description="Require only that an evaluation object was returned.",
            predicate=lambda member: member.evaluation is not None,
        ),
    )

    assert validity_metric.successes == (True, False)
    assert validity_metric.value == pytest.approx(0.5)
    assert completion_metric.successes == (True, True)
    assert completion_metric.value == pytest.approx(1.0)
    assert validity_metric.execution_failure_seeds == ()


@pytest.mark.parametrize(
    ("accepted_seeds", "expected"),
    [(frozenset(), 0.0), (frozenset({1, 2, 3}), 1.0)],
)
def test_fraction_includes_zero_and_one_endpoints(
    accepted_seeds: frozenset[int],
    expected: float,
) -> None:
    ensemble = _parameter_ensemble(seeds=(1, 2, 3))
    metric = compute_robustness_fraction(
        ensemble,
        criterion=RobustnessSuccessCriterion(
            key="endpoint_fixture",
            description="Fixture criterion for endpoint coverage.",
            predicate=lambda member: member.seed in accepted_seeds,
        ),
    )

    assert metric.value == expected


def test_predicate_exception_propagates_as_metric_configuration_failure() -> None:
    ensemble = _parameter_ensemble(seeds=(1,))

    def fail(member: DisorderEnsembleMember) -> bool:
        del member
        raise RuntimeError("criterion failed")

    with pytest.raises(RuntimeError, match="criterion failed"):
        compute_robustness_fraction(
            ensemble,
            criterion=RobustnessSuccessCriterion(
                key="failing_criterion",
                description="A deliberately failing test criterion.",
                predicate=fail,
            ),
        )


@pytest.mark.parametrize("result", [1, 1.0, np.bool_(True), None])
def test_predicate_must_return_an_actual_boolean(result: object) -> None:
    ensemble = _parameter_ensemble(seeds=(1,))
    with pytest.raises(TypeError, match="must return a boolean"):
        compute_robustness_fraction(
            ensemble,
            criterion=RobustnessSuccessCriterion(
                key="nonboolean_criterion",
                description="A deliberately invalid result type.",
                predicate=lambda member: result,  # type: ignore[arg-type,return-value]
            ),
        )


@pytest.mark.parametrize("key", ["", "contains-hyphen", "with space"])
def test_criterion_requires_stable_key(key: str) -> None:
    with pytest.raises(ValueError, match="criterion key"):
        RobustnessSuccessCriterion(
            key=key,
            description="Valid description.",
            predicate=lambda member: True,
        )


def test_metric_record_is_immutable_and_validates_failure_partition() -> None:
    request = DisorderEnsembleRequest(seeds=(1, 2))
    metric = RobustnessFractionMetric(
        criterion_key="manual_fixture",
        criterion_description="  A manually constructed fixture.  ",
        request=request,
        successes=(True, False),
        execution_failure_indices=(1,),
    )

    assert metric.criterion_description == "A manually constructed fixture."
    with pytest.raises(FrozenInstanceError):
        metric.successes = (False, False)  # type: ignore[misc]
    with pytest.raises(ValueError, match="cannot be marked successful"):
        RobustnessFractionMetric(
            criterion_key="invalid_fixture",
            criterion_description="Invalid execution failure partition.",
            request=request,
            successes=(True, True),
            execution_failure_indices=(1,),
        )
