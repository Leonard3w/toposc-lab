from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import BasisLayout
from toposc_lab.evaluation import (
    GeometryEvaluationConfig,
    GeometryEvaluationRun,
    GeometryModelAdapter,
    ModelGeometryRequirements,
    evaluate_geometry,
)
from toposc_lab.geometry import Geometry, chain
from toposc_lab.robustness import (
    DISORDER_ENSEMBLE_EXECUTION_VERSION,
    DisorderEnsembleFailureStage,
    DisorderEnsembleRequest,
    DisorderOutcome,
    NodeRemovalRealization,
    apply_random_node_removal,
    apply_uniform_coordinate_perturbation,
    apply_uniform_parameter_perturbation,
    execute_disorder_ensemble,
)


class _GeometryModel(BaseModel):
    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry

    @property
    def parameters(self) -> dict[str, object]:
        return {"onsite_scale": 1.0}

    @property
    def basis_layout(self) -> BasisLayout:
        return BasisLayout(spatial_shape=(self.geometry.n_sites,))

    def hamiltonian(self) -> np.ndarray:
        return np.diag(np.linspace(-1.0, 1.0, self.geometry.n_sites))


def _model_factory(geometry: Geometry) -> BaseModel:
    return _GeometryModel(geometry)


def _coordinate_realization_factory(
    geometry: Geometry,
    *,
    width: float = 0.2,
) -> Callable[[int], DisorderOutcome]:
    def realize(seed: int) -> DisorderOutcome:
        return apply_uniform_coordinate_perturbation(
            geometry,
            width=width,
            seed=seed,
        )

    return realize


def test_request_normalizes_explicit_seed_order_and_result_preserves_it() -> None:
    request = DisorderEnsembleRequest(seeds=[7, np.int64(3), 11])  # type: ignore[arg-type]
    result = execute_disorder_ensemble(
        request,
        realization_factory=lambda seed: apply_uniform_parameter_perturbation(
            {"mass": 0.5},
            widths={"mass": 0.2},
            seed=seed,
        ),
    )

    assert request.seeds == (7, 3, 11)
    assert result.execution_version == DISORDER_ENSEMBLE_EXECUTION_VERSION
    assert result.evaluations_requested is False
    assert tuple(member.sample_index for member in result.members) == (0, 1, 2)
    assert tuple(member.seed for member in result.members) == request.seeds
    assert all(member.disorder is not None for member in result.members)
    assert all(member.evaluation is None for member in result.members)
    assert all(member.failure is None for member in result.members)
    assert tuple(
        member.disorder.provenance.seed
        for member in result.members
        if member.disorder is not None
    ) == request.seeds


def test_same_request_reproduces_all_member_snapshots() -> None:
    geometry = chain(4)
    request = DisorderEnsembleRequest(seeds=(2, 5, 9))
    factory = _coordinate_realization_factory(geometry)

    first = execute_disorder_ensemble(request, realization_factory=factory)
    second = execute_disorder_ensemble(request, realization_factory=factory)

    assert tuple(
        member.disorder.provenance
        for member in first.members
        if member.disorder is not None
    ) == tuple(
        member.disorder.provenance
        for member in second.members
        if member.disorder is not None
    )


def test_specialized_node_removal_outcomes_are_retained_without_losing_mapping() -> None:
    geometry = chain(6)
    result = execute_disorder_ensemble(
        DisorderEnsembleRequest(seeds=(3, 5)),
        realization_factory=lambda seed: apply_random_node_removal(
            geometry,
            removal_probability=0.4,
            seed=seed,
        ),
    )

    for member in result.members:
        assert isinstance(member.disorder, NodeRemovalRealization)
        assert member.disorder.old_to_new == {
            old_site: new_site
            for new_site, old_site in enumerate(member.disorder.surviving_sites)
        }


def test_optional_evaluation_callback_keeps_phase_7_run_separate() -> None:
    geometry = chain(3)
    adapter = GeometryModelAdapter(model_factory=_model_factory)

    def evaluate(disorder: DisorderOutcome, seed: int) -> GeometryEvaluationRun:
        assert isinstance(disorder.state, Geometry)
        return evaluate_geometry(
            disorder.state,
            adapter=adapter,
            config=GeometryEvaluationConfig(low_energy_count=3),
            seed=seed,
            code_version="phase-8.9-test",
        )

    result = execute_disorder_ensemble(
        DisorderEnsembleRequest(seeds=(13, 17)),
        realization_factory=_coordinate_realization_factory(geometry),
        evaluation_factory=evaluate,
    )

    assert result.evaluations_requested is True
    for member in result.members:
        assert member.failure is None
        assert member.disorder is not None
        assert member.evaluation is not None
        assert member.evaluation.is_valid
        assert member.evaluation.reproducibility is not None
        assert member.evaluation.reproducibility.seed == member.seed
        assert member.evaluation.reproducibility.geometry_id == (
            member.disorder.provenance.result.identifier
        )


def test_invalid_phase_7_run_is_not_reclassified_as_ensemble_failure() -> None:
    geometry = chain(3)
    invalid_adapter = GeometryModelAdapter(
        model_factory=_model_factory,
        requirements=ModelGeometryRequirements(required_spatial_axes=(1,)),
    )

    def evaluate(disorder: DisorderOutcome, seed: int) -> GeometryEvaluationRun:
        assert isinstance(disorder.state, Geometry)
        return evaluate_geometry(
            disorder.state,
            adapter=invalid_adapter,
            seed=seed,
            code_version="phase-8.9-test",
        )

    result = execute_disorder_ensemble(
        DisorderEnsembleRequest(seeds=(1,)),
        realization_factory=_coordinate_realization_factory(geometry),
        evaluation_factory=evaluate,
    )

    member = result.members[0]
    assert member.failure is None
    assert member.evaluation is not None
    assert not member.evaluation.is_valid
    assert member.evaluation.validity.errors


def test_realization_exception_is_recorded_and_later_seeds_continue() -> None:
    geometry = chain(3)
    calls: list[int] = []

    def realize(seed: int) -> DisorderOutcome:
        calls.append(seed)
        if seed == 2:
            raise RuntimeError("realization unavailable")
        return apply_uniform_coordinate_perturbation(
            geometry,
            width=0.2,
            seed=seed,
        )

    result = execute_disorder_ensemble(
        DisorderEnsembleRequest(seeds=(1, 2, 3)),
        realization_factory=realize,
    )

    assert calls == [1, 2, 3]
    failed = result.members[1]
    assert failed.disorder is None
    assert failed.evaluation is None
    assert failed.failure is not None
    assert failed.failure.stage is (
        DisorderEnsembleFailureStage.DISORDER_REALIZATION
    )
    assert failed.failure.error_type == "RuntimeError"
    assert failed.failure.message == "realization unavailable"
    assert result.members[2].disorder is not None


def test_evaluation_exception_retains_disorder_and_later_seeds_continue() -> None:
    geometry = chain(3)
    evaluated: list[int] = []

    def evaluate(disorder: DisorderOutcome, seed: int) -> GeometryEvaluationRun:
        del disorder
        evaluated.append(seed)
        if seed == 2:
            raise ValueError("evaluation callback failed")
        return evaluate_geometry(
            geometry,
            adapter=GeometryModelAdapter(model_factory=_model_factory),
            seed=seed,
            code_version="phase-8.9-test",
        )

    result = execute_disorder_ensemble(
        DisorderEnsembleRequest(seeds=(1, 2, 3)),
        realization_factory=_coordinate_realization_factory(geometry),
        evaluation_factory=evaluate,
    )

    assert evaluated == [1, 2, 3]
    failed = result.members[1]
    assert failed.disorder is not None
    assert failed.evaluation is None
    assert failed.failure is not None
    assert failed.failure.stage is (
        DisorderEnsembleFailureStage.EVALUATION_CALLBACK
    )
    assert failed.failure.error_type == "ValueError"
    assert failed.failure.message == "evaluation callback failed"
    assert result.members[2].evaluation is not None


def test_outcomes_must_share_source_and_disorder_definition() -> None:
    geometry = chain(3)

    def changing_definition(seed: int) -> DisorderOutcome:
        return apply_uniform_coordinate_perturbation(
            geometry,
            width=0.1 if seed == 1 else 0.2,
            seed=seed,
        )

    with pytest.raises(ValueError, match="share one source and disorder definition"):
        execute_disorder_ensemble(
            DisorderEnsembleRequest(seeds=(1, 2)),
            realization_factory=changing_definition,
        )


def test_outcome_seed_must_match_requested_seed() -> None:
    with pytest.raises(ValueError, match="seed must match"):
        execute_disorder_ensemble(
            DisorderEnsembleRequest(seeds=(1,)),
            realization_factory=lambda seed: apply_uniform_parameter_perturbation(
                {"mass": 0.5},
                widths={"mass": 0.2},
                seed=seed + 1,
            ),
        )


def test_callback_return_contracts_are_enforced() -> None:
    request = DisorderEnsembleRequest(seeds=(1,))
    with pytest.raises(TypeError, match="implementing DisorderOutcome"):
        execute_disorder_ensemble(
            request,
            realization_factory=cast(Callable[[int], DisorderOutcome], lambda seed: seed),
        )

    with pytest.raises(TypeError, match="must return GeometryEvaluationRun"):
        execute_disorder_ensemble(
            request,
            realization_factory=lambda seed: apply_uniform_parameter_perturbation(
                {"mass": 0.5},
                widths={"mass": 0.2},
                seed=seed,
            ),
            evaluation_factory=cast(
                Callable[[DisorderOutcome, int], GeometryEvaluationRun],
                lambda disorder, seed: (disorder, seed),
            ),
        )


def test_structural_outcome_must_match_its_recorded_result_snapshot() -> None:
    valid = apply_uniform_parameter_perturbation(
        {"mass": 0.5},
        widths={"mass": 0.2},
        seed=1,
    )

    class _ForgedOutcome:
        state = {"mass": 99.0}
        provenance = valid.provenance

    with pytest.raises(ValueError, match="state does not match"):
        execute_disorder_ensemble(
            DisorderEnsembleRequest(seeds=(1,)),
            realization_factory=lambda seed: _ForgedOutcome(),
        )


def test_keyboard_interrupt_is_not_captured_as_member_failure() -> None:
    def interrupt(seed: int) -> DisorderOutcome:
        del seed
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute_disorder_ensemble(
            DisorderEnsembleRequest(seeds=(1,)),
            realization_factory=interrupt,
        )


@pytest.mark.parametrize(
    "seeds",
    [(), (1, 1), (-1,), (True,), (1.5,)],
)
def test_invalid_seed_sequences_are_rejected(seeds: tuple[object, ...]) -> None:
    expected_error = ValueError if seeds in ((), (1, 1), (-1,)) else TypeError
    with pytest.raises(expected_error):
        DisorderEnsembleRequest(seeds=seeds)  # type: ignore[arg-type]
