from __future__ import annotations

from collections.abc import Mapping
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
from toposc_lab.search import (
    BATCH_EVALUATION_EXECUTION_VERSION,
    BatchEvaluationCandidate,
    BatchEvaluationFailure,
    BatchEvaluationFailureStage,
    BatchEvaluationMember,
    BatchEvaluationRequest,
    BatchEvaluationResult,
    GeometrySamplingRecipe,
    RandomGeometrySamplingConfig,
    RandomModelParameterSamplingConfig,
    UniformFloatDistribution,
    execute_evaluation_batch,
    sample_random_geometries,
    sample_random_model_parameters,
)


class _BatchModel(BaseModel):
    def __init__(
        self,
        geometry: Geometry,
        parameters: Mapping[str, object],
    ) -> None:
        self.geometry = geometry
        self._parameters = dict(parameters)

    @property
    def parameters(self) -> dict[str, object]:
        return dict(self._parameters)

    @property
    def basis_layout(self) -> BasisLayout:
        return BasisLayout(
            spatial_shape=(self.geometry.n_sites,),
            components_per_site=1,
            ordering="site_major",
            component_labels=("orbital",),
        )

    def hamiltonian(self) -> np.ndarray:
        onsite_value = self._parameters["onsite"]
        hopping_value = self._parameters["hopping"]
        if isinstance(onsite_value, bool) or not isinstance(
            onsite_value,
            (int, float),
        ):
            raise TypeError("onsite must be numeric")
        if isinstance(hopping_value, bool) or not isinstance(
            hopping_value,
            (int, float),
        ):
            raise TypeError("hopping must be numeric")
        onsite = float(onsite_value)
        hopping = float(hopping_value)
        matrix = np.eye(self.geometry.n_sites, dtype=complex) * onsite
        for edge in self.geometry.edges:
            matrix[edge.source, edge.target] = hopping
            matrix[edge.target, edge.source] = hopping
        return matrix


def _candidate(
    *,
    n_sites: int = 3,
    onsite: float = 0.0,
    hopping: float = 1.0,
    seed: int | None = None,
) -> BatchEvaluationCandidate:
    return BatchEvaluationCandidate(
        geometry=chain(n_sites),
        model_parameters={"onsite": onsite, "hopping": hopping},
        evaluation_seed=seed,
    )


def _evaluate_candidate(
    candidate: BatchEvaluationCandidate,
) -> GeometryEvaluationRun:
    return evaluate_geometry(
        candidate.geometry,
        adapter=GeometryModelAdapter(
            model_factory=lambda geometry: _BatchModel(
                geometry,
                candidate.model_parameters,
            ),
            requirements=ModelGeometryRequirements(
                require_connected=True,
                require_edges=True,
            ),
        ),
        seed=candidate.evaluation_seed,
        code_version="phase-9.3-test",
    )


def test_batch_executes_real_geometry_evaluations_in_request_order() -> None:
    candidates = (
        _candidate(onsite=-0.5, seed=11),
        _candidate(onsite=0.0, seed=12),
        _candidate(onsite=0.5, seed=13),
    )
    request = BatchEvaluationRequest(candidates)

    result = execute_evaluation_batch(request, evaluator=_evaluate_candidate)

    assert result.execution_version == BATCH_EVALUATION_EXECUTION_VERSION
    assert tuple(member.candidate_index for member in result.members) == (0, 1, 2)
    assert all(member.failure is None for member in result.members)
    assert all(member.evaluation is not None for member in result.members)
    assert all(
        member.evaluation is not None and member.evaluation.is_valid
        for member in result.members
    )
    records = tuple(
        evaluation.reproducibility
        for member in result.members
        if (evaluation := member.evaluation) is not None
    )
    assert all(record is not None for record in records)
    assert tuple(record.seed for record in records if record is not None) == (
        11,
        12,
        13,
    )


def test_callback_failure_is_retained_and_later_candidates_continue() -> None:
    request = BatchEvaluationRequest(
        (
            _candidate(onsite=-0.5),
            _candidate(onsite=0.0),
            _candidate(onsite=0.5),
        )
    )
    calls: list[float] = []

    def evaluator(candidate: BatchEvaluationCandidate) -> GeometryEvaluationRun:
        onsite_value = candidate.model_parameters["onsite"]
        if isinstance(onsite_value, bool) or not isinstance(
            onsite_value,
            (int, float),
        ):
            raise TypeError("onsite must be numeric")
        onsite = float(onsite_value)
        calls.append(onsite)
        if onsite == 0.0:
            raise ValueError("synthetic candidate failure")
        return _evaluate_candidate(candidate)

    result = execute_evaluation_batch(request, evaluator=evaluator)

    assert calls == [-0.5, 0.0, 0.5]
    assert result.members[0].evaluation is not None
    assert result.members[0].failure is None
    assert result.members[1].evaluation is None
    assert result.members[1].failure == BatchEvaluationFailure(
        stage=BatchEvaluationFailureStage.EVALUATION_CALLBACK,
        error_type="ValueError",
        message="synthetic candidate failure",
    )
    assert result.members[2].evaluation is not None
    assert result.members[2].failure is None


def test_invalid_evaluation_run_is_not_relabelled_as_batch_failure() -> None:
    candidate = BatchEvaluationCandidate(
        geometry=Geometry(n_sites=2),
        model_parameters={"onsite": 0.0, "hopping": 1.0},
    )

    result = execute_evaluation_batch(
        BatchEvaluationRequest((candidate,)),
        evaluator=_evaluate_candidate,
    )

    member = result.members[0]
    assert member.failure is None
    assert member.evaluation is not None
    assert not member.evaluation.is_valid
    assert member.evaluation.reproducibility is None
    assert tuple(issue.code for issue in member.evaluation.validity.errors) == (
        "disconnected_components",
    )


def test_phase_9_sampler_outputs_are_combined_only_by_explicit_caller_pairing() -> None:
    geometries = sample_random_geometries(
        RandomGeometrySamplingConfig(
            recipes=(GeometrySamplingRecipe("chain", {"n_sites": 3}),),
            sample_count=2,
        ),
        seed=101,
    )
    parameters = sample_random_model_parameters(
        RandomModelParameterSamplingConfig(
            base_parameters={"onsite": 0.0, "hopping": 1.0},
            distributions={"onsite": UniformFloatDistribution(-0.5, 0.5)},
            sample_count=2,
        ),
        seed=202,
    )
    candidates = tuple(
        BatchEvaluationCandidate(
            geometry=geometry_sample.geometry,
            model_parameters=parameter_sample.parameters,
        )
        for geometry_sample, parameter_sample in zip(
            geometries.samples,
            parameters.samples,
            strict=True,
        )
    )

    result = execute_evaluation_batch(
        BatchEvaluationRequest(candidates),
        evaluator=_evaluate_candidate,
    )

    assert tuple(member.candidate for member in result.members) == candidates
    assert all(
        member.evaluation is not None and member.evaluation.is_valid
        for member in result.members
    )


@pytest.mark.parametrize("mismatch", ("geometry", "parameters", "seed"))
def test_reproducibility_mismatch_is_an_immediate_api_error(mismatch: str) -> None:
    candidate = _candidate(onsite=0.25, seed=17)

    def mismatched_evaluator(
        requested: BatchEvaluationCandidate,
    ) -> GeometryEvaluationRun:
        evaluated = BatchEvaluationCandidate(
            geometry=chain(4) if mismatch == "geometry" else requested.geometry,
            model_parameters=(
                {"onsite": -0.75, "hopping": 1.0}
                if mismatch == "parameters"
                else requested.model_parameters
            ),
            evaluation_seed=18 if mismatch == "seed" else requested.evaluation_seed,
        )
        return _evaluate_candidate(evaluated)

    with pytest.raises(ValueError, match=mismatch):
        execute_evaluation_batch(
            BatchEvaluationRequest((candidate,)),
            evaluator=mismatched_evaluator,
        )


def test_wrong_callback_return_type_is_an_immediate_api_error() -> None:
    calls = 0

    def wrong_evaluator(candidate: BatchEvaluationCandidate) -> str:
        nonlocal calls
        del candidate
        calls += 1
        return "not an evaluation run"

    with pytest.raises(TypeError, match="must return GeometryEvaluationRun"):
        execute_evaluation_batch(
            BatchEvaluationRequest((_candidate(), _candidate(onsite=1.0))),
            evaluator=wrong_evaluator,  # type: ignore[arg-type]
        )

    assert calls == 1


def test_keyboard_interrupt_is_not_converted_to_member_failure() -> None:
    def interrupted(candidate: BatchEvaluationCandidate) -> GeometryEvaluationRun:
        del candidate
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute_evaluation_batch(
            BatchEvaluationRequest((_candidate(),)),
            evaluator=interrupted,
        )


def test_candidate_parameters_and_request_sequence_are_defensively_immutable() -> None:
    nested = {"values": [1, 2]}
    candidate = BatchEvaluationCandidate(
        geometry=chain(3),
        model_parameters={
            "onsite": 0.0,
            "hopping": 1.0,
            "nested": nested,  # type: ignore[dict-item]
        },
    )
    source = [candidate]
    request = BatchEvaluationRequest(source)  # type: ignore[arg-type]
    nested["values"].append(3)
    source.append(_candidate(onsite=2.0))

    assert len(request.candidates) == 1
    assert candidate.model_parameters["nested"] == {"values": (1, 2)}
    with pytest.raises(TypeError):
        candidate.model_parameters["onsite"] = 1.0  # type: ignore[index]


@pytest.mark.parametrize("seed", (-1, -10))
def test_candidate_rejects_negative_evaluation_seed(seed: int) -> None:
    with pytest.raises(ValueError, match="evaluation_seed must be nonnegative"):
        _candidate(seed=seed)


@pytest.mark.parametrize("seed", (True, 1.5, "1"))
def test_candidate_rejects_noninteger_evaluation_seed(seed: object) -> None:
    with pytest.raises(TypeError, match="evaluation_seed must be an integer"):
        BatchEvaluationCandidate(
            geometry=chain(3),
            model_parameters={"onsite": 0.0, "hopping": 1.0},
            evaluation_seed=seed,  # type: ignore[arg-type]
        )


def test_request_rejects_empty_or_invalid_candidate_collections() -> None:
    with pytest.raises(ValueError, match="at least one"):
        BatchEvaluationRequest(())
    with pytest.raises(TypeError, match="BatchEvaluationCandidate"):
        BatchEvaluationRequest(("candidate",))  # type: ignore[arg-type]


def test_batch_rejects_invalid_request_or_evaluator_api() -> None:
    request = BatchEvaluationRequest((_candidate(),))

    with pytest.raises(TypeError, match="request must be BatchEvaluationRequest"):
        execute_evaluation_batch(object(), evaluator=_evaluate_candidate)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="evaluator must be callable"):
        execute_evaluation_batch(request, evaluator=None)  # type: ignore[arg-type]


def test_member_requires_exactly_one_evaluation_or_failure() -> None:
    candidate = _candidate()
    evaluation = _evaluate_candidate(candidate)
    failure = BatchEvaluationFailure(
        stage=BatchEvaluationFailureStage.EVALUATION_CALLBACK,
        error_type="ValueError",
        message="failure",
    )

    with pytest.raises(ValueError, match="exactly one"):
        BatchEvaluationMember(
            candidate_index=0,
            candidate=candidate,
            evaluation=None,
            failure=None,
        )
    with pytest.raises(ValueError, match="exactly one"):
        BatchEvaluationMember(
            candidate_index=0,
            candidate=candidate,
            evaluation=evaluation,
            failure=failure,
        )


def test_result_requires_exact_request_order_and_candidate_identity() -> None:
    first = _candidate(onsite=0.0)
    second = _candidate(onsite=1.0)
    request = BatchEvaluationRequest((first, second))
    members = (
        BatchEvaluationMember(0, first, _evaluate_candidate(first), None),
        BatchEvaluationMember(1, second, _evaluate_candidate(second), None),
    )

    assert BatchEvaluationResult(request, members).members == members
    with pytest.raises(ValueError, match="indices"):
        BatchEvaluationResult(
            request,
            (
                BatchEvaluationMember(1, first, _evaluate_candidate(first), None),
                members[1],
            ),
        )
    equivalent_but_distinct = _candidate(onsite=0.0)
    with pytest.raises(ValueError, match="exact requested candidates"):
        BatchEvaluationResult(
            request,
            (
                BatchEvaluationMember(
                    0,
                    equivalent_but_distinct,
                    _evaluate_candidate(equivalent_but_distinct),
                    None,
                ),
                members[1],
            ),
        )
