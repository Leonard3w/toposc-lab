from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
from typing import Literal

import numpy as np
import pytest

from toposc_lab.core.results import BasisLayout, SimulationResult
from toposc_lab.evaluation import (
    CandidateIssueCategory,
    CandidateIssueSeverity,
    CandidateValidityIssue,
    CandidateValidityReport,
    GeometryEvaluation,
    GeometryEvaluationRun,
    ObjectiveDirection,
    create_reproducibility_record,
)
from toposc_lab.geometry import chain
from toposc_lab.search import (
    BASELINE_STATISTICS_VERSION,
    BaselineSuccessCriterion,
    BaselineUncertaintyMethod,
    BatchEvaluationCandidate,
    BatchEvaluationFailure,
    BatchEvaluationFailureStage,
    BatchEvaluationMember,
    BatchEvaluationRequest,
    BatchEvaluationResult,
    CandidateRankingConfig,
    CandidateRankingCriterion,
    CandidateRankingEntry,
    CandidateRankingResult,
    CandidateRankingValueKind,
    SearchBaselineStatistics,
    SearchBaselineTrial,
    compute_search_baseline_statistics,
    rank_evaluated_candidates,
)

_Status = float | Literal["invalid", "failure"]


def _candidate(trial_index: int, candidate_index: int) -> BatchEvaluationCandidate:
    return BatchEvaluationCandidate(
        geometry=chain(2),
        model_parameters={
            "trial_label": trial_index,
            "candidate_label": candidate_index,
        },
        evaluation_seed=1000 + 10 * trial_index + candidate_index,
    )


def _valid_run(
    candidate: BatchEvaluationCandidate,
    *,
    quality: float,
) -> GeometryEvaluationRun:
    simulation = SimulationResult(
        model_name="baseline_test_model",
        eigenvalues=np.asarray((-quality, quality)),
        eigenvectors=np.eye(2, dtype=complex),
        basis_layout=BasisLayout(spatial_shape=(2,)),
        parameters=candidate.model_parameters,
    )
    evaluation = GeometryEvaluation(
        gap=2.0 * quality,
        low_energy_states={0: -quality, 1: quality},
        zero_mode_count=0,
        geometry_descriptors={"quality": quality},
    )
    reproducibility = create_reproducibility_record(
        candidate.geometry,
        seed=candidate.evaluation_seed,
        model_name="baseline_test_model",
        model_parameters=candidate.model_parameters,
        solver_name="exact_diagonalization",
        solver_settings={},
        evaluation_settings={},
        code_version="phase-9.7-test",
    )
    return GeometryEvaluationRun(
        simulation_result=simulation,
        evaluation=evaluation,
        validity=CandidateValidityReport(()),
        reproducibility=reproducibility,
    )


def _invalid_run() -> GeometryEvaluationRun:
    return GeometryEvaluationRun(
        simulation_result=None,
        evaluation=None,
        validity=CandidateValidityReport(
            (
                CandidateValidityIssue(
                    code="disconnected_components",
                    severity=CandidateIssueSeverity.ERROR,
                    category=CandidateIssueCategory.GEOMETRY,
                    message="synthetic invalid candidate",
                    path="geometry",
                ),
            )
        ),
    )


def _ranking(
    trial_index: int,
    statuses: tuple[_Status, ...],
    *,
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE,
) -> CandidateRankingResult:
    candidates = tuple(
        _candidate(trial_index, candidate_index) for candidate_index in range(len(statuses))
    )
    members: list[BatchEvaluationMember] = []
    for candidate_index, (candidate, status) in enumerate(zip(candidates, statuses, strict=True)):
        if status == "failure":
            members.append(
                BatchEvaluationMember(
                    candidate_index,
                    candidate,
                    None,
                    BatchEvaluationFailure(
                        stage=BatchEvaluationFailureStage.EVALUATION_CALLBACK,
                        error_type="RuntimeError",
                        message="synthetic callback failure",
                    ),
                )
            )
        elif status == "invalid":
            members.append(
                BatchEvaluationMember(
                    candidate_index,
                    candidate,
                    _invalid_run(),
                    None,
                )
            )
        else:
            members.append(
                BatchEvaluationMember(
                    candidate_index,
                    candidate,
                    _valid_run(candidate, quality=status),
                    None,
                )
            )
    batch = BatchEvaluationResult(
        request=BatchEvaluationRequest(candidates),
        members=tuple(members),
    )
    config = CandidateRankingConfig(
        (
            CandidateRankingCriterion(
                "quality",
                direction,
                CandidateRankingValueKind.REAL,
            ),
        )
    )

    def ranking_values(member: BatchEvaluationMember) -> dict[str, object]:
        assert member.evaluation is not None
        assert member.evaluation.evaluation is not None
        return {"quality": member.evaluation.evaluation.geometry_descriptors["quality"]}

    return rank_evaluated_candidates(
        batch,
        config=config,
        value_factory=ranking_values,
    )


def _trial(
    trial_index: int,
    statuses: tuple[_Status, ...],
    *,
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE,
) -> SearchBaselineTrial:
    return SearchBaselineTrial(
        trial_key=f"trial_{trial_index}",
        ranking=_ranking(trial_index, statuses, direction=direction),
    )


def _quality(entry: CandidateRankingEntry) -> float:
    return float(entry.values["quality"])


def test_baseline_measures_per_trial_hit_rate_and_retains_candidate_failures() -> None:
    trials = (
        _trial(0, (0.9, 0.2, 0.1)),
        _trial(1, (0.7, 0.6, 0.1)),
        _trial(2, ("invalid", 0.85, "failure")),
        _trial(3, ("failure", "invalid", 0.4)),
    )
    calls: list[tuple[int, int]] = []

    def is_strong(entry: CandidateRankingEntry) -> bool:
        parameters = entry.member.candidate.model_parameters
        calls.append((int(parameters["trial_label"]), int(parameters["candidate_label"])))
        return _quality(entry) >= 0.8

    criterion = BaselineSuccessCriterion(
        key="synthetic_strong_candidate",
        description="Synthetic predeclared quality criterion.",
        predicate=is_strong,
    )

    statistics = compute_search_baseline_statistics(
        trials,
        criterion=criterion,
        confidence_level=0.95,
    )

    assert statistics.statistics_version == BASELINE_STATISTICS_VERSION
    assert statistics.criterion is criterion
    assert statistics.uncertainty_method is BaselineUncertaintyMethod.WILSON_SCORE
    assert statistics.trial_count == 4
    assert statistics.trial_successes == (True, False, True, False)
    assert statistics.successful_trial_count == 2
    assert statistics.trial_success_fraction == 0.5
    assert statistics.trial_success_standard_error == 0.25
    assert statistics.critical_value == pytest.approx(1.959963984540054)
    assert statistics.trial_success_confidence_interval == pytest.approx(
        (0.15003898915214947, 0.8499610108478506)
    )
    assert statistics.candidates_per_trial == 3
    assert statistics.total_candidate_count == 12
    assert statistics.successful_candidate_count == 2
    assert statistics.candidate_success_fraction == pytest.approx(1.0 / 6.0)
    assert statistics.invalid_candidate_count == 2
    assert statistics.callback_failure_count == 2
    assert calls == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 1),
        (3, 2),
    ]

    third = statistics.outcomes[2]
    assert third.criterion is criterion
    assert third.candidate_successes == (False, True, False)
    assert third.invalid_candidate_indices == (0,)
    assert third.callback_failure_indices == (2,)
    assert third.successful_candidate_indices == (1,)


@pytest.mark.parametrize(
    ("statuses", "expected_interval_edge"),
    [
        (((0.1,), (0.2,), (0.3,)), 0.5614970317550454),
        (((0.9,), (0.9,), (0.9,)), 0.4385029682449546),
    ],
)
def test_wilson_interval_remains_non_degenerate_at_observed_endpoints(
    statuses: tuple[tuple[_Status, ...], ...],
    expected_interval_edge: float,
) -> None:
    criterion = BaselineSuccessCriterion(
        "strong",
        "Synthetic criterion fixed before the test outcomes.",
        lambda entry: entry.values["quality"] >= 0.8,
    )
    statistics = compute_search_baseline_statistics(
        tuple(_trial(index, values) for index, values in enumerate(statuses)),
        criterion=criterion,
        confidence_level=0.95,
    )

    lower, upper = statistics.trial_success_confidence_interval
    assert statistics.trial_success_standard_error == 0.0
    if statistics.trial_success_fraction == 0.0:
        assert lower == 0.0
        assert upper == pytest.approx(expected_interval_edge)
    else:
        assert lower == pytest.approx(expected_interval_edge)
        assert upper == 1.0


def test_success_predicate_is_not_short_circuited_after_trial_hit() -> None:
    trial = _trial(0, (0.9, 0.95, 0.1))
    calls: list[int] = []

    def predicate(entry: CandidateRankingEntry) -> bool:
        calls.append(int(entry.candidate_index))
        return _quality(entry) >= 0.8

    statistics = compute_search_baseline_statistics(
        (trial,),
        criterion=BaselineSuccessCriterion(
            "all_decisions",
            "Every rankable candidate receives a decision.",
            predicate,
        ),
        confidence_level=0.9,
    )

    assert calls == [0, 1, 2]
    assert statistics.outcomes[0].candidate_successes == (True, True, False)


def test_duplicate_or_incompatible_trials_fail_before_predicate_execution() -> None:
    reference = _trial(0, (0.5, 0.4))
    duplicate = SearchBaselineTrial("trial_0", _ranking(1, (0.5, 0.4)))
    different_count = _trial(2, (0.5,))
    different_config = _trial(
        3,
        (0.5, 0.4),
        direction=ObjectiveDirection.MINIMIZE,
    )
    calls = 0

    def predicate(entry: CandidateRankingEntry) -> bool:
        nonlocal calls
        calls += 1
        return True

    criterion = BaselineSuccessCriterion(
        "never_called",
        "Protocol validation precedes candidate decisions.",
        predicate,
    )
    with pytest.raises(ValueError, match="trial keys must be unique"):
        compute_search_baseline_statistics(
            (reference, duplicate), criterion=criterion, confidence_level=0.95
        )
    with pytest.raises(ValueError, match="same candidate count"):
        compute_search_baseline_statistics(
            (reference, different_count), criterion=criterion, confidence_level=0.95
        )
    with pytest.raises(ValueError, match="same ranking configuration"):
        compute_search_baseline_statistics(
            (reference, different_config), criterion=criterion, confidence_level=0.95
        )
    assert calls == 0


def test_predicate_contract_errors_propagate_without_partial_statistics() -> None:
    trials = (_trial(0, (0.5,)), _trial(1, (0.6,)))
    calls: list[int] = []

    def raises(entry: CandidateRankingEntry) -> bool:
        trial_label = int(entry.member.candidate.model_parameters["trial_label"])
        calls.append(trial_label)
        if trial_label == 1:
            raise RuntimeError("synthetic criterion failure")
        return False

    with pytest.raises(RuntimeError, match="criterion failure"):
        compute_search_baseline_statistics(
            trials,
            criterion=BaselineSuccessCriterion(
                "raising",
                "Synthetic failure propagation.",
                raises,
            ),
            confidence_level=0.95,
        )
    assert calls == [0, 1]

    with pytest.raises(TypeError, match="must return a boolean"):
        compute_search_baseline_statistics(
            trials,
            criterion=BaselineSuccessCriterion(
                "wrong_type",
                "Synthetic wrong return type.",
                lambda entry: 1,  # type: ignore[arg-type,return-value]
            ),
            confidence_level=0.95,
        )


@pytest.mark.parametrize("confidence_level", (0.0, 1.0, -0.1, math.inf, math.nan))
def test_confidence_level_must_be_finite_and_strictly_inside_unit_interval(
    confidence_level: float,
) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        compute_search_baseline_statistics(
            (_trial(0, (0.5,)),),
            criterion=BaselineSuccessCriterion("criterion", "Description.", lambda entry: False),
            confidence_level=confidence_level,
        )


def test_baseline_api_rejects_invalid_top_level_contracts() -> None:
    criterion = BaselineSuccessCriterion("criterion", "Description.", lambda entry: False)
    trial = _trial(0, (0.5,))

    with pytest.raises(ValueError, match="must not be empty"):
        compute_search_baseline_statistics((), criterion=criterion, confidence_level=0.95)
    with pytest.raises(TypeError, match="SearchBaselineTrial"):
        compute_search_baseline_statistics(  # type: ignore[arg-type]
            (object(),), criterion=criterion, confidence_level=0.95
        )
    with pytest.raises(TypeError, match="BaselineSuccessCriterion"):
        compute_search_baseline_statistics(  # type: ignore[arg-type]
            (trial,), criterion=object(), confidence_level=0.95
        )
    with pytest.raises(TypeError, match="real number"):
        compute_search_baseline_statistics(  # type: ignore[arg-type]
            (trial,), criterion=criterion, confidence_level=True
        )
    with pytest.raises(TypeError, match="BaselineUncertaintyMethod"):
        compute_search_baseline_statistics(  # type: ignore[arg-type]
            (trial,),
            criterion=criterion,
            confidence_level=0.95,
            uncertainty_method="wilson_score",
        )


def test_trials_criteria_and_statistics_are_immutable_and_validated() -> None:
    ranking = _ranking(0, (0.9,))
    criterion = BaselineSuccessCriterion("criterion", "  Description.  ", lambda entry: True)
    trial = SearchBaselineTrial("trial_0", ranking)
    statistics = compute_search_baseline_statistics(
        (trial,), criterion=criterion, confidence_level=0.95
    )

    assert criterion.description == "Description."
    assert statistics.outcomes[0].trial is trial
    with pytest.raises(FrozenInstanceError):
        statistics.confidence_level = 0.9  # type: ignore[misc]
    with pytest.raises(ValueError, match="exact baseline criterion"):
        SearchBaselineStatistics(
            criterion=BaselineSuccessCriterion(
                "other",
                "A different criterion.",
                lambda entry: True,
            ),
            outcomes=statistics.outcomes,
            confidence_level=0.95,
        )


def test_warnings_preserve_statistical_and_scientific_limits() -> None:
    statistics = compute_search_baseline_statistics(
        (_trial(0, ("invalid", "failure")),),
        criterion=BaselineSuccessCriterion(
            "criterion",
            "Explicit synthetic criterion.",
            lambda entry: False,
        ),
        confidence_level=0.95,
    )

    joined = " ".join(statistics.warnings)
    assert "frozen before outcomes" in joined
    assert "representative independent attempts" in joined
    assert "candidate success fraction is descriptive" in joined
    assert "not evidence of topology" in joined
    assert "remain unsuccessful in the denominator" in joined
    assert not hasattr(statistics, "p_value")
    assert not hasattr(statistics, "experiment_seed")
