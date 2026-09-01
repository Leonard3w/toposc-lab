from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import numpy as np
import pytest

from toposc_lab.core.results import BasisLayout, SimulationResult
from toposc_lab.evaluation import (
    BasicScoreComponent,
    CandidateIssueCategory,
    CandidateIssueSeverity,
    CandidateValidityIssue,
    CandidateValidityReport,
    GeometryEvaluation,
    GeometryEvaluationRun,
    ObjectiveDirection,
    compute_basic_scalar_score,
    create_reproducibility_record,
)
from toposc_lab.geometry import chain
from toposc_lab.search import (
    CANDIDATE_RANKING_VERSION,
    BatchEvaluationCandidate,
    BatchEvaluationFailure,
    BatchEvaluationFailureStage,
    BatchEvaluationMember,
    BatchEvaluationRequest,
    BatchEvaluationResult,
    CandidateRankingConfig,
    CandidateRankingCriterion,
    CandidateRankingValueKind,
    UnrankedCandidateReason,
    rank_evaluated_candidates,
)


def _candidate(index: int) -> BatchEvaluationCandidate:
    return BatchEvaluationCandidate(
        geometry=chain(2),
        model_parameters={"candidate_label": index},
        evaluation_seed=100 + index,
    )


def _valid_run(
    candidate: BatchEvaluationCandidate,
    *,
    quality: float,
    cost: int,
    eligible: bool = True,
) -> GeometryEvaluationRun:
    simulation = SimulationResult(
        model_name="ranking_test_model",
        eigenvalues=np.asarray((-quality, quality)),
        eigenvectors=np.eye(2, dtype=complex),
        basis_layout=BasisLayout(spatial_shape=(2,)),
        parameters=candidate.model_parameters,
    )
    evaluation = GeometryEvaluation(
        gap=2.0 * quality,
        low_energy_states={0: -quality, 1: quality},
        zero_mode_count=0,
        geometry_descriptors={
            "quality": quality,
            "cost": cost,
            "eligible": eligible,
        },
    )
    reproducibility = create_reproducibility_record(
        candidate.geometry,
        seed=candidate.evaluation_seed,
        model_name="ranking_test_model",
        model_parameters=candidate.model_parameters,
        solver_name="exact_diagonalization",
        solver_settings={},
        evaluation_settings={},
        code_version="phase-9.5-test",
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


def _valid_batch(
    values: tuple[tuple[float, int], ...],
) -> BatchEvaluationResult:
    candidates = tuple(_candidate(index) for index in range(len(values)))
    request = BatchEvaluationRequest(candidates)
    members = tuple(
        BatchEvaluationMember(
            index,
            candidate,
            _valid_run(candidate, quality=quality, cost=cost),
            None,
        )
        for index, (candidate, (quality, cost)) in enumerate(zip(candidates, values, strict=True))
    )
    return BatchEvaluationResult(request=request, members=members)


def _lexicographic_config() -> CandidateRankingConfig:
    return CandidateRankingConfig(
        (
            CandidateRankingCriterion(
                "quality",
                ObjectiveDirection.MAXIMIZE,
                CandidateRankingValueKind.REAL,
            ),
            CandidateRankingCriterion(
                "cost",
                ObjectiveDirection.MINIMIZE,
                CandidateRankingValueKind.INTEGER,
            ),
        )
    )


def _descriptor_values(member: BatchEvaluationMember) -> dict[str, object]:
    assert member.evaluation is not None
    assert member.evaluation.evaluation is not None
    descriptors = member.evaluation.evaluation.geometry_descriptors
    return {"quality": descriptors["quality"], "cost": descriptors["cost"]}


def test_lexicographic_ranking_respects_direction_ties_and_batch_order() -> None:
    batch = _valid_batch(((0.8, 3), (0.9, 5), (0.9, 4), (0.9, 4)))

    result = rank_evaluated_candidates(
        batch,
        config=_lexicographic_config(),
        value_factory=_descriptor_values,
    )

    assert result.version == CANDIDATE_RANKING_VERSION
    assert result.source is batch
    assert tuple(entry.candidate_index for entry in result.entries) == (0, 1, 2, 3)
    assert tuple(entry.candidate_index for entry in result.ranked_entries) == (
        2,
        3,
        1,
        0,
    )
    assert tuple(entry.rank for entry in result.ranked_entries) == (1, 1, 3, 4)
    assert result.ranked_entries[0].values == {"quality": 0.9, "cost": 4}


def test_invalid_runs_and_callback_failures_remain_visible_but_unranked() -> None:
    valid = _candidate(0)
    invalid = _candidate(1)
    failed = _candidate(2)
    batch = BatchEvaluationResult(
        request=BatchEvaluationRequest((valid, invalid, failed)),
        members=(
            BatchEvaluationMember(
                0,
                valid,
                _valid_run(valid, quality=0.5, cost=2),
                None,
            ),
            BatchEvaluationMember(1, invalid, _invalid_run(), None),
            BatchEvaluationMember(
                2,
                failed,
                None,
                BatchEvaluationFailure(
                    stage=BatchEvaluationFailureStage.EVALUATION_CALLBACK,
                    error_type="ValueError",
                    message="synthetic callback failure",
                ),
            ),
        ),
    )
    calls: list[int] = []

    def values(member: BatchEvaluationMember) -> dict[str, object]:
        calls.append(member.candidate_index)
        return _descriptor_values(member)

    result = rank_evaluated_candidates(
        batch,
        config=_lexicographic_config(),
        value_factory=values,
    )

    assert calls == [0]
    assert tuple(entry.candidate_index for entry in result.entries) == (0, 1, 2)
    assert result.entries[0].rank == 1
    assert result.entries[0].unranked_reason is None
    assert result.entries[1].rank is None
    assert result.entries[1].values == {}
    assert result.entries[1].unranked_reason is UnrankedCandidateReason.INVALID_EVALUATION
    assert result.entries[2].unranked_reason is UnrankedCandidateReason.EVALUATION_CALLBACK_FAILURE
    assert result.ranked_entries == (result.entries[0],)


def test_existing_basic_scalar_score_can_supply_one_explicit_ranking_value() -> None:
    batch = _valid_batch(((0.2, 1), (0.8, 1), (0.5, 1)))
    config = CandidateRankingConfig(
        (
            CandidateRankingCriterion(
                "engineering_score",
                ObjectiveDirection.MAXIMIZE,
                CandidateRankingValueKind.REAL,
            ),
        )
    )

    def score_value(member: BatchEvaluationMember) -> dict[str, object]:
        assert member.evaluation is not None
        assert member.evaluation.evaluation is not None
        score = compute_basic_scalar_score(
            member.evaluation.evaluation,
            weights={BasicScoreComponent.NORMALIZED_GAP: 1.0},
            gap_scale=1.0,
        )
        return {"engineering_score": score.value}

    result = rank_evaluated_candidates(
        batch,
        config=config,
        value_factory=score_value,
    )

    assert tuple(entry.candidate_index for entry in result.ranked_entries) == (1, 2, 0)
    assert "engineering ordering" in " ".join(result.warnings)
    assert "scientific discovery" in " ".join(result.warnings)


def test_boolean_integer_and_real_values_are_normalized_and_immutable() -> None:
    batch = _valid_batch(((0.75, 4),))
    config = CandidateRankingConfig(
        (
            CandidateRankingCriterion(
                "eligible",
                ObjectiveDirection.MAXIMIZE,
                CandidateRankingValueKind.BOOLEAN,
            ),
            CandidateRankingCriterion(
                "cost",
                ObjectiveDirection.MINIMIZE,
                CandidateRankingValueKind.INTEGER,
            ),
            CandidateRankingCriterion(
                "quality",
                ObjectiveDirection.MAXIMIZE,
                CandidateRankingValueKind.REAL,
            ),
        )
    )

    result = rank_evaluated_candidates(
        batch,
        config=config,
        value_factory=lambda member: {
            "eligible": True,
            "cost": np.int64(4),
            "quality": np.float64(0.75),
        },
    )

    values = result.entries[0].values
    assert values == {"eligible": True, "cost": 4, "quality": 0.75}
    assert type(values["cost"]) is int
    assert type(values["quality"]) is float
    with pytest.raises(TypeError):
        values["quality"] = 0.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.entries[0].rank = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("returned", "exception", "message"),
    [
        ({"quality": 0.5}, ValueError, "missing=.*cost"),
        (
            {"quality": 0.5, "cost": 2, "extra": 1},
            ValueError,
            "additional=.*extra",
        ),
        ({"quality": np.nan, "cost": 2}, ValueError, "must be finite"),
        ({"quality": 0.5, "cost": True}, TypeError, "must be an integer"),
        (cast(dict[str, object], None), TypeError, "must return a mapping"),
    ],
)
def test_invalid_value_factory_results_are_contract_errors(
    returned: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        rank_evaluated_candidates(
            _valid_batch(((0.5, 2),)),
            config=_lexicographic_config(),
            value_factory=lambda member: returned,
        )


def test_value_factory_exception_propagates_without_partial_ranking() -> None:
    batch = _valid_batch(((0.5, 2), (0.6, 3)))
    calls: list[int] = []

    def failing(member: BatchEvaluationMember) -> dict[str, object]:
        calls.append(member.candidate_index)
        if member.candidate_index == 1:
            raise RuntimeError("ranking policy failed")
        return _descriptor_values(member)

    with pytest.raises(RuntimeError, match="ranking policy failed"):
        rank_evaluated_candidates(
            batch,
            config=_lexicographic_config(),
            value_factory=failing,
        )

    assert calls == [0, 1]


@pytest.mark.parametrize(
    ("criteria", "exception", "message"),
    [
        ((), ValueError, "must not be empty"),
        (
            (
                CandidateRankingCriterion(
                    "duplicate",
                    ObjectiveDirection.MAXIMIZE,
                    CandidateRankingValueKind.REAL,
                ),
                CandidateRankingCriterion(
                    "duplicate",
                    ObjectiveDirection.MINIMIZE,
                    CandidateRankingValueKind.REAL,
                ),
            ),
            ValueError,
            "names must be unique",
        ),
        ((object(),), TypeError, "CandidateRankingCriterion"),
    ],
)
def test_ranking_config_requires_unique_typed_criteria(
    criteria: object,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        CandidateRankingConfig(criteria)  # type: ignore[arg-type]


def test_ranking_api_rejects_wrong_top_level_contracts() -> None:
    batch = _valid_batch(((0.5, 2),))
    config = _lexicographic_config()

    with pytest.raises(TypeError, match="BatchEvaluationResult"):
        rank_evaluated_candidates(  # type: ignore[arg-type]
            object(), config=config, value_factory=_descriptor_values
        )
    with pytest.raises(TypeError, match="CandidateRankingConfig"):
        rank_evaluated_candidates(  # type: ignore[arg-type]
            batch, config=object(), value_factory=_descriptor_values
        )
    with pytest.raises(TypeError, match="value_factory must be callable"):
        rank_evaluated_candidates(  # type: ignore[arg-type]
            batch, config=config, value_factory=None
        )


def test_ranking_adds_no_selection_persistence_or_source_mutation() -> None:
    batch = _valid_batch(((0.5, 2), (0.6, 3)))
    original_members = batch.members

    result = rank_evaluated_candidates(
        batch,
        config=_lexicographic_config(),
        value_factory=_descriptor_values,
    )

    assert batch.members == original_members
    assert not hasattr(result, "top_candidates")
    assert not hasattr(result, "selection")
    assert not hasattr(result, "save")
