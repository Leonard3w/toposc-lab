from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
from toposc_lab.geometry import Geometry, GeometryEdge, chain
from toposc_lab.search import (
    BatchEvaluationCandidate,
    BatchEvaluationMember,
    BatchEvaluationRequest,
    BatchEvaluationResult,
    CandidateRankingConfig,
    CandidateRankingCriterion,
    CandidateRankingResult,
    CandidateRankingValueKind,
    rank_evaluated_candidates,
)
from toposc_lab.visualization import (
    TopCandidateVisualization,
    plot_top_ranked_candidates,
)


def _geometry(index: int) -> Geometry:
    if index == 0:
        return Geometry(
            n_sites=3,
            edges=(GeometryEdge(2, 1), GeometryEdge(1, 0)),
        )
    if index == 1:
        return Geometry(
            n_sites=3,
            edges=(GeometryEdge(0, 1), GeometryEdge(1, 2)),
            coordinates=np.asarray(((0.0, 0.0, 1.0), (1.0, 0.0, 2.0), (1.0, 1.0, 3.0))),
        )
    return chain(3)


def _candidate(index: int) -> BatchEvaluationCandidate:
    return BatchEvaluationCandidate(
        geometry=_geometry(index),
        model_parameters={"candidate_label": index},
        evaluation_seed=500 + index,
    )


def _valid_run(
    candidate: BatchEvaluationCandidate,
    *,
    quality: float,
) -> GeometryEvaluationRun:
    simulation = SimulationResult(
        model_name="candidate_plot_test",
        eigenvalues=np.asarray((-quality, 0.0, quality)),
        eigenvectors=np.eye(3, dtype=complex),
        basis_layout=BasisLayout(spatial_shape=(3,)),
        parameters=candidate.model_parameters,
    )
    evaluation = GeometryEvaluation(
        gap=quality,
        low_energy_states={0: -quality, 1: 0.0, 2: quality},
        zero_mode_count=1,
        geometry_descriptors={"quality": quality},
    )
    reproducibility = create_reproducibility_record(
        candidate.geometry,
        seed=candidate.evaluation_seed,
        model_name="candidate_plot_test",
        model_parameters=candidate.model_parameters,
        solver_name="exact_diagonalization",
        solver_settings={},
        evaluation_settings={},
        code_version="phase-9.6-test",
    )
    return GeometryEvaluationRun(
        simulation_result=simulation,
        evaluation=evaluation,
        validity=CandidateValidityReport(()),
        reproducibility=reproducibility,
    )


def _ranking() -> CandidateRankingResult:
    qualities = (0.9, 0.9, 0.8, 0.7)
    candidates = tuple(_candidate(index) for index in range(len(qualities)))
    batch = BatchEvaluationResult(
        request=BatchEvaluationRequest(candidates),
        members=tuple(
            BatchEvaluationMember(
                index,
                candidate,
                _valid_run(candidate, quality=quality),
                None,
            )
            for index, (candidate, quality) in enumerate(zip(candidates, qualities, strict=True))
        ),
    )
    config = CandidateRankingConfig(
        (
            CandidateRankingCriterion(
                "quality",
                ObjectiveDirection.MAXIMIZE,
                CandidateRankingValueKind.REAL,
            ),
        )
    )

    def values(member: BatchEvaluationMember) -> dict[str, object]:
        assert member.evaluation is not None
        assert member.evaluation.evaluation is not None
        return {"quality": member.evaluation.evaluation.geometry_descriptors["quality"]}

    return rank_evaluated_candidates(batch, config=config, value_factory=values)


def _all_unranked() -> CandidateRankingResult:
    candidate = _candidate(0)
    invalid = GeometryEvaluationRun(
        simulation_result=None,
        evaluation=None,
        validity=CandidateValidityReport(
            (
                CandidateValidityIssue(
                    code="disconnected_components",
                    severity=CandidateIssueSeverity.ERROR,
                    category=CandidateIssueCategory.GEOMETRY,
                    message="synthetic invalid candidate",
                ),
            )
        ),
    )
    batch = BatchEvaluationResult(
        request=BatchEvaluationRequest((candidate,)),
        members=(BatchEvaluationMember(0, candidate, invalid, None),),
    )
    config = CandidateRankingConfig(
        (
            CandidateRankingCriterion(
                "quality",
                ObjectiveDirection.MAXIMIZE,
                CandidateRankingValueKind.REAL,
            ),
        )
    )
    return rank_evaluated_candidates(
        batch,
        config=config,
        value_factory=lambda member: {"quality": 0.0},
    )


def test_visualization_includes_complete_top_rank_tie() -> None:
    ranking = _ranking()

    result = plot_top_ranked_candidates(
        ranking,
        maximum_rank=1,
        columns=3,
        show=False,
    )

    assert isinstance(result, TopCandidateVisualization)
    assert tuple(entry.candidate_index for entry in result.entries) == (0, 1)
    assert tuple(entry.rank for entry in result.entries) == (1, 1)
    assert result.columns == 2
    assert len(result.axes) == 2
    assert "Rank 1 · candidate 0" in result.axes[0].get_title()
    assert "quality=0.9" in result.axes[0].get_title()
    assert "schematic circular layout; no coordinates" in result.axes[0].get_title()
    assert "x-y projection" in result.axes[1].get_title()
    assert "engineering visualization" in result.figure.texts[0].get_text()
    plt.close(result.figure)


def test_rank_cutoff_includes_every_entry_at_or_above_it() -> None:
    ranking = _ranking()

    result = plot_top_ranked_candidates(
        ranking,
        maximum_rank=3,
        columns=2,
        show=False,
    )

    assert tuple(entry.candidate_index for entry in result.entries) == (0, 1, 2)
    assert tuple(entry.rank for entry in result.entries) == (1, 1, 3)
    assert result.maximum_rank == 3
    assert result.columns == 2
    assert len(result.figure.axes) == 4
    assert not result.figure.axes[-1].axison
    plt.close(result.figure)


def test_default_view_displays_stored_edge_orientation_without_mutating_geometry() -> None:
    ranking = _ranking()
    source_geometry = ranking.source.members[0].candidate.geometry
    original_edges = source_geometry.edges

    result = plot_top_ranked_candidates(
        ranking,
        maximum_rank=1,
        show=False,
    )

    arrow_annotations = [
        text for text in result.axes[0].texts if getattr(text, "arrow_patch", None) is not None
    ]
    assert len(arrow_annotations) == len(source_geometry.edges)
    assert source_geometry.edges == original_edges
    assert source_geometry.coordinates is None
    plt.close(result.figure)


def test_show_flag_is_forwarded_only_after_complete_figure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(plt, "show", lambda: calls.append("show"))

    result = plot_top_ranked_candidates(
        _ranking(),
        maximum_rank=1,
        show=True,
    )

    assert calls == ["show"]
    plt.close(result.figure)


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"maximum_rank": 0}, ValueError, "maximum_rank must be positive"),
        ({"maximum_rank": True}, TypeError, "maximum_rank must be an integer"),
        ({"maximum_rank": 1, "columns": 0}, ValueError, "columns must be positive"),
        ({"maximum_rank": 1, "columns": 1.5}, TypeError, "columns must be an integer"),
    ],
)
def test_visualization_rejects_invalid_layout_contracts(
    kwargs: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        plot_top_ranked_candidates(_ranking(), show=False, **kwargs)  # type: ignore[arg-type]


def test_visualization_rejects_wrong_ranking_and_empty_ranked_view() -> None:
    with pytest.raises(TypeError, match="CandidateRankingResult"):
        plot_top_ranked_candidates(object(), maximum_rank=1, show=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no candidates"):
        plot_top_ranked_candidates(_all_unranked(), maximum_rank=1, show=False)


def test_plot_failure_closes_partial_figure() -> None:
    existing = set(plt.get_fignums())

    with pytest.raises(ValueError, match="site_size"):
        plot_top_ranked_candidates(_ranking(), maximum_rank=1, site_size=0.0, show=False)

    assert set(plt.get_fignums()) == existing


def test_visualization_writes_no_files_or_scientific_labels(tmp_path: Path) -> None:
    result = plot_top_ranked_candidates(_ranking(), maximum_rank=1, show=False)

    assert list(tmp_path.iterdir()) == []
    assert not hasattr(result, "save")
    joined = " ".join(result.warnings)
    assert "not evidence of topology" in joined
    assert "not a scientific eligibility" in joined
    plt.close(result.figure)
