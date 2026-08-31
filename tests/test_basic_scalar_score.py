from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from toposc_lab.evaluation import (
    BasicScalarScore,
    BasicScoreComponent,
    GeometryEvaluation,
    compute_basic_scalar_score,
)
from toposc_lab.observables.localization import LocalizationProfile
from toposc_lab.observables.majorana import MajoranaDiagnostics
from toposc_lab.topology import NumericalConfidence, TopologyMethod, TopologyResult


def _localization_profile(edge_weight: float) -> LocalizationProfile:
    return LocalizationProfile(
        probability=np.array([edge_weight, 1.0 - edge_weight]),
        component_probabilities=np.array([[edge_weight], [1.0 - edge_weight]]),
        center_of_mass=np.array([1.0 - edge_weight]),
        inverse_participation_ratio=0.8,
        participation_ratio=1.25,
        edge_weight=edge_weight,
        bulk_weight=1.0 - edge_weight,
        is_edge_localized=edge_weight >= 0.5,
        component_labels=("orbital",),
    )


def _majorana_diagnostics(self_conjugacy: float) -> MajoranaDiagnostics:
    return MajoranaDiagnostics(
        site_probability=np.array([0.5, 0.5]),
        particle_probability=np.array([0.25, 0.25]),
        hole_probability=np.array([0.25, 0.25]),
        polarization=np.array([self_conjugacy + 0.0j, 0.0 + 0.0j]),
        polarization_magnitude=np.array([self_conjugacy, 0.0]),
        total_polarization=self_conjugacy + 0.0j,
        self_conjugacy=self_conjugacy,
        polarization_norm=self_conjugacy,
        particle_weight=0.5,
        hole_weight=0.5,
    )


def _topology_result(
    *,
    method: TopologyMethod,
    is_topological: bool | None,
) -> TopologyResult:
    is_resolved = is_topological is not None
    return TopologyResult(
        invariant_value=(1 if is_topological else 0) if is_resolved else None,
        is_topological=is_topological,
        invariant_group="Z",
        method=method,
        applicability_assumptions=("The method-specific assumptions hold.",),
        confidence=NumericalConfidence(
            is_resolved=is_resolved,
            is_quantized=True if is_resolved else None,
            minimum_gap=0.4,
            gap_kind="finite_energy_gap",
            quantization_error=0.0 if is_resolved else None,
            maximum_residual=1.0e-14,
            convergence_checked=True,
        ),
        warnings=(),
    )


def _complete_evaluation() -> GeometryEvaluation:
    return GeometryEvaluation(
        gap=1.0,
        low_energy_states={0: 0.0, 1: 0.1},
        zero_mode_count=1,
        ipr={0: 0.4, 1: 0.8},
        localization={0: _localization_profile(0.25), 1: _localization_profile(0.75)},
        majorana_metrics={
            0: _majorana_diagnostics(0.3),
            1: _majorana_diagnostics(0.9),
        },
        topology=(
            _topology_result(
                method=TopologyMethod.PFAFFIAN_1D,
                is_topological=True,
            ),
            _topology_result(
                method=TopologyMethod.REAL_SPACE_WINDING_1D,
                is_topological=False,
            ),
        ),
    )


def test_score_exposes_every_normalized_component_and_contribution() -> None:
    weights = {component: 1.0 for component in BasicScoreComponent}

    score = compute_basic_scalar_score(
        _complete_evaluation(),
        weights=weights,
        gap_scale=1.0,
    )

    expected_components = {
        BasicScoreComponent.NORMALIZED_GAP: 0.5,
        BasicScoreComponent.ZERO_MODE_PRESENCE: 1.0,
        BasicScoreComponent.MAXIMUM_IPR: 0.8,
        BasicScoreComponent.MAXIMUM_BOUNDARY_WEIGHT: 0.75,
        BasicScoreComponent.MAXIMUM_MAJORANA_SELF_CONJUGACY: 0.9,
        BasicScoreComponent.TOPOLOGICAL_METHOD_FRACTION: 0.5,
    }
    assert score.components == expected_components
    assert all(
        weight == pytest.approx(1.0 / 6.0)
        for weight in score.normalized_weights.values()
    )
    assert score.value == pytest.approx(sum(expected_components.values()) / 6.0)
    assert sum(score.contributions.values()) == pytest.approx(score.value)
    for component, value in expected_components.items():
        assert score.contributions[component] == pytest.approx(value / 6.0)


def test_selected_subset_uses_explicit_weights_and_gap_scale() -> None:
    score = compute_basic_scalar_score(
        _complete_evaluation(),
        weights={
            BasicScoreComponent.NORMALIZED_GAP: 3.0,
            BasicScoreComponent.ZERO_MODE_PRESENCE: 1.0,
        },
        gap_scale=3.0,
    )

    assert score.components == {
        BasicScoreComponent.NORMALIZED_GAP: 0.25,
        BasicScoreComponent.ZERO_MODE_PRESENCE: 1.0,
    }
    assert score.normalized_weights == {
        BasicScoreComponent.NORMALIZED_GAP: 0.75,
        BasicScoreComponent.ZERO_MODE_PRESENCE: 0.25,
    }
    assert score.value == pytest.approx(0.4375)


def test_score_is_standalone_and_does_not_mutate_evaluation() -> None:
    evaluation = _complete_evaluation()
    original_ipr = dict(evaluation.ipr)

    score = compute_basic_scalar_score(
        evaluation,
        weights={BasicScoreComponent.MAXIMUM_IPR: 1.0},
    )

    assert isinstance(score, BasicScalarScore)
    assert not hasattr(evaluation, "score")
    assert evaluation.ipr == original_ipr


def test_result_mappings_are_defensively_copied_and_read_only() -> None:
    weights = {BasicScoreComponent.NORMALIZED_GAP: 1.0}
    score = compute_basic_scalar_score(_complete_evaluation(), weights=weights)
    weights[BasicScoreComponent.NORMALIZED_GAP] = 5.0

    assert score.normalized_weights[BasicScoreComponent.NORMALIZED_GAP] == 1.0
    with pytest.raises(TypeError):
        score.components[BasicScoreComponent.NORMALIZED_GAP] = 0.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        score.value = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("weights", "exception", "message"),
    [
        ({}, ValueError, "must not be empty"),
        ({BasicScoreComponent.NORMALIZED_GAP: 0.0}, ValueError, "must be positive"),
        ({BasicScoreComponent.NORMALIZED_GAP: -1.0}, ValueError, "must be positive"),
        ({BasicScoreComponent.NORMALIZED_GAP: np.inf}, ValueError, "must be finite"),
        ({BasicScoreComponent.NORMALIZED_GAP: True}, TypeError, "real number"),
        ({"normalized_gap": 1.0}, TypeError, "BasicScoreComponent"),
    ],
)
def test_invalid_weights_are_rejected(
    weights: object,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        compute_basic_scalar_score(
            _complete_evaluation(),
            weights=weights,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("gap_scale", [0.0, -1.0, np.inf, np.nan])
def test_gap_scale_must_be_positive_and_finite(gap_scale: float) -> None:
    with pytest.raises(ValueError, match="gap_scale"):
        compute_basic_scalar_score(
            _complete_evaluation(),
            weights={BasicScoreComponent.NORMALIZED_GAP: 1.0},
            gap_scale=gap_scale,
        )


@pytest.mark.parametrize(
    ("evaluation", "component", "message"),
    [
        (GeometryEvaluation(), BasicScoreComponent.NORMALIZED_GAP, "evaluation.gap"),
        (
            GeometryEvaluation(),
            BasicScoreComponent.ZERO_MODE_PRESENCE,
            "zero_mode_count",
        ),
        (GeometryEvaluation(), BasicScoreComponent.MAXIMUM_IPR, "evaluation.ipr"),
        (
            GeometryEvaluation(),
            BasicScoreComponent.MAXIMUM_BOUNDARY_WEIGHT,
            "localization profile",
        ),
        (
            GeometryEvaluation(),
            BasicScoreComponent.MAXIMUM_MAJORANA_SELF_CONJUGACY,
            "Majorana diagnostic",
        ),
        (
            GeometryEvaluation(),
            BasicScoreComponent.TOPOLOGICAL_METHOD_FRACTION,
            "topology result",
        ),
    ],
)
def test_requested_missing_component_is_not_silently_dropped(
    evaluation: GeometryEvaluation,
    component: BasicScoreComponent,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_basic_scalar_score(evaluation, weights={component: 1.0})


def test_unresolved_topology_result_is_not_excluded_from_fraction() -> None:
    evaluation = GeometryEvaluation(
        topology=(
            _topology_result(
                method=TopologyMethod.PFAFFIAN_1D,
                is_topological=True,
            ),
            _topology_result(
                method=TopologyMethod.REAL_SPACE_WINDING_1D,
                is_topological=None,
            ),
        )
    )

    with pytest.raises(ValueError, match="all topology results to be resolved"):
        compute_basic_scalar_score(
            evaluation,
            weights={BasicScoreComponent.TOPOLOGICAL_METHOD_FRACTION: 1.0},
        )


@pytest.mark.parametrize(
    ("component", "evaluation", "message"),
    [
        (
            BasicScoreComponent.MAXIMUM_IPR,
            GeometryEvaluation(low_energy_states={0: 0.0}, ipr={0: 1.1}),
            "maximum_ipr",
        ),
        (
            BasicScoreComponent.MAXIMUM_BOUNDARY_WEIGHT,
            GeometryEvaluation(
                low_energy_states={0: 0.0},
                localization={0: _localization_profile(1.1)},
            ),
            "maximum_boundary_weight",
        ),
        (
            BasicScoreComponent.MAXIMUM_MAJORANA_SELF_CONJUGACY,
            GeometryEvaluation(
                low_energy_states={0: 0.0},
                majorana_metrics={0: _majorana_diagnostics(1.1)},
            ),
            "maximum_majorana_self_conjugacy",
        ),
    ],
)
def test_dimensionless_components_must_be_normalized(
    component: BasicScoreComponent,
    evaluation: GeometryEvaluation,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_basic_scalar_score(evaluation, weights={component: 1.0})


def test_maximum_summary_validates_every_underlying_value() -> None:
    evaluation = GeometryEvaluation(
        low_energy_states={0: 0.0, 1: 0.1},
        localization={
            0: _localization_profile(-0.1),
            1: _localization_profile(0.75),
        },
    )

    with pytest.raises(ValueError, match="maximum_boundary_weight input"):
        compute_basic_scalar_score(
            evaluation,
            weights={BasicScoreComponent.MAXIMUM_BOUNDARY_WEIGHT: 1.0},
        )


def test_warnings_state_engineering_and_component_specific_limits() -> None:
    score = compute_basic_scalar_score(
        _complete_evaluation(),
        weights={component: 1.0 for component in BasicScoreComponent},
    )

    joined = " ".join(score.warnings)
    assert "engineering convenience" in joined
    assert "identical components" in joined
    assert "does not establish Majorana" in joined
    assert "Maximum state summaries" in joined
    assert "physical location" in joined
    assert "does not replace the separate method results" in joined
