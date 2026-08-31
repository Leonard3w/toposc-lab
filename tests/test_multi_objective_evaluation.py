from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from toposc_lab.evaluation import (
    GeometryEvaluation,
    MultiObjectiveEvaluation,
    ObjectiveDirection,
    ObjectiveQuantity,
    ObjectiveSpec,
    ObjectiveValue,
    evaluate_multi_objectives,
)
from toposc_lab.observables.localization import LocalizationProfile
from toposc_lab.observables.majorana import MajoranaDiagnostics
from toposc_lab.topology import NumericalConfidence, TopologyMethod, TopologyResult


def _localization_profile(edge_weight: float = 0.8) -> LocalizationProfile:
    return LocalizationProfile(
        probability=np.array([0.8, 0.2]),
        component_probabilities=np.array([[0.8], [0.2]]),
        center_of_mass=np.array([0.2]),
        inverse_participation_ratio=0.68,
        participation_ratio=1.0 / 0.68,
        edge_weight=edge_weight,
        bulk_weight=1.0 - edge_weight,
        is_edge_localized=True,
        component_labels=("orbital",),
    )


def _majorana_diagnostics(self_conjugacy: float = 0.9) -> MajoranaDiagnostics:
    return MajoranaDiagnostics(
        site_probability=np.array([0.8, 0.2]),
        particle_probability=np.array([0.4, 0.1]),
        hole_probability=np.array([0.4, 0.1]),
        polarization=np.array([self_conjugacy + 0.0j, 0.0 + 0.0j]),
        polarization_magnitude=np.array([self_conjugacy, 0.0]),
        total_polarization=self_conjugacy + 0.0j,
        self_conjugacy=self_conjugacy,
        polarization_norm=self_conjugacy,
        particle_weight=0.5,
        hole_weight=0.5,
    )


def _topology_result(is_topological: bool | None = True) -> TopologyResult:
    is_resolved = is_topological is not None
    return TopologyResult(
        invariant_value=(-1 if is_topological else 1) if is_resolved else None,
        is_topological=is_topological,
        invariant_group="Z2",
        method=TopologyMethod.PFAFFIAN_1D,
        applicability_assumptions=("The class-D Pfaffian assumptions hold.",),
        confidence=NumericalConfidence(
            is_resolved=is_resolved,
            is_quantized=True if is_resolved else None,
            minimum_gap=0.4,
            gap_kind="endpoint_energy_gap",
            quantization_error=0.0 if is_resolved else None,
            maximum_residual=1.0e-14,
            convergence_checked=True,
        ),
        warnings=(),
    )


def _complete_evaluation() -> GeometryEvaluation:
    return GeometryEvaluation(
        gap=0.4,
        low_energy_states={2: 0.0},
        zero_mode_count=1,
        ipr={2: 0.68},
        localization={2: _localization_profile()},
        majorana_metrics={2: _majorana_diagnostics()},
        topology=(_topology_result(),),
        geometry_descriptors={
            "site_count": 2,
            "is_connected": True,
            "mean_degree": 1.0,
        },
        warnings=("The spectral convention is finite-system specific.",),
    )


def _all_specs() -> tuple[ObjectiveSpec, ...]:
    maximize = ObjectiveDirection.MAXIMIZE
    return (
        ObjectiveSpec("spectral_gap", ObjectiveQuantity.GAP, maximize),
        ObjectiveSpec("zero_modes", ObjectiveQuantity.ZERO_MODE_COUNT, maximize),
        ObjectiveSpec(
            "state_2_ipr",
            ObjectiveQuantity.STATE_IPR,
            maximize,
            state_index=2,
        ),
        ObjectiveSpec(
            "state_2_boundary_weight",
            ObjectiveQuantity.STATE_BOUNDARY_WEIGHT,
            maximize,
            state_index=2,
        ),
        ObjectiveSpec(
            "state_2_majorana",
            ObjectiveQuantity.STATE_MAJORANA_SELF_CONJUGACY,
            maximize,
            state_index=2,
        ),
        ObjectiveSpec(
            "pfaffian_topological",
            ObjectiveQuantity.TOPOLOGY_CLASSIFICATION,
            maximize,
            topology_method=TopologyMethod.PFAFFIAN_1D,
        ),
        ObjectiveSpec(
            "small_site_count",
            ObjectiveQuantity.GEOMETRY_DESCRIPTOR,
            ObjectiveDirection.MINIMIZE,
            descriptor_name="site_count",
        ),
        ObjectiveSpec(
            "connected",
            ObjectiveQuantity.GEOMETRY_DESCRIPTOR,
            maximize,
            descriptor_name="is_connected",
        ),
    )


def test_multi_objective_result_preserves_separate_raw_values() -> None:
    result = evaluate_multi_objectives(
        _complete_evaluation(),
        objectives=_all_specs(),
    )

    assert result.objectives["spectral_gap"].value == pytest.approx(0.4)
    assert result.objectives["zero_modes"].value == 1
    assert result.objectives["state_2_ipr"].value == pytest.approx(0.68)
    assert result.objectives["state_2_boundary_weight"].value == pytest.approx(0.8)
    assert result.objectives["state_2_majorana"].value == pytest.approx(0.9)
    assert result.objectives["pfaffian_topological"].value is True
    assert result.objectives["small_site_count"].value == 2
    assert result.objectives["connected"].value is True
    assert not hasattr(result, "score")
    assert not hasattr(result, "rank")


def test_direction_is_retained_without_transforming_value() -> None:
    result = evaluate_multi_objectives(
        _complete_evaluation(),
        objectives=(
            ObjectiveSpec(
                "small_gap",
                ObjectiveQuantity.GAP,
                ObjectiveDirection.MINIMIZE,
            ),
        ),
    )

    objective = result.objectives["small_gap"]
    assert objective.value == pytest.approx(0.4)
    assert objective.spec.direction is ObjectiveDirection.MINIMIZE


def test_result_is_immutable_and_does_not_mutate_source_evaluation() -> None:
    evaluation = _complete_evaluation()
    original_descriptors = dict(evaluation.geometry_descriptors)
    result = evaluate_multi_objectives(evaluation, objectives=_all_specs())

    assert evaluation.geometry_descriptors == original_descriptors
    with pytest.raises(TypeError):
        result.objectives["new"] = result.objectives["spectral_gap"]  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.warnings = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        (
            {"name": "not valid", "quantity": ObjectiveQuantity.GAP},
            ValueError,
            "Python-style identifier",
        ),
        (
            {"name": "gap", "quantity": "gap"},
            TypeError,
            "ObjectiveQuantity",
        ),
        (
            {"name": "gap", "quantity": ObjectiveQuantity.GAP, "state_index": 0},
            ValueError,
            "does not accept a selector",
        ),
        (
            {"name": "ipr", "quantity": ObjectiveQuantity.STATE_IPR},
            ValueError,
            "requires state_index",
        ),
        (
            {
                "name": "ipr",
                "quantity": ObjectiveQuantity.STATE_IPR,
                "state_index": -1,
            },
            ValueError,
            "non-negative",
        ),
        (
            {"name": "topology", "quantity": ObjectiveQuantity.TOPOLOGY_CLASSIFICATION},
            ValueError,
            "requires topology_method",
        ),
        (
            {"name": "descriptor", "quantity": ObjectiveQuantity.GEOMETRY_DESCRIPTOR},
            ValueError,
            "requires descriptor_name",
        ),
    ],
)
def test_objective_spec_rejects_ambiguous_selectors(
    kwargs: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    kwargs.setdefault("direction", ObjectiveDirection.MAXIMIZE)
    with pytest.raises(exception, match=message):
        ObjectiveSpec(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("evaluation", "spec", "message"),
    [
        (
            GeometryEvaluation(),
            ObjectiveSpec("gap", ObjectiveQuantity.GAP, ObjectiveDirection.MAXIMIZE),
            "evaluation.gap",
        ),
        (
            GeometryEvaluation(),
            ObjectiveSpec(
                "zeros",
                ObjectiveQuantity.ZERO_MODE_COUNT,
                ObjectiveDirection.MAXIMIZE,
            ),
            "zero_mode_count",
        ),
        (
            GeometryEvaluation(),
            ObjectiveSpec(
                "ipr",
                ObjectiveQuantity.STATE_IPR,
                ObjectiveDirection.MAXIMIZE,
                state_index=0,
            ),
            "IPR for state 0",
        ),
        (
            GeometryEvaluation(),
            ObjectiveSpec(
                "boundary",
                ObjectiveQuantity.STATE_BOUNDARY_WEIGHT,
                ObjectiveDirection.MAXIMIZE,
                state_index=0,
            ),
            "localization for state 0",
        ),
        (
            GeometryEvaluation(),
            ObjectiveSpec(
                "majorana",
                ObjectiveQuantity.STATE_MAJORANA_SELF_CONJUGACY,
                ObjectiveDirection.MAXIMIZE,
                state_index=0,
            ),
            "Majorana diagnostics for state 0",
        ),
        (
            GeometryEvaluation(),
            ObjectiveSpec(
                "topology",
                ObjectiveQuantity.TOPOLOGY_CLASSIFICATION,
                ObjectiveDirection.MAXIMIZE,
                topology_method=TopologyMethod.PFAFFIAN_1D,
            ),
            "requires topology method pfaffian_1d",
        ),
        (
            GeometryEvaluation(),
            ObjectiveSpec(
                "sites",
                ObjectiveQuantity.GEOMETRY_DESCRIPTOR,
                ObjectiveDirection.MINIMIZE,
                descriptor_name="site_count",
            ),
            "geometry descriptor 'site_count'",
        ),
        (
            GeometryEvaluation(geometry_descriptors={"diameter": None}),
            ObjectiveSpec(
                "diameter",
                ObjectiveQuantity.GEOMETRY_DESCRIPTOR,
                ObjectiveDirection.MINIMIZE,
                descriptor_name="diameter",
            ),
            "defined geometry descriptor",
        ),
    ],
)
def test_missing_requested_quantity_is_not_silently_dropped(
    evaluation: GeometryEvaluation,
    spec: ObjectiveSpec,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_multi_objectives(evaluation, objectives=(spec,))


def test_unresolved_topology_classification_is_rejected() -> None:
    evaluation = GeometryEvaluation(topology=(_topology_result(None),))
    spec = ObjectiveSpec(
        "topology",
        ObjectiveQuantity.TOPOLOGY_CLASSIFICATION,
        ObjectiveDirection.MAXIMIZE,
        topology_method=TopologyMethod.PFAFFIAN_1D,
    )

    with pytest.raises(ValueError, match="resolved topology classification"):
        evaluate_multi_objectives(evaluation, objectives=(spec,))


def test_duplicate_objective_names_are_rejected() -> None:
    specs = (
        ObjectiveSpec("duplicate", ObjectiveQuantity.GAP, ObjectiveDirection.MAXIMIZE),
        ObjectiveSpec(
            "duplicate",
            ObjectiveQuantity.ZERO_MODE_COUNT,
            ObjectiveDirection.MINIMIZE,
        ),
    )

    with pytest.raises(ValueError, match="names must be unique"):
        evaluate_multi_objectives(_complete_evaluation(), objectives=specs)


@pytest.mark.parametrize("objectives", [(), "gap", (object(),)])
def test_objective_collection_must_be_nonempty_and_typed(objectives: object) -> None:
    with pytest.raises((TypeError, ValueError), match="objectives"):
        evaluate_multi_objectives(
            _complete_evaluation(),
            objectives=objectives,  # type: ignore[arg-type]
        )


def test_state_probability_like_objective_must_be_normalized() -> None:
    evaluation = GeometryEvaluation(
        low_energy_states={0: 0.0},
        localization={0: _localization_profile(edge_weight=1.1)},
    )
    spec = ObjectiveSpec(
        "boundary",
        ObjectiveQuantity.STATE_BOUNDARY_WEIGHT,
        ObjectiveDirection.MAXIMIZE,
        state_index=0,
    )

    with pytest.raises(ValueError, match="between zero and one"):
        evaluate_multi_objectives(evaluation, objectives=(spec,))


def test_source_and_quantity_specific_warnings_are_preserved() -> None:
    result = evaluate_multi_objectives(
        _complete_evaluation(),
        objectives=_all_specs(),
    )

    joined = " ".join(result.warnings)
    assert "finite-system specific" in joined
    assert "neither a scalar score nor a candidate ranking" in joined
    assert "caller-selected preferences" in joined
    assert "finite-system convention" in joined
    assert "does not establish Majorana" in joined
    assert "degenerate subspaces" in joined
    assert "method's assumptions" in joined
    assert "structural quantities" in joined


def test_result_container_validates_key_and_spec_name_agreement() -> None:
    objective = ObjectiveValue(
        spec=ObjectiveSpec(
            "gap",
            ObjectiveQuantity.GAP,
            ObjectiveDirection.MAXIMIZE,
        ),
        value=0.4,
    )

    with pytest.raises(ValueError, match="specification name"):
        MultiObjectiveEvaluation(
            objectives={"other": objective},
            warnings=("Engineering result.",),
        )
