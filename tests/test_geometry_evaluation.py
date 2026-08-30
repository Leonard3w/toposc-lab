from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from toposc_lab.evaluation import GeometryEvaluation
from toposc_lab.observables.localization import LocalizationProfile
from toposc_lab.observables.majorana import MajoranaDiagnostics
from toposc_lab.topology import (
    NumericalConfidence,
    TopologyMethod,
    TopologyResult,
)


def _localization_profile() -> LocalizationProfile:
    return LocalizationProfile(
        probability=np.array([0.8, 0.2]),
        component_probabilities=np.array([[0.8], [0.2]]),
        center_of_mass=np.array([0.2]),
        inverse_participation_ratio=0.68,
        participation_ratio=1.0 / 0.68,
        edge_weight=0.8,
        bulk_weight=0.2,
        is_edge_localized=True,
        component_labels=("orbital",),
    )


def _majorana_diagnostics() -> MajoranaDiagnostics:
    return MajoranaDiagnostics(
        site_probability=np.array([0.8, 0.2]),
        particle_probability=np.array([0.4, 0.1]),
        hole_probability=np.array([0.4, 0.1]),
        polarization=np.array([0.8 + 0.0j, 0.2 + 0.0j]),
        polarization_magnitude=np.array([0.8, 0.2]),
        total_polarization=1.0 + 0.0j,
        self_conjugacy=1.0,
        polarization_norm=1.0,
        particle_weight=0.5,
        hole_weight=0.5,
    )


def _topology_result() -> TopologyResult:
    return TopologyResult(
        invariant_value=-1,
        is_topological=True,
        invariant_group="Z2",
        method=TopologyMethod.PFAFFIAN_1D,
        applicability_assumptions=("The Hamiltonian is in symmetry class D.",),
        confidence=NumericalConfidence(
            is_resolved=True,
            is_quantized=True,
            minimum_gap=0.4,
            gap_kind="endpoint_energy_gap",
            quantization_error=0.0,
            maximum_residual=1.0e-14,
            convergence_checked=False,
        ),
        warnings=("Finite-size convergence has not been checked.",),
    )


def test_empty_evaluation_represents_not_yet_computed_quantities() -> None:
    result = GeometryEvaluation()

    assert result.gap is None
    assert result.low_energy_states == {}
    assert result.zero_mode_count is None
    assert result.ipr == {}
    assert result.localization == {}
    assert result.majorana_metrics == {}
    assert result.topology == ()
    assert result.geometry_descriptors == {}
    assert result.warnings == ()


def test_evaluation_aggregates_existing_scientific_result_types() -> None:
    localization = _localization_profile()
    majorana = _majorana_diagnostics()
    topology = _topology_result()

    result = GeometryEvaluation(
        gap=0.4,
        low_energy_states={2: -0.2, 3: 0.2},
        zero_mode_count=0,
        ipr={2: 0.68, 3: 0.68},
        localization={2: localization},
        majorana_metrics={2: majorana},
        topology=(topology,),
        geometry_descriptors={"site_count": 2, "is_connected": True},
        warnings=("The gap convention must be supplied by the evaluator.",),
    )

    assert result.gap == pytest.approx(0.4)
    assert result.low_energy_states == {2: -0.2, 3: 0.2}
    assert result.localization[2] is localization
    assert result.majorana_metrics[2] is majorana
    assert result.topology == (topology,)
    assert result.geometry_descriptors["site_count"] == 2


def test_input_mappings_are_defensively_copied_and_read_only() -> None:
    low_energy_states = {0: 0.0}
    descriptors = {"site_count": 1}
    result = GeometryEvaluation(
        low_energy_states=low_energy_states,
        zero_mode_count=1,
        geometry_descriptors=descriptors,
    )
    low_energy_states[0] = 1.0
    descriptors["site_count"] = 2

    assert result.low_energy_states[0] == 0.0
    assert result.geometry_descriptors["site_count"] == 1
    with pytest.raises(TypeError):
        result.low_energy_states[0] = 1.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.gap = 0.2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"gap": -0.1}, ValueError, "gap must be non-negative"),
        (
            {"low_energy_states": {0: np.inf}},
            ValueError,
            "low_energy_states\\[0\\] must be finite",
        ),
        (
            {"low_energy_states": {0: 0.0}, "zero_mode_count": 2},
            ValueError,
            "must not exceed",
        ),
        (
            {"low_energy_states": {0: 0.0}, "ipr": {1: 0.5}},
            ValueError,
            "states absent",
        ),
        (
            {"geometry_descriptors": {"mean degree": 2.0}},
            ValueError,
            "Python-style identifiers",
        ),
        ({"warnings": ("",)}, ValueError, "empty messages"),
    ],
)
def test_evaluation_rejects_inconsistent_or_nonfinite_data(
    kwargs: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        GeometryEvaluation(**kwargs)  # type: ignore[arg-type]


def test_evaluation_rejects_wrong_specialized_result_types() -> None:
    with pytest.raises(TypeError, match="LocalizationProfile"):
        GeometryEvaluation(
            low_energy_states={0: 0.0},
            localization={0: object()},  # type: ignore[dict-item]
        )

    with pytest.raises(TypeError, match="TopologyResult"):
        GeometryEvaluation(topology=(object(),))  # type: ignore[arg-type]
