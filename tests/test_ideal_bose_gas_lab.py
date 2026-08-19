from __future__ import annotations

from toposc_lab.bosons.ideal_bose_gas import (
    IdealBoseEinsteinCondensationParameters,
    IdealBoseGasParameters,
)
from toposc_lab.visualization.ideal_bose_gas_lab import (
    BOSONIC_EXCESS_PERCENT_RANGE,
    STATE_FRACTION_LOG_RANGE,
    bose_statistics_learning_figure,
    ideal_bec_learning_figure,
    ideal_bose_gas_live_material,
)


def test_learning_material_contains_the_bose_classical_comparison() -> None:
    comparison, figure = bose_statistics_learning_figure(
        IdealBoseGasParameters(maximum_mode_index=3)
    )

    assert comparison.expected_particle_number > 0.0
    assert len(figure.data) == 5
    assert not figure.frames
    assert comparison.ground_state_enhancement > 1.0
    assert figure.data[0].zmin == STATE_FRACTION_LOG_RANGE[0]
    assert figure.data[0].zmax == STATE_FRACTION_LOG_RANGE[1]
    assert figure.data[2].zmin == BOSONIC_EXCESS_PERCENT_RANGE[0]
    assert figure.data[2].zmax == BOSONIC_EXCESS_PERCENT_RANGE[1]


def test_legacy_lab_entry_point_returns_the_learning_material() -> None:
    comparison, figure = ideal_bose_gas_live_material(
        IdealBoseGasParameters(maximum_mode_index=3)
    )

    assert comparison.expected_particle_number > 0.0
    assert len(figure.data) == 5


def test_bec_learning_figure_marks_the_condensate_and_thermal_cloud() -> None:
    result, figure = ideal_bec_learning_figure(
        IdealBoseEinsteinCondensationParameters(temperature=50.0e-9)
    )

    assert result.condensate_fraction > 0.0
    assert len(figure.data) == 4
    assert figure.layout.barmode == "stack"
