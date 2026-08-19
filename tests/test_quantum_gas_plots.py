from __future__ import annotations

from toposc_lab.gases.ideal_quantum_gases import (
    IdealQuantumGasParameters,
    analyze_ideal_quantum_gas,
)
from toposc_lab.visualization.quantum_gas_plots import (
    MODE_OCCUPATION_LOG_RANGE,
    bose_condensation_figure,
    quantum_gas_state_figure,
    quantum_statistics_schematic,
)


def test_state_figure_has_fixed_mode_occupation_scale() -> None:
    result = analyze_ideal_quantum_gas(IdealQuantumGasParameters(), "classical")
    figure = quantum_gas_state_figure(
        result,
        maximum_mode_index=6,
        scale_mode="fixed",
    )

    assert len(figure.data) == 1 + 1
    assert figure.data[0].zmin == MODE_OCCUPATION_LOG_RANGE[0]
    assert figure.data[0].zmax == MODE_OCCUPATION_LOG_RANGE[1]
    assert figure.layout.yaxis2.range == MODE_OCCUPATION_LOG_RANGE


def test_bose_condensation_figure_contains_phase_curve_and_population_bars() -> None:
    result = analyze_ideal_quantum_gas(
        IdealQuantumGasParameters(temperature=50.0e-9),
        "boson",
    )
    figure = bose_condensation_figure(result)

    assert len(figure.data) == 4
    assert figure.layout.barmode == "stack"


def test_statistics_schematic_contains_classical_and_bose_counting_rules() -> None:
    figure = quantum_statistics_schematic()

    assert len(figure.data) == 7
    assert len(figure.layout.annotations) == 3
