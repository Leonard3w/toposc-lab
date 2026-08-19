from __future__ import annotations

import pytest

from toposc_lab.gases.ideal_quantum_gases import (
    BOLTZMANN_CONSTANT,
    IdealQuantumGasParameters,
    analyze_ideal_quantum_gas,
    momentum_state_slice,
)


def test_classical_gas_recovers_equipartition_at_fixed_particle_number() -> None:
    parameters = IdealQuantumGasParameters(temperature=100.0e-9)
    result = analyze_ideal_quantum_gas(parameters, "classical")

    assert result.thermal_particle_number == pytest.approx(parameters.particle_number)
    assert result.condensate_particle_number == 0.0
    assert result.mean_energy_per_particle == pytest.approx(
        1.5 * BOLTZMANN_CONSTANT * parameters.temperature
    )


def test_bose_gas_condenses_below_its_three_dimensional_critical_temperature() -> None:
    parameters = IdealQuantumGasParameters(temperature=50.0e-9)
    result = analyze_ideal_quantum_gas(parameters, "boson")
    _, _, _, occupations = momentum_state_slice(result, maximum_mode_index=6)

    assert result.critical_temperature is not None
    assert result.parameters.temperature < result.critical_temperature
    assert result.condensate_fraction > 0.0
    assert result.thermal_particle_number + result.condensate_particle_number == pytest.approx(
        result.parameters.particle_number
    )
    assert occupations[6, 6] == pytest.approx(result.condensate_particle_number)


def test_warm_bose_gas_has_no_condensate() -> None:
    cold = analyze_ideal_quantum_gas(
        IdealQuantumGasParameters(temperature=50.0e-9),
        "boson",
    )
    warm = analyze_ideal_quantum_gas(
        IdealQuantumGasParameters(temperature=2.0 * cold.critical_temperature),
        "boson",
    )

    assert warm.condensate_fraction == 0.0
    assert warm.thermal_particle_number == pytest.approx(warm.parameters.particle_number)
