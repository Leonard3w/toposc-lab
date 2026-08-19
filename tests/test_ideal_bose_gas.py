from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.bosons.ideal_bose_gas import (
    BOLTZMANN_CONSTANT,
    IdealBoseEinsteinCondensationParameters,
    IdealBoseGasParameters,
    analyze_ideal_bose_gas,
    analyze_ideal_bose_einstein_condensation,
    bose_einstein_occupation,
    compare_bose_to_classical,
    maxwell_boltzmann_occupation,
    representative_particles,
)


def test_finite_box_builds_all_visible_momentum_modes() -> None:
    result = analyze_ideal_bose_gas(
        IdealBoseGasParameters(maximum_mode_index=2)
    )

    assert result.energies.shape == (5, 5)
    assert result.occupations.shape == (5, 5)
    assert result.expected_particle_number == pytest.approx(np.sum(result.occupations))
    assert result.ground_state_occupation == pytest.approx(
        1.0
        / np.expm1(
            -result.parameters.chemical_potential
            / (BOLTZMANN_CONSTANT * result.parameters.temperature)
        )
    )


def test_bose_distribution_responds_to_temperature_and_chemical_potential() -> None:
    cold = analyze_ideal_bose_gas(
        IdealBoseGasParameters(temperature=80.0e-9)
    )
    warm = analyze_ideal_bose_gas(
        IdealBoseGasParameters(temperature=200.0e-9)
    )
    low_mu = analyze_ideal_bose_gas(
        IdealBoseGasParameters(chemical_potential=-80.0e-9 * BOLTZMANN_CONSTANT)
    )
    near_zero_mu = analyze_ideal_bose_gas(
        IdealBoseGasParameters(chemical_potential=-5.0e-9 * BOLTZMANN_CONSTANT)
    )

    assert warm.expected_particle_number > cold.expected_particle_number
    assert near_zero_mu.ground_state_occupation > low_mu.ground_state_occupation


def test_representative_particles_follow_periodic_box_and_mode_weights() -> None:
    result = analyze_ideal_bose_gas(IdealBoseGasParameters())
    positions, momenta = representative_particles(result, n_particles=40, seed=4)

    assert positions.shape == (40, 2)
    assert momenta.shape == (40, 2)
    assert np.all(positions >= 0.0)
    assert np.all(positions < result.parameters.box_length)


def test_bose_occupation_rejects_mu_at_or_above_ground_state() -> None:
    with pytest.raises(ValueError, match="below"):
        bose_einstein_occupation(np.array([0.0]), 100.0e-9, 0.0)


def test_classical_reference_has_the_same_population_but_no_bose_enhancement() -> None:
    comparison = compare_bose_to_classical(
        IdealBoseGasParameters(
            temperature=100.0e-9,
            chemical_potential=-5.0e-9 * BOLTZMANN_CONSTANT,
            maximum_mode_index=4,
        )
    )

    assert np.sum(comparison.classical_occupations) == pytest.approx(
        comparison.expected_particle_number
    )
    assert comparison.bose_ground_state_fraction > (
        comparison.classical_ground_state_fraction
    )
    assert comparison.ground_state_enhancement > 1.0


def test_classical_occupations_are_normalized_to_the_requested_population() -> None:
    occupations = maxwell_boltzmann_occupation(
        np.array([0.0, 2.0e-30, 4.0e-30]),
        temperature=100.0e-9,
        expected_particle_number=12.0,
    )

    assert np.sum(occupations) == pytest.approx(12.0)
    assert occupations[0] > occupations[-1]


def test_three_dimensional_ideal_gas_bec_has_the_known_condensate_fraction() -> None:
    parameters = IdealBoseEinsteinCondensationParameters(
        temperature=50.0e-9,
        particle_number=100_000,
        box_length=20.0e-6,
        particle_mass_amu=87.0,
    )
    result = analyze_ideal_bose_einstein_condensation(parameters)

    expected_fraction = 1.0 - (parameters.temperature / result.critical_temperature) ** 1.5
    assert result.critical_temperature > parameters.temperature
    assert result.condensate_fraction == pytest.approx(expected_fraction)
    assert result.condensate_number + result.thermal_number == pytest.approx(
        parameters.particle_number
    )


def test_three_dimensional_ideal_gas_has_no_condensate_above_critical_temperature() -> None:
    cold = analyze_ideal_bose_einstein_condensation(
        IdealBoseEinsteinCondensationParameters(temperature=50.0e-9)
    )
    warm = analyze_ideal_bose_einstein_condensation(
        IdealBoseEinsteinCondensationParameters(
            temperature=2.0 * cold.critical_temperature
        )
    )

    assert warm.condensate_fraction == 0.0
    assert warm.thermal_number == warm.parameters.particle_number
