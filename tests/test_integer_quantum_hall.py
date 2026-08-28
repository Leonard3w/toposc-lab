from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from toposc_lab.quantum_hall.integer_quantum_hall import (
    CONDUCTANCE_QUANTUM_SIEMENS,
    FLUX_QUANTUM_WEBER,
    IQHEParameters,
    VON_KLITZING_CONSTANT_OHM,
    broadened_density_of_states,
    conductivity_from_resistivity,
    edge_mode_spectrum,
    filling_factor,
    iqhe_state,
    plateau_scan,
    quantized_transport_tensor,
)


def test_filling_factor_counts_flux_quanta_per_electron_density() -> None:
    parameters = IQHEParameters(
        magnetic_field_tesla=4.0,
        electron_density_m2=2.1e15,
    )

    assert filling_factor(parameters) == pytest.approx(
        parameters.electron_density_m2 * FLUX_QUANTUM_WEBER / 4.0
    )
    assert filling_factor(parameters, 8.0) == pytest.approx(
        0.5 * filling_factor(parameters)
    )


def test_exact_plateau_tensor_has_tong_signs_and_is_its_own_inverse_pair() -> None:
    transport = quantized_transport_tensor(3)

    assert transport.longitudinal_resistivity_ohm == 0.0
    assert transport.hall_resistivity_ohm == pytest.approx(
        -VON_KLITZING_CONSTANT_OHM / 3.0
    )
    assert transport.longitudinal_conductivity_siemens == 0.0
    assert transport.hall_conductivity_siemens == pytest.approx(
        3.0 * CONDUCTANCE_QUANTUM_SIEMENS
    )
    assert transport.resistivity_matrix_ohm @ transport.conductivity_matrix_siemens == (
        pytest.approx(np.eye(2), abs=1.0e-14)
    )


def test_general_resistivity_inversion_preserves_tensor_identity() -> None:
    rho_xx = np.asarray([0.0, 500.0, 1_200.0])
    rho_xy = np.asarray([-12_906.0, -8_604.0, -6_453.0])
    sigma_xx, sigma_xy = conductivity_from_resistivity(rho_xx, rho_xy)

    for xx, xy, sxx, sxy in zip(rho_xx, rho_xy, sigma_xx, sigma_xy, strict=True):
        resistivity = np.asarray(((xx, xy), (-xy, xx)))
        conductivity = np.asarray(((sxx, sxy), (-sxy, sxx)))
        assert resistivity @ conductivity == pytest.approx(np.eye(2), abs=1.0e-12)


def test_thermal_number_equation_reconstructs_requested_density() -> None:
    parameters = IQHEParameters(
        magnetic_field_tesla=3.7,
        electron_density_m2=2.8e15,
        temperature_kelvin=4.2,
        maximum_level=6,
    )
    state = iqhe_state(parameters)

    assert np.sum(state.density_contributions_m2) == pytest.approx(
        parameters.electron_density_m2,
        rel=2.0e-12,
    )
    assert np.all((state.occupations >= 0.0) & (state.occupations <= 1.0))


def test_zero_temperature_integer_filling_places_mu_in_landau_gap() -> None:
    density = 3.0 * 5.0 / FLUX_QUANTUM_WEBER
    parameters = IQHEParameters(
        magnetic_field_tesla=5.0,
        electron_density_m2=density,
        temperature_kelvin=0.0,
    )
    state = iqhe_state(parameters)

    assert state.filling_factor == pytest.approx(3.0)
    assert state.occupations[:3] == pytest.approx(np.ones(3))
    assert state.occupations[3] == 0.0
    assert state.energies_joule[2] < state.chemical_potential_joule
    assert state.chemical_potential_joule < state.energies_joule[3]


def test_plateau_scan_matches_exact_quantization_at_integer_filling() -> None:
    field = 5.0
    density = 2.0 * field / FLUX_QUANTUM_WEBER
    parameters = IQHEParameters(
        magnetic_field_tesla=field,
        electron_density_m2=density,
        plateau_transition_width=0.02,
        field_scan_min_tesla=4.5,
        field_scan_max_tesla=5.5,
        field_scan_points=101,
    )
    state = iqhe_state(parameters)
    scan = plateau_scan(parameters)

    assert state.effective_plateau_index == pytest.approx(2.0, rel=1.0e-12)
    assert state.transport.hall_resistivity_ohm == pytest.approx(
        -VON_KLITZING_CONSTANT_OHM / 2.0,
        rel=1.0e-12,
    )
    assert scan.classical_hall_resistivity_ohm[50] == pytest.approx(
        -VON_KLITZING_CONSTANT_OHM / 2.0
    )


def test_unresolved_spin_mode_uses_even_plateau_steps() -> None:
    field = 6.0
    density = 4.0 * field / FLUX_QUANTUM_WEBER
    state = iqhe_state(
        IQHEParameters(
            magnetic_field_tesla=field,
            electron_density_m2=density,
            spin_mode="degenerate",
            plateau_transition_width=0.03,
        )
    )

    assert state.nearest_plateau_index == 4
    assert state.effective_plateau_index == pytest.approx(4.0, rel=1.0e-12)


def test_broadened_dos_preserves_visible_landau_state_density() -> None:
    parameters = IQHEParameters(maximum_level=5, landau_broadening_mev=0.2)
    state = iqhe_state(parameters)
    density_of_states = broadened_density_of_states(parameters, state, energy_points=2_001)
    integrated_density = np.trapezoid(
        density_of_states.density_per_joule_m2,
        density_of_states.energy_joule,
    )
    visible = state.orbital_indices <= parameters.maximum_level
    expected = (
        parameters.magnetic_field_tesla
        / FLUX_QUANTUM_WEBER
        * np.sum(state.branch_degeneracies[visible])
    )

    assert integrated_density == pytest.approx(expected, rel=8.0e-5)
    integrated_occupied_density = np.trapezoid(
        density_of_states.occupied_density_per_joule_m2,
        density_of_states.energy_joule,
    )
    expected_occupied_density = np.sum(state.density_contributions_m2[visible])
    assert integrated_occupied_density == pytest.approx(
        expected_occupied_density,
        rel=8.0e-5,
    )
    assert np.all(
        density_of_states.occupied_density_per_joule_m2
        <= density_of_states.density_per_joule_m2
    )


def test_smooth_edge_has_opposite_chiralities_and_two_modes_at_nu_two() -> None:
    field = 5.0
    parameters = IQHEParameters(
        magnetic_field_tesla=field,
        electron_density_m2=2.0 * field / FLUX_QUANTUM_WEBER,
        temperature_kelvin=0.0,
    )
    spectrum = edge_mode_spectrum(parameters)

    assert spectrum.mode_count_per_edge == 2
    assert spectrum.left_mode_count == 2
    assert spectrum.right_mode_count == 2
    assert np.all(spectrum.crossing_velocity_m_s[spectrum.crossing_sides < 0] > 0.0)
    assert np.all(spectrum.crossing_velocity_m_s[spectrum.crossing_sides > 0] < 0.0)
    assert spectrum.potential_joule == pytest.approx(spectrum.potential_joule[::-1])
    assert spectrum.drift_velocity_m_s == pytest.approx(
        -spectrum.drift_velocity_m_s[::-1]
    )


def test_edge_current_is_quantized_and_reverses_with_hall_bias() -> None:
    field = 4.0
    density = 3.0 * field / FLUX_QUANTUM_WEBER
    positive = edge_mode_spectrum(
        IQHEParameters(
            magnetic_field_tesla=field,
            electron_density_m2=density,
            temperature_kelvin=0.0,
            hall_voltage_microvolt=80.0,
        )
    )
    negative = edge_mode_spectrum(
        IQHEParameters(
            magnetic_field_tesla=field,
            electron_density_m2=density,
            temperature_kelvin=0.0,
            hall_voltage_microvolt=-80.0,
        )
    )

    assert positive.mode_count_per_edge == 3
    assert positive.edge_current_ampere == pytest.approx(
        3.0 * CONDUCTANCE_QUANTUM_SIEMENS * 80.0e-6
    )
    assert negative.edge_current_ampere == pytest.approx(-positive.edge_current_ampere)


def test_spin_degenerate_edge_crossings_carry_their_multiplicity() -> None:
    field = 6.0
    spectrum = edge_mode_spectrum(
        IQHEParameters(
            magnetic_field_tesla=field,
            electron_density_m2=4.0 * field / FLUX_QUANTUM_WEBER,
            temperature_kelvin=0.0,
            spin_mode="degenerate",
        )
    )

    assert spectrum.mode_count_per_edge == 4
    assert np.all(spectrum.crossing_degeneracies == 2.0)


def test_invalid_iqhe_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="magnetic_field_tesla"):
        IQHEParameters(magnetic_field_tesla=0.0)
    with pytest.raises(ValueError, match="spin_mode"):
        IQHEParameters(spin_mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="field_scan_min_tesla"):
        replace(IQHEParameters(), field_scan_min_tesla=10.0, field_scan_max_tesla=9.0)
    with pytest.raises(ValueError, match="energy_points"):
        broadened_density_of_states(IQHEParameters(), energy_points=50)
    with pytest.raises(ValueError, match="edge_sample_width_l_b"):
        IQHEParameters(edge_sample_width_l_b=5.0, edge_smoothness_l_b=3.0)
    with pytest.raises(ValueError, match="grid_points"):
        edge_mode_spectrum(IQHEParameters(), grid_points=100)
