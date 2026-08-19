from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from toposc_lab.quantum_hall.landau_levels import (
    ELEMENTARY_CHARGE,
    HBAR,
    LandauLevelParameters,
    cyclotron_trajectory,
    electric_field_energies_joule,
    landau_gauge_state,
    landau_level_energies_joule,
    landau_scales,
    symmetric_gauge_lll_state,
)


def test_characteristic_scales_and_degeneracy_have_expected_values() -> None:
    parameters = LandauLevelParameters()
    scales = landau_scales(parameters)

    assert scales.magnetic_length_m == pytest.approx(25.6556e-9, rel=2.0e-4)
    assert scales.cyclotron_energy_joule * 1.0e3 / ELEMENTARY_CHARGE == pytest.approx(
        1.7279,
        rel=2.0e-4,
    )
    assert scales.flux_quanta == pytest.approx(241.799, rel=2.0e-4)


def test_magnetic_scaling_and_equal_level_spacing() -> None:
    parameters = LandauLevelParameters(magnetic_field_tesla=2.0)
    lower_field = landau_scales(replace(parameters, magnetic_field_tesla=0.5))
    higher_field = landau_scales(parameters)
    energies = landau_level_energies_joule(parameters)

    assert lower_field.magnetic_length_m / higher_field.magnetic_length_m == pytest.approx(2.0)
    assert higher_field.flux_quanta / lower_field.flux_quanta == pytest.approx(4.0)
    assert np.diff(energies) == pytest.approx(higher_field.cyclotron_energy_joule)


def test_electric_dispersion_has_exact_e_cross_b_group_velocity() -> None:
    parameters = LandauLevelParameters(electric_field_v_per_m=2400.0)
    k_l_b = np.linspace(-1.0, 1.0, 101)
    energy = electric_field_energies_joule(
        parameters,
        k_l_b,
        np.asarray([0]),
    )[0]
    scales = landau_scales(parameters)
    derivative_by_k = np.gradient(energy, k_l_b) * scales.magnetic_length_m
    group_velocity = derivative_by_k / HBAR

    assert group_velocity == pytest.approx(
        parameters.electric_field_v_per_m / parameters.magnetic_field_tesla,
        rel=1.0e-10,
    )


def test_landau_gauge_state_is_normalized_and_moves_with_k_and_electric_field() -> None:
    parameters = LandauLevelParameters(
        selected_level=2,
        maximum_level=4,
        wave_number_l_b=1.25,
        electric_field_v_per_m=3000.0,
        view_extent_l_b=10.0,
    )
    zero_field_state = landau_gauge_state(parameters)
    shifted_state = landau_gauge_state(parameters, include_electric_shift=True)
    scales = landau_scales(parameters)

    normalization = np.trapezoid(
        zero_field_state.transverse_probability_density,
        zero_field_state.x_over_l_b,
    )
    assert normalization == pytest.approx(1.0, rel=1.0e-7)
    assert zero_field_state.center_over_l_b == pytest.approx(-1.25)
    assert shifted_state.center_over_l_b == pytest.approx(
        -1.25 + scales.electric_displacement_m / scales.magnetic_length_m
    )


def test_symmetric_gauge_lll_density_is_normalized_and_peaks_on_expected_ring() -> None:
    parameters = LandauLevelParameters(
        angular_momentum=5,
        view_extent_l_b=9.0,
    )
    state = symmetric_gauge_lll_state(parameters, grid_points=401)
    integral_x = np.trapezoid(
        state.probability_density_l_b2,
        state.x_over_l_b,
        axis=1,
    )
    normalization = np.trapezoid(integral_x, state.y_over_l_b)

    assert normalization == pytest.approx(1.0, rel=2.0e-5)
    assert state.peak_radius_over_l_b == pytest.approx(np.sqrt(10.0))
    numerical_peak = state.radius_over_l_b[np.argmax(state.radial_density_l_b2)]
    assert numerical_peak == pytest.approx(np.sqrt(10.0), abs=0.03)


def test_cyclotron_guiding_center_drifts_at_e_over_b() -> None:
    parameters = LandauLevelParameters(electric_field_v_per_m=-1800.0)
    trajectory = cyclotron_trajectory(parameters, cycles=1.5, samples=91)
    scales = landau_scales(parameters)
    physical_displacement = (
        trajectory.guiding_center_y_over_l_b[-1] - trajectory.guiding_center_y_over_l_b[0]
    ) * scales.magnetic_length_m

    assert physical_displacement == pytest.approx(scales.drift_velocity_m_s * trajectory.time_s[-1])
    assert trajectory.x_over_l_b.shape == (91,)


def test_invalid_landau_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="magnetic_field_tesla"):
        LandauLevelParameters(magnetic_field_tesla=0.0)

    with pytest.raises(ValueError, match="selected_level"):
        LandauLevelParameters(maximum_level=2, selected_level=3)
