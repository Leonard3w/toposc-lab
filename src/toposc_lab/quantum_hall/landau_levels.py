"""Continuum Landau-level physics used by the interactive learning lab.

The conventions follow Section 1.4 of David Tong's *The Quantum Hall
Effect* (arXiv:1606.06687v2): the particle has charge ``-e``, ``e`` below is
the positive elementary-charge magnitude, and ``B > 0`` points along ``+z``.
The electric-field formulas use the Hamiltonian written in Eq. (1.23).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial, pi

import numpy as np
from scipy.constants import elementary_charge as ELEMENTARY_CHARGE
from scipy.constants import hbar as HBAR
from scipy.constants import m_e as ELECTRON_MASS
from scipy.constants import physical_constants
from scipy.special import eval_hermite


BOHR_MAGNETON = float(physical_constants["Bohr magneton"][0])
BOLTZMANN_CONSTANT = float(physical_constants["Boltzmann constant"][0])


def _require_finite(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class LandauLevelParameters:
    """Physical and display parameters for the Section 1.4 learning lab."""

    magnetic_field_tesla: float = 1.0
    effective_mass_ratio: float = 0.067
    electric_field_v_per_m: float = 0.0
    sample_width_m: float = 1.0e-6
    sample_height_m: float = 1.0e-6
    maximum_level: int = 5
    selected_level: int = 0
    wave_number_l_b: float = 0.0
    angular_momentum: int = 0
    g_factor: float = -0.44
    include_zeeman: bool = False
    view_extent_l_b: float = 6.0
    orbit_radius_l_b: float = 2.0
    orbit_phase_radians: float = 0.0

    def __post_init__(self) -> None:
        positive_values = {
            "magnetic_field_tesla": self.magnetic_field_tesla,
            "effective_mass_ratio": self.effective_mass_ratio,
            "sample_width_m": self.sample_width_m,
            "sample_height_m": self.sample_height_m,
            "view_extent_l_b": self.view_extent_l_b,
            "orbit_radius_l_b": self.orbit_radius_l_b,
        }
        for name, value in positive_values.items():
            _require_finite(name, float(value))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")

        for name, value in {
            "electric_field_v_per_m": self.electric_field_v_per_m,
            "wave_number_l_b": self.wave_number_l_b,
            "g_factor": self.g_factor,
            "orbit_phase_radians": self.orbit_phase_radians,
        }.items():
            _require_finite(name, float(value))

        if not 0 <= self.maximum_level <= 30:
            raise ValueError("maximum_level must lie between 0 and 30")
        if not 0 <= self.selected_level <= self.maximum_level:
            raise ValueError("selected_level must lie between 0 and maximum_level")
        if not 0 <= self.angular_momentum <= 100:
            raise ValueError("angular_momentum must lie between 0 and 100")

    @property
    def effective_mass_kg(self) -> float:
        """Effective particle mass in kilograms."""
        return self.effective_mass_ratio * ELECTRON_MASS

    @property
    def sample_area_m2(self) -> float:
        """Rectangular sample area in square metres."""
        return self.sample_width_m * self.sample_height_m


@dataclass(frozen=True)
class LandauScales:
    """Derived magnetic, electric and finite-sample scales."""

    magnetic_length_m: float
    cyclotron_frequency_rad_s: float
    cyclotron_period_s: float
    cyclotron_energy_joule: float
    zeeman_splitting_joule: float
    drift_velocity_m_s: float
    electric_displacement_m: float
    flux_quantum_weber: float
    sample_flux_weber: float
    flux_quanta: float


@dataclass(frozen=True)
class LandauGaugeState:
    """A normalized transverse Landau-gauge oscillator state."""

    x_over_l_b: np.ndarray
    y_over_l_b: np.ndarray
    transverse_probability_density: np.ndarray
    probability_map: np.ndarray
    center_over_l_b: float
    wave_number_l_b: float


@dataclass(frozen=True)
class SymmetricGaugeState:
    """A normalized lowest-Landau-level state in symmetric gauge."""

    x_over_l_b: np.ndarray
    y_over_l_b: np.ndarray
    probability_density_l_b2: np.ndarray
    phase_radians: np.ndarray
    radius_over_l_b: np.ndarray
    radial_density_l_b2: np.ndarray
    peak_radius_over_l_b: float


@dataclass(frozen=True)
class CyclotronTrajectory:
    """Classical cyclotron motion with the quantum E-cross-B drift."""

    time_s: np.ndarray
    phase_radians: np.ndarray
    x_over_l_b: np.ndarray
    y_over_l_b: np.ndarray
    guiding_center_x_over_l_b: np.ndarray
    guiding_center_y_over_l_b: np.ndarray


def landau_scales(parameters: LandauLevelParameters) -> LandauScales:
    """Compute the characteristic scales appearing throughout Section 1.4."""
    magnetic_field = parameters.magnetic_field_tesla
    mass = parameters.effective_mass_kg
    magnetic_length = np.sqrt(HBAR / (ELEMENTARY_CHARGE * magnetic_field))
    cyclotron_frequency = ELEMENTARY_CHARGE * magnetic_field / mass
    cyclotron_energy = HBAR * cyclotron_frequency
    flux_quantum = 2.0 * pi * HBAR / ELEMENTARY_CHARGE
    sample_flux = magnetic_field * parameters.sample_area_m2

    return LandauScales(
        magnetic_length_m=float(magnetic_length),
        cyclotron_frequency_rad_s=float(cyclotron_frequency),
        cyclotron_period_s=float(2.0 * pi / cyclotron_frequency),
        cyclotron_energy_joule=float(cyclotron_energy),
        zeeman_splitting_joule=float(parameters.g_factor * BOHR_MAGNETON * magnetic_field),
        drift_velocity_m_s=float(parameters.electric_field_v_per_m / magnetic_field),
        electric_displacement_m=float(
            mass * parameters.electric_field_v_per_m / (ELEMENTARY_CHARGE * magnetic_field**2)
        ),
        flux_quantum_weber=float(flux_quantum),
        sample_flux_weber=float(sample_flux),
        flux_quanta=float(sample_flux / flux_quantum),
    )


def _level_array(
    parameters: LandauLevelParameters,
    levels: np.ndarray | None,
) -> np.ndarray:
    if levels is None:
        return np.arange(parameters.maximum_level + 1, dtype=float)

    values = np.asarray(levels, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("levels must be finite and non-negative")
    return values


def landau_level_energies_joule(
    parameters: LandauLevelParameters,
    levels: np.ndarray | None = None,
    *,
    spin_projection: int | None = None,
) -> np.ndarray:
    """Return ``E_n = hbar omega_B (n + 1/2)`` with optional Zeeman energy.

    ``spin_projection`` is ``+1`` or ``-1`` and adds
    ``s g mu_B B / 2``.  The separation of the two branches is therefore
    ``g mu_B B``, as in the spin aside in the lecture notes.
    """
    level_values = _level_array(parameters, levels)
    scales = landau_scales(parameters)
    energies = scales.cyclotron_energy_joule * (level_values + 0.5)

    if spin_projection is not None:
        if spin_projection not in (-1, 1):
            raise ValueError("spin_projection must be -1, +1 or None")
        energies = energies + 0.5 * spin_projection * scales.zeeman_splitting_joule

    return np.asarray(energies, dtype=float)


def electric_field_energies_joule(
    parameters: LandauLevelParameters,
    k_l_b: np.ndarray,
    levels: np.ndarray | None = None,
    *,
    spin_projection: int | None = None,
) -> np.ndarray:
    """Return Eq. (1.25) on a dimensionless ``k l_B`` axis.

    Completing the square reduces Eq. (1.25) to
    ``E_nk = E_n + e E l_B (k l_B) - m E^2/(2 B^2)``.  The derivative with
    respect to ``hbar k`` is exactly the signed drift velocity ``E/B``.
    """
    dimensionless_k = np.asarray(k_l_b, dtype=float)
    if np.any(~np.isfinite(dimensionless_k)):
        raise ValueError("k_l_b must be finite")

    base = landau_level_energies_joule(
        parameters,
        levels,
        spin_projection=spin_projection,
    )
    scales = landau_scales(parameters)
    electric_shift = (
        ELEMENTARY_CHARGE
        * parameters.electric_field_v_per_m
        * scales.magnetic_length_m
        * dimensionless_k
    )
    common_offset = (
        -0.5
        * parameters.effective_mass_kg
        * (parameters.electric_field_v_per_m / parameters.magnetic_field_tesla) ** 2
    )
    return np.asarray(base)[..., np.newaxis] + electric_shift + common_offset


def landau_gauge_state(
    parameters: LandauLevelParameters,
    *,
    grid_points: int = 241,
    include_electric_shift: bool = False,
) -> LandauGaugeState:
    """Evaluate the selected ``psi_nk`` as a strip in Landau gauge.

    The returned transverse density is ``l_B |f_n(x)|^2`` and integrates to
    one on an infinite x-axis.  The two-dimensional map repeats this density
    along y because the plane-wave factor ``exp(i k y)`` has constant modulus.
    """
    if grid_points < 51:
        raise ValueError("grid_points must be at least 51")

    extent = parameters.view_extent_l_b
    x_over_l_b = np.linspace(-extent, extent, grid_points)
    y_over_l_b = np.linspace(-extent, extent, grid_points)
    scales = landau_scales(parameters)
    electric_displacement = (
        scales.electric_displacement_m / scales.magnetic_length_m if include_electric_shift else 0.0
    )
    oscillator_coordinate = x_over_l_b + parameters.wave_number_l_b - electric_displacement
    level = parameters.selected_level
    hermite = eval_hermite(level, oscillator_coordinate)
    normalization = np.sqrt(pi) * (2.0**level) * factorial(level)
    transverse_density = hermite**2 * np.exp(-(oscillator_coordinate**2)) / normalization
    probability_map = np.repeat(
        transverse_density[np.newaxis, :],
        grid_points,
        axis=0,
    )

    return LandauGaugeState(
        x_over_l_b=x_over_l_b,
        y_over_l_b=y_over_l_b,
        transverse_probability_density=transverse_density,
        probability_map=probability_map,
        center_over_l_b=float(electric_displacement - parameters.wave_number_l_b),
        wave_number_l_b=parameters.wave_number_l_b,
    )


def symmetric_gauge_lll_state(
    parameters: LandauLevelParameters,
    *,
    grid_points: int = 241,
) -> SymmetricGaugeState:
    """Evaluate the normalized ``psi_LLL,m`` from Eq. (1.30)."""
    if grid_points < 51:
        raise ValueError("grid_points must be at least 51")

    extent = parameters.view_extent_l_b
    coordinates = np.linspace(-extent, extent, grid_points)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    radius_squared = x_grid**2 + y_grid**2
    angular_momentum = parameters.angular_momentum
    normalization = 2.0 * pi * (2.0**angular_momentum) * factorial(angular_momentum)
    density = radius_squared**angular_momentum * np.exp(-0.5 * radius_squared) / normalization
    phase = -angular_momentum * np.arctan2(y_grid, x_grid)

    radius = np.linspace(0.0, extent, grid_points)
    radial_density = radius ** (2 * angular_momentum) * np.exp(-0.5 * radius**2) / normalization

    return SymmetricGaugeState(
        x_over_l_b=coordinates,
        y_over_l_b=coordinates,
        probability_density_l_b2=density,
        phase_radians=phase,
        radius_over_l_b=radius,
        radial_density_l_b2=radial_density,
        peak_radius_over_l_b=float(np.sqrt(2.0 * angular_momentum)),
    )


def cyclotron_trajectory(
    parameters: LandauLevelParameters,
    *,
    cycles: float = 2.0,
    samples: int = 181,
) -> CyclotronTrajectory:
    """Return cyclotron motion plus the ``E/B`` guiding-centre drift."""
    if not np.isfinite(cycles) or cycles <= 0.0:
        raise ValueError("cycles must be positive and finite")
    if samples < 20:
        raise ValueError("samples must be at least 20")

    scales = landau_scales(parameters)
    phase = np.linspace(0.0, cycles * 2.0 * pi, samples)
    time = phase / scales.cyclotron_frequency_rad_s
    center_x = np.full(
        samples,
        scales.electric_displacement_m / scales.magnetic_length_m,
    )
    center_y = scales.drift_velocity_m_s * time / scales.magnetic_length_m
    orbit_phase = phase + parameters.orbit_phase_radians

    return CyclotronTrajectory(
        time_s=time,
        phase_radians=phase,
        x_over_l_b=center_x - parameters.orbit_radius_l_b * np.sin(orbit_phase),
        y_over_l_b=center_y + parameters.orbit_radius_l_b * np.cos(orbit_phase),
        guiding_center_x_over_l_b=center_x,
        guiding_center_y_over_l_b=center_y,
    )
