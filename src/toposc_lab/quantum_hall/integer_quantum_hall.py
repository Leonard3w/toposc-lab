"""Integer-quantum-Hall calculations for the Section 2 learning lab.

The conventions follow David Tong, *The Quantum Hall Effect*, Section 2:

* the electron has charge ``-e`` with ``e > 0``;
* the magnetic field points along ``+z``;
* resistivity and conductivity tensors are written as

  ``rho = [[rho_xx, rho_xy], [-rho_xy, rho_xx]]`` and
  ``sigma = [[sigma_xx, sigma_xy], [-sigma_xy, sigma_xx]]``;
* consequently a filled electron Landau level at ``B > 0`` has
  ``rho_xy = -h/(nu e^2)`` and ``sigma_xy = +nu e^2/h``.

The finite-width plateaux implemented here are a transparent phenomenological
preview.  They are not presented as a microscopic disorder calculation: the
later disorder lab will replace their adjustable transition width and
longitudinal peaks by localization data.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, pi
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

from toposc_lab.quantum_hall.landau_levels import (
    BOHR_MAGNETON,
    BOLTZMANN_CONSTANT,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    HBAR,
)


SpinMode = Literal["spinless", "degenerate", "zeeman"]

PLANCK_CONSTANT = 2.0 * pi * HBAR
FLUX_QUANTUM_WEBER = PLANCK_CONSTANT / ELEMENTARY_CHARGE
CONDUCTANCE_QUANTUM_SIEMENS = ELEMENTARY_CHARGE**2 / PLANCK_CONSTANT
VON_KLITZING_CONSTANT_OHM = PLANCK_CONSTANT / ELEMENTARY_CHARGE**2
JOULE_PER_MEV = 1.0e-3 * ELEMENTARY_CHARGE


def _require_finite(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class IQHEParameters:
    """Physical and numerical controls for the first IQHE dashboard."""

    magnetic_field_tesla: float = 5.0
    electron_density_m2: float = 10.0 / FLUX_QUANTUM_WEBER
    effective_mass_ratio: float = 0.067
    temperature_kelvin: float = 1.5
    maximum_level: int = 8
    spin_mode: SpinMode = "spinless"
    g_factor: float = -0.44
    landau_broadening_mev: float = 0.30
    plateau_transition_width: float = 0.08
    longitudinal_peak_resistivity_ohm: float = 1_500.0
    field_scan_min_tesla: float = 1.5
    field_scan_max_tesla: float = 12.0
    field_scan_points: int = 1_001
    edge_sample_width_l_b: float = 24.0
    edge_smoothness_l_b: float = 3.0
    edge_confinement_energy_mev: float = 45.0
    edge_confinement_power: float = 2.0
    edge_maximum_level: int = 4
    hall_voltage_microvolt: float = 100.0
    skipping_orbit_radius_l_b: float = 1.5
    edge_animation_speed: float = 1.0

    def __post_init__(self) -> None:
        positive = {
            "magnetic_field_tesla": self.magnetic_field_tesla,
            "electron_density_m2": self.electron_density_m2,
            "effective_mass_ratio": self.effective_mass_ratio,
            "landau_broadening_mev": self.landau_broadening_mev,
            "plateau_transition_width": self.plateau_transition_width,
            "field_scan_min_tesla": self.field_scan_min_tesla,
            "field_scan_max_tesla": self.field_scan_max_tesla,
            "edge_sample_width_l_b": self.edge_sample_width_l_b,
            "edge_smoothness_l_b": self.edge_smoothness_l_b,
            "edge_confinement_energy_mev": self.edge_confinement_energy_mev,
            "edge_confinement_power": self.edge_confinement_power,
            "skipping_orbit_radius_l_b": self.skipping_orbit_radius_l_b,
            "edge_animation_speed": self.edge_animation_speed,
        }
        for name, value in positive.items():
            _require_finite(name, float(value))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")

        _require_finite("temperature_kelvin", self.temperature_kelvin)
        if self.temperature_kelvin < 0.0:
            raise ValueError("temperature_kelvin must be non-negative")

        _require_finite("g_factor", self.g_factor)
        _require_finite("hall_voltage_microvolt", self.hall_voltage_microvolt)
        _require_finite(
            "longitudinal_peak_resistivity_ohm",
            self.longitudinal_peak_resistivity_ohm,
        )
        if self.longitudinal_peak_resistivity_ohm < 0.0:
            raise ValueError("longitudinal_peak_resistivity_ohm must be non-negative")

        if not 0 <= self.maximum_level <= 40:
            raise ValueError("maximum_level must lie between 0 and 40")
        if not 0 <= self.edge_maximum_level <= 12:
            raise ValueError("edge_maximum_level must lie between 0 and 12")
        if self.edge_maximum_level > self.maximum_level:
            raise ValueError("edge_maximum_level must not exceed maximum_level")
        if self.spin_mode not in {"spinless", "degenerate", "zeeman"}:
            raise ValueError("spin_mode must be spinless, degenerate or zeeman")
        if self.plateau_transition_width > 0.5:
            raise ValueError("plateau_transition_width must not exceed 0.5")
        if self.field_scan_min_tesla >= self.field_scan_max_tesla:
            raise ValueError("field_scan_min_tesla must be smaller than field_scan_max_tesla")
        if not 101 <= self.field_scan_points <= 2_001:
            raise ValueError("field_scan_points must lie between 101 and 2001")
        if self.edge_sample_width_l_b <= 2.0 * self.edge_smoothness_l_b:
            raise ValueError("edge_sample_width_l_b must exceed twice edge_smoothness_l_b")

    @property
    def effective_mass_kg(self) -> float:
        return self.effective_mass_ratio * ELECTRON_MASS

    @property
    def plateau_step(self) -> int:
        """Spacing of visible plateaux for the selected spin treatment."""
        return 2 if self.spin_mode == "degenerate" else 1


@dataclass(frozen=True)
class TransportTensor:
    """Independent entries of isotropic Hall-response tensors."""

    longitudinal_resistivity_ohm: float
    hall_resistivity_ohm: float
    longitudinal_conductivity_siemens: float
    hall_conductivity_siemens: float

    @property
    def resistivity_matrix_ohm(self) -> np.ndarray:
        xx = self.longitudinal_resistivity_ohm
        xy = self.hall_resistivity_ohm
        return np.asarray(((xx, xy), (-xy, xx)), dtype=float)

    @property
    def conductivity_matrix_siemens(self) -> np.ndarray:
        xx = self.longitudinal_conductivity_siemens
        xy = self.hall_conductivity_siemens
        return np.asarray(((xx, xy), (-xy, xx)), dtype=float)


@dataclass(frozen=True)
class IQHEState:
    """Landau-level occupations and transport at the selected field."""

    filling_factor: float
    chemical_potential_joule: float
    orbital_indices: np.ndarray
    spin_projections: np.ndarray
    branch_degeneracies: np.ndarray
    energies_joule: np.ndarray
    occupations: np.ndarray
    density_contributions_m2: np.ndarray
    nearest_plateau_index: int
    effective_plateau_index: float
    transport: TransportTensor


@dataclass(frozen=True)
class PlateauScan:
    """Field scan of the clean Hall line and phenomenological plateaux."""

    magnetic_field_tesla: np.ndarray
    filling_factor: np.ndarray
    effective_plateau_index: np.ndarray
    classical_hall_resistivity_ohm: np.ndarray
    hall_resistivity_ohm: np.ndarray
    longitudinal_resistivity_ohm: np.ndarray
    hall_conductivity_siemens: np.ndarray
    longitudinal_conductivity_siemens: np.ndarray


@dataclass(frozen=True)
class DensityOfStates:
    """Broadened Landau-level DOS and its occupied part."""

    energy_joule: np.ndarray
    density_per_joule_m2: np.ndarray
    occupied_density_per_joule_m2: np.ndarray


@dataclass(frozen=True)
class EdgeModeSpectrum:
    """Smooth-edge spectrum, drift velocities and Fermi crossings."""

    x_over_l_b: np.ndarray
    k_l_b: np.ndarray
    potential_joule: np.ndarray
    potential_gradient_joule_per_m: np.ndarray
    drift_velocity_m_s: np.ndarray
    orbital_indices: np.ndarray
    spin_projections: np.ndarray
    branch_degeneracies: np.ndarray
    energies_joule: np.ndarray
    chemical_potential_joule: float
    crossing_branch_indices: np.ndarray
    crossing_sides: np.ndarray
    crossing_x_over_l_b: np.ndarray
    crossing_k_l_b: np.ndarray
    crossing_velocity_m_s: np.ndarray
    crossing_degeneracies: np.ndarray
    mode_count_per_edge: int
    edge_current_ampere: float

    @property
    def left_mode_count(self) -> int:
        return int(np.rint(np.sum(self.crossing_degeneracies[self.crossing_sides < 0])))

    @property
    def right_mode_count(self) -> int:
        return int(np.rint(np.sum(self.crossing_degeneracies[self.crossing_sides > 0])))


def filling_factor(parameters: IQHEParameters, magnetic_field_tesla: float | None = None) -> float:
    """Return ``nu = n_e h/(e B) = n_e Phi_0/B``."""
    magnetic_field = (
        parameters.magnetic_field_tesla
        if magnetic_field_tesla is None
        else float(magnetic_field_tesla)
    )
    _require_finite("magnetic_field_tesla", magnetic_field)
    if magnetic_field <= 0.0:
        raise ValueError("magnetic_field_tesla must be positive")
    return float(parameters.electron_density_m2 * FLUX_QUANTUM_WEBER / magnetic_field)


def conductivity_from_resistivity(
    longitudinal_resistivity_ohm: np.ndarray | float,
    hall_resistivity_ohm: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the isotropic Hall resistivity tensor with the stated signs."""
    rho_xx = np.asarray(longitudinal_resistivity_ohm, dtype=float)
    rho_xy = np.asarray(hall_resistivity_ohm, dtype=float)
    denominator = rho_xx**2 + rho_xy**2
    sigma_xx = np.divide(
        rho_xx,
        denominator,
        out=np.full(np.broadcast_shapes(rho_xx.shape, rho_xy.shape), np.nan),
        where=denominator > 0.0,
    )
    sigma_xy = np.divide(
        -rho_xy,
        denominator,
        out=np.full(np.broadcast_shapes(rho_xx.shape, rho_xy.shape), np.nan),
        where=denominator > 0.0,
    )
    return sigma_xx, sigma_xy


def quantized_transport_tensor(plateau_index: int) -> TransportTensor:
    """Return the exact clean integer-plateau response for ``nu > 0``."""
    if isinstance(plateau_index, bool) or not isinstance(plateau_index, (int, np.integer)):
        raise TypeError("plateau_index must be an integer")
    if plateau_index <= 0:
        raise ValueError("plateau_index must be positive")

    rho_xy = -VON_KLITZING_CONSTANT_OHM / plateau_index
    return TransportTensor(
        longitudinal_resistivity_ohm=0.0,
        hall_resistivity_ohm=rho_xy,
        longitudinal_conductivity_siemens=0.0,
        hall_conductivity_siemens=plateau_index * CONDUCTANCE_QUANTUM_SIEMENS,
    )


def _branch_spectrum(
    parameters: IQHEParameters,
    magnetic_field_tesla: float,
    orbital_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    orbitals = np.arange(orbital_count, dtype=int)
    cyclotron_energy = (
        HBAR * ELEMENTARY_CHARGE * magnetic_field_tesla / parameters.effective_mass_kg
    )
    base_energy = cyclotron_energy * (orbitals + 0.5)

    if parameters.spin_mode == "zeeman":
        orbital_indices = np.repeat(orbitals, 2)
        spin_projections = np.tile(np.asarray((-1, 1), dtype=int), orbital_count)
        degeneracies = np.ones(2 * orbital_count, dtype=float)
        energies = np.repeat(base_energy, 2) + (
            0.5
            * spin_projections
            * parameters.g_factor
            * BOHR_MAGNETON
            * magnetic_field_tesla
        )
    else:
        orbital_indices = orbitals
        spin_projections = np.zeros(orbital_count, dtype=int)
        degeneracy = 2.0 if parameters.spin_mode == "degenerate" else 1.0
        degeneracies = np.full(orbital_count, degeneracy)
        energies = base_energy

    order = np.argsort(energies, kind="stable")
    return (
        orbital_indices[order],
        spin_projections[order],
        degeneracies[order],
        np.asarray(energies[order], dtype=float),
    )


def _zero_temperature_occupations(
    filling: float,
    degeneracies: np.ndarray,
    energies: np.ndarray,
) -> tuple[np.ndarray, float]:
    occupations = np.zeros_like(energies)
    remaining = filling
    last_filled = -1

    for index, degeneracy in enumerate(degeneracies):
        if remaining <= 1.0e-12:
            break
        fraction = min(1.0, remaining / degeneracy)
        occupations[index] = fraction
        remaining -= fraction * degeneracy
        if fraction > 0.0:
            last_filled = index
        if fraction < 1.0:
            return occupations, float(energies[index])

    if last_filled < 0:
        return occupations, float(energies[0] - (energies[1] - energies[0]))
    if last_filled + 1 < energies.size:
        chemical_potential = 0.5 * (energies[last_filled] + energies[last_filled + 1])
    else:
        chemical_potential = energies[last_filled]
    return occupations, float(chemical_potential)


def _thermal_occupations(
    target_density_m2: float,
    state_density_m2: np.ndarray,
    energies: np.ndarray,
    temperature_kelvin: float,
) -> tuple[np.ndarray, float]:
    thermal_energy = BOLTZMANN_CONSTANT * temperature_kelvin
    energy_span = max(float(np.ptp(energies)), thermal_energy)
    lower = float(energies[0] - energy_span - 60.0 * thermal_energy)
    upper = float(energies[-1] + energy_span + 60.0 * thermal_energy)
    reference_energy = float(energies[0])

    def relative_density_residual(scaled_chemical_potential: float) -> float:
        chemical_potential = reference_energy + scaled_chemical_potential * thermal_energy
        occupations = expit((chemical_potential - energies) / thermal_energy)
        return float(np.dot(state_density_m2, occupations) / target_density_m2 - 1.0)

    scaled_lower = (lower - reference_energy) / thermal_energy
    scaled_upper = (upper - reference_energy) / thermal_energy
    scaled_chemical_potential = brentq(
        relative_density_residual,
        scaled_lower,
        scaled_upper,
        xtol=1.0e-13,
        rtol=1.0e-13,
    )
    chemical_potential = reference_energy + scaled_chemical_potential * thermal_energy
    occupations = expit((chemical_potential - energies) / thermal_energy)
    return np.asarray(occupations, dtype=float), float(chemical_potential)


def _phenomenological_transport(
    filling_values: np.ndarray,
    parameters: IQHEParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(filling_values, dtype=float)
    step = parameters.plateau_step
    maximum_filling = max(float(np.max(values)), float(step))
    transition_count = int(ceil(maximum_filling / step)) + 2
    transition_centres = step * (np.arange(transition_count, dtype=float) + 0.5)
    scaled_distance = (
        values[..., np.newaxis] - transition_centres
    ) / parameters.plateau_transition_width

    effective_index = step * np.sum(0.5 * (1.0 + np.tanh(scaled_distance)), axis=-1)
    longitudinal = parameters.longitudinal_peak_resistivity_ohm * np.sum(
        np.exp(-0.5 * scaled_distance**2),
        axis=-1,
    )

    plateau_visible = effective_index >= 0.25 * step
    hall_resistivity = np.divide(
        -VON_KLITZING_CONSTANT_OHM,
        effective_index,
        out=np.full_like(effective_index, np.nan),
        where=plateau_visible,
    )
    sigma_xx, sigma_xy = conductivity_from_resistivity(longitudinal, hall_resistivity)
    return effective_index, longitudinal, hall_resistivity, sigma_xx, sigma_xy


def iqhe_state(parameters: IQHEParameters) -> IQHEState:
    """Solve occupations and response at the selected magnetic field."""
    filling = filling_factor(parameters)
    states_per_orbital = 2 if parameters.spin_mode in {"degenerate", "zeeman"} else 1
    orbital_count = max(
        parameters.maximum_level + 2,
        int(ceil(filling / states_per_orbital)) + 6,
    )
    orbitals, spins, degeneracies, energies = _branch_spectrum(
        parameters,
        parameters.magnetic_field_tesla,
        orbital_count,
    )
    degeneracy_density = (
        ELEMENTARY_CHARGE * parameters.magnetic_field_tesla / PLANCK_CONSTANT
    )
    state_density = degeneracies * degeneracy_density

    if parameters.temperature_kelvin == 0.0:
        occupations, chemical_potential = _zero_temperature_occupations(
            filling,
            degeneracies,
            energies,
        )
    else:
        occupations, chemical_potential = _thermal_occupations(
            parameters.electron_density_m2,
            state_density,
            energies,
            parameters.temperature_kelvin,
        )

    effective, rho_xx, rho_xy, sigma_xx, sigma_xy = _phenomenological_transport(
        np.asarray([filling]),
        parameters,
    )
    step = parameters.plateau_step
    nearest_plateau = int(step * np.floor(filling / step + 0.5))

    return IQHEState(
        filling_factor=filling,
        chemical_potential_joule=chemical_potential,
        orbital_indices=orbitals,
        spin_projections=spins,
        branch_degeneracies=degeneracies,
        energies_joule=energies,
        occupations=occupations,
        density_contributions_m2=state_density * occupations,
        nearest_plateau_index=nearest_plateau,
        effective_plateau_index=float(effective[0]),
        transport=TransportTensor(
            longitudinal_resistivity_ohm=float(rho_xx[0]),
            hall_resistivity_ohm=float(rho_xy[0]),
            longitudinal_conductivity_siemens=float(sigma_xx[0]),
            hall_conductivity_siemens=float(sigma_xy[0]),
        ),
    )


def plateau_scan(parameters: IQHEParameters) -> PlateauScan:
    """Scan the fixed-density response over magnetic field."""
    fields = np.linspace(
        parameters.field_scan_min_tesla,
        parameters.field_scan_max_tesla,
        parameters.field_scan_points,
    )
    fillings = parameters.electron_density_m2 * FLUX_QUANTUM_WEBER / fields
    effective, rho_xx, rho_xy, sigma_xx, sigma_xy = _phenomenological_transport(
        fillings,
        parameters,
    )
    classical_hall = -fields / (parameters.electron_density_m2 * ELEMENTARY_CHARGE)
    return PlateauScan(
        magnetic_field_tesla=fields,
        filling_factor=fillings,
        effective_plateau_index=effective,
        classical_hall_resistivity_ohm=classical_hall,
        hall_resistivity_ohm=rho_xy,
        longitudinal_resistivity_ohm=rho_xx,
        hall_conductivity_siemens=sigma_xy,
        longitudinal_conductivity_siemens=sigma_xx,
    )


def broadened_density_of_states(
    parameters: IQHEParameters,
    state: IQHEState | None = None,
    *,
    energy_points: int = 601,
) -> DensityOfStates:
    """Return a normalized Gaussian preview of broadened Landau levels."""
    if energy_points < 101:
        raise ValueError("energy_points must be at least 101")
    selected_state = iqhe_state(parameters) if state is None else state
    visible = selected_state.orbital_indices <= parameters.maximum_level
    energies = selected_state.energies_joule[visible]
    degeneracies = selected_state.branch_degeneracies[visible]
    occupations = selected_state.occupations[visible]
    broadening = parameters.landau_broadening_mev * JOULE_PER_MEV
    thermal_margin = 8.0 * BOLTZMANN_CONSTANT * parameters.temperature_kelvin
    margin = max(4.0 * broadening, thermal_margin)
    energy_axis = np.linspace(
        max(0.0, float(np.min(energies) - margin)),
        float(np.max(energies) + margin),
        energy_points,
    )
    degeneracy_density = (
        ELEMENTARY_CHARGE * parameters.magnetic_field_tesla / PLANCK_CONSTANT
    )
    offsets = (energy_axis[:, np.newaxis] - energies[np.newaxis, :]) / broadening
    gaussian = np.exp(-0.5 * offsets**2) / (np.sqrt(2.0 * pi) * broadening)
    density = degeneracy_density * (gaussian @ degeneracies)
    occupied_density = degeneracy_density * (gaussian @ (degeneracies * occupations))

    return DensityOfStates(
        energy_joule=energy_axis,
        density_per_joule_m2=np.asarray(density, dtype=float),
        occupied_density_per_joule_m2=np.asarray(occupied_density, dtype=float),
    )


def _edge_confinement_profile(
    parameters: IQHEParameters,
    x_over_l_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a symmetric soft-wall potential and its physical derivative."""
    half_width = 0.5 * parameters.edge_sample_width_l_b
    smoothness = parameters.edge_smoothness_l_b
    power = parameters.edge_confinement_power
    left_coordinate = (-half_width - x_over_l_b) / smoothness
    right_coordinate = (x_over_l_b - half_width) / smoothness
    left_soft_wall = np.logaddexp(0.0, left_coordinate)
    right_soft_wall = np.logaddexp(0.0, right_coordinate)
    normalization = np.log(2.0) ** power

    potential_mev = parameters.edge_confinement_energy_mev * (
        left_soft_wall**power + right_soft_wall**power
    ) / normalization
    derivative_mev_per_l_b = (
        parameters.edge_confinement_energy_mev
        * power
        / (normalization * smoothness)
        * (
            -(left_soft_wall ** (power - 1.0)) * expit(left_coordinate)
            + (right_soft_wall ** (power - 1.0)) * expit(right_coordinate)
        )
    )
    magnetic_length = np.sqrt(
        HBAR / (ELEMENTARY_CHARGE * parameters.magnetic_field_tesla)
    )
    return (
        np.asarray(potential_mev * JOULE_PER_MEV, dtype=float),
        np.asarray(derivative_mev_per_l_b * JOULE_PER_MEV / magnetic_length, dtype=float),
    )


def _linear_crossings(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    signs = np.sign(y_values)
    exact = np.flatnonzero(signs == 0.0)
    changes = np.flatnonzero(signs[:-1] * signs[1:] < 0.0)
    roots = [float(x_values[index]) for index in exact]
    for index in changes:
        x_left = x_values[index]
        x_right = x_values[index + 1]
        y_left = y_values[index]
        y_right = y_values[index + 1]
        roots.append(float(x_left - y_left * (x_right - x_left) / (y_right - y_left)))
    if not roots:
        return np.asarray([], dtype=float)
    return np.unique(np.asarray(roots, dtype=float))


def edge_mode_spectrum(
    parameters: IQHEParameters,
    *,
    grid_points: int = 801,
) -> EdgeModeSpectrum:
    """Compute Tong's smooth-edge approximation ``E_n(k)=E_n+V(-k l_B^2)``."""
    if grid_points < 201:
        raise ValueError("grid_points must be at least 201")

    half_width = 0.5 * parameters.edge_sample_width_l_b
    x_over_l_b = np.linspace(-half_width, half_width, grid_points)
    k_l_b = -x_over_l_b
    potential, gradient = _edge_confinement_profile(parameters, x_over_l_b)
    drift_velocity = -gradient / (
        ELEMENTARY_CHARGE * parameters.magnetic_field_tesla
    )
    orbitals, spins, degeneracies, bulk_energies = _branch_spectrum(
        parameters,
        parameters.magnetic_field_tesla,
        parameters.edge_maximum_level + 1,
    )
    energies = bulk_energies[:, np.newaxis] + potential[np.newaxis, :]
    chemical_potential = iqhe_state(parameters).chemical_potential_joule

    crossing_branch_indices: list[int] = []
    crossing_sides: list[int] = []
    crossing_positions: list[float] = []
    crossing_velocities: list[float] = []
    crossing_degeneracies: list[float] = []
    for branch_index, branch_energy in enumerate(energies):
        roots = _linear_crossings(x_over_l_b, branch_energy - chemical_potential)
        for root in roots:
            if abs(root) <= 1.0e-10:
                continue
            crossing_branch_indices.append(branch_index)
            crossing_sides.append(-1 if root < 0.0 else 1)
            crossing_positions.append(root)
            crossing_velocities.append(float(np.interp(root, x_over_l_b, drift_velocity)))
            crossing_degeneracies.append(float(degeneracies[branch_index]))

    branch_indices_array = np.asarray(crossing_branch_indices, dtype=int)
    sides_array = np.asarray(crossing_sides, dtype=int)
    positions_array = np.asarray(crossing_positions, dtype=float)
    velocities_array = np.asarray(crossing_velocities, dtype=float)
    crossing_degeneracies_array = np.asarray(crossing_degeneracies, dtype=float)
    left_count = int(np.rint(np.sum(crossing_degeneracies_array[sides_array < 0])))
    right_count = int(np.rint(np.sum(crossing_degeneracies_array[sides_array > 0])))
    mode_count = min(left_count, right_count)
    hall_voltage_volt = parameters.hall_voltage_microvolt * 1.0e-6

    return EdgeModeSpectrum(
        x_over_l_b=x_over_l_b,
        k_l_b=k_l_b,
        potential_joule=potential,
        potential_gradient_joule_per_m=gradient,
        drift_velocity_m_s=np.asarray(drift_velocity, dtype=float),
        orbital_indices=orbitals,
        spin_projections=spins,
        branch_degeneracies=degeneracies,
        energies_joule=np.asarray(energies, dtype=float),
        chemical_potential_joule=float(chemical_potential),
        crossing_branch_indices=branch_indices_array,
        crossing_sides=sides_array,
        crossing_x_over_l_b=positions_array,
        crossing_k_l_b=-positions_array,
        crossing_velocity_m_s=velocities_array,
        crossing_degeneracies=crossing_degeneracies_array,
        mode_count_per_edge=mode_count,
        edge_current_ampere=float(
            mode_count * CONDUCTANCE_QUANTUM_SIEMENS * hall_voltage_volt
        ),
    )


__all__ = [
    "CONDUCTANCE_QUANTUM_SIEMENS",
    "DensityOfStates",
    "EdgeModeSpectrum",
    "FLUX_QUANTUM_WEBER",
    "IQHEParameters",
    "IQHEState",
    "JOULE_PER_MEV",
    "PLANCK_CONSTANT",
    "PlateauScan",
    "SpinMode",
    "TransportTensor",
    "VON_KLITZING_CONSTANT_OHM",
    "broadened_density_of_states",
    "conductivity_from_resistivity",
    "edge_mode_spectrum",
    "filling_factor",
    "iqhe_state",
    "plateau_scan",
    "quantized_transport_tensor",
]
