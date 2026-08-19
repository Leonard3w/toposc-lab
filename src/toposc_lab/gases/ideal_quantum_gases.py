"""Consistent three-dimensional ideal classical and Bose gases.

All three calculators use the same homogeneous cubic box, temperature, mass
and *fixed total particle number*.  The chemical potential is solved from the
number equation rather than entered independently.  This makes a comparison
between the three statistical laws physically meaningful.

The thermodynamics use the standard three-dimensional thermodynamic-limit
formulae.  ``momentum_state_slice`` provides a finite n_z=0 box-mode slice for
visualization only; it does not secretly set the thermodynamic cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gamma


BOLTZMANN_CONSTANT = 1.380649e-23  # J/K, exact SI value.
REDUCED_PLANCK_CONSTANT = 1.054571817e-34  # J s.
ATOMIC_MASS_UNIT = 1.66053906660e-27  # kg.
ZETA_THREE_HALVES = 2.612375348685488

GasStatistic = Literal["classical", "boson"]


class IdealQuantumGasParameters(PydanticBaseModel):
    """External conditions of a one-component, homogeneous 3D ideal gas."""

    temperature: float = Field(default=80.0e-9, gt=0.0)
    particle_number: int = Field(default=100_000, ge=1)
    box_length: float = Field(default=20.0e-6, gt=0.0)
    particle_mass_amu: float = Field(default=87.0, gt=0.0)


@dataclass(frozen=True)
class QuantumGasResult:
    """Thermodynamic state obtained at a fixed total particle number."""

    statistic: GasStatistic
    parameters: IdealQuantumGasParameters
    particle_mass_kg: float
    volume: float
    number_density: float
    thermal_wavelength: float
    phase_space_density: float
    chemical_potential: float
    fugacity: float
    thermal_particle_number: float
    condensate_particle_number: float
    mean_energy: float
    entropy_over_kb: float
    critical_temperature: float | None

    @property
    def reduced_chemical_potential(self) -> float:
        """Return mu/(k_B T), the parameter directly used in occupations."""
        return float(
            self.chemical_potential
            / (BOLTZMANN_CONSTANT * self.parameters.temperature)
        )

    @property
    def condensate_fraction(self) -> float:
        """Return N0/N; it is non-zero only for the Bose calculator."""
        return float(
            self.condensate_particle_number / self.parameters.particle_number
        )

    @property
    def mean_energy_per_particle(self) -> float:
        """Return the total mean energy divided by total particle number."""
        return float(self.mean_energy / self.parameters.particle_number)

    @property
    def classical_regime(self) -> bool:
        """Whether n lambda_T^3 < 0.1, a conservative classical-gas criterion."""
        return self.phase_space_density < 0.1


def _positive_finite(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def thermal_de_broglie_wavelength(temperature: float, mass_kg: float) -> float:
    """Return lambda_T = sqrt(2 pi hbar^2/(m k_B T))."""
    temperature_value = _positive_finite("temperature", temperature)
    mass_value = _positive_finite("mass_kg", mass_kg)
    return float(
        np.sqrt(
            2.0
            * np.pi
            * REDUCED_PLANCK_CONSTANT**2
            / (mass_value * BOLTZMANN_CONSTANT * temperature_value)
        )
    )


def _occupation_from_reduced_energy(
    reduced_energy: np.ndarray | float,
    reduced_chemical_potential: float,
    statistic: GasStatistic,
) -> np.ndarray:
    """Evaluate the one-state occupation as a function of epsilon/(k_B T)."""
    energy = np.asarray(reduced_energy, dtype=float)
    exponent = energy - reduced_chemical_potential
    if statistic == "classical":
        return np.exp(np.clip(-exponent, -745.0, 700.0))
    if statistic == "boson":
        if np.any(exponent < 0.0):
            raise ValueError("Bose chemical potential cannot lie above the ground state")
        result = np.zeros_like(exponent)
        finite = exponent < 700.0
        nonzero = finite & (exponent > 0.0)
        result[nonzero] = 1.0 / np.expm1(exponent[nonzero])
        return result
    raise ValueError(f"unknown statistic: {statistic}")


def _quantum_integral(
    order: float,
    reduced_chemical_potential: float,
    statistic: GasStatistic,
) -> float:
    """Return the dimensionless Bose integral g_order(z).

    The normalisation is chosen such that the number equation is
    ``n lambda_T^3 = f_(3/2)(z)``.  The Maxwell--Boltzmann value is simply
    ``exp(mu/(k_B T))``.
    """
    if statistic == "classical":
        return float(np.exp(reduced_chemical_potential))

    def integrand(energy: float) -> float:
        if energy == 0.0 and statistic == "boson" and reduced_chemical_potential == 0.0:
            # The value at this isolated integration endpoint does not affect
            # the integral; its limiting singularity is integrable for order>1.
            return 0.0
        occupation = _occupation_from_reduced_energy(
            energy,
            reduced_chemical_potential,
            statistic,
        )
        return float(energy ** (order - 1.0) * occupation / gamma(order))

    value, _ = quad(integrand, 0.0, np.inf, epsabs=1.0e-9, epsrel=1.0e-9, limit=200)
    return float(value)


def _solve_reduced_chemical_potential(
    phase_space_density: float,
    statistic: GasStatistic,
) -> tuple[float, float, float]:
    """Solve n lambda_T^3=f_3/2(z), returning alpha, f3/2 and N0/N."""
    if statistic == "classical":
        alpha = float(np.log(phase_space_density))
        return alpha, phase_space_density, 0.0

    if statistic == "boson":
        if phase_space_density >= ZETA_THREE_HALVES:
            return 0.0, ZETA_THREE_HALVES, float(
                1.0 - ZETA_THREE_HALVES / phase_space_density
            )
        root = brentq(
            lambda alpha: _quantum_integral(1.5, alpha, "boson") - phase_space_density,
            -60.0,
            -1.0e-12,
            xtol=1.0e-11,
        )
        return float(root), phase_space_density, 0.0

    raise ValueError(f"unknown statistic: {statistic}")


def bose_critical_temperature(number_density: float, mass_kg: float) -> float:
    """Return the 3D homogeneous ideal-Bose-gas critical temperature."""
    density = _positive_finite("number_density", number_density)
    mass = _positive_finite("mass_kg", mass_kg)
    return float(
        2.0
        * np.pi
        * REDUCED_PLANCK_CONSTANT**2
        / (mass * BOLTZMANN_CONSTANT)
        * (density / ZETA_THREE_HALVES) ** (2.0 / 3.0)
    )


def analyze_ideal_quantum_gas(
    parameters: IdealQuantumGasParameters,
    statistic: GasStatistic,
) -> QuantumGasResult:
    """Solve classical or Bose ideal-gas statistics at fixed N, T, L and m."""
    if not isinstance(parameters, IdealQuantumGasParameters):
        raise TypeError("parameters must be an IdealQuantumGasParameters instance")

    volume = parameters.box_length**3
    number_density = parameters.particle_number / volume
    mass = parameters.particle_mass_amu * ATOMIC_MASS_UNIT
    thermal_wavelength = thermal_de_broglie_wavelength(parameters.temperature, mass)
    phase_space_density = number_density * thermal_wavelength**3
    alpha, thermal_phase_space_density, condensate_fraction = _solve_reduced_chemical_potential(
        phase_space_density,
        statistic,
    )
    thermal_particle_number = (
        parameters.particle_number * thermal_phase_space_density / phase_space_density
    )
    condensate_particle_number = parameters.particle_number * condensate_fraction
    energy_integral = _quantum_integral(2.5, alpha, statistic)
    thermal_prefactor = volume / thermal_wavelength**3
    mean_energy = 1.5 * BOLTZMANN_CONSTANT * parameters.temperature * thermal_prefactor * energy_integral
    entropy_over_kb = 2.5 * thermal_prefactor * energy_integral - alpha * thermal_particle_number

    return QuantumGasResult(
        statistic=statistic,
        parameters=parameters,
        particle_mass_kg=mass,
        volume=volume,
        number_density=number_density,
        thermal_wavelength=thermal_wavelength,
        phase_space_density=phase_space_density,
        chemical_potential=alpha * BOLTZMANN_CONSTANT * parameters.temperature,
        fugacity=float(np.exp(min(alpha, 700.0))),
        thermal_particle_number=float(thermal_particle_number),
        condensate_particle_number=float(condensate_particle_number),
        mean_energy=float(mean_energy),
        entropy_over_kb=float(entropy_over_kb),
        critical_temperature=(
            bose_critical_temperature(number_density, mass)
            if statistic == "boson"
            else None
        ),
    )


def momentum_state_slice(
    result: QuantumGasResult,
    *,
    maximum_mode_index: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return n_z=0 mode labels, reduced energies and mean state occupations.

    In a Bose-condensed state, the k=0 cell contains the macroscopic N0.  The
    remaining cells show the thermal one-state occupation.  This function is a
    visualization slice; thermodynamic values in ``QuantumGasResult`` remain
    cutoff independent.
    """
    if maximum_mode_index < 1:
        raise ValueError("maximum_mode_index must be at least one")
    labels = np.arange(-maximum_mode_index, maximum_mode_index + 1)
    n_x, n_y = np.meshgrid(labels, labels, indexing="xy")
    momentum_unit = 2.0 * np.pi * REDUCED_PLANCK_CONSTANT / result.parameters.box_length
    energies = momentum_unit**2 * (n_x**2 + n_y**2) / (2.0 * result.particle_mass_kg)
    reduced_energies = energies / (BOLTZMANN_CONSTANT * result.parameters.temperature)
    occupations = _occupation_from_reduced_energy(
        reduced_energies,
        result.reduced_chemical_potential,
        result.statistic,
    )
    if result.statistic == "boson" and result.condensate_particle_number > 0.0:
        centre = maximum_mode_index
        occupations[centre, centre] = result.condensate_particle_number
    return n_x, n_y, reduced_energies, occupations
