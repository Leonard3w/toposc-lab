"""Explicit ensemble calculations for ideal classical and Bose gases.

The module keeps the ensembles genuinely distinct:

* canonical: fixed temperature, particle number and volume;
* grand canonical: fixed temperature, chemical potential and volume;
* microcanonical: fixed total energy, particle number and volume.

Classical dynamics is represented by ballistic particles because an ideal
classical gas has well-defined positions and momenta.  The Bose
microcanonical calculator instead enumerates finite-system Fock states; it
does not invent classical trajectories for a stationary quantum equilibrium
state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field

from toposc_lab.gases.ideal_quantum_gases import (
    ATOMIC_MASS_UNIT,
    BOLTZMANN_CONSTANT,
    REDUCED_PLANCK_CONSTANT,
    _occupation_from_reduced_energy,
    _quantum_integral,
    thermal_de_broglie_wavelength,
)


class ClassicalCanonicalParameters(PydanticBaseModel):
    """Fixed-N, fixed-T classical ideal gas in a periodic cubic box."""

    temperature: float = Field(default=80.0e-9, gt=0.0)
    particle_number: int = Field(default=200, ge=2, le=100_000)
    box_length: float = Field(default=20.0e-6, gt=0.0)
    particle_mass_amu: float = Field(default=87.0, gt=0.0)


class ClassicalGrandCanonicalParameters(PydanticBaseModel):
    """Fixed-T, fixed-mu classical ideal gas in a periodic cubic box."""

    temperature: float = Field(default=80.0e-9, gt=0.0)
    chemical_potential: float = Field(default=-300.0e-9 * BOLTZMANN_CONSTANT)
    box_length: float = Field(default=20.0e-6, gt=0.0)
    particle_mass_amu: float = Field(default=87.0, gt=0.0)


class ClassicalMicrocanonicalParameters(PydanticBaseModel):
    """Fixed-E, fixed-N classical ideal gas in a periodic cubic box."""

    particle_number: int = Field(default=200, ge=2, le=100_000)
    total_energy: float = Field(default=24_000.0e-9 * BOLTZMANN_CONSTANT, gt=0.0)
    box_length: float = Field(default=20.0e-6, gt=0.0)
    particle_mass_amu: float = Field(default=87.0, gt=0.0)


@dataclass(frozen=True)
class ClassicalEnsembleResult:
    """Thermodynamics of an ideal classical gas in one specified ensemble."""

    ensemble: str
    temperature: float
    particle_number_mean: float
    particle_number_fixed: int | None
    number_variance: float
    total_energy_mean: float
    energy_variance: float
    total_energy_fixed: float | None
    chemical_potential: float
    volume: float
    box_length: float
    particle_mass_kg: float
    thermal_wavelength: float
    phase_space_density: float
    entropy_over_kb: float


@dataclass(frozen=True)
class ClassicalParticleSample:
    """A representative projected particle sample for the classical animation."""

    positions: np.ndarray
    velocities: np.ndarray


class BoseGrandCanonicalParameters(PydanticBaseModel):
    """Fixed-T, fixed-mu homogeneous three-dimensional ideal Bose gas.

    ``mu`` must be strictly below the ground-state energy, set to zero here.
    This describes the normal Bose gas approaching condensation from above.
    At mu=0 the thermodynamic grand-canonical condensate population is not
    determined by the reservoir alone, so that singular point is excluded.
    """

    temperature: float = Field(default=80.0e-9, gt=0.0)
    chemical_potential: float = Field(
        default=-15.0e-9 * BOLTZMANN_CONSTANT,
        lt=0.0,
    )
    box_length: float = Field(default=20.0e-6, gt=0.0)
    particle_mass_amu: float = Field(default=87.0, gt=0.0)


@dataclass(frozen=True)
class BoseGrandCanonicalResult:
    """Normal-phase grand-canonical ideal Bose-gas observables."""

    parameters: BoseGrandCanonicalParameters
    particle_mass_kg: float
    volume: float
    thermal_wavelength: float
    mean_particle_number: float
    number_variance: float
    mean_energy: float
    entropy_over_kb: float
    ground_state_occupation: float

    @property
    def reduced_chemical_potential(self) -> float:
        """Return mu/(k_B T)."""
        return float(
            self.parameters.chemical_potential
            / (BOLTZMANN_CONSTANT * self.parameters.temperature)
        )


class BoseMicrocanonicalParameters(PydanticBaseModel):
    """Small finite Bose system exactly enumerated at fixed N and energy.

    The system is a one-dimensional periodic mode set with energies
    epsilon_n=n^2 epsilon_1.  Its deliberately small limits make exact Fock
    state counting practical and transparent.
    """

    particle_number: int = Field(default=8, ge=1, le=14)
    energy_quanta: int = Field(default=8, ge=0, le=40)
    maximum_mode_index: int = Field(default=4, ge=1, le=6)
    box_length: float = Field(default=20.0e-6, gt=0.0)
    particle_mass_amu: float = Field(default=87.0, gt=0.0)


@dataclass(frozen=True)
class BoseMicrocanonicalResult:
    """Uniform average over exact finite-system Bose Fock states at fixed E,N."""

    parameters: BoseMicrocanonicalParameters
    mode_quantum_numbers: np.ndarray
    mode_energy_quanta: np.ndarray
    mean_occupations: np.ndarray
    microstate_count: int
    entropy_over_kb: float
    energy_quantum: float
    inferred_temperature: float | None


def _classical_thermodynamic_values(
    *,
    temperature: float,
    particle_number: float,
    box_length: float,
    particle_mass_amu: float,
) -> tuple[float, float, float, float, float]:
    """Return volume, mass, lambda_T, n lambda_T^3 and mu for classical gas."""
    volume = box_length**3
    mass = particle_mass_amu * ATOMIC_MASS_UNIT
    wavelength = thermal_de_broglie_wavelength(temperature, mass)
    phase_space_density = particle_number / volume * wavelength**3
    chemical_potential = BOLTZMANN_CONSTANT * temperature * np.log(phase_space_density)
    return volume, mass, wavelength, phase_space_density, chemical_potential


def analyze_classical_canonical(
    parameters: ClassicalCanonicalParameters,
) -> ClassicalEnsembleResult:
    """Calculate exact ideal-gas canonical means and fluctuations."""
    particle_number = float(parameters.particle_number)
    volume, mass, wavelength, phase_density, chemical_potential = _classical_thermodynamic_values(
        temperature=parameters.temperature,
        particle_number=particle_number,
        box_length=parameters.box_length,
        particle_mass_amu=parameters.particle_mass_amu,
    )
    thermal_energy = 1.5 * particle_number * BOLTZMANN_CONSTANT * parameters.temperature
    return ClassicalEnsembleResult(
        ensemble="canonical",
        temperature=parameters.temperature,
        particle_number_mean=particle_number,
        particle_number_fixed=parameters.particle_number,
        number_variance=0.0,
        total_energy_mean=thermal_energy,
        energy_variance=1.5 * particle_number * (BOLTZMANN_CONSTANT * parameters.temperature) ** 2,
        total_energy_fixed=None,
        chemical_potential=float(chemical_potential),
        volume=volume,
        box_length=parameters.box_length,
        particle_mass_kg=mass,
        thermal_wavelength=wavelength,
        phase_space_density=phase_density,
        entropy_over_kb=particle_number * (2.5 - np.log(phase_density)),
    )


def analyze_classical_grand_canonical(
    parameters: ClassicalGrandCanonicalParameters,
) -> ClassicalEnsembleResult:
    """Calculate a classical grand-canonical gas with Poisson number statistics."""
    mass = parameters.particle_mass_amu * ATOMIC_MASS_UNIT
    volume = parameters.box_length**3
    wavelength = thermal_de_broglie_wavelength(parameters.temperature, mass)
    alpha = parameters.chemical_potential / (BOLTZMANN_CONSTANT * parameters.temperature)
    phase_density = float(np.exp(alpha))
    particle_number = volume / wavelength**3 * phase_density
    thermal_energy = 1.5 * particle_number * BOLTZMANN_CONSTANT * parameters.temperature
    return ClassicalEnsembleResult(
        ensemble="grand_canonical",
        temperature=parameters.temperature,
        particle_number_mean=float(particle_number),
        particle_number_fixed=None,
        number_variance=float(particle_number),
        total_energy_mean=float(thermal_energy),
        energy_variance=float(3.75 * particle_number * (BOLTZMANN_CONSTANT * parameters.temperature) ** 2),
        total_energy_fixed=None,
        chemical_potential=parameters.chemical_potential,
        volume=volume,
        box_length=parameters.box_length,
        particle_mass_kg=mass,
        thermal_wavelength=wavelength,
        phase_space_density=phase_density,
        entropy_over_kb=float(particle_number * (2.5 - alpha)),
    )


def analyze_classical_microcanonical(
    parameters: ClassicalMicrocanonicalParameters,
) -> ClassicalEnsembleResult:
    """Calculate a finite-N microcanonical ideal gas and its inferred temperature."""
    particle_number = float(parameters.particle_number)
    # From rho(E) proportional to E^(3N/2-1), using 1/T=dS/dE.
    temperature = parameters.total_energy / (
        BOLTZMANN_CONSTANT * (1.5 * particle_number - 1.0)
    )
    volume, mass, wavelength, phase_density, chemical_potential = _classical_thermodynamic_values(
        temperature=temperature,
        particle_number=particle_number,
        box_length=parameters.box_length,
        particle_mass_amu=parameters.particle_mass_amu,
    )
    return ClassicalEnsembleResult(
        ensemble="microcanonical",
        temperature=float(temperature),
        particle_number_mean=particle_number,
        particle_number_fixed=parameters.particle_number,
        number_variance=0.0,
        total_energy_mean=parameters.total_energy,
        energy_variance=0.0,
        total_energy_fixed=parameters.total_energy,
        chemical_potential=float(chemical_potential),
        volume=volume,
        box_length=parameters.box_length,
        particle_mass_kg=mass,
        thermal_wavelength=wavelength,
        phase_space_density=phase_density,
        # Thermodynamic-limit Sackur--Tetrode value at the temperature inferred
        # from the fixed energy.  This avoids taking a logarithm of a
        # dimensionful energy and is the appropriate extensive comparison to
        # the canonical calculation.
        entropy_over_kb=float(particle_number * (2.5 - np.log(phase_density))),
    )


def sample_classical_particles(
    result: ClassicalEnsembleResult,
    *,
    visible_particle_count: int = 150,
    seed: int = 17,
) -> ClassicalParticleSample:
    """Draw representative classical positions and velocities for visualisation.

    For the microcanonical case, the displayed sample is rescaled so that its
    kinetic energy per particle is exactly the system value.  Canonical and
    grand-canonical samples show their physically allowed thermal energy
    fluctuations.
    """
    if visible_particle_count <= 0:
        raise ValueError("visible_particle_count must be positive")
    generator = np.random.default_rng(seed)
    if result.particle_number_fixed is not None:
        count = min(visible_particle_count, result.particle_number_fixed)
    else:
        count = visible_particle_count
    positions = generator.uniform(0.0, result.box_length, size=(count, 3))
    velocity_scale = np.sqrt(BOLTZMANN_CONSTANT * result.temperature / result.particle_mass_kg)
    velocities = generator.normal(0.0, velocity_scale, size=(count, 3))
    if result.ensemble == "microcanonical":
        visible_energy = result.total_energy_mean * count / result.particle_number_mean
        velocities *= np.sqrt(
            2.0 * visible_energy
            / (result.particle_mass_kg * np.sum(velocities**2))
        )
    return ClassicalParticleSample(positions=positions, velocities=velocities)


def analyze_bose_grand_canonical(
    parameters: BoseGrandCanonicalParameters,
) -> BoseGrandCanonicalResult:
    """Calculate the normal Bose gas at fixed T, mu and V in three dimensions."""
    mass = parameters.particle_mass_amu * ATOMIC_MASS_UNIT
    volume = parameters.box_length**3
    wavelength = thermal_de_broglie_wavelength(parameters.temperature, mass)
    alpha = parameters.chemical_potential / (BOLTZMANN_CONSTANT * parameters.temperature)
    prefactor = volume / wavelength**3
    occupation_integral = _quantum_integral(1.5, alpha, "boson")
    mean_particle_number = prefactor * occupation_integral
    number_variance = prefactor * _quantum_integral(0.5, alpha, "boson")
    mean_energy = 1.5 * BOLTZMANN_CONSTANT * parameters.temperature * prefactor * _quantum_integral(
        2.5,
        alpha,
        "boson",
    )
    entropy = 2.5 * prefactor * _quantum_integral(2.5, alpha, "boson") - alpha * mean_particle_number
    ground_occupation = float(1.0 / np.expm1(-alpha))
    return BoseGrandCanonicalResult(
        parameters=parameters,
        particle_mass_kg=mass,
        volume=volume,
        thermal_wavelength=wavelength,
        mean_particle_number=float(mean_particle_number),
        number_variance=float(number_variance),
        mean_energy=float(mean_energy),
        entropy_over_kb=float(entropy),
        ground_state_occupation=ground_occupation,
    )


def bose_grand_canonical_momentum_slice(
    result: BoseGrandCanonicalResult,
    *,
    maximum_mode_index: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the n_z=0 slice of grand-canonical Bose state occupations."""
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
        "boson",
    )
    return n_x, n_y, reduced_energies, occupations


def _microcanonical_bose_count(
    particle_number: int,
    energy_quanta: int,
    mode_energy_quanta: np.ndarray,
    *,
    collect_occupations: bool,
) -> tuple[int, np.ndarray | None]:
    """Exactly count Fock states satisfying sum n_i=N and sum n_i e_i=E."""
    state = np.zeros(len(mode_energy_quanta), dtype=int)
    occupation_sum = np.zeros(len(mode_energy_quanta), dtype=float)
    count = 0
    maximum_count = 300_000

    def recurse(mode_index: int, remaining_particles: int, remaining_energy: int) -> None:
        nonlocal count
        if count > maximum_count:
            raise ValueError(
                "too many microstates for exact enumeration; reduce N, energy or mode cutoff"
            )
        if mode_index == len(mode_energy_quanta) - 1:
            energy = int(mode_energy_quanta[mode_index])
            if energy == 0:
                valid = remaining_energy == 0
                occupancy = remaining_particles
            else:
                valid = remaining_energy == remaining_particles * energy
                occupancy = remaining_particles
            if valid:
                state[mode_index] = occupancy
                count += 1
                if collect_occupations:
                    occupation_sum[:] += state
            return

        energy = int(mode_energy_quanta[mode_index])
        if energy == 0:
            allowed_occupancies = range(remaining_particles + 1)
        else:
            allowed_occupancies = range(min(remaining_particles, remaining_energy // energy) + 1)
        for occupancy in allowed_occupancies:
            state[mode_index] = occupancy
            recurse(
                mode_index + 1,
                remaining_particles - occupancy,
                remaining_energy - occupancy * energy,
            )
        state[mode_index] = 0

    recurse(0, particle_number, energy_quanta)
    return count, occupation_sum if collect_occupations else None


def analyze_bose_microcanonical(
    parameters: BoseMicrocanonicalParameters,
) -> BoseMicrocanonicalResult:
    """Exactly enumerate a small one-dimensional Bose microcanonical ensemble."""
    labels = np.array(
        [0, *[sign * index for index in range(1, parameters.maximum_mode_index + 1) for sign in (-1, 1)]],
        dtype=int,
    )
    energy_quanta = labels**2
    count, occupation_sum = _microcanonical_bose_count(
        parameters.particle_number,
        parameters.energy_quanta,
        energy_quanta,
        collect_occupations=True,
    )
    if count == 0 or occupation_sum is None:
        raise ValueError("no Bose Fock state exists for the requested N and energy")
    mass = parameters.particle_mass_amu * ATOMIC_MASS_UNIT
    energy_quantum = (2.0 * np.pi * REDUCED_PLANCK_CONSTANT / parameters.box_length) ** 2 / (2.0 * mass)

    inferred_temperature: float | None = None
    if parameters.energy_quanta >= 1:
        lower_count, _ = _microcanonical_bose_count(
            parameters.particle_number,
            parameters.energy_quanta - 1,
            energy_quanta,
            collect_occupations=False,
        )
        upper_count, _ = _microcanonical_bose_count(
            parameters.particle_number,
            parameters.energy_quanta + 1,
            energy_quanta,
            collect_occupations=False,
        )
        if lower_count > 0 and upper_count > 0:
            entropy_slope = 0.5 * (np.log(upper_count) - np.log(lower_count))
            if entropy_slope > 0.0:
                inferred_temperature = float(energy_quantum / (BOLTZMANN_CONSTANT * entropy_slope))

    return BoseMicrocanonicalResult(
        parameters=parameters,
        mode_quantum_numbers=labels,
        mode_energy_quanta=energy_quanta,
        mean_occupations=occupation_sum / count,
        microstate_count=count,
        entropy_over_kb=float(np.log(count)),
        energy_quantum=float(energy_quantum),
        inferred_temperature=inferred_temperature,
    )
