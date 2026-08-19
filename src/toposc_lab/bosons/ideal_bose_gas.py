"""Finite-mode ideal Bose gas in a two-dimensional periodic box.

The module deliberately uses the grand-canonical ensemble.  Temperature and
chemical potential are externally imposed, and the expected particle number
therefore responds to them.  In addition to the Bose distribution, this
module constructs a *classical Maxwell--Boltzmann reference gas with the same
mean particle number*.  That reference makes the specifically bosonic effect
visible: bosons accumulate in already favourable low-energy states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field


BOLTZMANN_CONSTANT = 1.380649e-23  # J/K, exact SI value.
REDUCED_PLANCK_CONSTANT = 1.054571817e-34  # J s.
ATOMIC_MASS_UNIT = 1.66053906660e-27  # kg.
RIEMANN_ZETA_THREE_HALVES = 2.612375348685488  # zeta(3/2).


class IdealBoseGasParameters(PydanticBaseModel):
    """External conditions and finite momentum grid of the statistical lab.

    All values use SI units.  The workspace converts its friendly nK, um and
    atomic-mass-unit inputs before instantiating this object.
    """

    temperature: float = Field(default=120.0e-9, gt=0.0)
    chemical_potential: float = Field(
        default=-25.0e-9 * BOLTZMANN_CONSTANT,
        lt=0.0,
        description="Chemical potential mu in Joules; it must lie below epsilon_0=0.",
    )
    box_length: float = Field(default=20.0e-6, gt=0.0)
    particle_mass_amu: float = Field(default=87.0, gt=0.0)
    maximum_mode_index: int = Field(default=7, ge=1, le=30)


class IdealBoseEinsteinCondensationParameters(PydanticBaseModel):
    """Conditions for the three-dimensional homogeneous ideal-gas BEC limit.

    Unlike the two-dimensional finite-box statistics material above, this
    model describes the standard three-dimensional thermodynamic-limit result
    at *fixed total particle number*.  This distinction matters: a uniform
    infinite 2D ideal gas does not have a non-zero-temperature BEC transition.
    """

    temperature: float = Field(default=80.0e-9, gt=0.0)
    particle_number: int = Field(default=100_000, ge=1)
    box_length: float = Field(default=20.0e-6, gt=0.0)
    particle_mass_amu: float = Field(default=87.0, gt=0.0)


@dataclass(frozen=True)
class IdealBoseGasResult:
    """Thermodynamics and all finite-box momentum-state occupations."""

    parameters: IdealBoseGasParameters
    particle_mass_kg: float
    quantum_numbers_x: np.ndarray
    quantum_numbers_y: np.ndarray
    momenta_x: np.ndarray
    momenta_y: np.ndarray
    energies: np.ndarray
    occupations: np.ndarray
    expected_particle_number: float
    mean_energy: float
    entropy_over_kb: float
    number_variance: float

    @property
    def ground_state_occupation(self) -> float:
        """Return the expected occupation of the k=(0,0) mode."""
        ground_index = np.unravel_index(np.argmin(self.energies), self.energies.shape)
        return float(self.occupations[ground_index])

    @property
    def mean_energy_per_particle(self) -> float:
        """Return E/N, or zero for an empty numerical state set."""
        if self.expected_particle_number == 0.0:
            return 0.0
        return float(self.mean_energy / self.expected_particle_number)

    @property
    def occupation_probabilities(self) -> np.ndarray:
        """Return normalized state weights for representative-particle samples."""
        return self.occupations.reshape(-1) / self.expected_particle_number

    @property
    def ground_state_fraction(self) -> float:
        """Return the fraction of the mean population in the k=(0, 0) mode."""
        return float(self.ground_state_occupation / self.expected_particle_number)

    @property
    def effective_number_of_modes(self) -> float:
        """Return 1/sum(p_k**2), an intuitive count of populated modes.

        It equals one if all population is in one momentum state and grows
        when the distribution is spread over many modes.  It is not a new
        thermodynamic observable; it is a compact teaching diagnostic.
        """
        probabilities = self.occupation_probabilities
        return float(1.0 / np.sum(probabilities**2))


@dataclass(frozen=True)
class BoseClassicalComparison:
    """Bose gas together with a classical gas at the same mean population.

    Matching the classical particle number is essential.  Otherwise a change
    could be caused simply by there being more particles rather than by Bose
    statistics.  With this comparison, any difference in the state fractions
    is a difference in statistical law alone.
    """

    bose: IdealBoseGasResult
    classical_occupations: np.ndarray

    @property
    def expected_particle_number(self) -> float:
        """Mean particle number shared by both comparison distributions."""
        return self.bose.expected_particle_number

    @property
    def ground_state_occupation(self) -> float:
        """Bose occupation of the k=(0, 0) mode, for UI compatibility."""
        return self.bose.ground_state_occupation

    @property
    def classical_ground_state_occupation(self) -> float:
        """Classical occupation of the k=(0, 0) mode at the same mean N."""
        ground_index = np.unravel_index(
            np.argmin(self.bose.energies), self.bose.energies.shape
        )
        return float(self.classical_occupations[ground_index])

    @property
    def bose_ground_state_fraction(self) -> float:
        """Fraction of bosons in the ground momentum state."""
        return self.bose.ground_state_fraction

    @property
    def classical_ground_state_fraction(self) -> float:
        """Ground-state fraction predicted by classical statistics."""
        return float(self.classical_ground_state_occupation / self.expected_particle_number)

    @property
    def ground_state_enhancement(self) -> float:
        """Bose ground-state fraction divided by the classical fraction."""
        return float(
            self.bose_ground_state_fraction / self.classical_ground_state_fraction
        )


@dataclass(frozen=True)
class IdealBoseEinsteinCondensationResult:
    """Analytic 3D ideal-gas Bose--Einstein-condensation observables."""

    parameters: IdealBoseEinsteinCondensationParameters
    particle_mass_kg: float
    volume: float
    number_density: float
    critical_temperature: float
    condensate_number: float
    thermal_number: float

    @property
    def condensate_fraction(self) -> float:
        """Return the fraction of all particles in the k=0 condensate."""
        return float(self.condensate_number / self.parameters.particle_number)

    @property
    def reduced_temperature(self) -> float:
        """Return T/T_c, the natural control variable of this transition."""
        return float(self.parameters.temperature / self.critical_temperature)


def _positive_finite(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def bose_einstein_occupation(
    energies: np.ndarray,
    temperature: float,
    chemical_potential: float,
) -> np.ndarray:
    """Evaluate n_k = 1 / (exp(beta (epsilon_k-mu)) - 1) mode by mode."""
    values = np.asarray(energies, dtype=float)
    temperature_value = _positive_finite("temperature", temperature)

    if values.ndim == 0 or values.size == 0:
        raise ValueError("energies must be a non-empty array")
    if not np.all(np.isfinite(values)):
        raise ValueError("energies must be finite")
    if np.any(values <= chemical_potential):
        raise ValueError("chemical_potential must be below every included energy")

    exponent = (values - chemical_potential) / (BOLTZMANN_CONSTANT * temperature_value)
    occupations = np.zeros_like(exponent)
    finite_exponent = exponent < 700.0
    occupations[finite_exponent] = 1.0 / np.expm1(exponent[finite_exponent])
    return occupations


def maxwell_boltzmann_occupation(
    energies: np.ndarray,
    temperature: float,
    expected_particle_number: float,
) -> np.ndarray:
    """Return a classical occupation distribution normalized to a given mean N.

    The Maxwell--Boltzmann law is ``n_k proportional to exp(-beta epsilon_k)``.
    The normalization is chosen such that its sum equals the Bose-gas mean
    population.  This isolates the effect of the statistics from the trivial
    effect of changing the total amount of gas.
    """
    values = np.asarray(energies, dtype=float)
    temperature_value = _positive_finite("temperature", temperature)
    population = _positive_finite("expected_particle_number", expected_particle_number)
    if values.ndim == 0 or values.size == 0:
        raise ValueError("energies must be a non-empty array")
    if not np.all(np.isfinite(values)):
        raise ValueError("energies must be finite")

    shifted_energies = values - np.min(values)
    weights = np.exp(-shifted_energies / (BOLTZMANN_CONSTANT * temperature_value))
    return population * weights / np.sum(weights)


def analyze_ideal_bose_gas(params: IdealBoseGasParameters) -> IdealBoseGasResult:
    """Construct box modes and evaluate Bose statistics for the given T and mu.

    A periodic square box has k_i=2 pi n_i/L and epsilon_k=hbar^2 k^2/(2m).
    The finite cutoff is an explicit teaching device: it makes every state
    visible and avoids claiming a thermodynamic-limit phase transition.
    """
    if not isinstance(params, IdealBoseGasParameters):
        raise TypeError("params must be an IdealBoseGasParameters instance")

    mass = params.particle_mass_amu * ATOMIC_MASS_UNIT
    mode_indices = np.arange(-params.maximum_mode_index, params.maximum_mode_index + 1)
    quantum_numbers_x, quantum_numbers_y = np.meshgrid(
        mode_indices,
        mode_indices,
        indexing="xy",
    )
    momentum_unit = 2.0 * np.pi * REDUCED_PLANCK_CONSTANT / params.box_length
    momenta_x = momentum_unit * quantum_numbers_x
    momenta_y = momentum_unit * quantum_numbers_y
    energies = (momenta_x**2 + momenta_y**2) / (2.0 * mass)
    occupations = bose_einstein_occupation(
        energies,
        params.temperature,
        params.chemical_potential,
    )

    expected_particle_number = float(np.sum(occupations))
    mean_energy = float(np.sum(energies * occupations))
    entropy_terms = (1.0 + occupations) * np.log1p(occupations)
    positive_occupations = occupations > 0.0
    entropy_terms[positive_occupations] -= (
        occupations[positive_occupations] * np.log(occupations[positive_occupations])
    )

    return IdealBoseGasResult(
        parameters=params,
        particle_mass_kg=mass,
        quantum_numbers_x=quantum_numbers_x,
        quantum_numbers_y=quantum_numbers_y,
        momenta_x=momenta_x,
        momenta_y=momenta_y,
        energies=energies,
        occupations=occupations,
        expected_particle_number=expected_particle_number,
        mean_energy=mean_energy,
        entropy_over_kb=float(np.sum(entropy_terms)),
        number_variance=float(np.sum(occupations * (1.0 + occupations))),
    )


def compare_bose_to_classical(
    params: IdealBoseGasParameters,
) -> BoseClassicalComparison:
    """Evaluate Bose and equal-population classical distributions side by side."""
    bose = analyze_ideal_bose_gas(params)
    return BoseClassicalComparison(
        bose=bose,
        classical_occupations=maxwell_boltzmann_occupation(
            bose.energies,
            params.temperature,
            bose.expected_particle_number,
        ),
    )


def ideal_bec_critical_temperature(
    *,
    particle_number: int,
    box_length: float,
    particle_mass_amu: float,
) -> float:
    """Return the 3D ideal-gas critical temperature in Kelvin.

    For number density ``n=N/V`` the standard thermodynamic-limit result is

    ``T_c = 2 pi hbar^2/(m k_B) * (n/zeta(3/2))^(2/3)``.
    """
    if particle_number <= 0:
        raise ValueError("particle_number must be positive")
    length = _positive_finite("box_length", box_length)
    mass_amu = _positive_finite("particle_mass_amu", particle_mass_amu)
    density = particle_number / length**3
    mass = mass_amu * ATOMIC_MASS_UNIT
    return float(
        (2.0 * np.pi * REDUCED_PLANCK_CONSTANT**2 / (mass * BOLTZMANN_CONSTANT))
        * (density / RIEMANN_ZETA_THREE_HALVES) ** (2.0 / 3.0)
    )


def ideal_bec_condensate_fraction(
    temperature: float,
    critical_temperature: float,
) -> float:
    """Return N_0/N for a three-dimensional homogeneous ideal Bose gas.

    Below ``T_c``, thermal excited states can hold only
    ``N(T/T_c)^(3/2)`` particles; the remainder occupies k=0.  Above ``T_c``
    the condensate fraction is zero.
    """
    temperature_value = _positive_finite("temperature", temperature)
    critical_temperature_value = _positive_finite(
        "critical_temperature", critical_temperature
    )
    reduced_temperature = temperature_value / critical_temperature_value
    return float(max(0.0, 1.0 - reduced_temperature ** 1.5))


def analyze_ideal_bose_einstein_condensation(
    params: IdealBoseEinsteinCondensationParameters,
) -> IdealBoseEinsteinCondensationResult:
    """Evaluate the analytic condensate and thermal-cloud populations."""
    if not isinstance(params, IdealBoseEinsteinCondensationParameters):
        raise TypeError("params must be an IdealBoseEinsteinCondensationParameters instance")

    critical_temperature = ideal_bec_critical_temperature(
        particle_number=params.particle_number,
        box_length=params.box_length,
        particle_mass_amu=params.particle_mass_amu,
    )
    condensate_fraction = ideal_bec_condensate_fraction(
        params.temperature,
        critical_temperature,
    )
    condensate_number = params.particle_number * condensate_fraction
    volume = params.box_length**3
    return IdealBoseEinsteinCondensationResult(
        parameters=params,
        particle_mass_kg=params.particle_mass_amu * ATOMIC_MASS_UNIT,
        volume=volume,
        number_density=params.particle_number / volume,
        critical_temperature=critical_temperature,
        condensate_number=condensate_number,
        thermal_number=params.particle_number - condensate_number,
    )


def representative_particles(
    result: IdealBoseGasResult,
    *,
    n_particles: int,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample initial positions and momenta from the calculated one-body state.

    Positions are uniform because a homogeneous ideal gas has translational
    symmetry.  Momentum modes are sampled with probability n_k / sum_q n_q.
    The samples are for visualization only; the exact physics remains in the
    expectation values stored in :class:`IdealBoseGasResult`.
    """
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")

    random_generator = np.random.default_rng(seed)
    mode_indices = random_generator.choice(
        result.occupations.size,
        size=n_particles,
        p=result.occupation_probabilities,
    )
    flattened_momenta = np.column_stack(
        (result.momenta_x.reshape(-1), result.momenta_y.reshape(-1))
    )
    positions = random_generator.uniform(
        0.0,
        result.parameters.box_length,
        size=(n_particles, 2),
    )
    return positions, flattened_momenta[mode_indices]
