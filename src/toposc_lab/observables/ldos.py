from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from toposc_lab.core.results import BasisLayout, SimulationResult
from toposc_lab.observables.results import ObservableRecord


@dataclass(frozen=True)
class LocalDensityOfStates:
    """Energy- and site-resolved local density of states."""

    energy_values: np.ndarray
    values: np.ndarray
    component_values: np.ndarray
    component_labels: tuple[str, ...]

    def to_observable_record(self) -> ObservableRecord:
        """Return a standardized numerical LDOS record."""
        return ObservableRecord(
            kind="local_density_of_states",
            arrays={
                "energy_values": self.energy_values,
                "values": self.values,
                "component_values": self.component_values,
            },
            metadata={"component_labels": self.component_labels},
        )


def local_density_of_states(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    basis_layout: BasisLayout,
    energy_values: np.ndarray,
    broadening: float,
) -> LocalDensityOfStates:
    r"""Calculate the Lorentzian-broadened LDOS for an arbitrary site layout.

    The site-resolved quantity is

    ``rho_i(E) = sum_n sum_a |psi_n(i, a)|^2 L_eta(E - E_n)``,

    where ``a`` runs over internal components and ``L_eta`` is a normalized
    Lorentzian. No coordinates, neighbor relations, or lattice type enter the
    calculation. For an irregular geometry, use ``spatial_shape=(n_sites,)``;
    regular multidimensional lattices may retain their grid shape.
    """
    spectrum = np.asarray(eigenvalues, dtype=float)
    vectors = np.asarray(eigenvectors, dtype=complex)
    energies = np.asarray(energy_values, dtype=float)

    if spectrum.ndim != 1 or spectrum.size == 0:
        raise ValueError("eigenvalues must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(spectrum)):
        raise ValueError("eigenvalues must contain only finite values")
    if vectors.ndim != 2:
        raise ValueError("eigenvectors must be a two-dimensional array")
    if vectors.shape != (basis_layout.dimension, spectrum.size):
        raise ValueError("eigenvectors must match the basis layout and eigenvalues")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("eigenvectors must contain only finite values")
    if energies.ndim != 1 or energies.size == 0:
        raise ValueError("energy_values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(energies)):
        raise ValueError("energy_values must contain only finite values")
    if isinstance(broadening, bool) or not isinstance(broadening, Real):
        raise TypeError("broadening must be a real number")
    broadening = float(broadening)
    if not np.isfinite(broadening) or broadening <= 0.0:
        raise ValueError("broadening must be finite and positive")

    site_major_vectors = basis_layout.to_site_major(vectors)
    component_probabilities = np.abs(site_major_vectors) ** 2
    component_probabilities = component_probabilities.reshape(
        basis_layout.n_sites,
        basis_layout.components_per_site,
        spectrum.size,
    )
    lorentzians = broadening / np.pi / (
        (energies[:, np.newaxis] - spectrum[np.newaxis, :]) ** 2
        + broadening**2
    )
    component_values = np.einsum(
        "scn,en->esc",
        component_probabilities,
        lorentzians,
    ).reshape(
        energies.size,
        *basis_layout.spatial_shape,
        basis_layout.components_per_site,
    )

    return LocalDensityOfStates(
        energy_values=energies,
        values=np.sum(component_values, axis=-1),
        component_values=component_values,
        component_labels=basis_layout.component_labels,
    )


def local_density_of_states_from_result(
    result: SimulationResult,
    energy_values: np.ndarray,
    broadening: float,
) -> LocalDensityOfStates:
    """Calculate LDOS from a standardized eigensolver result."""
    return local_density_of_states(
        eigenvalues=result.eigenvalues,
        eigenvectors=result.eigenvectors,
        basis_layout=result.basis_layout,
        energy_values=energy_values,
        broadening=broadening,
    )
