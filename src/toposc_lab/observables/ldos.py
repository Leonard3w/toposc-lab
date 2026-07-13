from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toposc_lab.core.results import SimulationResult


@dataclass(frozen=True)
class LocalDensityOfStates:
    """Energie- und ortsaufgelöste lokale Zustandsdichte."""

    energy_values: np.ndarray
    values: np.ndarray
    component_values: np.ndarray
    component_labels: tuple[str, ...]


def local_density_of_states_from_result(
    result: SimulationResult,
    energy_values: np.ndarray,
    broadening: float,
) -> LocalDensityOfStates:
    """
    Berechne die LDOS mit Lorentz-Verbreiterung.

    rho_i(E) = sum_n |psi_n(i)|^2 * eta / pi / ((E - E_n)^2 + eta^2)

    broadening eta ist die Energieauflösung der Analyse. Kleine Werte
    zeigen einzelne diskrete Zustände, größere Werte glätten Spektren.
    """
    energies = np.asarray(energy_values, dtype=float)

    if energies.ndim != 1 or energies.size == 0:
        raise ValueError(
            "energy_values must be a non-empty one-dimensional array"
        )

    if broadening <= 0.0:
        raise ValueError("broadening must be positive")

    layout = result.basis_layout
    vectors = result.site_major_eigenvectors()

    component_probabilities = np.abs(vectors) ** 2
    component_probabilities = component_probabilities.reshape(
        layout.n_sites,
        layout.components_per_site,
        result.n_states,
    )

    lorentzians = (
        broadening
        / np.pi
        / (
            (energies[:, np.newaxis] - result.eigenvalues[np.newaxis, :]) ** 2
            + broadening**2
        )
    )

    component_values = np.einsum(
        "scn,en->esc",
        component_probabilities,
        lorentzians,
    ).reshape(
        energies.size,
        *layout.spatial_shape,
        layout.components_per_site,
    )

    values = np.sum(component_values, axis=-1)

    return LocalDensityOfStates(
        energy_values=energies,
        values=values,
        component_values=component_values,
        component_labels=layout.component_labels,
    )
