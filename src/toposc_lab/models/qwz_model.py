from __future__ import annotations

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field

from toposc_lab.core.model import BaseModel
from toposc_lab.lattices.square import SquareLattice


class QWZModelParameters(PydanticBaseModel):
    """Parameter des zweidimensionalen Qi-Wu-Zhang-Modells."""

    n_x: int = Field(..., gt=1, description="Number of lattice sites in x direction.")
    n_y: int = Field(..., gt=1, description="Number of lattice sites in y direction.")
    mass: float = Field(..., description="QWZ mass parameter m.")
    boundary_x: str = Field(default="open", description="Boundary in x direction.")
    boundary_y: str = Field(default="open", description="Boundary in y direction.")


class QWZModel(BaseModel):
    """Reale Raumdarstellung des Qi-Wu-Zhang-Chern-Isolators."""

    def __init__(self, params: QWZModelParameters) -> None:
        self.params = params
        self.lattice = SquareLattice(
            n_x=params.n_x,
            n_y=params.n_y,
            boundary_x=params.boundary_x,
            boundary_y=params.boundary_y,
        )

    def hamiltonian(self) -> np.ndarray:
        """Baue die QWZ-Hamiltonmatrix auf dem quadratischen Gitter."""
        mass = self.params.mass

        sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
        sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

        onsite_term = mass * sigma_z

        # Hopping-Terme erzeugen im Impulsraum die QWZ-Struktur.
        hopping_x = (sigma_z - 1.0j * sigma_x) / 2.0
        hopping_y = (sigma_z - 1.0j * sigma_y) / 2.0

        dimension = 2 * self.lattice.n_sites
        hamiltonian = np.zeros((dimension, dimension), dtype=complex)

        for site in range(self.lattice.n_sites):
            state = slice(2 * site, 2 * site + 2)
            hamiltonian[state, state] += onsite_term

        # Die Bond-Richtung entscheidet, welcher Hopping-Term verwendet wird.
        for bond in self.lattice.bonds:
            source_state = slice(2 * bond.source, 2 * bond.source + 2)
            target_state = slice(2 * bond.target, 2 * bond.target + 2)

            hopping = hopping_x if bond.direction == (1, 0) else hopping_y

            hamiltonian[source_state, target_state] += hopping
            hamiltonian[target_state, source_state] += hopping.conj().T

        return hamiltonian