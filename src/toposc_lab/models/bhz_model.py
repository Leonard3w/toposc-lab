from __future__ import annotations

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field

from toposc_lab.core.model import BaseModel
from toposc_lab.lattices.square import SquareLattice


class BHZModelParameters(PydanticBaseModel):
    """Parameter des zweidimensionalen BHZ-Modells."""

    n_x: int = Field(..., gt=1, description="Number of lattice sites in x direction.")
    n_y: int = Field(..., gt=1, description="Number of lattice sites in y direction.")
    mass: float = Field(..., description="Mass parameter m.")
    boundary_x: str = Field(default="open", description="Boundary in x direction.")
    boundary_y: str = Field(default="open", description="Boundary in y direction.")


class BHZModel(BaseModel):
    """
    Vereinfachtes Gitter-BHZ-Modell.

    Jeder Platz besitzt vier Zustände: zwei Orbitale für Spin up und
    zwei Orbitale für Spin down.
    """

    def __init__(self, params: BHZModelParameters) -> None:
        self.params = params
        self.lattice = SquareLattice(
            n_x=params.n_x,
            n_y=params.n_y,
            boundary_x=params.boundary_x,
            boundary_y=params.boundary_y,
        )

    def hamiltonian(self) -> np.ndarray:
        """Baue die BHZ-Hamiltonmatrix auf dem quadratischen Gitter."""
        mass = self.params.mass

        sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
        sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        zero = np.zeros((2, 2), dtype=complex)

        # QWZ-artiger Spin-up-Block.
        hopping_up_x = (sigma_z - 1.0j * sigma_x) / 2.0
        hopping_up_y = (sigma_z - 1.0j * sigma_y) / 2.0

        # Zeitumgekehrter Spin-down-Block.
        hopping_down_x = (sigma_z + 1.0j * sigma_x) / 2.0
        hopping_down_y = (sigma_z - 1.0j * sigma_y) / 2.0

        onsite_term = np.block(
            [
                [mass * sigma_z, zero],
                [zero, mass * sigma_z],
            ]
        )

        hopping_x = np.block(
            [
                [hopping_up_x, zero],
                [zero, hopping_down_x],
            ]
        )

        hopping_y = np.block(
            [
                [hopping_up_y, zero],
                [zero, hopping_down_y],
            ]
        )

        dimension = 4 * self.lattice.n_sites
        hamiltonian = np.zeros((dimension, dimension), dtype=complex)

        for site in range(self.lattice.n_sites):
            state = slice(4 * site, 4 * site + 4)
            hamiltonian[state, state] += onsite_term

        for bond in self.lattice.bonds:
            source_state = slice(4 * bond.source, 4 * bond.source + 4)
            target_state = slice(4 * bond.target, 4 * bond.target + 4)

            hopping = hopping_x if bond.direction == (1, 0) else hopping_y

            hamiltonian[source_state, target_state] += hopping
            hamiltonian[target_state, source_state] += hopping.conj().T

        return hamiltonian