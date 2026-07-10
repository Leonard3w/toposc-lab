from __future__ import annotations

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field

from toposc_lab.core.model import BaseModel


class BHZModelParameters(PydanticBaseModel):
    """Parameter des zweidimensionalen BHZ-Modells."""

    n_x: int = Field(..., gt=1, description="Number of lattice sites in x direction.")
    n_y: int = Field(..., gt=1, description="Number of lattice sites in y direction.")

    # Steuert, ob das System trivial oder topologisch ist.
    mass: float = Field(..., description="Mass parameter m.")

    boundary_x: str = Field(default="open", description="Boundary in x direction.")
    boundary_y: str = Field(default="open", description="Boundary in y direction.")


class BHZModel(BaseModel):
    """
    Ein vereinfachtes Gitter-BHZ-Modell.

    Jeder Gitterplatz besitzt vier Zustände:
    - zwei Orbitale für Spin up
    - zwei Orbitale für Spin down

    Die beiden Spin-Blöcke besitzen entgegengesetzte Chiralität.
    Dadurch bleibt die Zeitumkehrsymmetrie erhalten.
    """

    def __init__(self, params: BHZModelParameters) -> None:
        self.params = params
        self._validate_boundaries()

    def _validate_boundaries(self) -> None:
        valid_boundaries = ("open", "periodic")

        if self.params.boundary_x not in valid_boundaries:
            raise ValueError("boundary_x must be either open or periodic")

        if self.params.boundary_y not in valid_boundaries:
            raise ValueError("boundary_y must be either open or periodic")

    def _site_index(self, x: int, y: int) -> int:
        """Mappe die Gitterkoordinate (x, y) auf einen eindimensionalen Index."""
        return x * self.params.n_y + y

    def hamiltonian(self) -> np.ndarray:
        """Baue die Hamilton-Matrix für ein endliches BHZ-Gitter."""
        n_x = self.params.n_x
        n_y = self.params.n_y
        mass = self.params.mass

        # Pauli-Matrizen wirken auf die zwei Orbitale eines Spin-Blocks.
        sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
        sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

        # QWZ-artiger Block für Spin up.
        hopping_up_x = (sigma_z - 1.0j * sigma_x) / 2.0
        hopping_up_y = (sigma_z - 1.0j * sigma_y) / 2.0

        # Zeitumgekehrter Block für Spin down.
        # Das Vorzeichen des sin(k_x)-Terms ist gegenüber Spin up umgedreht.
        hopping_down_x = (sigma_z + 1.0j * sigma_x) / 2.0
        hopping_down_y = (sigma_z - 1.0j * sigma_y) / 2.0

        # Basis pro Gitterplatz:
        # (up orbital 1, up orbital 2, down orbital 1, down orbital 2)
        onsite_term = np.block(
            [
                [mass * sigma_z, np.zeros((2, 2), dtype=complex)],
                [np.zeros((2, 2), dtype=complex), mass * sigma_z],
            ]
        )

        hopping_x = np.block(
            [
                [hopping_up_x, np.zeros((2, 2), dtype=complex)],
                [np.zeros((2, 2), dtype=complex), hopping_down_x],
            ]
        )

        hopping_y = np.block(
            [
                [hopping_up_y, np.zeros((2, 2), dtype=complex)],
                [np.zeros((2, 2), dtype=complex), hopping_down_y],
            ]
        )

        # Vier Zustände pro Gitterplatz.
        dimension = 4 * n_x * n_y
        hamiltonian = np.zeros((dimension, dimension), dtype=complex)

        for x in range(n_x):
            for y in range(n_y):
                site = self._site_index(x, y)
                state = slice(4 * site, 4 * site + 4)

                hamiltonian[state, state] += onsite_term

                # Nachbar in x-Richtung.
                if x < n_x - 1:
                    next_site = self._site_index(x + 1, y)
                    next_state = slice(4 * next_site, 4 * next_site + 4)

                    hamiltonian[state, next_state] += hopping_x
                    hamiltonian[next_state, state] += hopping_x.conj().T

                elif self.params.boundary_x == "periodic":
                    next_site = self._site_index(0, y)
                    next_state = slice(4 * next_site, 4 * next_site + 4)

                    hamiltonian[state, next_state] += hopping_x
                    hamiltonian[next_state, state] += hopping_x.conj().T

                # Nachbar in y-Richtung.
                if y < n_y - 1:
                    next_site = self._site_index(x, y + 1)
                    next_state = slice(4 * next_site, 4 * next_site + 4)

                    hamiltonian[state, next_state] += hopping_y
                    hamiltonian[next_state, state] += hopping_y.conj().T

                elif self.params.boundary_y == "periodic":
                    next_site = self._site_index(x, 0)
                    next_state = slice(4 * next_site, 4 * next_site + 4)

                    hamiltonian[state, next_state] += hopping_y
                    hamiltonian[next_state, state] += hopping_y.conj().T

        return hamiltonian