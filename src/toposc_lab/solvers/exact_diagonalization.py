from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import SimulationResult


@dataclass(frozen=True)
class EigenSystem:
    """
    Legacy-Ergebnis der direkten Diagonalisierung.

    Für neue Forschungsauswertungen sollte solve_model() verwendet werden.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


class ExactDiagonalizationSolver:
    """Diagonalisierung hermitescher endlichdimensionaler Hamiltonmatrizen."""

    def solve(self, hamiltonian: np.ndarray) -> EigenSystem:
        """
        Legacy-Schnittstelle: Diagonalisiere nur eine Matrix.

        Sie bleibt erhalten, damit bestehende Beispiele nicht brechen.
        """
        matrix = np.asarray(hamiltonian, dtype=complex)

        if matrix.ndim != 2:
            raise ValueError("Hamiltonian must be a two-dimensional matrix")

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Hamiltonian must be square")

        if not np.allclose(matrix, matrix.conj().T):
            raise ValueError("Hamiltonian must be Hermitian")

        eigenvalues, eigenvectors = np.linalg.eigh(matrix)

        return EigenSystem(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )

    def solve_model(
        self,
        model: BaseModel,
        metadata: Mapping[str, Any] | None = None,
    ) -> SimulationResult:
        """
        Diagonalisiere ein Modell und liefere ein standardisiertes Ergebnis.

        Das Ergebnis kennt danach Modell, Parameter, Komponenten und
        Basisordnung. Damit können Plots und Observablen modellunabhängig
        arbeiten.
        """
        eigensystem = self.solve(model.hamiltonian())

        return SimulationResult(
            model_name=model.model_name,
            eigenvalues=eigensystem.eigenvalues,
            eigenvectors=eigensystem.eigenvectors,
            basis_layout=model.basis_layout,
            parameters=model.parameters,
            metadata={} if metadata is None else metadata,
        )