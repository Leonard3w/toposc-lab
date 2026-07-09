from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EigenSystem:
    """Container for eigenvalues and eigenvectors"""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


class ExactDiagonalizationSolver:
    def solve(self, hamiltonian: np.ndarray) -> EigenSystem:

        if hamiltonian.ndim != 2:
            raise ValueError("Hamiltonian must be a two-dimensional matrix")
        if hamiltonian.shape[0] != hamiltonian.shape[1]:
            raise ValueError("Hamiltonian must be square.")
        if not np.allclose(hamiltonian, hamiltonian.conj().T):
            raise ValueError("Hamiltonian must be Hermitian.")

        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

        return EigenSystem(eigenvalues=eigenvalues, eigenvectors=eigenvectors)
