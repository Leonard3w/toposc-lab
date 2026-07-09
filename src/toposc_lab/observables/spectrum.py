import numpy as np


def positive_energies(eigenvalues: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    """Return positive eigenvalues above a numerical tolerance."""
    return eigenvalues[eigenvalues > tolerance]


def energy_gap(eigenvalues: np.ndarray, tolerance: float = 1e-10) -> float:
    """Return the smallest positive excitation energy."""
    positive = positive_energies(eigenvalues, tolerance=tolerance)

    if positive.size == 0:
        return 0.0

    return float(np.min(positive))
