from __future__ import annotations

import numpy as np


def positive_energies(eigenvalues: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    """Return positive eigenvalues above a numerical tolerance."""
    energies = np.asarray(eigenvalues, dtype=float)

    return energies[energies > tolerance]


def lowest_abs_energy(eigenvalues: np.ndarray) -> float:
    """Return the eigenvalue closest to zero in absolute value."""
    energies = np.asarray(eigenvalues, dtype=float)

    if energies.size == 0:
        return 0.0

    return float(np.min(np.abs(energies)))


def count_zero_modes(eigenvalues: np.ndarray, tolerance: float = 1e-10) -> int:
    """Count eigenvalues with absolute value below a numerical tolerance."""
    energies = np.asarray(eigenvalues, dtype=float)

    return int(np.sum(np.abs(energies) <= tolerance))


def edge_gap(eigenvalues: np.ndarray) -> float:
    """
    Return the edge gap.

    For the current implementation this is the energy closest to zero.
    This is useful for detecting Majorana zero modes.
    """
    return lowest_abs_energy(eigenvalues)


def bulk_gap(eigenvalues: np.ndarray, tolerance: float = 1e-10) -> float:
    """
    Return the smallest positive excitation energy above a tolerance.

    This ignores exact or numerical zero modes and therefore behaves like
    a simple bulk-gap estimate.
    """
    positive = positive_energies(eigenvalues, tolerance=tolerance)

    if positive.size == 0:
        return 0.0

    return float(np.min(positive))


def energy_gap(eigenvalues: np.ndarray, tolerance: float = 1e-10) -> float:
    """
    Return the smallest positive excitation energy.

    This keeps the old behavior for backward compatibility.
    """
    return bulk_gap(eigenvalues, tolerance=tolerance)