from __future__ import annotations

from numbers import Real

import numpy as np


def positive_energies(eigenvalues: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    """Return positive eigenvalues above a numerical tolerance."""
    energies = np.asarray(eigenvalues, dtype=float)

    return energies[energies > tolerance]


def lowest_abs_energy(eigenvalues: np.ndarray) -> float:
    """Return the smallest absolute eigenvalue of a non-empty spectrum.

    This quantity measures proximity to zero only. It is not by itself a bulk
    gap or evidence for a Majorana mode.
    """
    energies = np.asarray(eigenvalues, dtype=float)

    if energies.ndim != 1:
        raise ValueError("eigenvalues must be one-dimensional")
    if energies.size == 0:
        raise ValueError("eigenvalues must not be empty")
    if not np.all(np.isfinite(energies)):
        raise ValueError("eigenvalues must contain only finite values")

    return float(np.min(np.abs(energies)))


def count_zero_modes(eigenvalues: np.ndarray, tolerance: float = 1e-10) -> int:
    """Count eigenvalues with absolute value below a numerical tolerance."""
    energies = np.asarray(eigenvalues, dtype=float)

    return int(np.sum(np.abs(energies) <= tolerance))


def edge_gap(eigenvalues: np.ndarray) -> float:
    """Return the historical edge-gap estimate ``min(abs(E))``.

    This function does not inspect eigenvectors and therefore cannot establish
    that the nearest-zero state is spatially localized at an edge.
    """
    return lowest_abs_energy(eigenvalues)


def spectral_gap(
    eigenvalues: np.ndarray,
    reference_energy: float = 0.0,
    tolerance: float = 1e-10,
) -> float:
    """Return the full spectral separation across a reference energy.

    The result is ``E_above - E_below``, where ``E_above`` is the lowest state
    above ``reference_energy`` and ``E_below`` the highest state below it. If a
    state lies within ``tolerance`` of the reference, the spectral gap is zero.
    Both sides of the reference must be represented in the finite spectrum.
    For a particle-hole-symmetric BdG spectrum without zero modes, this full
    gap is twice the smallest positive quasiparticle excitation energy.
    """
    energies = np.asarray(eigenvalues, dtype=float)
    if energies.ndim != 1:
        raise ValueError("eigenvalues must be one-dimensional")
    if energies.size == 0:
        raise ValueError("eigenvalues must not be empty")
    if not np.all(np.isfinite(energies)):
        raise ValueError("eigenvalues must contain only finite values")
    reference_energy = _finite_real(reference_energy, name="reference_energy")
    tolerance = _nonnegative_finite_real(tolerance, name="tolerance")

    offsets = energies - reference_energy
    if np.any(np.abs(offsets) <= tolerance):
        return 0.0

    below = energies[offsets < -tolerance]
    above = energies[offsets > tolerance]
    if below.size == 0 or above.size == 0:
        raise ValueError("spectrum must contain states on both sides of reference_energy")
    return float(np.min(above) - np.max(below))


def bulk_gap(eigenvalues: np.ndarray, tolerance: float = 1e-10) -> float:
    """
    Return the smallest positive eigenvalue above a zero-mode tolerance.

    The function filters exact or numerical zero modes, but it does not inspect
    eigenvectors and therefore cannot by itself distinguish bulk from boundary
    states. The historical name is retained for backward compatibility.
    """
    positive = positive_energies(eigenvalues, tolerance=tolerance)

    if positive.size == 0:
        return 0.0

    return float(np.min(positive))


def energy_gap(eigenvalues: np.ndarray, tolerance: float = 1e-10) -> float:
    """
    Return the historical positive, zero-mode-filtered excitation estimate.

    This keeps the old behavior for backward compatibility.
    """
    return bulk_gap(eigenvalues, tolerance=tolerance)


def _finite_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _nonnegative_finite_real(value: float, *, name: str) -> float:
    value = _finite_real(value, name=name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value
