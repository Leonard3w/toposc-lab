from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SymmetryCheckResult:
    """Numerisches Ergebnis einer physikalischen Symmetrieprüfung."""

    name: str
    satisfied: bool
    residual: float
    tolerance: float


def _validate_square_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    """Prüfe, ob eine Matrix quadratisch ist."""
    values = np.asarray(matrix, dtype=complex)

    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{name} must be a square matrix")

    return values


def _validate_symmetry_operator(
    hamiltonian: np.ndarray,
    operator: np.ndarray,
) -> np.ndarray:
    """Prüfe Größe und Unitarität des unitären Symmetrieanteils."""
    values = _validate_square_matrix(operator, "operator")

    if values.shape != hamiltonian.shape:
        raise ValueError("operator must have the same shape as hamiltonian")

    identity = np.eye(values.shape[0], dtype=complex)

    if not np.allclose(values @ values.conj().T, identity):
        raise ValueError("operator must be unitary")

    return values


def _result(
    name: str,
    difference: np.ndarray,
    tolerance: float,
) -> SymmetryCheckResult:
    """Erzeuge ein einheitliches Prüfergebnis aus einer Differenzmatrix."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    residual = float(np.max(np.abs(difference)))

    return SymmetryCheckResult(
        name=name,
        satisfied=residual <= tolerance,
        residual=residual,
        tolerance=tolerance,
    )


def check_hermitian(
    hamiltonian: np.ndarray,
    tolerance: float = 1e-10,
) -> SymmetryCheckResult:
    """
    Prüfe Hermitizität: H = H†.

    Jede physikalische Hamiltonmatrix muss diese Bedingung erfüllen.
    """
    hamiltonian = _validate_square_matrix(hamiltonian, "hamiltonian")

    return _result(
        name="Hermitian",
        difference=hamiltonian - hamiltonian.conj().T,
        tolerance=tolerance,
    )


def check_particle_hole_symmetry(
    hamiltonian: np.ndarray,
    operator: np.ndarray,
    tolerance: float = 1e-10,
) -> SymmetryCheckResult:
    """
    Prüfe Teilchen-Loch-Symmetrie: C H* C† = -H.

    Der Operator ist der unitäre Anteil von C = operator · K,
    wobei K komplexe Konjugation bedeutet.
    """
    hamiltonian = _validate_square_matrix(hamiltonian, "hamiltonian")
    operator = _validate_symmetry_operator(hamiltonian, operator)

    transformed_hamiltonian = operator @ hamiltonian.conj() @ operator.conj().T

    return _result(
        name="Particle-hole symmetry",
        difference=transformed_hamiltonian + hamiltonian,
        tolerance=tolerance,
    )


def check_chiral_symmetry(
    hamiltonian: np.ndarray,
    operator: np.ndarray,
    tolerance: float = 1e-10,
) -> SymmetryCheckResult:
    """
    Prüfe chirale Symmetrie: S H S† = -H.

    Beim SSH-Modell ist S beispielsweise der Operator, der den
    beiden Sublattices entgegengesetzte Vorzeichen gibt.
    """
    hamiltonian = _validate_square_matrix(hamiltonian, "hamiltonian")
    operator = _validate_symmetry_operator(hamiltonian, operator)

    transformed_hamiltonian = operator @ hamiltonian @ operator.conj().T

    return _result(
        name="Chiral symmetry",
        difference=transformed_hamiltonian + hamiltonian,
        tolerance=tolerance,
    )


def check_time_reversal_symmetry(
    hamiltonian: np.ndarray,
    operator: np.ndarray,
    tolerance: float = 1e-10,
) -> SymmetryCheckResult:
    """
    Prüfe Zeitumkehrsymmetrie: T H* T† = H.

    Der Operator ist der unitäre Anteil von T = operator · K.
    """
    hamiltonian = _validate_square_matrix(hamiltonian, "hamiltonian")
    operator = _validate_symmetry_operator(hamiltonian, operator)

    transformed_hamiltonian = operator @ hamiltonian.conj() @ operator.conj().T

    return _result(
        name="Time-reversal symmetry",
        difference=transformed_hamiltonian - hamiltonian,
        tolerance=tolerance,
    )