"""Bott index for finite two-dimensional real-space Hamiltonians."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from toposc_lab.observables.results import ObservableRecord
from toposc_lab.topology.symmetry import (
    AltlandZirnbauerClass,
    SymmetryClassification,
)


@dataclass(frozen=True, slots=True)
class BottIndexResult:
    """Bott estimate with numerical reliability diagnostics.

    The sign uses the supplied ``(x, y)`` orientation and the commutator
    convention ``U_x U_y U_x^dagger U_y^dagger``. ``bott_index`` is ``None``
    when the trace-log estimate is not integer-quantized within the requested
    tolerance.
    """

    bott_estimate: float
    bott_index: int | None
    quantization_error: float
    is_quantized: bool
    occupied_state_count: int
    unoccupied_state_count: int
    fermi_energy: float
    minimum_fermi_distance: float
    minimum_projected_position_singular_value: float
    minimum_branch_cut_distance: float
    maximum_hermiticity_residual: float
    maximum_unitarity_residual: float
    commutator_eigenphases: np.ndarray
    coordinate_periods: np.ndarray
    tolerance: float
    quantization_tolerance: float

    def to_observable_record(self) -> ObservableRecord:
        """Return a standardized numerical Bott-index record."""
        return ObservableRecord(
            kind="bott_index",
            scalars={
                "bott_estimate": self.bott_estimate,
                "bott_index": self.bott_index,
                "quantization_error": self.quantization_error,
                "is_quantized": self.is_quantized,
                "occupied_state_count": self.occupied_state_count,
                "unoccupied_state_count": self.unoccupied_state_count,
                "fermi_energy": self.fermi_energy,
                "minimum_fermi_distance": self.minimum_fermi_distance,
                "minimum_projected_position_singular_value": (
                    self.minimum_projected_position_singular_value
                ),
                "minimum_branch_cut_distance": self.minimum_branch_cut_distance,
                "maximum_hermiticity_residual": (
                    self.maximum_hermiticity_residual
                ),
                "maximum_unitarity_residual": self.maximum_unitarity_residual,
                "tolerance": self.tolerance,
                "quantization_tolerance": self.quantization_tolerance,
            },
            arrays={
                "commutator_eigenphases": self.commutator_eigenphases,
                "coordinate_periods": self.coordinate_periods,
            },
            metadata={
                "method": "polar_projected_position_unitaries",
                "commutator_convention": "Ux_Uy_Ux_dagger_Uy_dagger",
                "applicable_az_classes": ["A", "C", "D"],
                "sign_depends_on_xy_orientation": True,
                "requires_explicit_coordinate_periods": True,
                "requires_mobility_or_spectral_gap_assumption": True,
            },
        )


def bott_index(
    hamiltonian: np.ndarray,
    basis_coordinates: np.ndarray,
    coordinate_periods: np.ndarray | tuple[float, float],
    classification: SymmetryClassification,
    *,
    fermi_energy: float = 0.0,
    tolerance: float = 1.0e-10,
    quantization_tolerance: float = 1.0e-6,
) -> BottIndexResult:
    r"""Evaluate the occupied-projector Bott index of a finite 2D system.

    ``basis_coordinates`` contains one ``(x, y)`` row per Hamiltonian basis
    state, including repeated rows for internal orbitals. The two explicit
    coordinate periods define the phase operators
    ``exp(2 pi i x/L_x)`` and ``exp(2 pi i y/L_y)``; they are intentionally not
    inferred for irregular or nonperiodic geometries. The Fermi energy must
    lie outside the finite numerical spectrum, and the projected phase
    operators must be nonsingular so their polar factors are well-defined.
    A thermodynamic spectral or mobility gap remains a physical assumption
    that cannot be established from one finite Hamiltonian alone.
    """
    if not isinstance(classification, SymmetryClassification):
        raise TypeError("classification must be a SymmetryClassification")
    if classification.altland_zirnbauer_class not in {
        AltlandZirnbauerClass.A,
        AltlandZirnbauerClass.C,
        AltlandZirnbauerClass.D,
    }:
        raise ValueError("Bott index is implemented only for class A, C, or D")

    tolerance = _nonnegative_finite_real(tolerance, name="tolerance")
    quantization_tolerance = _nonnegative_finite_real(
        quantization_tolerance,
        name="quantization_tolerance",
    )
    fermi_energy = _finite_real(fermi_energy, name="fermi_energy")
    matrix = _finite_square_matrix(hamiltonian, name="hamiltonian")
    hermiticity_residual = _maximum_abs(matrix - matrix.conj().T)
    if hermiticity_residual > tolerance:
        raise ValueError("hamiltonian must be Hermitian within tolerance")

    coordinates = np.asarray(basis_coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape != (matrix.shape[0], 2):
        raise ValueError("basis_coordinates must have shape (dimension, 2)")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("basis_coordinates must contain only finite values")
    centered_coordinates = coordinates - np.mean(coordinates, axis=0)
    if np.linalg.matrix_rank(centered_coordinates) < 2:
        raise ValueError("basis_coordinates must span two spatial dimensions")

    periods = np.asarray(coordinate_periods, dtype=float)
    if periods.shape != (2,):
        raise ValueError("coordinate_periods must contain exactly two values")
    if not np.all(np.isfinite(periods)) or np.any(periods <= 0.0):
        raise ValueError("coordinate_periods must be finite and positive")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    fermi_distances = np.abs(eigenvalues - fermi_energy)
    minimum_fermi_distance = float(np.min(fermi_distances))
    if minimum_fermi_distance <= tolerance:
        raise ValueError("Bott index is undefined at a Fermi-level eigenstate")
    occupied = eigenvalues < fermi_energy
    occupied_state_count = int(np.count_nonzero(occupied))
    if occupied_state_count == 0 or occupied_state_count == matrix.shape[0]:
        raise ValueError("Fermi energy must separate occupied and unoccupied states")
    occupied_vectors = eigenvectors[:, occupied]

    phase_coordinates = np.exp(
        2.0j * np.pi * coordinates / periods[np.newaxis, :]
    )
    compressed_x = occupied_vectors.conj().T @ (
        phase_coordinates[:, 0, np.newaxis] * occupied_vectors
    )
    compressed_y = occupied_vectors.conj().T @ (
        phase_coordinates[:, 1, np.newaxis] * occupied_vectors
    )
    unitary_x, singular_x = _polar_unitary(compressed_x)
    unitary_y, singular_y = _polar_unitary(compressed_y)
    minimum_singular_value = float(min(np.min(singular_x), np.min(singular_y)))
    if minimum_singular_value <= tolerance:
        raise ValueError("projected position operator is numerically singular")

    occupied_identity = np.eye(occupied_state_count, dtype=complex)
    bott_commutator = (
        unitary_x
        @ unitary_y
        @ unitary_x.conj().T
        @ unitary_y.conj().T
    )
    maximum_unitarity_residual = max(
        _maximum_abs(unitary_x @ unitary_x.conj().T - occupied_identity),
        _maximum_abs(unitary_y @ unitary_y.conj().T - occupied_identity),
        _maximum_abs(
            bott_commutator @ bott_commutator.conj().T - occupied_identity
        ),
    )
    if maximum_unitarity_residual > tolerance:
        raise ValueError("polar position operators are not unitary within tolerance")

    commutator_eigenphases = np.sort(
        np.angle(np.linalg.eigvals(bott_commutator))
    )
    minimum_branch_cut_distance = float(
        np.min(np.pi - np.abs(commutator_eigenphases))
    )
    if minimum_branch_cut_distance <= tolerance:
        raise ValueError("Bott logarithm spectrum touches its branch cut")

    bott_estimate = float(np.sum(commutator_eigenphases) / (2.0 * np.pi))
    rounded_index = int(np.rint(bott_estimate))
    quantization_error = abs(bott_estimate - rounded_index)
    is_quantized = quantization_error <= quantization_tolerance

    return BottIndexResult(
        bott_estimate=bott_estimate,
        bott_index=rounded_index if is_quantized else None,
        quantization_error=quantization_error,
        is_quantized=is_quantized,
        occupied_state_count=occupied_state_count,
        unoccupied_state_count=matrix.shape[0] - occupied_state_count,
        fermi_energy=fermi_energy,
        minimum_fermi_distance=minimum_fermi_distance,
        minimum_projected_position_singular_value=minimum_singular_value,
        minimum_branch_cut_distance=minimum_branch_cut_distance,
        maximum_hermiticity_residual=hermiticity_residual,
        maximum_unitarity_residual=maximum_unitarity_residual,
        commutator_eigenphases=commutator_eigenphases,
        coordinate_periods=periods.copy(),
        tolerance=tolerance,
        quantization_tolerance=quantization_tolerance,
    )


def _polar_unitary(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    return left @ right, singular_values


def _finite_square_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=complex)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.size == 0:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _maximum_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


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
