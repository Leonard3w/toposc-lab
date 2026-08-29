"""Real-space winding marker for compatible one-dimensional chiral systems."""

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
class RealSpaceWindingResult:
    """Local marker and bulk-averaged integer winding estimate.

    The sign depends on the orientation of the position coordinate and on which
    chiral eigenspace is labeled positive. ``winding_number`` is ``None`` when
    the selected bulk average is not quantized within the requested tolerance.
    """

    positions: np.ndarray
    local_marker: np.ndarray
    bulk_mask: np.ndarray
    winding_estimate: float
    winding_number: int | None
    quantization_error: float
    is_quantized: bool
    zero_mode_count: int
    minimum_nonzero_abs_energy: float
    maximum_chiral_residual: float
    marker_imaginary_residual: float
    tolerance: float
    quantization_tolerance: float

    def to_observable_record(self) -> ObservableRecord:
        """Return a standardized numerical winding-marker record."""
        return ObservableRecord(
            kind="real_space_winding",
            scalars={
                "winding_estimate": self.winding_estimate,
                "winding_number": self.winding_number,
                "quantization_error": self.quantization_error,
                "is_quantized": self.is_quantized,
                "zero_mode_count": self.zero_mode_count,
                "minimum_nonzero_abs_energy": self.minimum_nonzero_abs_energy,
                "maximum_chiral_residual": self.maximum_chiral_residual,
                "marker_imaginary_residual": self.marker_imaginary_residual,
                "tolerance": self.tolerance,
                "quantization_tolerance": self.quantization_tolerance,
            },
            arrays={
                "positions": self.positions,
                "local_marker": self.local_marker,
                "bulk_mask": self.bulk_mask,
            },
            metadata={
                "method": "projected_position_commutator",
                "applicable_az_classes": ["AIII", "BDI", "CII"],
                "sign_depends_on_orientation": True,
                "requires_explicit_bulk_mask": True,
            },
        )


def real_space_winding_invariant(
    hamiltonian: np.ndarray,
    chiral_operator: np.ndarray,
    basis_positions: np.ndarray,
    bulk_mask: np.ndarray,
    classification: SymmetryClassification,
    *,
    position_scale: float = 1.0,
    zero_tolerance: float = 1.0e-10,
    quantization_tolerance: float = 1.0e-3,
) -> RealSpaceWindingResult:
    r"""Evaluate a bulk-averaged 1D real-space chiral winding marker.

    With ``Q=sign(H)``, ``P_+-=(I+-Gamma)/2``, and position ``X``, the local
    marker is the position-resolved trace of
    ``P_- Q P_+ [X, P_+ Q P_-]``. Internal basis states sharing a coordinate
    are summed before the explicit bulk mask is averaged. Numerical zero modes
    are omitted from ``Q``, which permits open topological chains with boundary
    zero modes while retaining their gapped bulk marker.
    """
    if not isinstance(classification, SymmetryClassification):
        raise TypeError("classification must be a SymmetryClassification")
    if classification.altland_zirnbauer_class not in {
        AltlandZirnbauerClass.AIII,
        AltlandZirnbauerClass.BDI,
        AltlandZirnbauerClass.CII,
    }:
        raise ValueError("real-space winding requires class AIII, BDI, or CII")
    if not classification.chiral_symmetry:
        raise ValueError("real-space winding requires declared chiral symmetry")

    zero_tolerance = _nonnegative_finite_real(
        zero_tolerance,
        name="zero_tolerance",
    )
    quantization_tolerance = _nonnegative_finite_real(
        quantization_tolerance,
        name="quantization_tolerance",
    )
    position_scale = _positive_finite_real(position_scale, name="position_scale")
    matrix = _finite_square_matrix(hamiltonian, name="hamiltonian")
    chiral = _finite_square_matrix(chiral_operator, name="chiral_operator")
    if chiral.shape != matrix.shape:
        raise ValueError("chiral_operator must match the Hamiltonian shape")

    tolerance = zero_tolerance
    hermiticity_residual = _maximum_abs(matrix - matrix.conj().T)
    if hermiticity_residual > tolerance:
        raise ValueError("hamiltonian must be Hermitian within zero_tolerance")
    identity = np.eye(matrix.shape[0], dtype=complex)
    chiral_residuals = (
        _maximum_abs(chiral @ chiral.conj().T - identity),
        _maximum_abs(chiral @ chiral - identity),
        _maximum_abs(chiral - chiral.conj().T),
        _maximum_abs(chiral @ matrix @ chiral.conj().T + matrix),
    )
    maximum_chiral_residual = max(chiral_residuals)
    if maximum_chiral_residual > tolerance:
        raise ValueError(
            "chiral operator or Hamiltonian violates chiral symmetry within tolerance"
        )

    positions = np.asarray(basis_positions, dtype=float)
    if positions.ndim != 1 or positions.size != matrix.shape[0]:
        raise ValueError("basis_positions must contain one coordinate per basis state")
    if not np.all(np.isfinite(positions)):
        raise ValueError("basis_positions must contain only finite values")
    unique_positions, position_indices = np.unique(positions, return_inverse=True)
    selected_bulk = np.asarray(bulk_mask)
    if selected_bulk.dtype != np.bool_ or selected_bulk.ndim != 1:
        raise TypeError("bulk_mask must be a one-dimensional boolean array")
    if selected_bulk.size != unique_positions.size:
        raise ValueError("bulk_mask must contain one entry per unique position")
    if not np.any(selected_bulk):
        raise ValueError("bulk_mask must select at least one position")

    position_operator = np.diag(positions / position_scale).astype(complex)
    position_chiral_residual = _maximum_abs(
        position_operator @ chiral - chiral @ position_operator
    )
    maximum_chiral_residual = max(
        maximum_chiral_residual,
        position_chiral_residual,
    )
    if position_chiral_residual > tolerance:
        raise ValueError("chiral_operator must not mix distinct positions")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    zero_modes = np.abs(eigenvalues) <= zero_tolerance
    nonzero_abs_energies = np.abs(eigenvalues[~zero_modes])
    if nonzero_abs_energies.size == 0:
        raise ValueError("hamiltonian must contain a nonzero gapped sector")
    spectral_sign = np.sign(eigenvalues)
    spectral_sign[zero_modes] = 0.0
    flattened = (eigenvectors * spectral_sign) @ eigenvectors.conj().T

    positive_chiral = 0.5 * (identity + chiral)
    negative_chiral = 0.5 * (identity - chiral)
    plus_minus = positive_chiral @ flattened @ negative_chiral
    minus_plus = negative_chiral @ flattened @ positive_chiral
    commutator = position_operator @ plus_minus - plus_minus @ position_operator
    basis_marker = np.diag(minus_plus @ commutator)

    local_marker_complex = np.zeros(unique_positions.size, dtype=complex)
    np.add.at(local_marker_complex, position_indices, basis_marker)
    marker_imaginary_residual = _maximum_abs(local_marker_complex.imag)
    if marker_imaginary_residual > tolerance:
        raise ValueError("real-space winding marker must be real within tolerance")
    local_marker = local_marker_complex.real
    winding_estimate = float(np.mean(local_marker[selected_bulk]))
    rounded_winding = int(np.rint(winding_estimate))
    quantization_error = abs(winding_estimate - rounded_winding)
    is_quantized = quantization_error <= quantization_tolerance

    return RealSpaceWindingResult(
        positions=unique_positions,
        local_marker=local_marker,
        bulk_mask=selected_bulk.copy(),
        winding_estimate=winding_estimate,
        winding_number=rounded_winding if is_quantized else None,
        quantization_error=quantization_error,
        is_quantized=is_quantized,
        zero_mode_count=int(np.count_nonzero(zero_modes)),
        minimum_nonzero_abs_energy=float(np.min(nonzero_abs_energies)),
        maximum_chiral_residual=maximum_chiral_residual,
        marker_imaginary_residual=marker_imaginary_residual,
        tolerance=zero_tolerance,
        quantization_tolerance=quantization_tolerance,
    )


def _finite_square_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=complex)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.size == 0:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _maximum_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


def _positive_finite_real(value: float, *, name: str) -> float:
    value = _nonnegative_finite_real(value, name=name)
    if value == 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_finite_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value
