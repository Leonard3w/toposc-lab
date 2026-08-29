"""Local Chern marker for finite two-dimensional real-space systems."""

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
class LocalChernMarkerResult:
    """Position-resolved marker and explicitly selected bulk estimate."""

    positions: np.ndarray
    local_marker: np.ndarray
    position_areas: np.ndarray
    bulk_mask: np.ndarray
    bulk_chern_estimate: float
    chern_number: int | None
    quantization_error: float
    is_quantized: bool
    occupied_state_count: int
    unoccupied_state_count: int
    fermi_energy: float
    minimum_fermi_distance: float
    finite_sample_trace_residual: float
    maximum_hermiticity_residual: float
    maximum_projector_residual: float
    tolerance: float
    quantization_tolerance: float

    def to_observable_record(self) -> ObservableRecord:
        """Return a standardized local-Chern-marker record."""
        return ObservableRecord(
            kind="local_chern_marker",
            scalars={
                "bulk_chern_estimate": self.bulk_chern_estimate,
                "chern_number": self.chern_number,
                "quantization_error": self.quantization_error,
                "is_quantized": self.is_quantized,
                "occupied_state_count": self.occupied_state_count,
                "unoccupied_state_count": self.unoccupied_state_count,
                "fermi_energy": self.fermi_energy,
                "minimum_fermi_distance": self.minimum_fermi_distance,
                "finite_sample_trace_residual": (
                    self.finite_sample_trace_residual
                ),
                "maximum_hermiticity_residual": (
                    self.maximum_hermiticity_residual
                ),
                "maximum_projector_residual": self.maximum_projector_residual,
                "tolerance": self.tolerance,
                "quantization_tolerance": self.quantization_tolerance,
            },
            arrays={
                "positions": self.positions,
                "local_marker": self.local_marker,
                "position_areas": self.position_areas,
                "bulk_mask": self.bulk_mask,
            },
            metadata={
                "method": "occupied_unoccupied_projected_positions",
                "marker_convention": "4pi_Im_diag_PXQYP",
                "applicable_az_classes": ["A", "C", "D"],
                "sign_depends_on_xy_orientation": True,
                "requires_explicit_position_areas": True,
                "requires_explicit_bulk_mask": True,
                "finite_sample_total_marker_cancels": True,
                "requires_mobility_or_spectral_gap_assumption": True,
            },
        )


def local_chern_marker(
    hamiltonian: np.ndarray,
    basis_coordinates: np.ndarray,
    position_areas: float | np.ndarray,
    bulk_mask: np.ndarray,
    classification: SymmetryClassification,
    *,
    fermi_energy: float = 0.0,
    tolerance: float = 1.0e-10,
    quantization_tolerance: float = 5.0e-3,
) -> LocalChernMarkerResult:
    r"""Evaluate a position-resolved Chern marker and its bulk average.

    One ``(x, y)`` coordinate is required per Hamiltonian basis state. Internal
    states at the same coordinate are summed. ``position_areas`` and
    ``bulk_mask`` refer to the resulting lexicographically sorted unique
    positions; a scalar area is broadcast for a regular lattice. The marker
    convention is ``4 pi Im diag(P X Q Y P) / area``, matching the Bott-index
    orientation used by this package.

    The full marker sum of a finite open system cancels between bulk and edge,
    so the caller must explicitly select a physically justified bulk region.
    A thermodynamic spectral or mobility gap remains a model assumption.
    """
    if not isinstance(classification, SymmetryClassification):
        raise TypeError("classification must be a SymmetryClassification")
    if classification.altland_zirnbauer_class not in {
        AltlandZirnbauerClass.A,
        AltlandZirnbauerClass.C,
        AltlandZirnbauerClass.D,
    }:
        raise ValueError("local Chern marker is implemented only for class A, C, or D")

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
    positions, position_indices = np.unique(
        coordinates,
        axis=0,
        return_inverse=True,
    )
    areas = _position_areas(position_areas, position_count=positions.shape[0])
    selected_bulk = np.asarray(bulk_mask)
    if selected_bulk.dtype != np.bool_ or selected_bulk.ndim != 1:
        raise TypeError("bulk_mask must be a one-dimensional boolean array")
    if selected_bulk.size != positions.shape[0]:
        raise ValueError("bulk_mask must contain one entry per unique position")
    if not np.any(selected_bulk):
        raise ValueError("bulk_mask must select at least one position")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    minimum_fermi_distance = float(np.min(np.abs(eigenvalues - fermi_energy)))
    if minimum_fermi_distance <= tolerance:
        raise ValueError("local Chern marker is undefined at a Fermi-level eigenstate")
    occupied = eigenvalues < fermi_energy
    occupied_state_count = int(np.count_nonzero(occupied))
    if occupied_state_count == 0 or occupied_state_count == matrix.shape[0]:
        raise ValueError("Fermi energy must separate occupied and unoccupied states")

    occupied_vectors = eigenvectors[:, occupied]
    projector = occupied_vectors @ occupied_vectors.conj().T
    complement = np.eye(matrix.shape[0], dtype=complex) - projector
    projector_residual = _maximum_abs(projector @ projector - projector)
    if projector_residual > tolerance:
        raise ValueError("occupied-state projector is not idempotent within tolerance")

    projected_marker = (
        projector
        @ (coordinates[:, 0, np.newaxis] * complement)
        @ (coordinates[:, 1, np.newaxis] * projector)
    )
    basis_marker = 4.0 * np.pi * np.imag(np.diag(projected_marker))
    position_marker = np.zeros(positions.shape[0], dtype=float)
    np.add.at(position_marker, position_indices, basis_marker)
    local_marker = position_marker / areas
    bulk_area = float(np.sum(areas[selected_bulk]))
    bulk_chern_estimate = float(np.sum(position_marker[selected_bulk]) / bulk_area)
    rounded_chern = int(np.rint(bulk_chern_estimate))
    quantization_error = abs(bulk_chern_estimate - rounded_chern)
    is_quantized = quantization_error <= quantization_tolerance

    return LocalChernMarkerResult(
        positions=positions,
        local_marker=local_marker,
        position_areas=areas,
        bulk_mask=selected_bulk.copy(),
        bulk_chern_estimate=bulk_chern_estimate,
        chern_number=rounded_chern if is_quantized else None,
        quantization_error=quantization_error,
        is_quantized=is_quantized,
        occupied_state_count=occupied_state_count,
        unoccupied_state_count=matrix.shape[0] - occupied_state_count,
        fermi_energy=fermi_energy,
        minimum_fermi_distance=minimum_fermi_distance,
        finite_sample_trace_residual=abs(float(np.sum(position_marker))),
        maximum_hermiticity_residual=hermiticity_residual,
        maximum_projector_residual=projector_residual,
        tolerance=tolerance,
        quantization_tolerance=quantization_tolerance,
    )


def _position_areas(values: float | np.ndarray, *, position_count: int) -> np.ndarray:
    if isinstance(values, Real) and not isinstance(values, bool):
        area = _positive_finite_real(values, name="position_areas")
        return np.full(position_count, area, dtype=float)
    areas = np.asarray(values, dtype=float)
    if areas.ndim != 1 or areas.size != position_count:
        raise ValueError("position_areas must contain one value per unique position")
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ValueError("position_areas must be finite and positive")
    return areas.copy()


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
    value = _finite_real(value, name=name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


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
