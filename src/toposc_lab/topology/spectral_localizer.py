"""Two-dimensional spectral localizer for finite real-space Hamiltonians."""

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
class SpectralLocalizerResult:
    """Localizer spectrum, gap, signature, and local Chern index."""

    probe_position: np.ndarray
    energy: float
    kappa: float
    signature: int
    local_chern_number: int | None
    is_invertible: bool
    localizer_gap: float
    positive_eigenvalue_count: int
    negative_eigenvalue_count: int
    zero_eigenvalue_count: int
    minimum_energy_distance: float
    maximum_hamiltonian_hermiticity_residual: float
    maximum_localizer_hermiticity_residual: float
    localizer_eigenvalues: np.ndarray
    tolerance: float

    def to_observable_record(self) -> ObservableRecord:
        """Return a standardized spectral-localizer record."""
        return ObservableRecord(
            kind="spectral_localizer_2d",
            scalars={
                "energy": self.energy,
                "kappa": self.kappa,
                "signature": self.signature,
                "local_chern_number": self.local_chern_number,
                "is_invertible": self.is_invertible,
                "localizer_gap": self.localizer_gap,
                "positive_eigenvalue_count": self.positive_eigenvalue_count,
                "negative_eigenvalue_count": self.negative_eigenvalue_count,
                "zero_eigenvalue_count": self.zero_eigenvalue_count,
                "minimum_energy_distance": self.minimum_energy_distance,
                "maximum_hamiltonian_hermiticity_residual": (
                    self.maximum_hamiltonian_hermiticity_residual
                ),
                "maximum_localizer_hermiticity_residual": (
                    self.maximum_localizer_hermiticity_residual
                ),
                "tolerance": self.tolerance,
            },
            arrays={
                "probe_position": self.probe_position,
                "localizer_eigenvalues": self.localizer_eigenvalues,
            },
            metadata={
                "method": "two_dimensional_spectral_localizer",
                "localizer_convention": "H_E_kappa_X_minus_iY",
                "index_convention": "half_signature",
                "applicable_az_classes": ["A", "C", "D"],
                "sign_depends_on_xy_orientation": True,
                "index_requires_nonzero_localizer_gap": True,
                "kappa_units": "energy_per_coordinate",
            },
        )


def spectral_localizer(
    hamiltonian: np.ndarray,
    basis_coordinates: np.ndarray,
    probe_position: np.ndarray | tuple[float, float],
    classification: SymmetryClassification,
    *,
    energy: float = 0.0,
    kappa: float = 0.2,
    tolerance: float = 1.0e-10,
) -> SpectralLocalizerResult:
    r"""Evaluate the 2D spectral localizer at one position-energy probe.

    In block form the convention is

    ``[[H-E, kappa*(X-x-i*(Y-y))],``
    `` [kappa*(X-x+i*(Y-y)), -(H-E)]]``.

    For a localizer gap greater than ``tolerance``, half of its signature is
    returned as ``local_chern_number``. A zero or unresolved localizer gap
    leaves that index undefined. ``kappa`` has units of energy per coordinate,
    so it must be rescaled inversely if spatial coordinates are rescaled.
    """
    if not isinstance(classification, SymmetryClassification):
        raise TypeError("classification must be a SymmetryClassification")
    if classification.altland_zirnbauer_class not in {
        AltlandZirnbauerClass.A,
        AltlandZirnbauerClass.C,
        AltlandZirnbauerClass.D,
    }:
        raise ValueError(
            "2D spectral-localizer signature is implemented only for class A, C, or D"
        )

    tolerance = _nonnegative_finite_real(tolerance, name="tolerance")
    energy = _finite_real(energy, name="energy")
    kappa = _positive_finite_real(kappa, name="kappa")
    matrix = _finite_square_matrix(hamiltonian, name="hamiltonian")
    hamiltonian_hermiticity_residual = _maximum_abs(matrix - matrix.conj().T)
    if hamiltonian_hermiticity_residual > tolerance:
        raise ValueError("hamiltonian must be Hermitian within tolerance")

    coordinates = np.asarray(basis_coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape != (matrix.shape[0], 2):
        raise ValueError("basis_coordinates must have shape (dimension, 2)")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("basis_coordinates must contain only finite values")
    centered_coordinates = coordinates - np.mean(coordinates, axis=0)
    if np.linalg.matrix_rank(centered_coordinates) < 2:
        raise ValueError("basis_coordinates must span two spatial dimensions")
    probe = np.asarray(probe_position, dtype=float)
    if probe.shape != (2,) or not np.all(np.isfinite(probe)):
        raise ValueError("probe_position must contain two finite coordinates")

    dimension = matrix.shape[0]
    shifted_hamiltonian = matrix - energy * np.eye(dimension, dtype=complex)
    shifted_position = kappa * (
        coordinates[:, 0]
        - probe[0]
        - 1.0j * (coordinates[:, 1] - probe[1])
    )
    localizer = np.zeros((2 * dimension, 2 * dimension), dtype=complex)
    localizer[:dimension, :dimension] = shifted_hamiltonian
    localizer[dimension:, dimension:] = -shifted_hamiltonian
    localizer[:dimension, dimension:] = np.diag(shifted_position)
    localizer[dimension:, :dimension] = np.diag(shifted_position.conj())

    localizer_hermiticity_residual = _maximum_abs(localizer - localizer.conj().T)
    if localizer_hermiticity_residual > tolerance:
        raise ValueError("spectral localizer is not Hermitian within tolerance")
    localizer_eigenvalues = np.linalg.eigvalsh(localizer)
    positive_count = int(np.count_nonzero(localizer_eigenvalues > tolerance))
    negative_count = int(np.count_nonzero(localizer_eigenvalues < -tolerance))
    zero_count = int(localizer_eigenvalues.size - positive_count - negative_count)
    signature = positive_count - negative_count
    localizer_gap = float(np.min(np.abs(localizer_eigenvalues)))
    is_invertible = localizer_gap > tolerance
    if is_invertible and signature % 2 != 0:
        raise RuntimeError("invertible 2D localizer must have an even signature")

    return SpectralLocalizerResult(
        probe_position=probe.copy(),
        energy=energy,
        kappa=kappa,
        signature=signature,
        local_chern_number=signature // 2 if is_invertible else None,
        is_invertible=is_invertible,
        localizer_gap=localizer_gap,
        positive_eigenvalue_count=positive_count,
        negative_eigenvalue_count=negative_count,
        zero_eigenvalue_count=zero_count,
        minimum_energy_distance=float(
            np.min(np.abs(np.linalg.eigvalsh(matrix) - energy))
        ),
        maximum_hamiltonian_hermiticity_residual=(
            hamiltonian_hermiticity_residual
        ),
        maximum_localizer_hermiticity_residual=localizer_hermiticity_residual,
        localizer_eigenvalues=localizer_eigenvalues,
        tolerance=tolerance,
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
