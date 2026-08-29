"""Pfaffian utilities and the restricted 1D Kitaev/Majorana invariant."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np

from toposc_lab.observables.results import ObservableRecord
from toposc_lab.topology.symmetry import (
    AltlandZirnbauerClass,
    SymmetryClassification,
)


@dataclass(frozen=True, slots=True)
class PfaffianInvariantResult:
    """Restricted 1D class-D/BDI Pfaffian ``Z2`` result.

    ``invariant = -1`` is the non-trivial Kitaev/Majorana phase and ``+1`` is
    trivial. ``minimum_endpoint_abs_energy`` only probes ``k=0`` and ``k=pi``;
    it is not a full bulk-gap calculation.
    """

    invariant: Literal[-1, 1]
    is_topological: bool
    pfaffian_zero: complex
    pfaffian_pi: complex
    pfaffian_product: float
    minimum_endpoint_abs_energy: float
    maximum_particle_hole_residual: float
    maximum_antisymmetry_residual: float
    tolerance: float

    def to_observable_record(self) -> ObservableRecord:
        """Return a standardized numerical invariant record."""
        return ObservableRecord(
            kind="pfaffian_invariant_1d",
            scalars={
                "invariant": self.invariant,
                "is_topological": self.is_topological,
                "pfaffian_zero_real": self.pfaffian_zero.real,
                "pfaffian_zero_imag": self.pfaffian_zero.imag,
                "pfaffian_pi_real": self.pfaffian_pi.real,
                "pfaffian_pi_imag": self.pfaffian_pi.imag,
                "pfaffian_product": self.pfaffian_product,
                "minimum_endpoint_abs_energy": self.minimum_endpoint_abs_energy,
                "maximum_particle_hole_residual": (
                    self.maximum_particle_hole_residual
                ),
                "maximum_antisymmetry_residual": (
                    self.maximum_antisymmetry_residual
                ),
                "tolerance": self.tolerance,
            },
            metadata={
                "method": "kitaev_majorana_pfaffian",
                "endpoint_momenta": ["0", "pi"],
                "applicable_az_classes": ["D", "BDI"],
                "requires_full_bulk_gap_check": True,
            },
        )


def pfaffian(matrix: np.ndarray, *, tolerance: float = 1.0e-12) -> complex:
    """Return the Pfaffian of a finite even-dimensional skew matrix.

    A pivoted skew-Gaussian elimination is used. The input must satisfy
    ``A.T = -A`` within the absolute tolerance; transpose, not Hermitian
    conjugation, is the relevant operation.
    """
    tolerance = _nonnegative_finite_real(tolerance, name="tolerance")
    values = np.asarray(matrix, dtype=complex)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.size == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if values.shape[0] % 2 != 0:
        raise ValueError("matrix dimension must be even")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must contain only finite values")
    if not np.allclose(values.T, -values, atol=tolerance, rtol=0.0):
        raise ValueError("matrix must be antisymmetric within tolerance")

    reduced = values.copy()
    result = 1.0 + 0.0j
    dimension = reduced.shape[0]
    for index in range(0, dimension - 1, 2):
        pivot = index + 1 + int(np.argmax(np.abs(reduced[index, index + 1 :])))
        if abs(reduced[index, pivot]) <= tolerance:
            return 0.0 + 0.0j
        if pivot != index + 1:
            reduced[[index + 1, pivot], :] = reduced[[pivot, index + 1], :]
            reduced[:, [index + 1, pivot]] = reduced[:, [pivot, index + 1]]
            result *= -1.0

        pivot_value = reduced[index, index + 1]
        result *= pivot_value
        if index + 2 < dimension:
            tail = slice(index + 2, dimension)
            tau = reduced[index, tail] / pivot_value
            column = reduced[tail, index + 1]
            reduced[tail, tail] += np.outer(tau, column) - np.outer(column, tau)

    return complex(result)


def one_dimensional_pfaffian_invariant(
    hamiltonian_zero: np.ndarray,
    hamiltonian_pi: np.ndarray,
    particle_hole_operator: np.ndarray,
    classification: SymmetryClassification,
    *,
    tolerance: float = 1.0e-10,
) -> PfaffianInvariantResult:
    r"""Evaluate the restricted Kitaev/Majorana Pfaffian invariant.

    The two matrices must be the Bloch BdG Hamiltonian at the particle-hole
    invariant momenta ``k=0`` and ``k=pi`` of a 1D translation-invariant
    periodic system. The convention must use a real unitary ``U_C`` with
    ``C=U_C K`` and ``C^2=+1``. In that convention ``B(k)=H(k) U_C`` is
    antisymmetric and has real Pfaffians at both endpoints. The caller must
    independently establish that the bulk stays gapped over the complete
    Brillouin zone; the two endpoint matrices cannot verify that assumption.
    """
    tolerance = _nonnegative_finite_real(tolerance, name="tolerance")
    if not isinstance(classification, SymmetryClassification):
        raise TypeError("classification must be a SymmetryClassification")
    if classification.altland_zirnbauer_class not in {
        AltlandZirnbauerClass.D,
        AltlandZirnbauerClass.BDI,
    }:
        raise ValueError("Pfaffian invariant is implemented only for class D or BDI")
    if classification.particle_hole is None or classification.particle_hole.square != 1:
        raise ValueError("Pfaffian invariant requires particle-hole symmetry with C^2=+1")

    zero = _finite_hermitian_matrix(hamiltonian_zero, name="hamiltonian_zero", tolerance=tolerance)
    pi = _finite_hermitian_matrix(hamiltonian_pi, name="hamiltonian_pi", tolerance=tolerance)
    if zero.shape != pi.shape:
        raise ValueError("endpoint Hamiltonians must have the same shape")
    if zero.shape[0] % 2 != 0:
        raise ValueError("BdG Hamiltonian dimension must be even")

    particle_hole = _finite_square_matrix(
        particle_hole_operator,
        name="particle_hole_operator",
    )
    if particle_hole.shape != zero.shape:
        raise ValueError("particle_hole_operator must match the Hamiltonian shape")
    identity = np.eye(zero.shape[0], dtype=complex)
    unitarity_residual = _maximum_abs(
        particle_hole @ particle_hole.conj().T - identity
    )
    square_residual = _maximum_abs(
        particle_hole @ particle_hole.conj() - identity
    )
    reality_residual = _maximum_abs(particle_hole.imag)
    if unitarity_residual > tolerance:
        raise ValueError("particle_hole_operator must be unitary within tolerance")
    if square_residual > tolerance:
        raise ValueError("particle_hole_operator must satisfy C^2=+1")
    if reality_residual > tolerance:
        raise ValueError("particle_hole_operator must use a real Nambu convention")

    particle_hole_residuals = tuple(
        _maximum_abs(
            particle_hole @ matrix.conj() @ particle_hole.conj().T + matrix
        )
        for matrix in (zero, pi)
    )
    maximum_particle_hole_residual = max(particle_hole_residuals)
    if maximum_particle_hole_residual > tolerance:
        raise ValueError(
            "endpoint Hamiltonians violate particle-hole symmetry within tolerance"
        )

    skew_zero = zero @ particle_hole
    skew_pi = pi @ particle_hole
    antisymmetry_residuals = (
        _maximum_abs(skew_zero.T + skew_zero),
        _maximum_abs(skew_pi.T + skew_pi),
    )
    maximum_antisymmetry_residual = max(antisymmetry_residuals)
    if maximum_antisymmetry_residual > tolerance:
        raise ValueError("H(k) U_C must be antisymmetric at k=0 and k=pi")

    pfaffian_zero = pfaffian(skew_zero, tolerance=tolerance)
    pfaffian_pi = pfaffian(skew_pi, tolerance=tolerance)
    reality_residual = max(abs(pfaffian_zero.imag), abs(pfaffian_pi.imag))
    if reality_residual > tolerance:
        raise ValueError("endpoint Pfaffians must be real in the supplied convention")
    if abs(pfaffian_zero) <= tolerance or abs(pfaffian_pi) <= tolerance:
        raise ValueError("Pfaffian invariant is undefined at an endpoint gap closing")

    product = float(pfaffian_zero.real * pfaffian_pi.real)
    invariant: Literal[-1, 1] = -1 if product < 0.0 else 1
    minimum_endpoint_abs_energy = min(
        float(np.min(np.abs(np.linalg.eigvalsh(zero)))),
        float(np.min(np.abs(np.linalg.eigvalsh(pi)))),
    )
    return PfaffianInvariantResult(
        invariant=invariant,
        is_topological=invariant == -1,
        pfaffian_zero=pfaffian_zero,
        pfaffian_pi=pfaffian_pi,
        pfaffian_product=product,
        minimum_endpoint_abs_energy=minimum_endpoint_abs_energy,
        maximum_particle_hole_residual=maximum_particle_hole_residual,
        maximum_antisymmetry_residual=maximum_antisymmetry_residual,
        tolerance=tolerance,
    )


def _finite_hermitian_matrix(
    matrix: np.ndarray,
    *,
    name: str,
    tolerance: float,
) -> np.ndarray:
    values = _finite_square_matrix(matrix, name=name)
    if not np.allclose(values, values.conj().T, atol=tolerance, rtol=0.0):
        raise ValueError(f"{name} must be Hermitian within tolerance")
    return values


def _finite_square_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=complex)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.size == 0:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _maximum_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


def _nonnegative_finite_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value
