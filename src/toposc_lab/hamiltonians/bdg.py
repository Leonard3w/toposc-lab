"""Generic Bogoliubov-de-Gennes matrix construction."""

from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from toposc_lab.hamiltonians.nambu import NambuBasis


ComplexMatrix = NDArray[np.complex128]


def build_bdg_hamiltonian(
    normal_state: ArrayLike,
    pairing: ArrayLike,
    *,
    basis: NambuBasis,
    validation_tolerance: float = 1e-10,
) -> ComplexMatrix:
    r"""Build a BdG matrix from normal-state and pairing blocks.

    The input blocks use the undoubled site-major single-particle order

    ``(site 0 component 0, ..., site N-1 component m-1)``.

    They define the sector-major block convention

    .. math::

        H_{\mathrm{BdG}} =
        \begin{pmatrix}
            h & \Delta \\
            -\Delta^* & -h^*
        \end{pmatrix}.

    The blocks are placed at the particle and hole indices specified by
    ``basis``, so the returned matrix follows its explicit ``site_major`` or
    ``component_major`` ordering. The builder requires the fermionic input
    conditions ``h = h^\dagger`` and ``Delta = -Delta^T`` within
    ``validation_tolerance`` and verifies that the result is Hermitian using
    the same absolute tolerance and ``rtol=0``.
    """
    validation_tolerance = _validate_tolerance(validation_tolerance)
    normal_matrix = _complex_block(
        normal_state,
        dimension=basis.normal_dimension,
        name="normal_state",
    )
    pairing_matrix = _complex_block(
        pairing,
        dimension=basis.normal_dimension,
        name="pairing",
    )
    _validate_hermitian(
        normal_matrix,
        name="normal_state",
        tolerance=validation_tolerance,
    )
    _validate_antisymmetric(
        pairing_matrix,
        name="pairing",
        tolerance=validation_tolerance,
    )

    particle_indices = np.asarray(basis.particle_indices, dtype=np.intp)
    hole_indices = np.asarray(basis.hole_indices, dtype=np.intp)
    hamiltonian = np.zeros((basis.dimension, basis.dimension), dtype=np.complex128)

    hamiltonian[np.ix_(particle_indices, particle_indices)] = normal_matrix
    hamiltonian[np.ix_(particle_indices, hole_indices)] = pairing_matrix
    hamiltonian[np.ix_(hole_indices, particle_indices)] = -pairing_matrix.conj()
    hamiltonian[np.ix_(hole_indices, hole_indices)] = -normal_matrix.conj()
    _validate_hermitian(
        hamiltonian,
        name="constructed BdG Hamiltonian",
        tolerance=validation_tolerance,
    )
    return hamiltonian


def _complex_block(
    values: ArrayLike,
    *,
    dimension: int,
    name: str,
) -> ComplexMatrix:
    try:
        matrix = np.asarray(values, dtype=np.complex128)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a square numeric matrix") from error

    expected_shape = (dimension, dimension)
    if matrix.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _validate_tolerance(tolerance: float) -> float:
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        raise TypeError("validation_tolerance must be a real number")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("validation_tolerance must be finite and non-negative")
    return tolerance


def _validate_hermitian(
    matrix: ComplexMatrix,
    *,
    name: str,
    tolerance: float,
) -> None:
    difference = matrix - matrix.conj().T
    if not np.allclose(difference, 0.0, rtol=0.0, atol=tolerance):
        residual = float(np.max(np.abs(difference)))
        raise ValueError(
            f"{name} must be Hermitian within tolerance {tolerance:g}; "
            f"maximum residual is {residual:g}"
        )


def _validate_antisymmetric(
    matrix: ComplexMatrix,
    *,
    name: str,
    tolerance: float,
) -> None:
    difference = matrix + matrix.T
    if not np.allclose(difference, 0.0, rtol=0.0, atol=tolerance):
        residual = float(np.max(np.abs(difference)))
        raise ValueError(
            f"{name} must be antisymmetric within tolerance {tolerance:g}; "
            f"maximum residual is {residual:g}"
        )
