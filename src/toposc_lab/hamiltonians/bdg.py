"""Generic Bogoliubov-de-Gennes matrix construction."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from toposc_lab.hamiltonians.nambu import NambuBasis


ComplexMatrix = NDArray[np.complex128]


def build_bdg_hamiltonian(
    normal_state: ArrayLike,
    pairing: ArrayLike,
    *,
    basis: NambuBasis,
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
    ``component_major`` ordering. For physical fermionic inputs
    ``h = h^\dagger`` and ``Delta = -Delta^T``, the result is Hermitian.
    """
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

    particle_indices = np.asarray(basis.particle_indices, dtype=np.intp)
    hole_indices = np.asarray(basis.hole_indices, dtype=np.intp)
    hamiltonian = np.zeros((basis.dimension, basis.dimension), dtype=np.complex128)

    hamiltonian[np.ix_(particle_indices, particle_indices)] = normal_matrix
    hamiltonian[np.ix_(particle_indices, hole_indices)] = pairing_matrix
    hamiltonian[np.ix_(hole_indices, particle_indices)] = -pairing_matrix.conj()
    hamiltonian[np.ix_(hole_indices, hole_indices)] = -normal_matrix.conj()
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
