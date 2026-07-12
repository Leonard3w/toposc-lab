import numpy as np
import pytest

from toposc_lab.observables.symmetries import (
    check_chiral_symmetry,
    check_hermitian,
    check_particle_hole_symmetry,
    check_time_reversal_symmetry,
)


def test_check_hermitian_accepts_hermitian_matrix() -> None:
    hamiltonian = np.array(
        [
            [1.0, 1.0j],
            [-1.0j, 2.0],
        ]
    )

    result = check_hermitian(hamiltonian)

    assert result.satisfied
    assert result.residual == pytest.approx(0.0)


def test_check_hermitian_rejects_non_hermitian_matrix() -> None:
    hamiltonian = np.array(
        [
            [1.0, 1.0],
            [0.0, 2.0],
        ]
    )

    result = check_hermitian(hamiltonian)

    assert not result.satisfied
    assert result.residual > 0.0


def test_check_particle_hole_symmetry() -> None:
    # BdG-Beispiel: H = sigma_z, Teilchen-Loch-Operator C = sigma_x K.
    hamiltonian = np.array(
        [
            [1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    particle_hole_operator = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )

    result = check_particle_hole_symmetry(
        hamiltonian,
        particle_hole_operator,
    )

    assert result.satisfied


def test_check_chiral_symmetry() -> None:
    # H = sigma_x, chiraler Operator S = sigma_z.
    hamiltonian = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    chiral_operator = np.array(
        [
            [1.0, 0.0],
            [0.0, -1.0],
        ]
    )

    result = check_chiral_symmetry(hamiltonian, chiral_operator)

    assert result.satisfied


def test_check_time_reversal_symmetry() -> None:
    # Reelle Hamiltonmatrix: Zeitumkehr ist hier nur komplexe Konjugation.
    hamiltonian = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    time_reversal_operator = np.eye(2)

    result = check_time_reversal_symmetry(
        hamiltonian,
        time_reversal_operator,
    )

    assert result.satisfied


def test_check_time_reversal_symmetry_detects_broken_symmetry() -> None:
    # sigma_y ist unter komplexer Konjugation nicht invariant.
    hamiltonian = np.array(
        [
            [0.0, -1.0j],
            [1.0j, 0.0],
        ]
    )
    time_reversal_operator = np.eye(2)

    result = check_time_reversal_symmetry(
        hamiltonian,
        time_reversal_operator,
    )

    assert not result.satisfied