import numpy as np
import pytest

from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver


def test_solver_diagonalizes_hermitian_matrix() -> None:
    hamiltonian = np.array(
        [
            [1.0, 0.2],
            [0.2, -1.0],
        ]
    )

    solver = ExactDiagonalizationSolver()
    result = solver.solve(hamiltonian)

    assert result.eigenvalues.shape == (2,)
    assert result.eigenvectors.shape == (2, 2)


def test_solver_returns_real_eigenvalues_in_ascending_order() -> None:
    hamiltonian = np.diag([2.0, -1.0, 0.5]).astype(complex)

    result = ExactDiagonalizationSolver().solve(hamiltonian)

    assert result.eigenvalues.dtype == np.dtype(float)
    assert np.array_equal(result.eigenvalues, [-1.0, 0.5, 2.0])


def test_solver_rejects_non_square_matrix() -> None:
    hamiltonian = np.zeros((2, 3))

    solver = ExactDiagonalizationSolver()

    with pytest.raises(ValueError):
        solver.solve(hamiltonian)


def test_solver_rejects_non_hermitian_matrix() -> None:
    hamiltonian = np.array(
        [
            [1.0, 2.0],
            [0.0, 1.0],
        ]
    )

    solver = ExactDiagonalizationSolver()

    with pytest.raises(ValueError):
        solver.solve(hamiltonian)
