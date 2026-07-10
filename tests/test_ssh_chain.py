import numpy as np
import pytest

from toposc_lab.models.ssh_chain import SSHChain, SSHChainParameters
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver


def test_ssh_hamiltonian_has_correct_shape() -> None:
    model = SSHChain(SSHChainParameters(n_cells=10, v=0.5, w=1.0))

    assert model.hamiltonian().shape == (20, 20)


def test_ssh_hamiltonian_is_hermitian() -> None:
    model = SSHChain(SSHChainParameters(n_cells=10, v=0.5, w=1.0))

    hamiltonian = model.hamiltonian()

    assert np.allclose(hamiltonian, hamiltonian.conj().T)


def test_invalid_boundary_raises_error() -> None:
    params = SSHChainParameters(n_cells=10, v=0.5, w=1.0, boundary="invalid")

    with pytest.raises(ValueError):
        SSHChain(params)


def test_topological_phase_has_near_zero_edge_states() -> None:
    model = SSHChain(
        SSHChainParameters(n_cells=30, v=0.3, w=1.0, boundary="open")
    )

    eigenvalues = ExactDiagonalizationSolver().solve(model.hamiltonian()).eigenvalues

    # v < w: zwei Randzustände liegen sehr nahe bei Energie 0.
    assert np.sort(np.abs(eigenvalues))[0] < 1e-6
    assert np.sort(np.abs(eigenvalues))[1] < 1e-6


def test_trivial_phase_has_no_near_zero_edge_states() -> None:
    model = SSHChain(
        SSHChainParameters(n_cells=30, v=1.0, w=0.3, boundary="open")
    )

    eigenvalues = ExactDiagonalizationSolver().solve(model.hamiltonian()).eigenvalues

    assert np.min(np.abs(eigenvalues)) > 0.1