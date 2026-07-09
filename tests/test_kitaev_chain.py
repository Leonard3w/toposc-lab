import numpy as np
import pytest

from toposc_lab.models.kitaev_chain import KitaevChain, KitaevChainParameters


def test_kitaev_hamiltonian_shape() -> None:
    params = KitaevChainParameters(
        n_sites=10,
        hopping=1.0,
        chemical_potential=0.0,
        pairing=0.5,
    )

    model = KitaevChain(params)
    hamiltonian = model.hamiltonian()

    assert hamiltonian.shape == (20, 20)


def test_kitaev_hamiltonian_is_hermitian() -> None:
    params = KitaevChainParameters(
        n_sites=10,
        hopping=1.0,
        chemical_potential=0.3,
        pairing=0.5,
    )

    model = KitaevChain(params)
    hamiltonian = model.hamiltonian()

    assert np.allclose(hamiltonian, hamiltonian.conj().T)


def test_invalid_boundary_raises_error() -> None:
    params = KitaevChainParameters(
        n_sites=10,
        hopping=1.0,
        chemical_potential=0.0,
        pairing=0.5,
        boundary="wrong",
    )

    with pytest.raises(ValueError):
        KitaevChain(params)
        