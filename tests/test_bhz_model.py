import numpy as np
import pytest

from toposc_lab.models.bhz_model import BHZModel, BHZModelParameters


def test_bhz_hamiltonian_has_correct_shape() -> None:
    model = BHZModel(
        BHZModelParameters(
            n_x=4,
            n_y=3,
            mass=1.0,
        )
    )

    # 4 * 3 Gitterplätze, jeweils vier interne Zustände.
    assert model.hamiltonian().shape == (4 * 4 * 3, 4 * 4 * 3)


def test_bhz_hamiltonian_is_hermitian() -> None:
    model = BHZModel(
        BHZModelParameters(
            n_x=4,
            n_y=4,
            mass=-1.0,
            boundary_x="periodic",
            boundary_y="periodic",
        )
    )

    hamiltonian = model.hamiltonian()

    # Eine physikalische Hamilton-Matrix muss hermitesch sein.
    assert np.allclose(hamiltonian, hamiltonian.conj().T)


def test_invalid_x_boundary_raises_error() -> None:
    params = BHZModelParameters(
        n_x=4,
        n_y=4,
        mass=1.0,
        boundary_x="invalid",
    )

    with pytest.raises(ValueError):
        BHZModel(params)


def test_invalid_y_boundary_raises_error() -> None:
    params = BHZModelParameters(
        n_x=4,
        n_y=4,
        mass=1.0,
        boundary_y="invalid",
    )

    with pytest.raises(ValueError):
        BHZModel(params)