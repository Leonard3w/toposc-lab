import numpy as np
import pytest

from toposc_lab.models.qwz_model import QWZModel, QWZModelParameters


def test_qwz_hamiltonian_has_correct_shape() -> None:
    model = QWZModel(QWZModelParameters(n_x=4, n_y=3, mass=1.0))

    # 4 * 3 Plätze, jeweils zwei interne Zustände.
    assert model.hamiltonian().shape == (24, 24)


def test_qwz_hamiltonian_is_hermitian() -> None:
    model = QWZModel(
        QWZModelParameters(
            n_x=4,
            n_y=4,
            mass=1.0,
            boundary_x="periodic",
            boundary_y="periodic",
        )
    )

    hamiltonian = model.hamiltonian()

    assert np.allclose(hamiltonian, hamiltonian.conj().T)


def test_invalid_x_boundary_raises_error() -> None:
    params = QWZModelParameters(
        n_x=4,
        n_y=4,
        mass=1.0,
        boundary_x="invalid",
    )

    with pytest.raises(ValueError):
        QWZModel(params)


def test_invalid_y_boundary_raises_error() -> None:
    params = QWZModelParameters(
        n_x=4,
        n_y=4,
        mass=1.0,
        boundary_y="invalid",
    )

    with pytest.raises(ValueError):
        QWZModel(params)