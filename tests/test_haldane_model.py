import numpy as np

from toposc_lab.models.graphene_model import (
    GrapheneModel,
    GrapheneParameters,
)
from toposc_lab.models.haldane_model import (
    HaldaneModel,
    HaldaneModelParameters,
)
from toposc_lab.observables.symmetries import (
    check_time_reversal_symmetry,
)
from toposc_lab.solvers.exact_diagonalization import (
    ExactDiagonalizationSolver,
)


def test_haldane_hamiltonian_has_expected_shape() -> None:
    model = HaldaneModel(
        HaldaneModelParameters(
            n_x=4,
            n_y=3,
        )
    )

    assert model.hamiltonian().shape == (2 * 4 * 3, 2 * 4 * 3)


def test_haldane_hamiltonian_is_hermitian() -> None:
    model = HaldaneModel(
        HaldaneModelParameters(
            n_x=4,
            n_y=4,
            next_nearest_neighbor_hopping=0.2,
            phase=np.pi / 2.0,
            sublattice_mass=0.1,
            boundary_x="periodic",
            boundary_y="periodic",
        )
    )

    hamiltonian = model.hamiltonian()

    assert np.allclose(hamiltonian, hamiltonian.conj().T)


def test_haldane_reduces_to_graphene_without_nnn_or_mass() -> None:
    graphene = GrapheneModel(
        GrapheneParameters(
            n_x=4,
            n_y=3,
            hopping=1.0,
        )
    )

    haldane = HaldaneModel(
        HaldaneModelParameters(
            n_x=4,
            n_y=3,
            nearest_neighbor_hopping=1.0,
            next_nearest_neighbor_hopping=0.0,
            sublattice_mass=0.0,
        )
    )

    assert np.allclose(
        haldane.hamiltonian(),
        graphene.hamiltonian(),
    )


def test_complex_haldane_hopping_breaks_spinless_time_reversal() -> None:
    model = HaldaneModel(
        HaldaneModelParameters(
            n_x=3,
            n_y=3,
            next_nearest_neighbor_hopping=0.2,
            phase=np.pi / 2.0,
            boundary_x="periodic",
            boundary_y="periodic",
        )
    )

    time_reversal_operator = np.eye(model.lattice.n_sites)

    result = check_time_reversal_symmetry(
        model.hamiltonian(),
        time_reversal_operator,
    )

    assert not result.satisfied


def test_haldane_term_opens_the_graphene_dirac_gap() -> None:
    """
    Bei 3 × 3 periodischen Zellen liegen die diskreten Impulse auf den
    Graphen-Dirac-Punkten. Ein Haldane-Term muss dort einen Gap öffnen.
    """
    model = HaldaneModel(
        HaldaneModelParameters(
            n_x=3,
            n_y=3,
            next_nearest_neighbor_hopping=0.2,
            phase=np.pi / 2.0,
            sublattice_mass=0.0,
            boundary_x="periodic",
            boundary_y="periodic",
        )
    )

    energies = ExactDiagonalizationSolver().solve_model(model).eigenvalues

    assert np.min(np.abs(energies)) > 0.5


def test_haldane_basis_layout_has_a_and_b_components() -> None:
    model = HaldaneModel(
        HaldaneModelParameters(
            n_x=3,
            n_y=4,
        )
    )

    assert model.basis_layout.spatial_shape == (3, 4)
    assert model.basis_layout.components_per_site == 2