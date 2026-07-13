import numpy as np

from toposc_lab.models.graphene_model import (
    GrapheneModel,
    GrapheneParameters,
)
from toposc_lab.observables.symmetries import check_chiral_symmetry
from toposc_lab.solvers.exact_diagonalization import (
    ExactDiagonalizationSolver,
)


def test_graphene_hamiltonian_has_expected_shape() -> None:
    model = GrapheneModel(
        GrapheneParameters(
            n_x=4,
            n_y=3,
            hopping=1.0,
        )
    )

    # Zwei Plätze A/B pro Einheitszelle.
    assert model.hamiltonian().shape == (2 * 4 * 3, 2 * 4 * 3)


def test_graphene_hamiltonian_is_hermitian() -> None:
    model = GrapheneModel(
        GrapheneParameters(
            n_x=4,
            n_y=4,
            hopping=1.0,
            boundary_x="periodic",
            boundary_y="periodic",
        )
    )

    hamiltonian = model.hamiltonian()

    assert np.allclose(hamiltonian, hamiltonian.conj().T)


def test_graphene_has_chiral_sublattice_symmetry() -> None:
    n_x = 4
    n_y = 3
    n_cells = n_x * n_y

    model = GrapheneModel(
        GrapheneParameters(
            n_x=n_x,
            n_y=n_y,
            hopping=1.0,
        )
    )

    # A erhält +1, B erhält -1.
    chiral_operator = np.diag([1.0, -1.0] * n_cells)

    result = check_chiral_symmetry(
        model.hamiltonian(),
        chiral_operator,
    )

    assert result.satisfied


def test_periodic_graphene_has_dirac_points_for_three_by_three_cells() -> None:
    """
    Bei 3 × 3 periodischen Einheitszellen liegen die diskreten k-Punkte
    genau auf den Dirac-Punkten des unendlichen Graphenmodells.
    """
    model = GrapheneModel(
        GrapheneParameters(
            n_x=3,
            n_y=3,
            hopping=1.0,
            boundary_x="periodic",
            boundary_y="periodic",
        )
    )

    energies = ExactDiagonalizationSolver().solve_model(model).eigenvalues

    assert np.min(np.abs(energies)) < 1e-10


def test_graphene_basis_layout_describes_a_and_b_components() -> None:
    model = GrapheneModel(
        GrapheneParameters(
            n_x=3,
            n_y=4,
        )
    )

    layout = model.basis_layout

    assert layout.spatial_shape == (3, 4)
    assert layout.components_per_site == 2
    assert layout.component_labels == ("A sublattice", "B sublattice")