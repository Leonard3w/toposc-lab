import numpy as np

from toposc_lab.models.kitaev_chain import (
    KitaevChain,
    KitaevChainParameters,
)
from toposc_lab.models.kitaev_ladder import (
    KitaevLadder,
    KitaevLadderParameters,
)
from toposc_lab.solvers.exact_diagonalization import (
    ExactDiagonalizationSolver,
)


def test_kitaev_ladder_has_expected_hamiltonian_shape() -> None:
    model = KitaevLadder(
        KitaevLadderParameters(
            n_legs=2,
            length=5,
            hopping=1.0,
            chemical_potential=0.2,
            pairing=0.5,
        )
    )

    # 2 Legs × 5 Plätze × Elektron/Loch.
    assert model.hamiltonian().shape == (20, 20)


def test_kitaev_ladder_hamiltonian_is_hermitian() -> None:
    model = KitaevLadder(
        KitaevLadderParameters(
            n_legs=2,
            length=5,
            hopping=1.0,
            chemical_potential=0.2,
            pairing=0.5,
            rung_hopping=0.3,
            rung_pairing=0.1,
        )
    )

    hamiltonian = model.hamiltonian()

    assert np.allclose(hamiltonian, hamiltonian.conj().T)


def test_zero_rung_coupling_gives_two_independent_kitaev_chains() -> None:
    """
    Ohne Sprossenkopplung muss das Leiter-Spektrum aus zwei identischen
    Kitaev-Ketten bestehen.
    """
    ladder = KitaevLadder(
        KitaevLadderParameters(
            n_legs=2,
            length=4,
            hopping=1.0,
            chemical_potential=0.3,
            pairing=0.5,
            rung_hopping=0.0,
            rung_pairing=0.0,
        )
    )

    chain = KitaevChain(
        KitaevChainParameters(
            n_sites=4,
            hopping=1.0,
            chemical_potential=0.3,
            pairing=0.5,
        )
    )

    solver = ExactDiagonalizationSolver()

    ladder_energies = solver.solve_model(ladder).eigenvalues
    chain_energies = solver.solve_model(chain).eigenvalues

    expected_energies = np.sort(np.tile(chain_energies, 2))

    assert np.allclose(ladder_energies, expected_energies)


def test_kitaev_ladder_basis_layout_has_two_spatial_dimensions() -> None:
    model = KitaevLadder(
        KitaevLadderParameters(
            n_legs=3,
            length=6,
            hopping=1.0,
            chemical_potential=0.0,
            pairing=1.0,
        )
    )

    layout = model.basis_layout

    assert layout.spatial_shape == (3, 6)
    assert layout.components_per_site == 2
    assert layout.ordering == "component_major"