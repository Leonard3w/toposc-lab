import numpy as np

from toposc_lab.models.bhz_model import BHZModel, BHZModelParameters
from toposc_lab.models.kitaev_chain import KitaevChain, KitaevChainParameters
from toposc_lab.models.qwz_model import QWZModel, QWZModelParameters
from toposc_lab.models.ssh_chain import SSHChain, SSHChainParameters
from toposc_lab.observables.symmetries import (
    check_chiral_symmetry,
    check_particle_hole_symmetry,
    check_time_reversal_symmetry,
)


def test_kitaev_chain_has_particle_hole_symmetry() -> None:
    n_sites = 10

    model = KitaevChain(
        KitaevChainParameters(
            n_sites=n_sites,
            hopping=1.0,
            chemical_potential=0.3,
            pairing=0.5,
            disorder_strength=0.4,
            disorder_seed=42,
        )
    )

    # In der Kitaev-BdG-Basis stehen zuerst alle Elektronen und danach
    # alle Löcher. Tau_x vertauscht beide Blöcke.
    identity = np.eye(n_sites)
    particle_hole_operator = np.block(
        [
            [np.zeros((n_sites, n_sites)), identity],
            [identity, np.zeros((n_sites, n_sites))],
        ]
    )

    result = check_particle_hole_symmetry(
        model.hamiltonian(),
        particle_hole_operator,
    )

    assert result.satisfied


def test_ssh_chain_has_chiral_symmetry() -> None:
    n_cells = 10

    model = SSHChain(
        SSHChainParameters(
            n_cells=n_cells,
            v=0.4,
            w=1.0,
            boundary="open",
        )
    )

    # A- und B-Sublattice bekommen entgegengesetzte Vorzeichen.
    chiral_operator = np.diag([1.0, -1.0] * n_cells)

    result = check_chiral_symmetry(
        model.hamiltonian(),
        chiral_operator,
    )

    assert result.satisfied


def test_qwz_model_breaks_spinless_time_reversal_symmetry() -> None:
    n_x = 4
    n_y = 4

    model = QWZModel(
        QWZModelParameters(
            n_x=n_x,
            n_y=n_y,
            mass=-1.0,
            boundary_x="periodic",
            boundary_y="periodic",
        )
    )

    # Für ein spinloses Modell wäre Zeitumkehr nur komplexe Konjugation.
    time_reversal_operator = np.eye(2 * n_x * n_y)

    result = check_time_reversal_symmetry(
        model.hamiltonian(),
        time_reversal_operator,
    )

    assert not result.satisfied


def test_bhz_model_has_time_reversal_symmetry() -> None:
    n_x = 3
    n_y = 3
    n_sites = n_x * n_y

    model = BHZModel(
        BHZModelParameters(
            n_x=n_x,
            n_y=n_y,
            mass=-1.0,
            boundary_x="periodic",
            boundary_y="periodic",
        )
    )

    # Der Zeitumkehr-Operator vertauscht die beiden Spin-Blöcke.
    # Pro Platz lautet er: [[0, I], [-I, 0]].
    orbital_identity = np.eye(2)
    time_reversal_per_site = np.block(
        [
            [np.zeros((2, 2)), orbital_identity],
            [-orbital_identity, np.zeros((2, 2))],
        ]
    )
    time_reversal_operator = np.kron(
        np.eye(n_sites),
        time_reversal_per_site,
    )

    result = check_time_reversal_symmetry(
        model.hamiltonian(),
        time_reversal_operator,
    )

    assert result.satisfied