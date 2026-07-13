from toposc_lab.lattices.chain import ChainLattice
from toposc_lab.models.kitaev_chain import KitaevChain, KitaevChainParameters
from toposc_lab.models.ssh_chain import SSHChain, SSHChainParameters
from toposc_lab.lattices.square import SquareLattice
from toposc_lab.models.bhz_model import BHZModel, BHZModelParameters
from toposc_lab.models.qwz_model import QWZModel, QWZModelParameters


def test_kitaev_chain_uses_chain_lattice() -> None:
    model = KitaevChain(
        KitaevChainParameters(
            n_sites=10,
            hopping=1.0,
            chemical_potential=0.0,
            pairing=0.5,
            boundary="periodic",
        )
    )

    assert isinstance(model.lattice, ChainLattice)
    assert model.lattice.n_sites == 10
    assert model.lattice.boundary == "periodic"


def test_ssh_chain_uses_chain_lattice() -> None:
    model = SSHChain(
        SSHChainParameters(
            n_cells=10,
            v=0.5,
            w=1.0,
            boundary="open",
        )
    )

    assert isinstance(model.lattice, ChainLattice)
    assert model.lattice.n_sites == 20
    assert model.lattice.boundary == "open"


def test_qwz_model_uses_square_lattice() -> None:
    model = QWZModel(
        QWZModelParameters(
            n_x=4,
            n_y=3,
            mass=-1.0,
            boundary_x="periodic",
            boundary_y="open",
        )
    )

    assert isinstance(model.lattice, SquareLattice)
    assert model.lattice.n_sites == 12
    assert model.lattice.boundary_x == "periodic"


def test_bhz_model_uses_square_lattice() -> None:
    model = BHZModel(
        BHZModelParameters(
            n_x=4,
            n_y=3,
            mass=-1.0,
            boundary_x="open",
            boundary_y="periodic",
        )
    )

    assert isinstance(model.lattice, SquareLattice)
    assert model.lattice.n_sites == 12
    assert model.lattice.boundary_y == "periodic"