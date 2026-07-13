import numpy as np
import pytest

from toposc_lab.lattices.chain import ChainLattice
from toposc_lab.lattices.square import SquareLattice


def test_open_chain_has_expected_coordinates_and_bonds() -> None:
    lattice = ChainLattice(length=4, boundary="open")

    assert lattice.n_sites == 4
    assert lattice.dimension == 1
    assert np.array_equal(lattice.coordinates[:, 0], [0, 1, 2, 3])
    assert [(bond.source, bond.target) for bond in lattice.bonds] == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]


def test_periodic_chain_connects_its_ends() -> None:
    lattice = ChainLattice(length=4, boundary="periodic")

    assert len(lattice.bonds) == 4
    assert lattice.neighbors(0) == (1, 3)


def test_square_lattice_maps_coordinates_to_indices() -> None:
    lattice = SquareLattice(n_x=3, n_y=4)

    assert lattice.n_sites == 12
    assert lattice.dimension == 2
    assert lattice.site_index(2, 3) == 11
    assert np.array_equal(lattice.coordinates[11], [2, 3])


def test_open_square_has_expected_number_of_bonds() -> None:
    lattice = SquareLattice(n_x=3, n_y=4)

    # x-Bonds: (3 - 1) * 4, y-Bonds: 3 * (4 - 1).
    assert len(lattice.bonds) == 17


def test_periodic_square_has_four_neighbors_per_site() -> None:
    lattice = SquareLattice(
        n_x=3,
        n_y=4,
        boundary_x="periodic",
        boundary_y="periodic",
    )

    assert len(lattice.bonds) == 24
    assert lattice.neighbors(lattice.site_index(0, 0)) == (1, 3, 4, 8)


def test_lattice_rejects_invalid_boundary() -> None:
    with pytest.raises(ValueError):
        ChainLattice(length=4, boundary="invalid")

    with pytest.raises(ValueError):
        SquareLattice(n_x=3, n_y=3, boundary_x="invalid")