import numpy as np
import pytest

from toposc_lab.lattices.ladder import LadderLattice


def test_open_two_leg_ladder_has_expected_geometry() -> None:
    lattice = LadderLattice(n_legs=2, length=4)

    assert lattice.n_sites == 8
    assert lattice.dimension == 2
    assert lattice.site_index(1, 3) == 7
    assert np.array_equal(lattice.coordinates[7], [3, 1])

    # Zwei Ketten mit je drei Längsbonds plus vier Rungs.
    assert len(lattice.bonds) == 10


def test_periodic_length_connects_chain_ends() -> None:
    lattice = LadderLattice(
        n_legs=2,
        length=4,
        boundary_length="periodic",
    )

    # Platz (leg=0, position=0) hat zwei Längs- und einen Quernachbarn.
    assert lattice.neighbors(lattice.site_index(0, 0)) == (1, 3, 4)


def test_periodic_legs_connects_outer_legs() -> None:
    lattice = LadderLattice(
        n_legs=3,
        length=3,
        boundary_legs="periodic",
    )

    # Der mittlere Platz der ersten Kette verbindet sich mit beiden anderen Legs.
    assert lattice.neighbors(lattice.site_index(0, 1)) == (0, 2, 4, 7)


def test_ladder_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        LadderLattice(n_legs=1, length=4)

    with pytest.raises(ValueError):
        LadderLattice(n_legs=2, length=1)

    with pytest.raises(ValueError):
        LadderLattice(n_legs=2, length=4, boundary_length="invalid")