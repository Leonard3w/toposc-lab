import numpy as np
import pytest

from toposc_lab.lattices.honeycomb import HoneycombLattice


def test_honeycomb_has_two_sites_per_unit_cell() -> None:
    lattice = HoneycombLattice(n_x=3, n_y=4)

    assert lattice.n_sites == 24
    assert lattice.dimension == 2
    assert lattice.site_index(2, 3, "B") == 23
    assert lattice.sublattice(0) == "A"
    assert lattice.sublattice(1) == "B"


def test_honeycomb_coordinates_have_nearest_neighbor_distance_one() -> None:
    lattice = HoneycombLattice(n_x=3, n_y=3)

    a_site = lattice.site_index(1, 1, "A")
    b_site = lattice.site_index(1, 1, "B")

    distance = np.linalg.norm(
        lattice.coordinates[a_site] - lattice.coordinates[b_site]
    )

    assert distance == pytest.approx(1.0)


def test_open_honeycomb_has_expected_number_of_bonds() -> None:
    lattice = HoneycombLattice(n_x=3, n_y=4)

    # Intracell: 3*4, x-Bonds: (3-1)*4, y-Bonds: 3*(4-1).
    assert len(lattice.bonds) == 29


def test_bulk_honeycomb_site_has_three_neighbors() -> None:
    lattice = HoneycombLattice(n_x=3, n_y=3)

    a_site = lattice.site_index(1, 1, "A")

    assert lattice.neighbors(a_site) == (3, 7, 9)


def test_periodic_honeycomb_connects_its_boundaries() -> None:
    lattice = HoneycombLattice(
        n_x=3,
        n_y=3,
        boundary_x="periodic",
        boundary_y="periodic",
    )

    a_site = lattice.site_index(0, 0, "A")

    assert len(lattice.bonds) == 27
    assert lattice.neighbors(a_site) == (1, 5, 13)


def test_honeycomb_rejects_invalid_sublattice() -> None:
    lattice = HoneycombLattice(n_x=3, n_y=3)

    with pytest.raises(ValueError):
        lattice.site_index(0, 0, "invalid")