import pytest

from toposc_lab.lattices.ribbon import RibbonLattice


def test_y_periodic_ribbon_has_open_x_edges() -> None:
    lattice = RibbonLattice(
        width=4,
        length=6,
        periodic_direction="y",
    )

    assert lattice.n_sites == 24
    assert lattice.open_direction == "x"

    # Ecke bei x=0: zwei periodische y-Nachbarn und ein x-Nachbar.
    assert lattice.neighbors(lattice.site_index(0, 0)) == (1, 5, 6)


def test_x_periodic_ribbon_has_open_y_edges() -> None:
    lattice = RibbonLattice(
        width=4,
        length=6,
        periodic_direction="x",
    )

    assert lattice.n_sites == 24
    assert lattice.open_direction == "y"

    # Ecke bei y=0: zwei periodische x-Nachbarn und ein y-Nachbar.
    assert lattice.neighbors(lattice.site_index(0, 0)) == (1, 4, 20)


def test_ribbon_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        RibbonLattice(width=1, length=6)

    with pytest.raises(ValueError):
        RibbonLattice(width=4, length=1)

    with pytest.raises(ValueError):
        RibbonLattice(width=4, length=6, periodic_direction="invalid")