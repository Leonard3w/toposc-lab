from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from toposc_lab.lattices.base import BaseLattice, LatticeBond
from toposc_lab.lattices.square import SquareLattice


@dataclass(frozen=True)
class RibbonLattice(BaseLattice):
    """
    Zweidimensionales Ribbon.

    Ein Ribbon ist in einer Richtung offen und in der anderen periodisch.
    Es ist ideal für Randzustandsanalysen in QWZ- und BHZ-Modellen.
    """

    width: int
    length: int
    periodic_direction: str = "y"

    _square_lattice: SquareLattice = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.width <= 1:
            raise ValueError("width must be greater than one")

        if self.length <= 1:
            raise ValueError("length must be greater than one")

        if self.periodic_direction not in ("x", "y"):
            raise ValueError("periodic_direction must be either x or y")

        if self.periodic_direction == "y":
            # Offen in x, periodisch in y.
            square_lattice = SquareLattice(
                n_x=self.width,
                n_y=self.length,
                boundary_x="open",
                boundary_y="periodic",
            )
        else:
            # Periodisch in x, offen in y.
            square_lattice = SquareLattice(
                n_x=self.length,
                n_y=self.width,
                boundary_x="periodic",
                boundary_y="open",
            )

        object.__setattr__(self, "_square_lattice", square_lattice)

    @property
    def coordinates(self) -> np.ndarray:
        """Koordinaten aller Plätze des Ribbons."""
        return self._square_lattice.coordinates

    @property
    def bonds(self) -> tuple[LatticeBond, ...]:
        """Nachbarverbindungen des Ribbons."""
        return self._square_lattice.bonds

    def site_index(self, x: int, y: int) -> int:
        """Mappe eine 2D-Koordinate auf den Platzindex."""
        return self._square_lattice.site_index(x, y)

    @property
    def open_direction(self) -> str:
        """Die Richtung mit offenen Rändern."""
        return "x" if self.periodic_direction == "y" else "y"