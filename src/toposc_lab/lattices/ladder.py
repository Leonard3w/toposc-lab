from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toposc_lab.lattices.base import BaseLattice, LatticeBond

@dataclass(frozen=True)
class LadderLattice(BaseLattice):
    """
    Quasi-eindimensionales Gitter aus mehreren gekoppelten Ketten.

    Die Koordinaten eines Platzes sind (position, leg).
    Die Richtung (1, 0) verläuft entlang einer Kette,
    die Richtung (0, 1) verbindet unterschiedliche Ketten.
    """

    n_legs: int
    length: int
    boundary_length: str = "open"
    boundary_legs: str = "open"

    def __post_init__(self) -> None:
        if self.n_legs <= 1:
            raise ValueError("n_legs must be greater than one")

        if self.length <= 1:
            raise ValueError("length must be greater than one")

        valid_boundaries = ("open", "periodic")

        if self.boundary_length not in valid_boundaries:
            raise ValueError("boundary_length must be either open or periodic")

        if self.boundary_legs not in valid_boundaries:
            raise ValueError("boundary_legs must be either open or periodic")
        
    def site_index(self, leg: int, position: int) -> int:
        """Mappe die Koordinate (leg, position) auf einen Platzindex."""
        if not 0 <= leg < self.n_legs:
            raise ValueError("leg is outside the lattice")

        if not 0 <= position < self.length:
            raise ValueError("position is outside the lattice")

        return leg * self.length + position
    
    @property
    def coordinates(self)->np.ndarray:
        return np.asarray(
            [
                (position, leg)
                for leg in range(self.n_legs)
                for position in range(self.length)
            ],
            dtype=int,
        )
    
    @property
    def bonds(self)-> tuple[LatticeBond,...]:
        bonds: list[LatticeBond] = []

        for leg in range(self.n_legs):
            for position in range(self.length - 1):
                bonds.append(
                    LatticeBond(
                        self.site_index(leg,position),
                        self.site_index(leg, position + 1),
                        direction=(1,0),
                    )
                )
            if self.boundary_length == "periodic" and self.length > 2:
                bonds.append(
                    LatticeBond(
                        self.site_index(leg, self.length - 1),
                        self.site_index(leg, 0),
                        direction=(1,0),
                    )
                )
        for leg in range(self.n_legs - 1):
            for position in range(self.length):
                bonds.append(
                    LatticeBond(
                        self.site_index(leg, position),
                        self.site_index(leg + 1, position),
                        direction=(0, 1),
                    )
                )

        # Periodische Querrichtung: Zylinder aus mehreren Ketten.
        # Für zwei Ketten wäre dies dieselbe Verbindung und wird nicht doppelt gezählt.
        if self.boundary_legs == "periodic" and self.n_legs > 2:
            for position in range(self.length):
                bonds.append(
                    LatticeBond(
                        self.site_index(self.n_legs - 1, position),
                        self.site_index(0, position),
                        direction=(0, 1),
                    )
                )

        return tuple(bonds)
    

        