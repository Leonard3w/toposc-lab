from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toposc_lab.lattices.base import BaseLattice, LatticeBond

@dataclass(frozen=True)
class ChainLattice(BaseLattice):
    """Eindimensionale Kette mit offenen oder periodischen Rändern."""

    length: int
    boundary: str = "open"

    def __post_init__(self)->None:
        if self.length <= 1:
            raise ValueError("length must be greater than one")
        
        if self.boundary not in ("open", "periodic"):
            raise ValueError("boundary must be either open or periodic")
        
    @property
    def coordinates(self)-> np.ndarray:
        """Koordinaten jedes Platzes entlang einer Kette"""
        return np.arange(self.length, dtype=int).reshape(-1,1)
    
    @property
    def bonds(self)->tuple[LatticeBond,...]:
        """Nächste Nachbar-Verbindungen der Kette."""
        bonds = [
            LatticeBond(site, site +1, direction=(1,))
            for site in range(self.length - 1)
        ]

        # Für zwei Plätze wäre die periodische Verbindung identisch zur
        # vorhandenen offenen Verbindung und würde sonst doppelt gezählt.
        if self.boundary == "periodic" and self.length > 2:
            bonds.append(
                LatticeBond(self.length - 1, 0, direction=(1,))
            )

        return tuple(bonds)