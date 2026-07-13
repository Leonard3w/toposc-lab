from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class LatticeBond:

    source: int
    target: int
    direction: tuple[int,...]


class BaseLattice(ABC):

    @property
    @abstractmethod
    def coordinates(self)->np.ndarray:
        """Gitterkoordinaten aller Plätze, Form: (n_sites, dimension)."""

    @property
    @abstractmethod
    def bonds(self)->tuple[LatticeBond,...]:
        """Alle eindeutigen Verbindungen zwischen benachbarten Plätzen."""

    @property
    def n_sites(self) -> int:
        """Gesamtzahl der Gitterplätze."""
        return int(self.coordinates.shape[0])

    @property
    def dimension(self) -> int:
        """Räumliche Dimension des Gitters."""
        return int(self.coordinates.shape[1])
    
    def neighbors(self, site:int)->tuple[int,...]:
        """Gibt alle direkten Nachbarplätze zurück."""
        if not 0 <= site < self.n_sites:
            raise ValueError("site is outside the lattice")
        
        neighbors: set[int] = set()

        for bond in self.bonds:
            if bond.source == site:
                neighbors.add(bond.target)
            elif bond.target == site:
                neighbors.add(bond.source)

        return tuple(sorted(neighbors))
