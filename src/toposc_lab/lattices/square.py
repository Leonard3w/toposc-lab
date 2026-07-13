from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toposc_lab.lattices.base import BaseLattice, LatticeBond


@dataclass(frozen=True)
class SquareLattice(BaseLattice):
    """Zweidimensionales quadratisches Gitter."""

    n_x: int
    n_y: int
    boundary_x: str = "open"
    boundary_y: str = "open"

    def __post_init__(self) -> None:
        if self.n_x <= 1 or self.n_y <= 1:
            raise ValueError("n_x and n_y must both be greater than one")

        valid_boundaries = ("open", "periodic")

        if self.boundary_x not in valid_boundaries:
            raise ValueError("boundary_x must be either open or periodic")

        if self.boundary_y not in valid_boundaries:
            raise ValueError("boundary_y must be either open or periodic")

    def site_index(self, x: int, y: int) -> int:
        """Mappe eine Koordinate (x, y) auf den zugehörigen Platzindex."""
        if not 0 <= x < self.n_x or not 0 <= y < self.n_y:
            raise ValueError("(x, y) is outside the lattice")

        return x * self.n_y + y

    @property
    def coordinates(self) -> np.ndarray:
        """Koordinaten aller Gitterplätze in der Reihenfolge von site_index()."""
        return np.asarray(
            [
                (x, y)
                for x in range(self.n_x)
                for y in range(self.n_y)
            ],
            dtype=int,
        )

    @property
    def bonds(self) -> tuple[LatticeBond, ...]:
        """Nächste-Nachbar-Verbindungen in x- und y-Richtung."""
        bonds: list[LatticeBond] = []

        for x in range(self.n_x):
            for y in range(self.n_y):
                site = self.site_index(x, y)

                # Verbindung in positiver x-Richtung.
                if x < self.n_x - 1:
                    bonds.append(
                        LatticeBond(
                            site,
                            self.site_index(x + 1, y),
                            direction=(1, 0),
                        )
                    )
                elif self.boundary_x == "periodic" and self.n_x > 2:
                    bonds.append(
                        LatticeBond(
                            site,
                            self.site_index(0, y),
                            direction=(1, 0),
                        )
                    )

                # Verbindung in positiver y-Richtung.
                if y < self.n_y - 1:
                    bonds.append(
                        LatticeBond(
                            site,
                            self.site_index(x, y + 1),
                            direction=(0, 1),
                        )
                    )
                elif self.boundary_y == "periodic" and self.n_y > 2:
                    bonds.append(
                        LatticeBond(
                            site,
                            self.site_index(x, 0),
                            direction=(0, 1),
                        )
                    )

        return tuple(bonds)
    