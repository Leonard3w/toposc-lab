from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toposc_lab.lattices.base import BaseLattice, LatticeBond


@dataclass(frozen=True)
class HoneycombLattice(BaseLattice):
    """
    Zweidimensionales Honeycomb- bzw. Wabengitter.

    Jede Einheitszelle enthält zwei Plätze, A und B.
    Ein Bulk-Platz besitzt drei nächste Nachbarn.
    """

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

    def site_index(self, x: int, y: int, sublattice: str) -> int:
        """Mappe (x, y, A/B) auf einen eindeutigen Platzindex."""
        if not 0 <= x < self.n_x or not 0 <= y < self.n_y:
            raise ValueError("(x, y) is outside the lattice")

        if sublattice not in ("A", "B"):
            raise ValueError("sublattice must be either A or B")

        sublattice_index = 0 if sublattice == "A" else 1
        cell_index = x * self.n_y + y

        return 2 * cell_index + sublattice_index

    def sublattice(self, site: int) -> str:
        """Gib die Sublattice-Bezeichnung eines Platzes zurück."""
        if not 0 <= site < self.n_sites:
            raise ValueError("site is outside the lattice")

        return "A" if site % 2 == 0 else "B"

    @property
    def coordinates(self) -> np.ndarray:
        """
        Physikalische 2D-Koordinaten aller A- und B-Plätze.

        Die Wahl der Gittervektoren erzeugt reguläre Sechsecke mit
        nächster-Nachbar-Abstand eins.
        """
        lattice_vector_x = np.array([np.sqrt(3.0), 0.0])
        lattice_vector_y = np.array([np.sqrt(3.0) / 2.0, 1.5])
        b_offset = np.array([np.sqrt(3.0) / 2.0, 0.5])

        coordinates: list[np.ndarray] = []

        for x in range(self.n_x):
            for y in range(self.n_y):
                a_coordinate = x * lattice_vector_x + y * lattice_vector_y

                coordinates.append(a_coordinate)
                coordinates.append(a_coordinate + b_offset)

        return np.asarray(coordinates)

    @property
    def bonds(self) -> tuple[LatticeBond, ...]:
        """
        Nächste-Nachbar-Bonds des Honeycomb-Gitters.

        Jeder A-Platz verbindet sich mit:
        - B derselben Einheitszelle,
        - B der vorherigen x-Zelle,
        - B der vorherigen y-Zelle.
        """
        bonds: list[LatticeBond] = []

        for x in range(self.n_x):
            for y in range(self.n_y):
                a_site = self.site_index(x, y, "A")

                # Verbindung innerhalb derselben Einheitszelle.
                bonds.append(
                    LatticeBond(
                        a_site,
                        self.site_index(x, y, "B"),
                        direction=(0, 0),
                    )
                )

                # Verbindung zur vorherigen Einheitszelle in x-Richtung.
                if x > 0:
                    bonds.append(
                        LatticeBond(
                            a_site,
                            self.site_index(x - 1, y, "B"),
                            direction=(-1, 0),
                        )
                    )
                elif self.boundary_x == "periodic":
                    bonds.append(
                        LatticeBond(
                            a_site,
                            self.site_index(self.n_x - 1, y, "B"),
                            direction=(-1, 0),
                        )
                    )

                # Verbindung zur vorherigen Einheitszelle in y-Richtung.
                if y > 0:
                    bonds.append(
                        LatticeBond(
                            a_site,
                            self.site_index(x, y - 1, "B"),
                            direction=(0, -1),
                        )
                    )
                elif self.boundary_y == "periodic":
                    bonds.append(
                        LatticeBond(
                            a_site,
                            self.site_index(x, self.n_y - 1, "B"),
                            direction=(0, -1),
                        )
                    )

        return tuple(bonds)