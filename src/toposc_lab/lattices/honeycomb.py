from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toposc_lab.lattices.base import BaseLattice, LatticeBond


@dataclass(frozen=True)
class OrientedNextNearestNeighborBond:
    """
    Gerichtete Verbindung zwischen zwei nächsten-nächsten Nachbarn.

    chirality = +1 bedeutet: Der Weg source -> intermediate -> target
    macht in der 2D-Einbettung eine Drehung gegen den Uhrzeigersinn.
    """

    source: int
    target: int
    intermediate: int
    chirality: int

    def __post_init__(self) -> None:
        if self.chirality not in (-1, 1):
            raise ValueError("chirality must be either -1 or 1")


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

    @staticmethod
    def _nearest_neighbor_displacement(
        direction: tuple[int, int],
    ) -> np.ndarray:
        """
        Physikalischer Verschiebungsvektor eines A-zu-B-NN-Bonds.

        direction ist die Verschiebung der Einheitszelle. Diese Darstellung
        funktioniert auch für Bonds über periodische Ränder.
        """
        lattice_vector_x = np.array([np.sqrt(3.0), 0.0])
        lattice_vector_y = np.array([np.sqrt(3.0) / 2.0, 1.5])
        b_offset = np.array([np.sqrt(3.0) / 2.0, 0.5])

        return (
            direction[0] * lattice_vector_x
            + direction[1] * lattice_vector_y
            + b_offset
        )

    @property
    def next_nearest_neighbor_bonds(
        self,
    ) -> tuple[OrientedNextNearestNeighborBond, ...]:
        """
        Liefere eindeutige gerichtete nächste-nächste-Nachbar-Bonds.

        Sie verbinden Plätze derselben Sublattice. Die Chirality ist die
        geometrische Information für komplexe Haldane- und Kane-Mele-Terme.
        """
        adjacency: dict[int, list[tuple[int, np.ndarray]]] = {
            site: [] for site in range(self.n_sites)
        }

        # Jeder NN-Bond wird in beide Laufrichtungen aufgenommen.
        for bond in self.bonds:
            displacement = self._nearest_neighbor_displacement(
                bond.direction
            )

            adjacency[bond.source].append(
                (bond.target, displacement)
            )
            adjacency[bond.target].append(
                (bond.source, -displacement)
            )

        next_nearest_bonds: dict[
            tuple[int, int],
            OrientedNextNearestNeighborBond,
        ] = {}

        for source in range(self.n_sites):
            for intermediate, first_step in adjacency[source]:
                for target, second_step in adjacency[intermediate]:
                    if target == source:
                        continue

                    # Ein Weg über zwei NN-Schritte verläuft als
                    # A -> B -> A oder B -> A -> B.
                    if self.sublattice(source) != self.sublattice(target):
                        continue

                    # Jede Verbindung wird nur einmal gespeichert. Die
                    # Richtung source -> target ist über source < target
                    # eindeutig festgelegt.
                    if target <= source:
                        continue

                    cross_product_z = (
                        first_step[0] * second_step[1]
                        - first_step[1] * second_step[0]
                    )

                    chirality = 1 if cross_product_z > 0.0 else -1
                    key = (source, target)

                    if key not in next_nearest_bonds:
                        next_nearest_bonds[key] = (
                            OrientedNextNearestNeighborBond(
                                source=source,
                                target=target,
                                intermediate=intermediate,
                                chirality=chirality,
                            )
                        )

        return tuple(
            next_nearest_bonds[key]
            for key in sorted(next_nearest_bonds)
        )
