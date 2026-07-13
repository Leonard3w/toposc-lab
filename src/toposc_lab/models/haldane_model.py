from __future__ import annotations

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import BasisLayout
from toposc_lab.lattices.honeycomb import HoneycombLattice


class HaldaneModelParameters(PydanticBaseModel):
    """
    Parameter des Haldane-Chern-Isolators auf dem Honeycomb-Gitter.

    Energieeinheit:
        Meist setzt man nearest_neighbor_hopping = 1.

    Annahmen:
        - spinlose, nichtwechselwirkende Fermionen,
        - ein Orbital pro Honeycomb-Platz,
        - komplexe NNN-Hoppings, aber null Gesamtfluss pro Einheitszelle.

    Referenz:
        F. D. M. Haldane, Phys. Rev. Lett. 61, 2015 (1988).
        DOI: 10.1103/PhysRevLett.61.2015
    """

    n_x: int = Field(..., gt=1)
    n_y: int = Field(..., gt=1)

    nearest_neighbor_hopping: float = Field(
        default=1.0,
        description="Real nearest-neighbor hopping t1.",
    )
    next_nearest_neighbor_hopping: float = Field(
        default=0.15,
        description="Magnitude t2 of complex next-nearest-neighbor hopping.",
    )
    phase: float = Field(
        default=float(np.pi / 2.0),
        description="Haldane phase phi in radians.",
    )
    sublattice_mass: float = Field(
        default=0.0,
        description="Staggered A/B onsite energy M.",
    )

    boundary_x: str = Field(default="open")
    boundary_y: str = Field(default="open")


class HaldaneModel(BaseModel):
    """
    Haldane-Modell in Realraumdarstellung.

    Die komplexen NNN-Hoppings erzeugen lokale Umlaufströme. Ihr
    Gesamtfluss pro Einheitszelle ist null, aber Zeitumkehrsymmetrie
    ist gebrochen.
    """

    def __init__(self, params: HaldaneModelParameters) -> None:
        self.params = params
        self.lattice = HoneycombLattice(
            n_x=params.n_x,
            n_y=params.n_y,
            boundary_x=params.boundary_x,
            boundary_y=params.boundary_y,
        )

    @property
    def basis_layout(self) -> BasisLayout:
        """Zwei Honeycomb-Komponenten A/B pro Einheitszelle."""
        return BasisLayout(
            spatial_shape=(self.params.n_x, self.params.n_y),
            components_per_site=2,
            ordering="site_major",
            component_labels=("A sublattice", "B sublattice"),
        )

    def bloch_hamiltonian(self, k_x: float, k_y: float) -> np.ndarray:
        """
        Baue die 2 x 2 Bulk-Hamiltonmatrix H(k_x, k_y).

        k_x und k_y sind dimensionslose Impulskoordinaten zu den beiden
        Einheitszellrichtungen und liegen konventionell im Intervall
        [-pi, pi). Diese Darstellung wird für Berry-Krümmung und
        Chern-Zahl mit periodischen Bulk-Randbedingungen verwendet.
        """
        nearest_hopping = self.params.nearest_neighbor_hopping
        next_nearest_hopping = self.params.next_nearest_neighbor_hopping
        phase = self.params.phase
        mass = self.params.sublattice_mass

        # Die drei unabhängigen Triangular-Lattice-Vektoren der A- bzw.
        # B-Sublattice. Die Vorzeichen kodieren die Haldane-Umlaufrichtung
        # auf der A-Sublattice; auf B ist sie entgegengesetzt.
        next_nearest_vectors = np.asarray(
            [
                (1.0, 0.0),
                (0.0, 1.0),
                (1.0, -1.0),
            ]
        )
        a_chiralities = np.asarray((-1.0, 1.0, 1.0))
        momenta = next_nearest_vectors @ np.asarray((k_x, k_y))

        a_diagonal = mass + 2.0 * next_nearest_hopping * np.sum(
            np.cos(momenta + a_chiralities * phase)
        )
        b_diagonal = -mass + 2.0 * next_nearest_hopping * np.sum(
            np.cos(momenta - a_chiralities * phase)
        )

        # Nächste-Nachbar-Graphen-Hopping zwischen A und B.
        off_diagonal = -nearest_hopping * (
            1.0 + np.exp(-1.0j * k_x) + np.exp(-1.0j * k_y)
        )

        return np.asarray(
            [
                [a_diagonal, off_diagonal],
                [off_diagonal.conjugate(), b_diagonal],
            ],
            dtype=complex,
        )

    def hamiltonian(self) -> np.ndarray:
        """Baue die hermitesche Haldane-Hamiltonmatrix."""
        dimension = self.lattice.n_sites

        hamiltonian = np.zeros(
            (dimension, dimension),
            dtype=complex,
        )

        # Gestaffelter Massenterm: +M auf A, -M auf B.
        for site in range(self.lattice.n_sites):
            sublattice_sign = (
                1.0 if self.lattice.sublattice(site) == "A" else -1.0
            )
            hamiltonian[site, site] = (
                sublattice_sign * self.params.sublattice_mass
            )

        # Reales Graphen-Hopping zwischen A und B.
        for bond in self.lattice.bonds:
            source = bond.source
            target = bond.target

            hamiltonian[source, target] += (
                -self.params.nearest_neighbor_hopping
            )
            hamiltonian[target, source] += (
                -self.params.nearest_neighbor_hopping
            )

        # Komplexes Haldane-Hopping zwischen gleichen Sublattices.
        for bond in self.lattice.next_nearest_neighbor_bonds:
            phase_factor = np.exp(
                1.0j * bond.chirality * self.params.phase
            )
            hopping = (
                self.params.next_nearest_neighbor_hopping
                * phase_factor
            )

            hamiltonian[bond.source, bond.target] += hopping
            hamiltonian[bond.target, bond.source] += hopping.conjugate()

        return hamiltonian
