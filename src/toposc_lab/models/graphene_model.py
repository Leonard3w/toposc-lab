from __future__ import annotations

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import BasisLayout
from toposc_lab.lattices.honeycomb import HoneycombLattice


class GrapheneParameters(PydanticBaseModel):
    """
    Parameter des spinlosen Nearest-Neighbor-Graphenmodells.

    Energieeinheit:
        Meist wird hopping = 1 gesetzt. Alle Energien werden dann in
        Einheiten des nächsten-Nachbar-Hoppings angegeben.

    Annahmen:
        - ein p_z-Orbital pro Kohlenstoffatom,
        - keine Elektron-Elektron-Wechselwirkung,
        - kein Spin und keine Spin-Bahn-Kopplung,
        - nur nächste Nachbarn.

    Referenz:
        A. H. Castro Neto et al., Rev. Mod. Phys. 81, 109 (2009).
        DOI: 10.1103/RevModPhys.81.109
    """

    n_x: int = Field(
        ...,
        gt=1,
        description="Number of honeycomb unit cells in x direction.",
    )
    n_y: int = Field(
        ...,
        gt=1,
        description="Number of honeycomb unit cells in y direction.",
    )
    hopping: float = Field(
        default=1.0,
        description="Nearest-neighbor hopping amplitude t.",
    )
    boundary_x: str = Field(
        default="open",
        description="Boundary in x direction: open or periodic.",
    )
    boundary_y: str = Field(
        default="open",
        description="Boundary in y direction: open or periodic.",
    )


class GrapheneModel(BaseModel):
    """
    Spinloses Tight-Binding-Modell von Graphen auf dem Honeycomb-Gitter.

    Jede Einheitszelle enthält zwei physikalische Plätze:
    die A- und B-Sublattice. Das Fehlen von Onsite-Terms und das reine
    A-B-Hopping führen zu chiraler Subgitter-Symmetrie.
    """

    def __init__(self, params: GrapheneParameters) -> None:
        self.params = params
        self.lattice = HoneycombLattice(
            n_x=params.n_x,
            n_y=params.n_y,
            boundary_x=params.boundary_x,
            boundary_y=params.boundary_y,
        )

    @property
    def basis_layout(self) -> BasisLayout:
        """
        Eine Einheitszelle ist ein 2D-Gitterplatz mit zwei Komponenten A/B.

        Das entspricht der Reihenfolge
        (A_00, B_00, A_01, B_01, ...).
        """
        return BasisLayout(
            spatial_shape=(self.params.n_x, self.params.n_y),
            components_per_site=2,
            ordering="site_major",
            component_labels=("A sublattice", "B sublattice"),
        )

    def hamiltonian(self) -> np.ndarray:
        """
        Baue H = -t sum_<i,j> (c_i† c_j + h.c.).

        Da HoneycombLattice nur A-B-Nächste-Nachbar-Bonds liefert,
        entstehen automatisch die charakteristischen Graphen-Bänder.
        """
        dimension = self.lattice.n_sites

        hamiltonian = np.zeros(
            (dimension, dimension),
            dtype=complex,
        )

        for bond in self.lattice.bonds:
            source = bond.source
            target = bond.target

            hamiltonian[source, target] = -self.params.hopping
            hamiltonian[target, source] = -self.params.hopping

        return hamiltonian
    