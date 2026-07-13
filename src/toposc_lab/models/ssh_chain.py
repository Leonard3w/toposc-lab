from __future__ import annotations

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field

from toposc_lab.core.model import BaseModel
from toposc_lab.lattices.chain import ChainLattice

from toposc_lab.core.results import BasisLayout


class SSHChainParameters(PydanticBaseModel):
    """Parameter der Su-Schrieffer-Heeger-Kette."""

    n_cells: int = Field(..., gt=1, description="Number of unit cells.")
    v: float = Field(..., description="Intracell hopping amplitude.")
    w: float = Field(..., description="Intercell hopping amplitude.")
    boundary: str = Field(default="open", description="Boundary: open or periodic.")


class SSHChain(BaseModel):
    """
    SSH-Kette mit zwei Plätzen A und B pro Einheitszelle.

    Die ChainLattice zählt die einzelnen A-/B-Plätze. Das SSH-Modell entscheidet
    anhand der Bond-Position, ob die Kopplung v oder w verwendet wird.
    """

    def __init__(self, params: SSHChainParameters) -> None:
        self.params = params
        self.lattice = ChainLattice(
            length=2 * params.n_cells,
            boundary=params.boundary,
        )
    @property
    def basis_layout(self) -> BasisLayout:
        """Ein orbitaler Zustand pro Gitterplatz."""
        return BasisLayout(
            spatial_shape=(self.lattice.n_sites,),
            components_per_site=1,
            ordering="site_major",
            component_labels=("orbital",),
        )    



    def hamiltonian(self) -> np.ndarray:
        """Baue die SSH-Hamiltonmatrix."""
        hamiltonian = np.zeros(
            (self.lattice.n_sites, self.lattice.n_sites),
            dtype=complex,
        )

        for bond in self.lattice.bonds:
            source = bond.source
            target = bond.target

            # Gerade Plätze sind A_i und ungerade Plätze B_i:
            # A_i -> B_i: v, B_i -> A_(i+1): w.
            hopping = self.params.v if source % 2 == 0 else self.params.w

            hamiltonian[source, target] = hopping
            hamiltonian[target, source] = hopping

        return hamiltonian