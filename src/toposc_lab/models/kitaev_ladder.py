from __future__ import annotations

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import BasisLayout
from toposc_lab.lattices.ladder import LadderLattice


class KitaevLadderParameters(PydanticBaseModel):
    """
    Parameter einer Kitaev-Ladder.

    Energieeinheit:
        Häufig setzt man hopping = 1. Dann werden alle anderen Energien
        relativ zum Hopping entlang der einzelnen Ketten angegeben.

    Annahmen:
        - nichtwechselwirkende fermionische Zustände,
        - p-Wave-Paarung,
        - endliche Leiter mit n_legs gekoppelten Kitaev-Ketten.
    """

    n_legs: int = Field(
        default=2,
        gt=1,
        description="Number of coupled Kitaev chains.",
    )
    length: int = Field(
        ...,
        gt=1,
        description="Number of lattice sites per chain.",
    )
    hopping: float = Field(
        ...,
        description="Nearest-neighbor hopping along each leg.",
    )
    chemical_potential: float = Field(
        ...,
        description="Chemical potential mu.",
    )
    pairing: float = Field(
        ...,
        description="p-wave pairing amplitude along each leg.",
    )
    rung_hopping: float = Field(
        default=0.0,
        description="Normal hopping between adjacent legs.",
    )
    rung_pairing: float = Field(
        default=0.0,
        description="p-wave pairing amplitude between adjacent legs.",
    )
    boundary_length: str = Field(
        default="open",
        description="Boundary along the chains: open or periodic.",
    )
    boundary_legs: str = Field(
        default="open",
        description="Boundary across the legs: open or periodic.",
    )


class KitaevLadder(BaseModel):
    """
    Mehrere gekoppelte Kitaev-Ketten auf einer LadderLattice.

    Die BdG-Basis lautet:

        (c_0, c_1, ..., c_N-1, c_0†, c_1†, ..., c_N-1†)

    Also erst alle Elektron- und danach alle Loch-Komponenten.
    """

    def __init__(self, params: KitaevLadderParameters) -> None:
        self.params = params
        self.lattice = LadderLattice(
            n_legs=params.n_legs,
            length=params.length,
            boundary_length=params.boundary_length,
            boundary_legs=params.boundary_legs,
        )

    @property
    def basis_layout(self) -> BasisLayout:
        """
        Blockweise BdG-Basis.

        Die räumliche Form ist (leg, position), weil die Gitterplätze
        intern legweise durchnummeriert sind.
        """
        return BasisLayout(
            spatial_shape=(self.params.n_legs, self.params.length),
            components_per_site=2,
            ordering="component_major",
            component_labels=("electron", "hole"),
        )

    def hamiltonian(self) -> np.ndarray:
        """Baue die hermitesche BdG-Hamiltonmatrix der Kitaev Ladder."""
        n_sites = self.lattice.n_sites

        normal_hamiltonian = np.zeros(
            (n_sites, n_sites),
            dtype=complex,
        )
        pairing_matrix = np.zeros(
            (n_sites, n_sites),
            dtype=complex,
        )

        # Lokaler chemischer Potentialterm auf jedem Platz.
        for site in range(n_sites):
            normal_hamiltonian[site, site] = -self.params.chemical_potential

        for bond in self.lattice.bonds:
            source = bond.source
            target = bond.target

            # (1, 0): entlang der Ketten; (0, 1): zwischen den Ketten.
            if bond.direction == (1, 0):
                hopping = self.params.hopping
                pairing = self.params.pairing
            elif bond.direction == (0, 1):
                hopping = self.params.rung_hopping
                pairing = self.params.rung_pairing
            else:
                raise ValueError(f"Unsupported ladder bond direction: {bond.direction}")

            # Normaler Tight-Binding-Term.
            normal_hamiltonian[source, target] = -hopping
            normal_hamiltonian[target, source] = -hopping

            # Antisymmetrische p-Wave-Paarung.
            pairing_matrix[source, target] = pairing
            pairing_matrix[target, source] = -pairing

        upper_block = np.hstack(
            (normal_hamiltonian, pairing_matrix)
        )
        lower_block = np.hstack(
            (-pairing_matrix.conj(), -normal_hamiltonian.conj())
        )

        return np.vstack((upper_block, lower_block))