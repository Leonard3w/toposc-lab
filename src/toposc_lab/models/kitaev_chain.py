from __future__ import annotations

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field

from toposc_lab.core.model import BaseModel
from toposc_lab.lattices.chain import ChainLattice

from toposc_lab.core.results import BasisLayout


class KitaevChainParameters(PydanticBaseModel):
    """Parameter der eindimensionalen Kitaev-Kette."""

    n_sites: int = Field(..., gt=1, description="Number of lattice sites.")
    hopping: float = Field(..., description="Nearest-neighbor hopping amplitude t.")
    chemical_potential: float = Field(..., description="Chemical potential mu.")
    pairing: float = Field(..., description="p-wave superconducting pairing Delta.")
    boundary: str = Field(default="open", description="Boundary: open or periodic.")
    disorder_strength: float = Field(
        default=0.0,
        ge=0.0,
        description="Strength W of onsite chemical-potential disorder.",
    )
    disorder_seed: int | None = Field(
        default=None,
        description="Optional seed for reproducible disorder.",
    )


class KitaevChain(BaseModel):
    """
    Kitaev-Kette mit optionalem Onsite-Disorder.

    Die Geometrie der Kette wird von ChainLattice bereitgestellt.
    Dieses Modell definiert nur die physikalischen Hopping- und Paarungsterme.
    """

    def __init__(self, params: KitaevChainParameters) -> None:
        self.params = params
        self.lattice = ChainLattice(
            length=params.n_sites,
            boundary=params.boundary,
        )
        self._disorder_profile = self._create_disorder_profile()

    @property
    def basis_layout(self) -> BasisLayout:
        """
        Kitaev nutzt eine blockweise BdG-Basis:
        erst alle Elektronen, dann alle Löcher.
        """
        return BasisLayout(
            spatial_shape=(self.lattice.n_sites,),
            components_per_site=2,
            ordering="component_major",
            component_labels=("electron", "hole"),
        )

    def _create_disorder_profile(self) -> np.ndarray:
        """Erzeuge lokale Abweichungen delta_mu_i des chemischen Potentials."""
        if self.params.disorder_strength == 0.0:
            return np.zeros(self.lattice.n_sites)

        random_generator = np.random.default_rng(self.params.disorder_seed)

        return random_generator.uniform(
            low=-self.params.disorder_strength / 2.0,
            high=self.params.disorder_strength / 2.0,
            size=self.lattice.n_sites,
        )

    @property
    def disorder_profile(self) -> np.ndarray:
        """Gib eine Kopie des verwendeten Disorder-Profils zurück."""
        return self._disorder_profile.copy()

    def hamiltonian(self) -> np.ndarray:
        """Baue die Bogoliubov-de-Gennes-Hamiltonmatrix."""
        n_sites = self.lattice.n_sites
        hopping = self.params.hopping
        chemical_potential = self.params.chemical_potential
        pairing = self.params.pairing

        normal_hamiltonian = np.zeros((n_sites, n_sites), dtype=complex)
        pairing_matrix = np.zeros((n_sites, n_sites), dtype=complex)

        for site in range(n_sites):
            normal_hamiltonian[site, site] = -(
                chemical_potential + self._disorder_profile[site]
            )

        # Die Lattice-Bonds enthalten automatisch offene oder periodische Nachbarn.
        for bond in self.lattice.bonds:
            source = bond.source
            target = bond.target

            normal_hamiltonian[source, target] = -hopping
            normal_hamiltonian[target, source] = -hopping

            # Antisymmetrische p-Wave-Paarung.
            pairing_matrix[source, target] = pairing
            pairing_matrix[target, source] = -pairing

        upper = np.hstack((normal_hamiltonian, pairing_matrix))
        lower = np.hstack(
            (
                -pairing_matrix.conj(),
                -normal_hamiltonian.conj(),
            )
        )

        return np.vstack((upper, lower))