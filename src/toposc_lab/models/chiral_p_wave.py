"""Spinless chiral p-wave superconductor on an arbitrary embedded geometry."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import BasisLayout
from toposc_lab.geometry import Geometry
from toposc_lab.hamiltonians import (
    NambuBasis,
    build_bdg_hamiltonian,
    build_chiral_p_wave_pairing,
    build_tight_binding_hamiltonian,
)


class ChiralPWaveParameters(PydanticBaseModel):
    """Parameters of a spinless ``p_x + i chirality p_y`` model."""

    hopping: float = Field(..., description="Spinless edge-hopping amplitude.")
    chemical_potential: float = Field(..., description="Chemical potential.")
    pairing: float = Field(..., description="Chiral p-wave pairing amplitude.")
    chirality: Literal[-1, 1] = Field(
        default=1,
        description="Chirality +1 or -1.",
    )
    plane_axes: tuple[int, int] = Field(
        default=(0, 1),
        description="Embedding axes defining the chiral pairing plane.",
    )


class ChiralPWaveModel(BaseModel):
    """Geometry-based spinless chiral p-wave BdG model."""

    def __init__(self, geometry: Geometry, params: ChiralPWaveParameters) -> None:
        self.geometry = geometry
        self.params = params

    @property
    def nambu_basis(self) -> NambuBasis:
        """Component-major spinless particle-hole basis."""
        return NambuBasis(n_sites=self.geometry.n_sites, ordering="component_major")

    @property
    def basis_layout(self) -> BasisLayout:
        """Result layout corresponding to the model's Nambu basis."""
        return self.nambu_basis.basis_layout

    def normal_hamiltonian(self) -> np.ndarray:
        """Build chemical-potential and spinless edge-hopping terms."""
        return build_tight_binding_hamiltonian(
            self.geometry,
            onsite=-self.params.chemical_potential,
            hopping=-self.params.hopping,
        )

    def pairing_matrix(self) -> np.ndarray:
        """Build the oriented chiral p-wave pairing block."""
        return build_chiral_p_wave_pairing(
            self.geometry,
            pairing=self.params.pairing,
            chirality=self.params.chirality,
            plane_axes=self.params.plane_axes,
        )

    def hamiltonian(self) -> np.ndarray:
        """Build the Hermitian chiral p-wave BdG Hamiltonian."""
        return build_bdg_hamiltonian(
            self.normal_hamiltonian(),
            self.pairing_matrix(),
            basis=self.nambu_basis,
        )
