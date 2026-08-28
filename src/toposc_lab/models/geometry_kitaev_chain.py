"""Kitaev chain implemented with the model-independent Geometry API."""

from __future__ import annotations

import numpy as np

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import BasisLayout
from toposc_lab.geometry import Geometry, chain, ring
from toposc_lab.hamiltonians import build_tight_binding_hamiltonian
from toposc_lab.models.kitaev_chain import KitaevChainParameters


class GeometryKitaevChain(BaseModel):
    """Spinless Kitaev chain whose connectivity is supplied by ``Geometry``.

    This transitional model keeps the established component-major Nambu basis
    ``(c_0, ..., c_{N-1}, c_0^dagger, ..., c_{N-1}^dagger)``. Its normal block
    uses the generic tight-binding builder. Pairing remains local to this model
    until the general BdG builder is introduced in Phase 3.
    """

    def __init__(self, params: KitaevChainParameters) -> None:
        self.params = params
        self.geometry = _chain_geometry(params.n_sites, boundary=params.boundary)
        self._disorder_profile = self._create_disorder_profile()

    @property
    def basis_layout(self) -> BasisLayout:
        """Describe the established component-major electron-hole basis."""
        return BasisLayout(
            spatial_shape=(self.geometry.n_sites,),
            components_per_site=2,
            ordering="component_major",
            component_labels=("electron", "hole"),
        )

    @property
    def disorder_profile(self) -> np.ndarray:
        """Return a copy of the reproducible onsite-disorder realization."""
        return self._disorder_profile.copy()

    def hamiltonian(self) -> np.ndarray:
        """Build the component-major Bogoliubov-de-Gennes Hamiltonian."""
        normal_hamiltonian = build_tight_binding_hamiltonian(
            self.geometry,
            onsite=lambda site: -(
                self.params.chemical_potential + self._disorder_profile[site]
            ),
            hopping=-self.params.hopping,
        )
        pairing_matrix = self._pairing_matrix()

        upper = np.hstack((normal_hamiltonian, pairing_matrix))
        lower = np.hstack(
            (
                -pairing_matrix.conj(),
                -normal_hamiltonian.conj(),
            )
        )
        return np.vstack((upper, lower))

    def _create_disorder_profile(self) -> np.ndarray:
        if self.params.disorder_strength == 0.0:
            return np.zeros(self.geometry.n_sites)

        random_generator = np.random.default_rng(self.params.disorder_seed)
        return random_generator.uniform(
            low=-self.params.disorder_strength / 2.0,
            high=self.params.disorder_strength / 2.0,
            size=self.geometry.n_sites,
        )

    def _pairing_matrix(self) -> np.ndarray:
        pairing_matrix = np.zeros(
            (self.geometry.n_sites, self.geometry.n_sites),
            dtype=complex,
        )
        for edge in self.geometry.edges:
            pairing_matrix[edge.source, edge.target] += self.params.pairing
            pairing_matrix[edge.target, edge.source] -= self.params.pairing
        return pairing_matrix


def _chain_geometry(n_sites: int, *, boundary: str) -> Geometry:
    if boundary == "open":
        return chain(n_sites)
    if boundary != "periodic":
        raise ValueError("boundary must be either open or periodic")
    if n_sites > 2:
        return ring(n_sites)

    open_chain = chain(n_sites)
    return Geometry(
        n_sites=n_sites,
        edges=open_chain.edges,
        coordinates=open_chain.coordinates,
        metadata={
            "generator": "two_site_periodic_chain",
            "boundary_condition": "periodic",
            "intrinsic_dimension": 1,
        },
    )
