"""Explicit basis conventions for Bogoliubov-de-Gennes Hamiltonians."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from toposc_lab.core.results import BasisLayout, BasisOrdering


NambuSector: TypeAlias = Literal["particle", "hole"]


@dataclass(frozen=True, slots=True)
class NambuState:
    """Logical identity of one state in a Nambu basis."""

    site: int
    component: int
    sector: NambuSector


@dataclass(frozen=True, slots=True)
class NambuBasis:
    """Index convention for particle and hole states on discrete sites.

    Hole states retain the same internal-component order as particle states.
    No time-reversal rotation or additional phase convention is applied.

    In ``component_major`` ordering the combined components
    ``(particle 0, ..., particle m-1, hole 0, ..., hole m-1)`` are each stored
    across all sites. For one normal component this is the established Kitaev
    order ``(c_0, ..., c_N-1, c_0^dagger, ..., c_N-1^dagger)``.

    In ``site_major`` ordering all ``2m`` Nambu components of one site are
    contiguous before the next site begins.
    """

    n_sites: int
    normal_components_per_site: int = 1
    ordering: BasisOrdering = "component_major"
    normal_component_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        n_sites = _positive_integer(self.n_sites, name="n_sites")
        normal_components = _positive_integer(
            self.normal_components_per_site,
            name="normal_components_per_site",
        )
        if self.ordering not in ("site_major", "component_major"):
            raise ValueError("ordering must be either 'site_major' or 'component_major'")

        labels = tuple(self.normal_component_labels)
        if not labels:
            labels = (
                ("spinless",)
                if normal_components == 1
                else tuple(f"component {index}" for index in range(normal_components))
            )
        if len(labels) != normal_components:
            raise ValueError(
                "normal_component_labels must match normal_components_per_site"
            )
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("normal_component_labels must be non-empty strings")
        if len(set(labels)) != len(labels):
            raise ValueError("normal_component_labels must be unique")

        object.__setattr__(self, "n_sites", n_sites)
        object.__setattr__(self, "normal_components_per_site", normal_components)
        object.__setattr__(self, "normal_component_labels", labels)

    @property
    def normal_dimension(self) -> int:
        """Dimension of the undoubled single-particle Hilbert space."""
        return self.n_sites * self.normal_components_per_site

    @property
    def nambu_components_per_site(self) -> int:
        """Number of particle and hole components together at one site."""
        return 2 * self.normal_components_per_site

    @property
    def dimension(self) -> int:
        """Total dimension after Nambu doubling."""
        return 2 * self.normal_dimension

    @property
    def component_labels(self) -> tuple[str, ...]:
        """Labels of combined particle and hole components."""
        if self.normal_components_per_site == 1:
            return ("electron", "hole")
        return tuple(
            f"{sector} {label}"
            for sector in ("particle", "hole")
            for label in self.normal_component_labels
        )

    @property
    def basis_layout(self) -> BasisLayout:
        """Return the equivalent generic result-layout description."""
        return BasisLayout(
            spatial_shape=(self.n_sites,),
            components_per_site=self.nambu_components_per_site,
            ordering=self.ordering,
            component_labels=self.component_labels,
        )

    @property
    def particle_indices(self) -> tuple[int, ...]:
        """Particle indices in site-major normal-state order."""
        return self._sector_indices("particle")

    @property
    def hole_indices(self) -> tuple[int, ...]:
        """Hole indices in site-major normal-state order."""
        return self._sector_indices("hole")

    @property
    def particle_hole_operator(self) -> NDArray[np.complex128]:
        r"""Unitary part of the antiunitary operator ``C = U_C K``.

        ``U_C`` exchanges particle and hole states at fixed site and internal
        component. With the unrotated hole convention used by this basis,
        ``U_C U_C^* = I`` and therefore ``C^2 = +1``.
        """
        operator = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
        for particle_index, hole_index in zip(
            self.particle_indices,
            self.hole_indices,
            strict=True,
        ):
            operator[particle_index, hole_index] = 1.0
            operator[hole_index, particle_index] = 1.0
        return operator

    def index(
        self,
        site: int,
        *,
        component: int = 0,
        sector: NambuSector,
    ) -> int:
        """Map a logical Nambu state to its flat matrix index."""
        site = self._validated_site(site)
        component = self._validated_component(component)
        sector_offset = self._sector_offset(sector)
        combined_component = sector_offset + component

        if self.ordering == "site_major":
            return site * self.nambu_components_per_site + combined_component
        return combined_component * self.n_sites + site

    def particle_index(self, site: int, *, component: int = 0) -> int:
        """Return the particle index for a site and normal component."""
        return self.index(site, component=component, sector="particle")

    def hole_index(self, site: int, *, component: int = 0) -> int:
        """Return the corresponding hole index."""
        return self.index(site, component=component, sector="hole")

    def decode(self, index: int) -> NambuState:
        """Decode a flat matrix index into site, component, and sector."""
        index = self._validated_flat_index(index)

        if self.ordering == "site_major":
            site, combined_component = divmod(index, self.nambu_components_per_site)
        else:
            combined_component, site = divmod(index, self.n_sites)

        if combined_component < self.normal_components_per_site:
            sector: NambuSector = "particle"
            component = combined_component
        else:
            sector = "hole"
            component = combined_component - self.normal_components_per_site

        return NambuState(site=site, component=component, sector=sector)

    def partner_index(self, index: int) -> int:
        """Return the index with particle and hole sectors exchanged."""
        state = self.decode(index)
        partner_sector: NambuSector = "hole" if state.sector == "particle" else "particle"
        return self.index(
            state.site,
            component=state.component,
            sector=partner_sector,
        )

    def permutation_to(self, ordering: BasisOrdering) -> NDArray[np.int64]:
        """Return indices ``p`` such that ``target_values = source_values[p]``."""
        if ordering not in ("site_major", "component_major"):
            raise ValueError("ordering must be either 'site_major' or 'component_major'")
        target_basis = replace(self, ordering=ordering)
        permutation = np.empty(self.dimension, dtype=np.int64)

        for source_index in range(self.dimension):
            state = self.decode(source_index)
            target_index = target_basis.index(
                state.site,
                component=state.component,
                sector=state.sector,
            )
            permutation[target_index] = source_index

        return permutation

    def reorder_states(
        self,
        states: np.ndarray,
        *,
        ordering: BasisOrdering,
    ) -> np.ndarray:
        """Reorder a state vector or eigenvector matrix to another ordering."""
        values = np.asarray(states)
        if values.ndim not in (1, 2) or values.shape[0] != self.dimension:
            raise ValueError(
                "states must be a vector or matrix with the Nambu dimension on axis 0"
            )
        return values[self.permutation_to(ordering)].copy()

    def reorder_matrix(
        self,
        matrix: np.ndarray,
        *,
        ordering: BasisOrdering,
    ) -> np.ndarray:
        """Reorder both axes of a square operator to another ordering."""
        values = np.asarray(matrix)
        if values.shape != (self.dimension, self.dimension):
            raise ValueError(
                f"matrix must have shape ({self.dimension}, {self.dimension})"
            )
        permutation = self.permutation_to(ordering)
        return values[np.ix_(permutation, permutation)].copy()

    def _sector_indices(self, sector: NambuSector) -> tuple[int, ...]:
        return tuple(
            self.index(site, component=component, sector=sector)
            for site in range(self.n_sites)
            for component in range(self.normal_components_per_site)
        )

    def _validated_site(self, site: int) -> int:
        site = _integer(site, name="site")
        if not 0 <= site < self.n_sites:
            raise ValueError(f"site {site} is outside the Nambu basis")
        return site

    def _validated_component(self, component: int) -> int:
        component = _integer(component, name="component")
        if not 0 <= component < self.normal_components_per_site:
            raise ValueError(f"component {component} is outside the Nambu basis")
        return component

    def _validated_flat_index(self, index: int) -> int:
        index = _integer(index, name="index")
        if not 0 <= index < self.dimension:
            raise ValueError(f"index {index} is outside the Nambu basis")
        return index

    def _sector_offset(self, sector: NambuSector) -> int:
        if sector == "particle":
            return 0
        if sector == "hole":
            return self.normal_components_per_site
        raise ValueError("sector must be either 'particle' or 'hole'")


def _positive_integer(value: int, *, name: str) -> int:
    value = _integer(value, name=name)
    if value < 1:
        raise ValueError(f"{name} must be at least one")
    return value


def _integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)
