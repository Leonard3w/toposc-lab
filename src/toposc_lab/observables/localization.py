from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np

from toposc_lab.core.results import BasisLayout, SimulationResult
from toposc_lab.geometry.base import Geometry

LatticeShape: TypeAlias = tuple[int, ...]


def _normalized_probability(probability: np.ndarray) -> np.ndarray:
    """Validate a probability distribution and normalize it to one."""
    values = np.asarray(probability, dtype=float)

    if values.ndim == 0 or values.size == 0:
        raise ValueError("probability must be a non-empty array")
    if not np.all(np.isfinite(values)):
        raise ValueError("probability must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError("probability must not contain negative values")

    total_weight = float(np.sum(values))
    if total_weight <= 0.0:
        raise ValueError("probability must have positive total weight")

    return values / total_weight


def inverse_participation_ratio(
    probability: np.ndarray,
    *,
    component_axis: int | None = None,
) -> float:
    r"""Calculate the site-resolved inverse participation ratio (IPR).

    Large values indicate that probability is concentrated on few lattice
    sites, while small values indicate an extended state. If internal degrees
    of freedom are present, pass their axis as ``component_axis``. Their
    probabilities are then combined per physical site before evaluating
    ``IPR = sum_i p_i**2``, with ``p_i = sum_a |psi_(i,a)|**2``.
    """
    values = _normalized_probability(probability)
    if component_axis is not None:
        if isinstance(component_axis, bool) or not isinstance(component_axis, int):
            raise TypeError("component_axis must be an integer")
        if values.ndim < 2:
            raise ValueError("component_axis requires at least one site axis")
        if not -values.ndim <= component_axis < values.ndim:
            raise ValueError("component_axis is outside the probability dimensions")
        values = np.sum(values, axis=component_axis)

    return float(np.sum(values**2))


def participation_ratio(
    probability: np.ndarray,
    *,
    component_axis: int | None = None,
) -> float:
    """Calculate the effective number of lattice sites occupied by a state."""
    return float(
        1.0
        / inverse_participation_ratio(
            probability,
            component_axis=component_axis,
        )
    )


def boundary_weight(
    probability: np.ndarray,
    boundary_sites: Collection[int],
) -> float:
    """Return normalized weight on explicitly declared physical boundary sites.

    Site indices refer to the flattened canonical site order. No boundary is
    inferred from coordination number, connectivity, or spatial coordinates.
    An empty collection therefore represents a geometry without a boundary.
    """
    values = _normalized_probability(probability).reshape(-1)
    sites = _validated_boundary_sites(boundary_sites, n_sites=values.size)
    if not sites:
        return 0.0
    return float(np.sum(values[np.asarray(sites, dtype=np.intp)]))


def boundary_weight_from_geometry(
    probability: np.ndarray,
    geometry: Geometry,
) -> float:
    """Return boundary weight using only a geometry's explicit boundary sites."""
    values = np.asarray(probability)
    if values.size != geometry.n_sites:
        raise ValueError("probability must contain one value per geometry site")
    return boundary_weight(values, geometry.boundary_sites)


def boundary_weight_from_result(
    result: SimulationResult,
    state_index: int,
    geometry: Geometry,
) -> float:
    """Map an eigenstate to sites and evaluate its explicit boundary weight."""
    density = site_probability_density_from_result(result, state_index)
    return boundary_weight_from_geometry(density.probability, geometry)


def is_boundary_localized(
    probability: np.ndarray,
    boundary_sites: Collection[int],
    threshold: float = 0.5,
) -> bool:
    """Classify localization using an explicit boundary and weight threshold."""
    threshold = _validated_localization_threshold(threshold)
    return boundary_weight(probability, boundary_sites) >= threshold


def is_boundary_localized_from_geometry(
    probability: np.ndarray,
    geometry: Geometry,
    threshold: float = 0.5,
) -> bool:
    """Classify localization on a geometry's explicitly declared boundary."""
    threshold = _validated_localization_threshold(threshold)
    return boundary_weight_from_geometry(probability, geometry) >= threshold


def _validated_boundary_sites(
    boundary_sites: Collection[int],
    *,
    n_sites: int,
) -> tuple[int, ...]:
    if isinstance(boundary_sites, (str, bytes)) or not isinstance(
        boundary_sites,
        Collection,
    ):
        raise TypeError("boundary_sites must be a collection of integers")

    sites: set[int] = set()
    for site in boundary_sites:
        if isinstance(site, bool) or not isinstance(site, Integral):
            raise TypeError("boundary_sites must contain only integers")
        site = int(site)
        if not 0 <= site < n_sites:
            raise ValueError("boundary site is outside the probability array")
        sites.add(site)
    return tuple(sorted(sites))


def _validated_localization_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a real number")
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and between zero and one")
    return threshold


def _validate_edge_width(
    probability: np.ndarray,
    edge_width: int,
    axis: int | None = None,
) -> None:
    """Validate a boundary width for all dimensions or one selected axis."""
    if edge_width <= 0:
        raise ValueError("edge_width must be positive")

    if axis is not None:
        if not 0 <= axis < probability.ndim:
            raise ValueError("axis is outside the available dimensions")
        if edge_width > probability.shape[axis]:
            raise ValueError("edge_width exceeds the selected axis size")
        return

    if any(edge_width > size for size in probability.shape):
        raise ValueError("edge_width exceeds a lattice dimension")


def left_edge_weight(
    probability: np.ndarray,
    edge_width: int = 1,
    axis: int = 0,
) -> float:
    """Return normalized weight at the lower boundary of a selected axis."""
    values = _normalized_probability(probability)
    _validate_edge_width(values, edge_width, axis)

    selection = [slice(None)] * values.ndim
    selection[axis] = slice(0, edge_width)

    return float(np.sum(values[tuple(selection)]))


def right_edge_weight(
    probability: np.ndarray,
    edge_width: int = 1,
    axis: int = 0,
) -> float:
    """Return normalized weight at the upper boundary of a selected axis."""
    values = _normalized_probability(probability)
    _validate_edge_width(values, edge_width, axis)

    selection = [slice(None)] * values.ndim
    selection[axis] = slice(values.shape[axis] - edge_width, values.shape[axis])

    return float(np.sum(values[tuple(selection)]))


def edge_weight(probability: np.ndarray, edge_width: int = 1) -> float:
    """Return normalized weight at every boundary of a lattice."""
    values = _normalized_probability(probability)
    _validate_edge_width(values, edge_width)

    grid_indices = np.indices(values.shape)
    edge_mask = np.zeros(values.shape, dtype=bool)

    for axis, axis_size in enumerate(values.shape):
        coordinates = grid_indices[axis]
        edge_mask |= coordinates < edge_width
        edge_mask |= coordinates >= axis_size - edge_width

    return float(np.sum(values[edge_mask]))


def bulk_weight(probability: np.ndarray, edge_width: int = 1) -> float:
    """Return normalized weight in the bulk, outside every boundary region."""
    return float(1.0 - edge_weight(probability, edge_width=edge_width))


def is_edge_localized(
    probability: np.ndarray,
    edge_width: int = 1,
    threshold: float = 0.5,
) -> bool:
    """Classify a state as edge-localized from its boundary weight."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")

    return edge_weight(probability, edge_width=edge_width) >= threshold


@dataclass(frozen=True)
class SiteProbabilityDensity:
    """Normalized probability of one eigenstate on physical sites."""

    probability: np.ndarray
    component_probabilities: np.ndarray
    component_labels: tuple[str, ...]


def site_probability_density(
    eigenvectors: np.ndarray,
    state_index: int,
    basis_layout: BasisLayout,
) -> SiteProbabilityDensity:
    r"""Map one eigenvector from its declared basis to physical sites.

    Eigenstates are columns. The basis layout determines whether internal
    components are stored site-major or component-major. For each site,
    ``p_i = sum_a |psi_(i,a)|**2`` is returned after normalization.
    """
    vectors = np.asarray(eigenvectors, dtype=complex)
    if vectors.ndim != 2:
        raise ValueError("eigenvectors must be a two-dimensional array")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("eigenvectors must contain only finite values")
    if not 0 <= state_index < vectors.shape[1]:
        raise ValueError("state_index is outside the available eigenvector range")

    site_major_vectors = basis_layout.to_site_major(vectors)
    component_probabilities = np.abs(site_major_vectors[:, state_index]) ** 2
    normalization = float(np.sum(component_probabilities))
    if normalization <= 0.0:
        raise ValueError("selected eigenvector must have positive norm")

    component_probabilities = component_probabilities.reshape(
        *basis_layout.spatial_shape,
        basis_layout.components_per_site,
    )
    component_probabilities = component_probabilities / normalization

    return SiteProbabilityDensity(
        probability=np.sum(component_probabilities, axis=-1),
        component_probabilities=component_probabilities,
        component_labels=basis_layout.component_labels,
    )


def site_probability_density_from_result(
    result: SimulationResult,
    state_index: int,
) -> SiteProbabilityDensity:
    """Map one state from a standardized eigensolver result to physical sites."""
    return site_probability_density(
        eigenvectors=result.eigenvectors,
        state_index=state_index,
        basis_layout=result.basis_layout,
    )


@dataclass(frozen=True)
class LocalizationProfile:
    """Räumliche Analyse eines einzelnen Eigenzustands."""

    # Gesamtwahrscheinlichkeit pro Gitterplatz.
    probability: np.ndarray

    # Wahrscheinlichkeit je interner Komponente pro Gitterplatz.
    # Letzte Achse: Komponente, z. B. Spin oder Orbital.
    component_probabilities: np.ndarray

    # Schwerpunkt der Wellenfunktion in Gitterkoordinaten.
    center_of_mass: np.ndarray

    # Maße für Lokalisierung.
    inverse_participation_ratio: float
    participation_ratio: float

    # Wahrscheinlichkeit in der Randregion.
    edge_weight: float

    # Probability in the bulk and automatic edge-state classification.
    bulk_weight: float
    is_edge_localized: bool

    # Labels der internen Komponenten.
    component_labels: tuple[str, ...]


def localization_profile(
    eigenvectors: np.ndarray,
    state_index: int,
    lattice_shape: LatticeShape,
    components_per_site: int = 1,
    component_labels: tuple[str, ...] | None = None,
    edge_width: int = 1,
) -> LocalizationProfile:
    """
    Berechne das Lokalisierungsprofil eines Eigenzustands.

    Die Eigenvektoren müssen spaltenweise gespeichert sein, wie beim
    ExactDiagonalizationSolver. Die Basis wird als site-major angenommen:
    Alle internen Komponenten eines Gitterplatzes stehen direkt hintereinander.

    Beispiele:
    - SSH: lattice_shape=(2 * n_cells,), components_per_site=1
    - Kitaev: lattice_shape=(n_sites,), components_per_site=2
    - QWZ: lattice_shape=(n_x, n_y), components_per_site=2
    - BHZ: lattice_shape=(n_x, n_y), components_per_site=4
    """
    if edge_width <= 0:
        raise ValueError("edge_width must be positive")

    layout = BasisLayout(
        spatial_shape=lattice_shape,
        components_per_site=components_per_site,
        component_labels=component_labels or (),
    )
    density = site_probability_density(
        eigenvectors=eigenvectors,
        state_index=state_index,
        basis_layout=layout,
    )
    probability = density.probability
    component_probabilities = density.component_probabilities

    inverse_participation_ratio_value = inverse_participation_ratio(probability)
    participation_ratio_value = participation_ratio(probability)

    # Schwerpunkt: Summe aus Koordinate mal Wahrscheinlichkeit.
    grid_indices = np.indices(lattice_shape)
    center_of_mass = np.asarray(
        [
            np.sum(grid_indices[axis] * probability)
            for axis in range(len(lattice_shape))
        ],
        dtype=float,
    )

    # Ein Platz gehört zum Rand, wenn er in mindestens einer Richtung
    # innerhalb der gewählten Randbreite liegt.
    edge_weight_value = edge_weight(probability, edge_width=edge_width)
    bulk_weight_value = bulk_weight(probability, edge_width=edge_width)

    return LocalizationProfile(
        probability=probability,
        component_probabilities=component_probabilities,
        center_of_mass=center_of_mass,
        inverse_participation_ratio=inverse_participation_ratio_value,
        participation_ratio=participation_ratio_value,
        edge_weight=edge_weight_value,
        bulk_weight=bulk_weight_value,
        is_edge_localized=is_edge_localized(probability, edge_width=edge_width),
        component_labels=density.component_labels,
    )


def localization_profile_from_result(
    result: SimulationResult,
    state_index: int,
    edge_width: int = 1,
) -> LocalizationProfile:
    """
    Berechne ein Lokalisierungsprofil direkt aus einem SimulationResult.

    Die gespeicherte Basisordnung wird automatisch in die einheitliche
    site-major-Ordnung gebracht. Damit funktioniert dieselbe Ortsanalyse
    ohne Sonderfall für Kitaev, SSH, QWZ und BHZ.
    """
    layout = result.basis_layout

    return localization_profile(
        eigenvectors=result.site_major_eigenvectors(),
        state_index=state_index,
        lattice_shape=layout.spatial_shape,
        components_per_site=layout.components_per_site,
        component_labels=layout.component_labels,
        edge_width=edge_width,
    )
