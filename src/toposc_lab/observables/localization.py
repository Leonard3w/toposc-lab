from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from toposc_lab.core.results import SimulationResult

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


def inverse_participation_ratio(probability: np.ndarray) -> float:
    """
    Calculate the inverse participation ratio (IPR).

    Large values indicate that probability is concentrated on few lattice
    sites, while small values indicate an extended state.
    """
    values = _normalized_probability(probability)

    return float(np.sum(values**2))


def participation_ratio(probability: np.ndarray) -> float:
    """Calculate the effective number of lattice sites occupied by a state."""
    return float(1.0 / inverse_participation_ratio(probability))


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
    vectors = np.asarray(eigenvectors, dtype=complex)

    if vectors.ndim != 2:
        raise ValueError("eigenvectors must be a two-dimensional array")

    if not 0 <= state_index < vectors.shape[1]:
        raise ValueError("state_index is outside the available eigenvector range")

    if not lattice_shape or any(size <= 0 for size in lattice_shape):
        raise ValueError("lattice_shape must contain positive dimensions")

    if components_per_site <= 0:
        raise ValueError("components_per_site must be positive")

    if edge_width <= 0:
        raise ValueError("edge_width must be positive")

    n_sites = int(np.prod(lattice_shape))
    expected_dimension = n_sites * components_per_site

    if vectors.shape[0] != expected_dimension:
        raise ValueError(
            "Eigenvector dimension does not match lattice_shape and components_per_site"
        )

    if component_labels is None:
        component_labels = tuple(
            f"component {index}" for index in range(components_per_site)
        )

    if len(component_labels) != components_per_site:
        raise ValueError("component_labels must match components_per_site")

    # Ein Eigenzustand ist eine Spalte der Eigenvektormatrix.
    wavefunction = vectors[:, state_index]

    # |psi|² liefert die Wahrscheinlichkeit jeder Basis-Komponente.
    component_probabilities = np.abs(wavefunction) ** 2

    normalization = float(np.sum(component_probabilities))
    if normalization <= 0.0:
        raise ValueError("Selected eigenvector has zero norm")

    # Numerische Normierung: Die Gesamtwahrscheinlichkeit wird exakt eins.
    component_probabilities /= normalization

    # Interne Komponenten eines Platzes werden zu einer Ortswahrscheinlichkeit addiert.
    component_probabilities = component_probabilities.reshape(
        *lattice_shape,
        components_per_site,
    )
    probability = np.sum(component_probabilities, axis=-1)

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
        component_labels=component_labels,
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
