from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

LatticeShape: TypeAlias = tuple[int, ...]


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

    inverse_participation_ratio = float(np.sum(probability**2))
    participation_ratio = float(1.0 / inverse_participation_ratio)

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
    edge_mask = np.zeros(lattice_shape, dtype=bool)

    for axis, axis_size in enumerate(lattice_shape):
        coordinates = grid_indices[axis]
        edge_mask |= coordinates < edge_width
        edge_mask |= coordinates >= axis_size - edge_width

    edge_weight = float(np.sum(probability[edge_mask]))

    return LocalizationProfile(
        probability=probability,
        component_probabilities=component_probabilities,
        center_of_mass=center_of_mass,
        inverse_participation_ratio=inverse_participation_ratio,
        participation_ratio=participation_ratio,
        edge_weight=edge_weight,
        component_labels=component_labels,
    )