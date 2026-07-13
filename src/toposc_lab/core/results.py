from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

BasisOrdering = Literal["site_major", "component_major"]


@dataclass(frozen=True)
class BasisLayout:
    """
    Beschreibt, wie die Basiszustände im Hamiltonoperator angeordnet sind.

    site_major:
        (Komponente 0, Komponente 1, ...) für Platz 0,
        dann für Platz 1, ...

    component_major:
        Erst alle Plätze von Komponente 0, dann alle Plätze von Komponente 1.
        Die Kitaev-Kette nutzt diese Ordnung für Elektronen und Löcher.
    """

    spatial_shape: tuple[int, ...]
    components_per_site: int = 1
    ordering: BasisOrdering = "site_major"
    component_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        shape = tuple(self.spatial_shape)

        if not shape or any(size <= 0 for size in shape):
            raise ValueError("spatial_shape must contain positive dimensions")

        if self.components_per_site <= 0:
            raise ValueError("components_per_site must be positive")

        if self.ordering not in ("site_major", "component_major"):
            raise ValueError(
                "ordering must be either 'site_major' or 'component_major'"
            )

        labels = self.component_labels
        if not labels:
            labels = tuple(
                f"component {index}"
                for index in range(self.components_per_site)
            )

        if len(labels) != self.components_per_site:
            raise ValueError(
                "component_labels must match components_per_site"
            )

        object.__setattr__(self, "spatial_shape", shape)
        object.__setattr__(self, "component_labels", tuple(labels))

    @property
    def n_sites(self) -> int:
        """Anzahl räumlicher Gitterplätze."""
        return int(np.prod(self.spatial_shape))

    @property
    def dimension(self) -> int:
        """Gesamtdimension eines Zustandsvektors."""
        return self.n_sites * self.components_per_site

    def to_site_major(self, eigenvectors: np.ndarray) -> np.ndarray:
        """
        Bringe Eigenvektoren in die einheitliche site-major-Reihenfolge.

        Das ist besonders wichtig für räumliche Observablen wie
        Lokalisierungsprofile oder Heatmaps.
        """
        vectors = np.asarray(eigenvectors, dtype=complex)

        if vectors.ndim != 2:
            raise ValueError("eigenvectors must be a two-dimensional array")

        if vectors.shape[0] != self.dimension:
            raise ValueError(
                "Eigenvector dimension does not match the basis layout"
            )

        if self.ordering == "site_major":
            return vectors.copy()

        n_states = vectors.shape[1]

        return (
            vectors.reshape(
                self.components_per_site,
                self.n_sites,
                n_states,
            )
            .transpose(1, 0, 2)
            .reshape(self.dimension, n_states)
        )

    def to_component_major(self, eigenvectors: np.ndarray) -> np.ndarray:
        """Bringe site-major-Eigenvektoren in component-major-Reihenfolge."""
        vectors = np.asarray(eigenvectors, dtype=complex)

        if vectors.ndim != 2:
            raise ValueError("eigenvectors must be a two-dimensional array")

        if vectors.shape[0] != self.dimension:
            raise ValueError(
                "Eigenvector dimension does not match the basis layout"
            )

        if self.ordering == "component_major":
            return vectors.copy()

        n_states = vectors.shape[1]

        return (
            vectors.reshape(
                self.n_sites,
                self.components_per_site,
                n_states,
            )
            .transpose(1, 0, 2)
            .reshape(self.dimension, n_states)
        )


@dataclass(frozen=True)
class SimulationResult:
    """
    Einheitliches Ergebnis einer Hamilton-Diagonalisierung.

    Es enthält nicht nur Eigenwerte und Eigenvektoren, sondern auch die
    Basisordnung, Modellparameter und optionale Metadaten.
    """

    model_name: str
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    basis_layout: BasisLayout
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        eigenvalues = np.asarray(self.eigenvalues, dtype=float).copy()
        eigenvectors = np.asarray(self.eigenvectors, dtype=complex).copy()

        if eigenvalues.ndim != 1:
            raise ValueError("eigenvalues must be one-dimensional")

        if eigenvectors.ndim != 2:
            raise ValueError("eigenvectors must be two-dimensional")

        if eigenvectors.shape[0] != self.basis_layout.dimension:
            raise ValueError(
                "Eigenvector dimension does not match the basis layout"
            )

        if eigenvectors.shape[1] != eigenvalues.size:
            raise ValueError(
                "Number of eigenvectors must match number of eigenvalues"
            )

        if not np.all(np.isfinite(eigenvalues)):
            raise ValueError("eigenvalues must be finite")

        if not np.all(np.isfinite(eigenvectors)):
            raise ValueError("eigenvectors must be finite")

        eigenvalues.setflags(write=False)
        eigenvectors.setflags(write=False)

        object.__setattr__(self, "eigenvalues", eigenvalues)
        object.__setattr__(self, "eigenvectors", eigenvectors)
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def dimension(self) -> int:
        """Dimension der Hamiltonmatrix."""
        return self.basis_layout.dimension

    @property
    def n_states(self) -> int:
        """Anzahl der Eigenzustände."""
        return self.eigenvalues.size

    def site_major_eigenvectors(self) -> np.ndarray:
        """
        Eigenvektoren in der Standardreihenfolge für Ortsanalysen.

        Diese Methode ersetzt das bisherige manuelle Umsortieren bei Kitaev.
        """
        return self.basis_layout.to_site_major(self.eigenvectors)


@dataclass(frozen=True)
class ParameterScanResult:
    """Standardisiertes Ergebnis eines eindimensionalen Parameterscans."""

    parameter_name: str
    parameter_values: np.ndarray
    results: tuple[SimulationResult, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.parameter_values).copy()

        if values.ndim != 1 or values.size == 0:
            raise ValueError(
                "parameter_values must be a non-empty one-dimensional array"
            )

        if not self.parameter_name:
            raise ValueError("parameter_name must not be empty")

        if len(self.results) != values.size:
            raise ValueError(
                "Each parameter value must have exactly one SimulationResult"
            )

        values.setflags(write=False)

        object.__setattr__(self, "parameter_values", values)
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def spectra(self) -> np.ndarray:
        """Eigenwertspektren aller Scanpunkte."""
        return np.asarray(
            [result.eigenvalues for result in self.results],
            dtype=float,
        )

    @property
    def gaps(self) -> np.ndarray:
        """Kleinster Energiebetrag für jeden Scanpunkt."""
        return np.min(np.abs(self.spectra), axis=1)