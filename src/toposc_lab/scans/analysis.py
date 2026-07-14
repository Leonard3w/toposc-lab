"""Standardisierte physikalische Auswertung von Parameterscans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toposc_lab.core.results import ParameterScanResult
from toposc_lab.observables.localization import localization_profile_from_result
from toposc_lab.observables.spectrum import bulk_gap, count_zero_modes, lowest_abs_energy


@dataclass(frozen=True)
class ParameterScanAnalysis:
    """Physikalische Observablen, die an jedem Punkt eines 1D-Scans entstehen."""

    scan: ParameterScanResult
    minimum_abs_energies: np.ndarray
    bulk_gaps: np.ndarray
    zero_mode_counts: np.ndarray
    inverse_participation_ratios: np.ndarray
    edge_weights: np.ndarray

    def observable_arrays(self) -> dict[str, np.ndarray]:
        """Liefere die Analyse in einer Form fuer StudyData und Visualisierungen."""
        return {
            "bulk_gaps": self.bulk_gaps,
            "zero_mode_counts": self.zero_mode_counts,
            "inverse_participation_ratios": self.inverse_participation_ratios,
            "edge_weights": self.edge_weights,
        }


def analyze_parameter_scan(
    scan: ParameterScanResult,
    *,
    zero_mode_tolerance: float = 1.0e-8,
    edge_width: int = 1,
) -> ParameterScanAnalysis:
    """Berechne zentrale Spektral- und Lokalisierungsobservablen pro Scanpunkt."""
    if zero_mode_tolerance < 0.0:
        raise ValueError("zero_mode_tolerance must be non-negative")

    if edge_width <= 0:
        raise ValueError("edge_width must be positive")

    minimum_abs_energies: list[float] = []
    bulk_gaps: list[float] = []
    zero_mode_counts: list[int] = []
    inverse_participation_ratios: list[float] = []
    edge_weights: list[float] = []

    for result in scan.results:
        state_index = int(np.argmin(np.abs(result.eigenvalues)))
        profile = localization_profile_from_result(
            result,
            state_index=state_index,
            edge_width=edge_width,
        )

        minimum_abs_energies.append(lowest_abs_energy(result.eigenvalues))
        bulk_gaps.append(bulk_gap(result.eigenvalues, zero_mode_tolerance))
        zero_mode_counts.append(
            count_zero_modes(result.eigenvalues, zero_mode_tolerance)
        )
        inverse_participation_ratios.append(profile.inverse_participation_ratio)
        edge_weights.append(profile.edge_weight)

    return ParameterScanAnalysis(
        scan=scan,
        minimum_abs_energies=np.asarray(minimum_abs_energies),
        bulk_gaps=np.asarray(bulk_gaps),
        zero_mode_counts=np.asarray(zero_mode_counts),
        inverse_participation_ratios=np.asarray(inverse_participation_ratios),
        edge_weights=np.asarray(edge_weights),
    )
