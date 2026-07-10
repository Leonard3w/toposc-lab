from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toposc_lab.models.bhz_model import BHZModel, BHZModelParameters
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver


@dataclass(frozen=True)
class BHZScanResult:
    """Ergebnisse eines BHZ-Scans über den Massenparameter."""

    mass_values: np.ndarray
    spectra: np.ndarray
    gaps: np.ndarray


def scan_bhz_mass(
    mass_values: np.ndarray,
    n_x: int = 6,
    n_y: int = 6,
    boundary_x: str = "periodic",
    boundary_y: str = "periodic",
) -> BHZScanResult:
    """
    Berechne das BHZ-Spektrum für verschiedene Massenwerte.

    Mit periodischen Rändern wird das Bulk-Spektrum ohne Randzustände
    berechnet. Die Lücke schließt bei mass = -2, 0 und 2.
    """
    solver = ExactDiagonalizationSolver()
    spectra_list: list[np.ndarray] = []
    gaps_list: list[float] = []

    for mass in mass_values:
        model = BHZModel(
            BHZModelParameters(
                n_x=n_x,
                n_y=n_y,
                mass=float(mass),
                boundary_x=boundary_x,
                boundary_y=boundary_y,
            )
        )

        eigenvalues = solver.solve(model.hamiltonian()).eigenvalues

        spectra_list.append(eigenvalues)

        # Kleinster Energiebetrag beschreibt die Lücke um E = 0.
        gaps_list.append(float(np.min(np.abs(eigenvalues))))

    return BHZScanResult(
        mass_values=np.asarray(mass_values),
        spectra=np.asarray(spectra_list),
        gaps=np.asarray(gaps_list),
    )