from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toposc_lab.models.qwz_model import QWZModel, QWZModelParameters
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver

@dataclass(frozen=True)
class QWZScanResult:

    mass_values: np.ndarray
    spectra: np.ndarray
    gaps: np.ndarray


def scan_qwz_mass(
            mass_values: np.ndarray,
            n_x: int = 8,
            n_y: int = 8,
            boundary_x: str = "periodic",
            boundary_y: str = "periodic",

    )->QWZScanResult:
        
        solver = ExactDiagonalizationSolver()
        spectra_list: list[np.ndarray] = []
        gaps_list: list[float] = []
        
        for mass in mass_values:
            model = QWZModel(
                QWZModelParameters(
                    n_x=n_x,
                    n_y=n_y,
                    mass=float(mass),
                    boundary_x=boundary_x,
                    boundary_y=boundary_y,
                )
            )

            eigenvalues = solver.solve(model.hamiltonian()).eigenvalues

            spectra_list.append(eigenvalues)

            gaps_list.append(float(np.min(np.abs(eigenvalues))))

        return QWZScanResult(
            mass_values=np.asarray(mass_values),
            spectra=np.asarray(spectra_list),
            gaps=np.asarray(gaps_list),
        )


