from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toposc_lab.core.results import ParameterScanResult
from toposc_lab.models.qwz_model import QWZModel, QWZModelParameters
from toposc_lab.scans.parameter_scan import simulation_scan


@dataclass(frozen=True)
class QWZScanResult:
    """QWZ-Massenscan mit vollständigen Einzelergebnissen."""

    scan: ParameterScanResult

    @property
    def mass_values(self) -> np.ndarray:
        return self.scan.parameter_values

    @property
    def spectra(self) -> np.ndarray:
        return self.scan.spectra

    @property
    def gaps(self) -> np.ndarray:
        return self.scan.gaps


def scan_qwz_mass(
    mass_values: np.ndarray,
    n_x: int = 8,
    n_y: int = 8,
    boundary_x: str = "periodic",
    boundary_y: str = "periodic",
) -> QWZScanResult:
    """Scanne das QWZ-Modell über den Massenparameter m."""

    def make_model(mass: float) -> QWZModel:
        return QWZModel(
            QWZModelParameters(
                n_x=n_x,
                n_y=n_y,
                mass=mass,
                boundary_x=boundary_x,
                boundary_y=boundary_y,
            )
        )

    scan = simulation_scan(
        parameter_name="mass",
        parameter_values=mass_values,
        model_factory=make_model,
        metadata={
            "n_x": n_x,
            "n_y": n_y,
            "boundary_x": boundary_x,
            "boundary_y": boundary_y,
        },
    )

    return QWZScanResult(scan=scan)