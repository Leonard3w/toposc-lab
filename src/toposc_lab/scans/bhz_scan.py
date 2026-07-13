from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toposc_lab.core.results import ParameterScanResult
from toposc_lab.models.bhz_model import BHZModel, BHZModelParameters
from toposc_lab.scans.parameter_scan import simulation_scan


@dataclass(frozen=True)
class BHZScanResult:
    """BHZ-Massenscan mit vollständigen Einzelergebnissen."""

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


def scan_bhz_mass(
    mass_values: np.ndarray,
    n_x: int = 6,
    n_y: int = 6,
    boundary_x: str = "periodic",
    boundary_y: str = "periodic",
) -> BHZScanResult:
    """Scanne das BHZ-Modell über den Massenparameter m."""

    def make_model(mass: float) -> BHZModel:
        return BHZModel(
            BHZModelParameters(
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

    return BHZScanResult(scan=scan)