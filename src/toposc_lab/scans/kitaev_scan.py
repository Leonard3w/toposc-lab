from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toposc_lab.core.results import ParameterScanResult
from toposc_lab.models.kitaev_chain import (
    KitaevChain,
    KitaevChainParameters,
)
from toposc_lab.scans.parameter_scan import simulation_scan


@dataclass(frozen=True)
class KitaevScanResult:
    """Kitaev-Scan mit standardisierten Einzelergebnissen."""

    scan: ParameterScanResult

    @property
    def mu_values(self) -> np.ndarray:
        return self.scan.parameter_values

    @property
    def spectra(self) -> np.ndarray:
        return self.scan.spectra

    @property
    def gaps(self) -> np.ndarray:
        return self.scan.gaps


def scan_kitaev_mu(
    mu_values: np.ndarray,
    L: int = 40,
    t: float = 1.0,
    delta: float = 1.0,
    periodic: bool = False,
) -> KitaevScanResult:
    """Scanne das Kitaev-Spektrum über das chemische Potential μ."""
    boundary = "periodic" if periodic else "open"

    def make_model(mu: float) -> KitaevChain:
        return KitaevChain(
            KitaevChainParameters(
                n_sites=L,
                hopping=t,
                chemical_potential=mu,
                pairing=delta,
                boundary=boundary,
            )
        )

    scan = simulation_scan(
        parameter_name="chemical_potential",
        parameter_values=mu_values,
        model_factory=make_model,
        metadata={
            "n_sites": L,
            "hopping": t,
            "pairing": delta,
            "boundary": boundary,
        },
    )

    return KitaevScanResult(scan=scan)