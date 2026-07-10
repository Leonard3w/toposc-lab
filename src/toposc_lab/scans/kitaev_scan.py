from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toposc_lab.models.kitaev_chain import KitaevChain, KitaevChainParameters
from toposc_lab.scans.parameter_scan import parameter_scan
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver


@dataclass(frozen=True)
class KitaevScanResult:
    """Result of a Kitaev chain parameter scan."""

    mu_values: np.ndarray
    spectra: np.ndarray
    gaps: np.ndarray


def scan_kitaev_mu(
    mu_values: np.ndarray,
    L: int = 40,
    t: float = 1.0,
    delta: float = 1.0,
    periodic: bool = False,
) -> KitaevScanResult:
    """
    Scan the Kitaev chain spectrum as a function of chemical potential mu.
    """
    solver = ExactDiagonalizationSolver()

    boundary = "periodic" if periodic else "open"

    def make_model(mu: float) -> KitaevChain:
        params = KitaevChainParameters(
            n_sites=L,
            hopping=t,
            chemical_potential=float(mu),
            pairing=delta,
            boundary=boundary,
        )

        return KitaevChain(params)

    def solve_spectrum(model: KitaevChain) -> np.ndarray:
        result = solver.solve(model.hamiltonian())
        return result.eigenvalues

    scanned_mu_values, spectra_list = parameter_scan(
        parameter_values=mu_values,
        model_factory=make_model,
        solver=solve_spectrum,
    )

    spectra = np.asarray(spectra_list)
    gaps = np.min(np.abs(spectra), axis=1)

    return KitaevScanResult(
        mu_values=np.asarray(scanned_mu_values),
        spectra=spectra,
        gaps=gaps,
    )