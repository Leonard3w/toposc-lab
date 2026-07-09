from __future__ import annotations

from dataclasses import dataclass

from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver

import numpy as np

from toposc_lab.models.kitaev_chain import KitaevChain, KitaevChainParameters

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
    spectra = []
    gaps = []
    solver = ExactDiagonalizationSolver()

    boundary = "periodic" if periodic else "open"

    for mu in mu_values:
        params = KitaevChainParameters(
            n_sites=L,
            hopping=t,
            chemical_potential=float(mu),
            pairing=delta,
            boundary=boundary,
        )

        model = KitaevChain(params)
        result = solver.solve(model.hamiltonian())
        eigenvalues = result.eigenvalues

        spectra.append(eigenvalues)
        gaps.append(float(np.min(np.abs(eigenvalues))))

    return KitaevScanResult(
        mu_values=np.asarray(mu_values),
        spectra=np.asarray(spectra),
        gaps=np.asarray(gaps),
    )
