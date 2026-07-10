from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from toposc_lab.models.ssh_chain import SSHChain, SSHChainParameters
from toposc_lab.observables.spectrum import count_zero_modes, edge_gap
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver


@dataclass(frozen=True)
class SSHScanResult:
    """Speichert die Ergebnisse eines SSH-Parameter-Scans."""

    # Werte des gescannten Parameters, beispielsweise v oder das Verhältnis v/w.
    parameter_values: np.ndarray

    # Energiespektren der Kette mit der gewählten Randbedingung.
    spectra: np.ndarray

    # Physikalischer Bulk-Gap, berechnet mit periodischen Randbedingungen.
    bulk_gaps: np.ndarray

    # Kleinster Energiebetrag der gewählten Kette.
    # Bei einer offenen topologischen Kette ist dieser nahe null.
    edge_gaps: np.ndarray

    # Anzahl der Zustände nahe Energie null.
    zero_mode_counts: np.ndarray


def _scan_ssh(
    parameter_values: np.ndarray,
    n_cells: int,
    parameter_to_hoppings: Callable[[float], tuple[float, float]],
    boundary: str,
) -> SSHScanResult:
    """
    Führt den gemeinsamen Teil aller SSH-Scans aus.

    Die gewählte Randbedingung bestimmt Spektrum und Edge States.
    Der Bulk-Gap wird zusätzlich mit periodischen Randbedingungen berechnet,
    damit er nicht durch Edge States verfälscht wird.
    """
    solver = ExactDiagonalizationSolver()

    spectra_list: list[np.ndarray] = []
    bulk_gaps_list: list[float] = []
    edge_gaps_list: list[float] = []
    zero_mode_counts_list: list[int] = []

    for parameter in parameter_values:
        # Aus dem gescannten Wert werden die beiden SSH-Hopping-Amplituden bestimmt.
        v, w = parameter_to_hoppings(float(parameter))

        # Diese Kette verwendet die vom Nutzer gewählte Randbedingung.
        # Bei "open" können hier topologische Edge States auftreten.
        model = SSHChain(
            SSHChainParameters(
                n_cells=n_cells,
                v=v,
                w=w,
                boundary=boundary,
            )
        )
        eigenvalues = solver.solve(model.hamiltonian()).eigenvalues

        # Die periodische Kette besitzt keine Ränder und damit keine Edge States.
        # Ihr kleinster Energiebetrag ist deshalb der echte Bulk-Gap.
        bulk_model = SSHChain(
            SSHChainParameters(
                n_cells=n_cells,
                v=v,
                w=w,
                boundary="periodic",
            )
        )
        bulk_eigenvalues = solver.solve(bulk_model.hamiltonian()).eigenvalues

        spectra_list.append(eigenvalues)
        bulk_gaps_list.append(float(np.min(np.abs(bulk_eigenvalues))))

        # Diese Werte gehören zur gewählten Kette, nicht zur periodischen Bulk-Kette.
        edge_gaps_list.append(edge_gap(eigenvalues))
        zero_mode_counts_list.append(
            count_zero_modes(eigenvalues, tolerance=1e-6)
        )

    return SSHScanResult(
        parameter_values=np.asarray(parameter_values),
        spectra=np.asarray(spectra_list),
        bulk_gaps=np.asarray(bulk_gaps_list),
        edge_gaps=np.asarray(edge_gaps_list),
        zero_mode_counts=np.asarray(zero_mode_counts_list),
    )


def scan_ssh_v(
    v_values: np.ndarray,
    n_cells: int = 30,
    w: float = 1.0,
    boundary: str = "open",
) -> SSHScanResult:
    """Scanne das Intrazell-Hopping v bei festem Interzell-Hopping w."""
    return _scan_ssh(
        parameter_values=v_values,
        n_cells=n_cells,
        parameter_to_hoppings=lambda v: (v, w),
        boundary=boundary,
    )


def scan_ssh_w(
    w_values: np.ndarray,
    n_cells: int = 30,
    v: float = 1.0,
    boundary: str = "open",
) -> SSHScanResult:
    """Scanne das Interzell-Hopping w bei festem Intrazell-Hopping v."""
    return _scan_ssh(
        parameter_values=w_values,
        n_cells=n_cells,
        parameter_to_hoppings=lambda w: (v, w),
        boundary=boundary,
    )


def scan_ssh_ratio(
    ratio_values: np.ndarray,
    n_cells: int = 30,
    w: float = 1.0,
    boundary: str = "open",
) -> SSHScanResult:
    """
    Scanne das Verhältnis v/w.

    Für eine offene SSH-Kette gilt:
    - v/w < 1: topologische Phase mit Edge States
    - v/w = 1: Phasenübergang, Bulk-Gap schließt
    - v/w > 1: triviale Phase ohne Edge States
    """
    return _scan_ssh(
        parameter_values=ratio_values,
        n_cells=n_cells,
        parameter_to_hoppings=lambda ratio: (ratio * w, w),
        boundary=boundary,
    )