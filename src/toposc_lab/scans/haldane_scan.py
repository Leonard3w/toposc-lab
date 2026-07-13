from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toposc_lab.core.results import ParameterScanResult
from toposc_lab.models.haldane_model import (
    HaldaneModel,
    HaldaneModelParameters,
)
from toposc_lab.observables.berry import berry_curvature, chern_number
from toposc_lab.solvers.exact_diagonalization import (
    ExactDiagonalizationSolver,
)


@dataclass(frozen=True)
class HaldaneMassScanResult:
    """
    Phasenscan des Haldane-Modells über die Sublattice-Masse M.

    scan enthält die vollständigen periodischen Realraum-Ergebnisse.
    bulk_gaps und chern_numbers sind die physikalischen Bulk-Observablen.
    """

    scan: ParameterScanResult
    bulk_gaps: np.ndarray
    chern_numbers: np.ndarray
    numerical_chern_numbers: np.ndarray

    @property
    def mass_values(self) -> np.ndarray:
        return self.scan.parameter_values

    @property
    def spectra(self) -> np.ndarray:
        return self.scan.spectra


def _bloch_band_gap(model: HaldaneModel, n_k: int) -> float:
    """Bestimme den indirekten Bulk-Gap zwischen unterem und oberem Band."""
    # Das Intervall [0, 2 pi) enthält bei n_k teilbar durch drei die
    # Graphen-Dirac-Punkte 2 pi / 3 und 4 pi / 3 exakt. Das ist wichtig,
    # damit ein Haldane-Gap-Close nicht nur angenähert erkannt wird.
    momenta = np.linspace(0.0, 2.0 * np.pi, n_k, endpoint=False)
    valence_band_maximum = -np.inf
    conduction_band_minimum = np.inf

    for k_x in momenta:
        for k_y in momenta:
            energies = np.linalg.eigvalsh(
                model.bloch_hamiltonian(float(k_x), float(k_y))
            )
            valence_band_maximum = max(valence_band_maximum, energies[0])
            conduction_band_minimum = min(conduction_band_minimum, energies[1])

    return float(max(0.0, conduction_band_minimum - valence_band_maximum))


def scan_haldane_mass(
    mass_values: np.ndarray,
    nearest_neighbor_hopping: float = 1.0,
    next_nearest_neighbor_hopping: float = 0.2,
    phase: float = float(np.pi / 2.0),
    n_k: int = 24,
    n_x: int = 3,
    n_y: int = 3,
) -> HaldaneMassScanResult:
    """
    Scanne den Haldane-Phasenübergang als Funktion der Masse M.

    Für phase = pi/2 liegen die erwarteten Übergänge bei
    M = +/- 3 sqrt(3) t2. Im topologischen Bereich ist |C| = 1,
    außerhalb ist C = 0. Für ein exakt aufgelöstes Gap sollte n_k
    durch drei teilbar sein.
    """
    values = np.asarray(mass_values, dtype=float)

    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "mass_values must be a non-empty one-dimensional array"
        )

    if n_k < 3:
        raise ValueError("n_k must be at least 3")

    solver = ExactDiagonalizationSolver()
    models: list[HaldaneModel] = []

    for mass in values:
        models.append(
            HaldaneModel(
                HaldaneModelParameters(
                    n_x=n_x,
                    n_y=n_y,
                    nearest_neighbor_hopping=nearest_neighbor_hopping,
                    next_nearest_neighbor_hopping=(
                        next_nearest_neighbor_hopping
                    ),
                    phase=phase,
                    sublattice_mass=float(mass),
                    boundary_x="periodic",
                    boundary_y="periodic",
                )
            )
        )

    scan = ParameterScanResult(
        parameter_name="sublattice_mass",
        parameter_values=values,
        results=tuple(solver.solve_model(model) for model in models),
        metadata={
            "nearest_neighbor_hopping": nearest_neighbor_hopping,
            "next_nearest_neighbor_hopping": next_nearest_neighbor_hopping,
            "phase": phase,
            "n_k": n_k,
        },
    )

    bulk_gaps: list[float] = []
    numerical_chern_numbers: list[float] = []
    chern_numbers: list[int] = []

    for model in models:
        berry_result = berry_curvature(
            model.bloch_hamiltonian,
            n_k=n_k,
        )

        bulk_gaps.append(_bloch_band_gap(model, n_k=n_k))
        numerical_chern_numbers.append(berry_result.chern_number)
        chern_numbers.append(
            chern_number(model.bloch_hamiltonian, n_k=n_k)
        )

    return HaldaneMassScanResult(
        scan=scan,
        bulk_gaps=np.asarray(bulk_gaps),
        chern_numbers=np.asarray(chern_numbers),
        numerical_chern_numbers=np.asarray(numerical_chern_numbers),
    )
