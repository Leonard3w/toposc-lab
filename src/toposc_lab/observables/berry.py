from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

BlochHamiltonian = Callable[[float, float], np.ndarray]


@dataclass(frozen=True)
class BerryCurvatureResult:
    """
    Diskrete Berry-Krümmung eines besetzten Bandsatzes im Impulsraum.

    berry_flux enthält den gauge-invarianten Fluss pro Plaquette.
    berry_curvature ist dieser Fluss dividiert durch die Plaquette-Fläche.
    """

    k_x: np.ndarray
    k_y: np.ndarray
    berry_flux: np.ndarray
    berry_curvature: np.ndarray
    chern_number: float


def _occupied_subspace(
    hamiltonian: np.ndarray,
    fermi_energy: float,
    gap_tolerance: float,
) -> np.ndarray:
    """Bestimme die besetzten Eigenzustände eines gapped Bloch-Punkts."""
    matrix = np.asarray(hamiltonian, dtype=complex)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Bloch Hamiltonian must be a square matrix")

    if not np.allclose(matrix, matrix.conj().T):
        raise ValueError("Bloch Hamiltonian must be Hermitian")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    if np.min(np.abs(eigenvalues - fermi_energy)) <= gap_tolerance:
        raise ValueError(
            "Chern number is undefined because the bulk gap closes "
            "at the Fermi energy"
        )

    occupied_indices = np.flatnonzero(eigenvalues < fermi_energy)

    if occupied_indices.size == 0 or occupied_indices.size == matrix.shape[0]:
        raise ValueError(
            "Fermi energy must leave at least one occupied and one empty band"
        )

    return eigenvectors[:, occupied_indices]


def _link_variable(
    subspace_a: np.ndarray,
    subspace_b: np.ndarray,
) -> complex:
    """Gauge-invariante U(1)-Verbindung zweier besetzter Unterräume."""
    overlap = subspace_a.conj().T @ subspace_b
    determinant = np.linalg.det(overlap)
    magnitude = abs(determinant)

    if magnitude <= 1e-14:
        raise ValueError(
            "Momentum grid is too coarse: occupied subspaces have zero overlap"
        )

    return determinant / magnitude


def berry_curvature(
    bloch_hamiltonian: BlochHamiltonian,
    n_k: int = 31,
    fermi_energy: float = 0.0,
    gap_tolerance: float = 1e-8,
) -> BerryCurvatureResult:
    """
    Berechne Berry-Krümmung und Chern-Zahl mit der Fukui-Methode.

    Das Impulsraum-Gitter ist periodisch. Die Formel verwendet nur
    Überlappungen benachbarter besetzter Eigenräume und ist daher gegen
    beliebige Phasen der numerischen Eigenvektoren invariant.
    """
    if n_k < 2:
        raise ValueError("n_k must be at least 2")

    if gap_tolerance < 0.0:
        raise ValueError("gap_tolerance must be non-negative")

    momenta = np.linspace(-np.pi, np.pi, n_k, endpoint=False)
    subspaces: list[list[np.ndarray]] = []
    n_occupied: int | None = None

    for k_x in momenta:
        row: list[np.ndarray] = []

        for k_y in momenta:
            occupied = _occupied_subspace(
                bloch_hamiltonian(float(k_x), float(k_y)),
                fermi_energy=fermi_energy,
                gap_tolerance=gap_tolerance,
            )

            if n_occupied is None:
                n_occupied = occupied.shape[1]
            elif occupied.shape[1] != n_occupied:
                raise ValueError(
                    "Number of occupied bands changes across momentum space"
                )

            row.append(occupied)

        subspaces.append(row)

    links_x = np.empty((n_k, n_k), dtype=complex)
    links_y = np.empty((n_k, n_k), dtype=complex)

    for index_x in range(n_k):
        for index_y in range(n_k):
            next_x = (index_x + 1) % n_k
            next_y = (index_y + 1) % n_k

            links_x[index_x, index_y] = _link_variable(
                subspaces[index_x][index_y],
                subspaces[next_x][index_y],
            )
            links_y[index_x, index_y] = _link_variable(
                subspaces[index_x][index_y],
                subspaces[index_x][next_y],
            )

    berry_flux = np.empty((n_k, n_k), dtype=float)

    for index_x in range(n_k):
        for index_y in range(n_k):
            next_x = (index_x + 1) % n_k
            next_y = (index_y + 1) % n_k

            plaquette_phase = (
                links_x[index_x, index_y]
                * links_y[next_x, index_y]
                * np.conjugate(links_x[index_x, next_y])
                * np.conjugate(links_y[index_x, index_y])
            )

            berry_flux[index_x, index_y] = np.angle(plaquette_phase)

    delta_k = 2.0 * np.pi / n_k
    curvature = berry_flux / delta_k**2
    chern = float(np.sum(berry_flux) / (2.0 * np.pi))

    return BerryCurvatureResult(
        k_x=momenta,
        k_y=momenta,
        berry_flux=berry_flux,
        berry_curvature=curvature,
        chern_number=chern,
    )


def chern_number(
    bloch_hamiltonian: BlochHamiltonian,
    n_k: int = 31,
    fermi_energy: float = 0.0,
    gap_tolerance: float = 1e-8,
    quantization_tolerance: float = 1e-6,
) -> int:
    """Berechne eine quantisierte Chern-Zahl und prüfe Rundungsfehler."""
    result = berry_curvature(
        bloch_hamiltonian,
        n_k=n_k,
        fermi_energy=fermi_energy,
        gap_tolerance=gap_tolerance,
    )

    rounded_chern = int(np.rint(result.chern_number))

    if abs(result.chern_number - rounded_chern) > quantization_tolerance:
        raise ValueError(
            "Chern number is not sufficiently quantized; increase n_k or "
            "check whether the system is gapped"
        )

    return rounded_chern
