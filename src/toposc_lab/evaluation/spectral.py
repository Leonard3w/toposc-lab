"""Basic eigenvalue-only evaluation for geometry candidates."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from toposc_lab.evaluation.results import GeometryEvaluation
from toposc_lab.observables.spectrum import count_zero_modes, spectral_gap

_EIGENSTATE_EVALUATION_PENDING_WARNING = (
    "low_energy_states were selected by energy proximity only; eigenstate "
    "diagnostics have not been evaluated."
)


def evaluate_spectrum(
    eigenvalues: np.ndarray,
    *,
    reference_energy: float = 0.0,
    zero_mode_tolerance: float = 1.0e-10,
    low_energy_count: int = 8,
) -> GeometryEvaluation:
    """Return the basic spectral part of a geometry evaluation.

    ``gap`` is the full spectral separation across ``reference_energy``. It is
    zero when an eigenvalue lies within ``zero_mode_tolerance`` of the
    reference, and it requires states on both sides. For a particle-hole
    symmetric BdG spectrum without zero modes, this full separation is twice
    the smallest positive quasiparticle excitation energy.

    At least ``low_energy_count`` states closest to the reference are retained.
    All states tied at the selection boundary are included, so a degenerate
    multiplet is never split by its eigensolver column index. No eigenvector,
    localization, Majorana, topology, or geometry analysis is performed.
    """
    requested_state_count = _positive_integer(
        low_energy_count,
        name="low_energy_count",
    )
    energies = np.asarray(eigenvalues, dtype=float)
    gap = spectral_gap(
        energies,
        reference_energy=reference_energy,
        tolerance=zero_mode_tolerance,
    )
    reference = float(reference_energy)
    offsets = energies - reference
    zero_mode_count = count_zero_modes(
        offsets,
        tolerance=zero_mode_tolerance,
    )
    low_energy_states = _select_low_energy_states(
        energies,
        reference_energy=reference,
        requested_state_count=requested_state_count,
    )

    warnings = (
        "gap is the full finite-spectrum separation across "
        f"reference_energy={reference:.17g} with "
        f"zero_mode_tolerance={float(zero_mode_tolerance):.17g}; it is not a "
        "zero-mode-filtered positive excitation gap.",
        "zero_mode_count uses the stated tolerance and is a numerical energy "
        "classification; it does not establish Majorana character, boundary "
        "localization, or topology.",
        _EIGENSTATE_EVALUATION_PENDING_WARNING,
    )
    return GeometryEvaluation(
        gap=gap,
        low_energy_states=low_energy_states,
        zero_mode_count=zero_mode_count,
        warnings=warnings,
    )


def _select_low_energy_states(
    eigenvalues: np.ndarray,
    *,
    reference_energy: float,
    requested_state_count: int,
) -> dict[int, float]:
    distances = np.abs(eigenvalues - reference_energy)
    ordered_indices = np.argsort(distances, kind="stable")
    boundary_position = min(requested_state_count, eigenvalues.size) - 1
    boundary_distance = distances[ordered_indices[boundary_position]]
    selected_indices = ordered_indices[distances[ordered_indices] <= boundary_distance]
    return {
        int(state_index): float(eigenvalues[state_index])
        for state_index in selected_indices
    }


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    integer_value = int(value)
    if integer_value <= 0:
        raise ValueError(f"{name} must be positive")
    return integer_value
