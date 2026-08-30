"""State-resolved evaluation on arbitrary discrete geometries."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from toposc_lab.core.results import BasisLayout
from toposc_lab.evaluation.results import GeometryEvaluation
from toposc_lab.evaluation.spectral import _EIGENSTATE_EVALUATION_PENDING_WARNING
from toposc_lab.geometry.base import Geometry
from toposc_lab.observables.localization import localization_profile_on_geometry

_FINITE_STATE_WARNING = (
    "IPR and boundary localization are finite-state diagnostics and do not "
    "establish Majorana character or topology."
)
_EXPLICIT_BOUNDARY_WARNING = (
    "localization uses only geometry.boundary_sites; low-degree sites and "
    "coordinate extrema are not inferred as physical boundaries."
)
_MISSING_BOUNDARY_WARNING = (
    "geometry has no explicit boundary sites; boundary weight is zero and no "
    "boundary-localized state is inferred."
)
_MISSING_COORDINATES_WARNING = (
    "geometry has no embedding coordinates; localization center_of_mass is "
    "an empty array."
)


def evaluate_eigenstates(
    evaluation: GeometryEvaluation,
    eigenvectors: np.ndarray,
    *,
    basis_layout: BasisLayout,
    geometry: Geometry,
    boundary_localization_threshold: float = 0.5,
) -> GeometryEvaluation:
    """Add IPR and geometry-aware localization for selected low-energy states.

    State indices come from ``evaluation.low_energy_states`` and refer to
    eigenvector columns. The returned evaluation preserves all spectral and
    already-populated non-eigenstate fields. Majorana and topology diagnostics
    are intentionally outside this Phase 7.3 operation.
    """
    if not isinstance(evaluation, GeometryEvaluation):
        raise TypeError("evaluation must be a GeometryEvaluation")
    if not isinstance(basis_layout, BasisLayout):
        raise TypeError("basis_layout must be a BasisLayout")
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be a Geometry")
    if not evaluation.low_energy_states:
        raise ValueError("evaluation must contain low-energy states")
    if basis_layout.n_sites != geometry.n_sites:
        raise ValueError("basis_layout and geometry must contain the same site count")

    localization = {
        state_index: localization_profile_on_geometry(
            eigenvectors=eigenvectors,
            state_index=state_index,
            basis_layout=basis_layout,
            geometry=geometry,
            boundary_threshold=boundary_localization_threshold,
        )
        for state_index in evaluation.low_energy_states
    }
    ipr = {
        state_index: profile.inverse_participation_ratio
        for state_index, profile in localization.items()
    }

    warnings = tuple(
        warning
        for warning in evaluation.warnings
        if warning != _EIGENSTATE_EVALUATION_PENDING_WARNING
    )
    warnings = _append_warning(warnings, _FINITE_STATE_WARNING)
    warnings = _append_warning(warnings, _EXPLICIT_BOUNDARY_WARNING)
    if not geometry.boundary_sites:
        warnings = _append_warning(warnings, _MISSING_BOUNDARY_WARNING)
    if geometry.coordinates is None:
        warnings = _append_warning(warnings, _MISSING_COORDINATES_WARNING)

    return replace(
        evaluation,
        ipr=ipr,
        localization=localization,
        warnings=warnings,
    )


def _append_warning(warnings: tuple[str, ...], message: str) -> tuple[str, ...]:
    if message in warnings:
        return warnings
    return warnings + (message,)
