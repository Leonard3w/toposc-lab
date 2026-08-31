"""Majorana-diagnostic integration for evaluated low-energy states."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from toposc_lab.evaluation.results import GeometryEvaluation
from toposc_lab.hamiltonians.nambu import NambuBasis
from toposc_lab.observables.majorana import MajoranaDiagnostics, majorana_diagnostics

_PROBABILITY_TOLERANCE = 1.0e-12
_BASIS_CONVENTION_WARNING = (
    "majorana_metrics use the unrotated matching-component particle-hole "
    "convention declared by nambu_basis."
)
_INTERPRETATION_WARNING = (
    "high Majorana self_conjugacy does not by itself establish zero energy, "
    "spatial separation, boundary localization, or topology."
)
_DEGENERACY_WARNING = (
    "individual-state Majorana diagnostics can change under eigensolver "
    "rotations within a degenerate subspace."
)
_SELECTION_WARNING = (
    "majorana_metrics cover every selected low-energy state, including states "
    "outside the numerical zero-mode tolerance."
)


def evaluate_majorana_diagnostics(
    evaluation: GeometryEvaluation,
    eigenvectors: np.ndarray,
    *,
    nambu_basis: NambuBasis,
) -> GeometryEvaluation:
    """Add basis-aware Majorana polarization for selected low-energy states.

    A complete Phase 7.3 IPR/localization result is required. Site-resolved
    probabilities from the declared Nambu convention are cross-checked against
    those localization profiles before diagnostics are accepted. The operation
    does not infer topology or classify a state as a spatially separated
    Majorana zero mode.
    """
    if not isinstance(evaluation, GeometryEvaluation):
        raise TypeError("evaluation must be a GeometryEvaluation")
    if not isinstance(nambu_basis, NambuBasis):
        raise TypeError("nambu_basis must be a NambuBasis")
    if not evaluation.low_energy_states:
        raise ValueError("evaluation must contain low-energy states")

    selected_states = set(evaluation.low_energy_states)
    if set(evaluation.ipr) != selected_states or set(evaluation.localization) != selected_states:
        raise ValueError(
            "Majorana integration requires complete Phase 7.3 IPR and localization"
        )

    metrics: dict[int, MajoranaDiagnostics] = {}
    for state_index in evaluation.low_energy_states:
        profile = evaluation.localization[state_index]
        if not np.isclose(
            evaluation.ipr[state_index],
            profile.inverse_participation_ratio,
            rtol=0.0,
            atol=_PROBABILITY_TOLERANCE,
        ):
            raise ValueError("IPR values do not match the localization profiles")
        if profile.component_labels != nambu_basis.component_labels:
            raise ValueError(
                "localization component labels do not match the Nambu basis"
            )
        if profile.probability.shape != (nambu_basis.n_sites,):
            raise ValueError(
                "localization site probabilities do not match the Nambu basis"
            )
        expected_component_shape = (
            nambu_basis.n_sites,
            nambu_basis.nambu_components_per_site,
        )
        if profile.component_probabilities.shape != expected_component_shape:
            raise ValueError(
                "localization component probabilities do not match the Nambu basis"
            )

        diagnostics = majorana_diagnostics(
            eigenvectors=eigenvectors,
            state_index=state_index,
            basis=nambu_basis,
        )
        if not np.allclose(
            diagnostics.site_probability,
            profile.probability,
            rtol=0.0,
            atol=_PROBABILITY_TOLERANCE,
        ):
            raise ValueError(
                "Nambu basis does not match the eigenstate localization basis"
            )
        metrics[state_index] = diagnostics

    warnings = evaluation.warnings
    for warning in (
        _BASIS_CONVENTION_WARNING,
        _INTERPRETATION_WARNING,
        _DEGENERACY_WARNING,
        _SELECTION_WARNING,
    ):
        warnings = _append_warning(warnings, warning)

    return replace(
        evaluation,
        majorana_metrics=metrics,
        warnings=warnings,
    )


def _append_warning(warnings: tuple[str, ...], message: str) -> tuple[str, ...]:
    if message in warnings:
        return warnings
    return warnings + (message,)
