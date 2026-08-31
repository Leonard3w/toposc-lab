"""Geometry-descriptor integration for scientific evaluations."""

from __future__ import annotations

from dataclasses import replace

from toposc_lab.evaluation.results import GeometryEvaluation
from toposc_lab.geometry.base import Geometry
from toposc_lab.geometry.descriptors import extract_geometry_descriptors

_WARNING_PREFIX = "[geometry descriptors] "
_SCOPE_WARNING = (
    "descriptors cover only the undirected one-skeleton; they do not encode "
    "stored edge orientation, pairing gauge, coordinates, boundary labels, "
    "metadata, or physical model terms."
)
_DISCONNECTED_WARNING = (
    "the geometry is disconnected; finite shortest-path statistics include "
    "only reachable unordered site pairs, with exclusions recorded by "
    "reachable_pair_fraction."
)
_NO_FINITE_PAIR_WARNING = (
    "the geometry has no distinct reachable site pair; finite shortest-path "
    "mean and maximum are unavailable."
)


def evaluate_geometry_descriptors(
    evaluation: GeometryEvaluation,
    geometry: Geometry,
) -> GeometryEvaluation:
    """Attach deterministic scalar graph descriptors to an evaluation."""
    if not isinstance(evaluation, GeometryEvaluation):
        raise TypeError("evaluation must be a GeometryEvaluation")
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be a Geometry")

    descriptors = extract_geometry_descriptors(geometry)
    warnings = tuple(
        warning
        for warning in evaluation.warnings
        if not warning.startswith(_WARNING_PREFIX)
    )
    warnings = _append_warning(warnings, _SCOPE_WARNING)
    if descriptors["component_count"] != 1:
        warnings = _append_warning(warnings, _DISCONNECTED_WARNING)
    if descriptors["connected_pair_count"] == 0:
        warnings = _append_warning(warnings, _NO_FINITE_PAIR_WARNING)

    return replace(
        evaluation,
        geometry_descriptors=descriptors,
        warnings=warnings,
    )


def _append_warning(warnings: tuple[str, ...], message: str) -> tuple[str, ...]:
    value = f"{_WARNING_PREFIX}{message}"
    if value in warnings:
        return warnings
    return warnings + (value,)
