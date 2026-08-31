"""Applicability-aware integration of topology diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import TypeAlias

from toposc_lab.evaluation.results import GeometryEvaluation
from toposc_lab.topology.dispatch import TopologyDispatchDecision
from toposc_lab.topology.results import (
    TopologyDiagnosticResult,
    TopologyMethod,
    TopologyResult,
    unify_topology_result,
)

TopologyIntegrationInput: TypeAlias = TopologyResult | TopologyDiagnosticResult

_WARNING_PREFIX = "[topology integration] "


def integrate_topology_results(
    evaluation: GeometryEvaluation,
    topology_results: Iterable[TopologyIntegrationInput],
    *,
    dispatch_decision: TopologyDispatchDecision,
    convergence_checked: bool = False,
) -> GeometryEvaluation:
    """Validate, unify, and attach topology results to an evaluation.

    ``dispatch_decision`` is the authoritative applicability record for the
    supplied physical dimension, symmetry class, and capabilities. Specialized
    diagnostics are converted to :class:`TopologyResult`; already unified
    results retain their own convergence status. This function does not execute
    invariant calculations.

    The supplied results replace any previous topology tuple. Warnings produced
    by an earlier call to this function are likewise replaced, while unrelated
    evaluation warnings are preserved.
    """
    if not isinstance(evaluation, GeometryEvaluation):
        raise TypeError("evaluation must be a GeometryEvaluation")
    if not isinstance(dispatch_decision, TopologyDispatchDecision):
        raise TypeError("dispatch_decision must be a TopologyDispatchDecision")
    if not isinstance(convergence_checked, bool):
        raise TypeError("convergence_checked must be a boolean")

    results_by_method: dict[TopologyMethod, TopologyResult] = {}
    for source_result in topology_results:
        unified = _unify_result(
            source_result,
            convergence_checked=convergence_checked,
        )
        dispatch_decision.require(unified.method)
        if unified.method in results_by_method:
            raise ValueError(
                "topology_results must contain at most one result per method"
            )
        results_by_method[unified.method] = unified

    topology = tuple(
        results_by_method[method]
        for method in TopologyMethod
        if method in results_by_method
    )
    warnings = tuple(
        warning
        for warning in evaluation.warnings
        if not warning.startswith(_WARNING_PREFIX)
    )
    for warning in dispatch_decision.warnings:
        warnings = _append_warning(warnings, f"dispatch: {warning}")
    for method in TopologyMethod:
        if method in dispatch_decision.rejected_methods:
            reasons = "; ".join(dispatch_decision.rejected_methods[method])
            warnings = _append_warning(
                warnings,
                f"{method.value} not applied: {reasons}",
            )
    missing_methods = tuple(
        method
        for method in dispatch_decision.applicable_methods
        if method not in results_by_method
    )
    if missing_methods:
        warnings = _append_warning(
            warnings,
            "applicable methods without supplied results: "
            + ", ".join(method.value for method in missing_methods),
        )

    for result in topology:
        for warning in result.warnings:
            warnings = _append_warning(
                warnings,
                f"{result.method.value}: {warning}",
            )
    if len(topology) == 1:
        warnings = _append_warning(
            warnings,
            "only one topology method was supplied; no independent topology "
            "cross-validation is available.",
        )
    elif len(topology) > 1:
        resolved_classifications = {
            result.is_topological
            for result in topology
            if result.is_topological is not None
        }
        if any(result.is_topological is None for result in topology):
            warnings = _append_warning(
                warnings,
                "at least one supplied topology method is numerically unresolved.",
            )
        if len(resolved_classifications) > 1:
            warnings = _append_warning(
                warnings,
                "supplied topology methods disagree on the topological classification.",
            )

    return replace(
        evaluation,
        topology=topology,
        warnings=warnings,
    )


def _unify_result(
    result: TopologyIntegrationInput,
    *,
    convergence_checked: bool,
) -> TopologyResult:
    if isinstance(result, TopologyResult):
        return result
    return unify_topology_result(
        result,
        convergence_checked=convergence_checked,
    )


def _append_warning(warnings: tuple[str, ...], message: str) -> tuple[str, ...]:
    value = f"{_WARNING_PREFIX}{message}"
    if value in warnings:
        return warnings
    return warnings + (value,)
