from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeVar

import numpy as np

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import ParameterScanResult
from toposc_lab.solvers.exact_diagonalization import (
    ExactDiagonalizationSolver,
)

Parameter = TypeVar("Parameter")
Model = TypeVar("Model")
Result = TypeVar("Result")


def parameter_scan(
    parameter_values: Iterable[Parameter],
    model_factory: Callable[[Parameter], Model],
    solver: Callable[[Model], Result],
) -> tuple[np.ndarray, list[Result]]:
    """
    Alte allgemeine Scan-Schnittstelle.

    Sie bleibt für Kompatibilität erhalten. Neue Scans verwenden
    simulation_scan() und erhalten ParameterScanResult.
    """
    values = np.asarray(list(parameter_values))
    results: list[Result] = []

    for value in values:
        results.append(solver(model_factory(value)))

    return values, results


def simulation_scan(
    parameter_name: str,
    parameter_values: Iterable[float],
    model_factory: Callable[[float], BaseModel],
    *,
    solver: ExactDiagonalizationSolver | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ParameterScanResult:
    """
    Standardisierter Scan über einen Modellparameter.

    Jeder Punkt enthält ein vollständiges SimulationResult statt nur
    eines losen Eigenwertarrays.
    """
    values = np.asarray(list(parameter_values), dtype=float)

    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "parameter_values must be a non-empty one-dimensional array"
        )

    active_solver = (
        ExactDiagonalizationSolver()
        if solver is None
        else solver
    )

    results = tuple(
        active_solver.solve_model(model_factory(float(value)))
        for value in values
    )

    return ParameterScanResult(
        parameter_name=parameter_name,
        parameter_values=values,
        results=results,
        metadata={} if metadata is None else metadata,
    )