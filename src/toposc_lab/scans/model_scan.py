"""Allgemeiner Parameterscan fuer Pydantic-basierte TopOSC-Lab-Modelle."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel as PydanticBaseModel

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import ParameterScanResult
from toposc_lab.scans.parameter_scan import simulation_scan
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver


def model_parameter_scan(
    parameter_name: str,
    parameter_values: Iterable[float],
    *,
    parameter_model: type[PydanticBaseModel],
    model_factory: Callable[[Any], BaseModel],
    base_parameters: Mapping[str, Any],
    solver: ExactDiagonalizationSolver | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ParameterScanResult:
    """Scanne einen numerischen Pydantic-Modellparameter.

    ``base_parameters`` enthaelt alle festen Parameter der Studie. Nur
    ``parameter_name`` wird fuer jeden Scanpunkt ersetzt und anschliessend
    erneut durch Pydantic validiert. Damit verhalten sich Python-API, CLI und
    GUI bei ungueltigen Modellparametern identisch.
    """
    if parameter_name not in parameter_model.model_fields:
        raise ValueError(
            f"Unknown parameter {parameter_name!r} for "
            f"{parameter_model.__name__}"
        )

    values = np.asarray(list(parameter_values), dtype=float)

    if values.ndim != 1 or values.size < 2:
        raise ValueError(
            "parameter_values must be a one-dimensional array with at least two values"
        )

    fixed_parameters = dict(base_parameters)
    fixed_parameters.pop(parameter_name, None)

    def build_model(value: float) -> BaseModel:
        parameters = parameter_model.model_validate(
            {**fixed_parameters, parameter_name: float(value)}
        )
        return model_factory(parameters)

    scan_metadata = {
        "base_parameters": fixed_parameters,
        **({} if metadata is None else metadata),
    }

    return simulation_scan(
        parameter_name=parameter_name,
        parameter_values=values,
        model_factory=build_model,
        solver=solver,
        metadata=scan_metadata,
    )
