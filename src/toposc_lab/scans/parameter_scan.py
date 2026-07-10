from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

import numpy as np

Parameter = TypeVar("Parameter")
Model = TypeVar("Model")
Result = TypeVar("Result")


def parameter_scan(
    parameter_values: Iterable[Parameter],
    model_factory: Callable[[Parameter], Model],
    solver: Callable[[Model], Result],
) -> tuple[np.ndarray, list[Result]]:
    """
    Generic one-parameter scan.

    For every value in parameter_values, this function builds a model with
    model_factory and evaluates it with solver.
    """
    values = np.asarray(list(parameter_values))

    results: list[Result] = []

    for value in values:
        model = model_factory(value)
        result = solver(model)
        results.append(result)

    return values, results