from __future__ import annotations

import numpy as np

from toposc_lab.scans.parameter_scan import parameter_scan


class DummyModel:
    def __init__(self, value: float) -> None:
        self.value = value


def test_parameter_scan_returns_values_and_results() -> None:
    parameter_values = np.array([-1.0, 0.0, 1.0])

    scanned_values, results = parameter_scan(
        parameter_values=parameter_values,
        model_factory=lambda value: DummyModel(float(value)),
        solver=lambda model: model.value**2,
    )

    assert np.allclose(scanned_values, parameter_values)
    assert results == [1.0, 0.0, 1.0]


def test_parameter_scan_uses_model_factory_for_each_value() -> None:
    parameter_values = np.array([0.0, 1.0, 2.0])

    scanned_values, results = parameter_scan(
        parameter_values=parameter_values,
        model_factory=lambda value: DummyModel(float(value) + 10.0),
        solver=lambda model: model.value,
    )

    assert np.allclose(scanned_values, parameter_values)
    assert results == [10.0, 11.0, 12.0]