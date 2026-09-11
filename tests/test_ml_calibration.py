from __future__ import annotations

import numpy as np

from toposc_lab.ml import (
    BootstrapUncertaintyRegressor,
    IntervalCalibrator,
    calibration_metrics,
)


def test_interval_calibration_is_fit_on_validation_and_measured_on_test() -> None:
    generator = np.random.default_rng(1210)
    train_x = np.linspace(-2.0, 2.0, 80).reshape(-1, 1)
    train_y = 0.5 + 1.2 * train_x[:, 0] + generator.normal(0.0, 0.2, len(train_x))
    validation_x = np.linspace(-1.9, 1.9, 40).reshape(-1, 1)
    validation_y = 0.5 + 1.2 * validation_x[:, 0] + generator.normal(0.0, 0.2, 40)
    test_x = np.linspace(-1.8, 1.8, 100).reshape(-1, 1)
    test_y = 0.5 + 1.2 * test_x[:, 0] + generator.normal(0.0, 0.2, 100)
    model = BootstrapUncertaintyRegressor(ensemble_size=20, seed=3).fit(train_x, train_y)
    validation_prediction = model.predict_with_uncertainty(validation_x)

    calibrator = IntervalCalibrator(coverage=0.9).fit(
        validation_y, validation_prediction
    )
    test_interval = calibrator.predict_interval(model.predict_with_uncertainty(test_x))
    metrics = calibration_metrics(
        test_y,
        test_interval,
        inherited_failure_modes=calibrator.failure_modes,
    )

    assert 0.8 <= metrics.empirical_coverage <= 1.0
    assert metrics.mean_interval_width > 0.0
    assert calibrator.validation_count == 40
    assert metrics.failure_modes == ()


def test_small_calibration_sets_document_coarse_coverage_failure_mode() -> None:
    features = np.arange(10, dtype=float).reshape(-1, 1)
    model = BootstrapUncertaintyRegressor(ensemble_size=4).fit(features, features[:, 0])
    prediction = model.predict_with_uncertainty(features[:4])
    calibrator = IntervalCalibrator().fit(features[:4, 0], prediction)

    assert "fewer than 20" in calibrator.failure_modes[0]
