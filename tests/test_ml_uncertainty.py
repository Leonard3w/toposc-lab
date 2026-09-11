from __future__ import annotations

import numpy as np

from toposc_lab.ml import BootstrapUncertaintyRegressor, uncertainty_diagnostics


def test_uncertainty_is_reproducible_and_tracks_held_out_error() -> None:
    train_x = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
    train_y = np.square(train_x[:, 0])
    test_x = np.array([[-0.8], [-0.2], [0.2], [0.8], [1.5], [2.0]])
    test_y = np.square(test_x[:, 0])
    config = {"ensemble_size": 20, "seed": 11, "distance_weight": 1.0}

    first = BootstrapUncertaintyRegressor(**config).fit(train_x, train_y)
    second = BootstrapUncertaintyRegressor(**config).fit(train_x, train_y)
    prediction = first.predict_with_uncertainty(test_x)

    np.testing.assert_array_equal(
        prediction.standard_deviation,
        second.predict_with_uncertainty(test_x).standard_deviation,
    )
    diagnostics = uncertainty_diagnostics(test_y, prediction)
    assert diagnostics.uncertainty_error_rank_correlation > 0.5
    assert diagnostics.high_uncertainty_has_larger_error
    assert prediction.standard_deviation[-1] > prediction.standard_deviation[0]


def test_uncertainty_components_are_finite_and_nonnegative() -> None:
    features = np.arange(20, dtype=float).reshape(-1, 1)
    targets = 2.0 * features[:, 0]
    model = BootstrapUncertaintyRegressor(ensemble_size=5).fit(features, targets)

    prediction = model.predict_with_uncertainty([[3.5], [30.0]])

    assert np.all(np.isfinite(prediction.mean))
    assert np.all(prediction.standard_deviation >= 0.0)
    assert prediction.nearest_training_distance[1] > prediction.nearest_training_distance[0]
