from __future__ import annotations

import numpy as np

from toposc_lab.ml import (
    GradientBoostingRegressor,
    MeanRegressor,
    RidgeRegressor,
    regression_metrics,
)


def test_linear_and_trivial_regressors_on_fixed_data() -> None:
    features = np.arange(40, dtype=float).reshape(-1, 1)
    targets = 1.5 + 2.25 * features[:, 0]
    train = np.arange(30)
    test = np.arange(30, 40)

    trivial = MeanRegressor().fit(features[train], targets[train])
    linear = RidgeRegressor(alpha=1e-10).fit(features[train], targets[train])

    trivial_metrics = regression_metrics(targets[test], trivial.predict(features[test]))
    linear_metrics = regression_metrics(targets[test], linear.predict(features[test]))
    assert linear_metrics.rmse < 1e-6
    assert linear_metrics.rmse < trivial_metrics.rmse


def test_boosting_is_reproducible_and_beats_mean_on_nonlinear_data() -> None:
    features = np.linspace(-2.0, 2.0, 120).reshape(-1, 1)
    targets = np.square(features[:, 0]) + 0.25 * features[:, 0]
    train = np.r_[0:45, 75:120]
    test = np.arange(45, 75)
    config = {"n_estimators": 80, "learning_rate": 0.08, "max_bins": 24}

    first = GradientBoostingRegressor(**config).fit(features[train], targets[train])
    second = GradientBoostingRegressor(**config).fit(features[train], targets[train])
    trivial = MeanRegressor().fit(features[train], targets[train])

    np.testing.assert_array_equal(first.predict(features[test]), second.predict(features[test]))
    assert regression_metrics(targets[test], first.predict(features[test])).rmse < regression_metrics(
        targets[test], trivial.predict(features[test])
    ).rmse
