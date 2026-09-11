from __future__ import annotations

import numpy as np

from toposc_lab.ml import (
    BalancedLogisticClassifier,
    classification_metrics,
    select_classification_threshold,
)


def test_balanced_classifier_reports_class_and_threshold_metrics() -> None:
    negative = np.linspace(-3.0, -0.2, 90)
    positive = np.linspace(0.2, 3.0, 10)
    features = np.concatenate((negative, positive)).reshape(-1, 1)
    labels = np.concatenate((np.zeros(90, dtype=int), np.ones(10, dtype=int)))
    classifier = BalancedLogisticClassifier(epochs=800).fit(features, labels)

    probabilities = classifier.predict_proba(features)
    threshold = select_classification_threshold(labels, probabilities)
    metrics = classification_metrics(labels, probabilities, threshold=threshold)

    assert metrics.positive_count == 10
    assert metrics.prevalence == 0.1
    assert metrics.balanced_accuracy > 0.95
    assert metrics.f1 > 0.9
    assert metrics.threshold == threshold


def test_threshold_selection_is_deterministic_and_validation_driven() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.45, 0.55, 0.9])

    first = select_classification_threshold(labels, probabilities)
    second = select_classification_threshold(labels, probabilities)

    assert first == second
    assert 0.0 <= first <= 1.0


def test_classifier_rejects_single_class_training_data() -> None:
    classifier = BalancedLogisticClassifier()
    try:
        classifier.fit(np.ones((4, 2)), np.ones(4))
    except ValueError as error:
        assert "both classes" in str(error)
    else:
        raise AssertionError("single-class training must be rejected")
