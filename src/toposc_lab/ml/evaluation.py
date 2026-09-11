"""Leakage-resistant evaluation protocol for Phase-12 regression surrogates."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from toposc_lab.data import (
    DATASET_SCHEMA_VERSION,
    DatasetSplit,
    DatasetSplitConfig,
    ExactPhysicsDataset,
    split_dataset,
)
from toposc_lab.ml.features import (
    HandcraftedFeatureSchema,
    extract_handcrafted_features,
    records_for_ids,
)
from toposc_lab.ml.models import (
    GradientBoostingRegressor,
    MeanRegressor,
    RegressionMetrics,
    Regressor,
    RidgeRegressor,
    regression_metrics,
)

EVALUATION_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class MetricInterval:
    confidence: float
    lower: float
    estimate: float
    upper: float
    bootstrap_samples: int


@dataclass(frozen=True, slots=True)
class RegressionEvaluationReport:
    """Validation-selected result; unselected models have no test metrics."""

    selected_model: str
    validation_metrics: Mapping[str, RegressionMetrics]
    selected_test_metrics: RegressionMetrics
    trivial_test_metrics: RegressionMetrics
    test_rmse_interval: MetricInterval
    split_seed: int
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    dataset_schema_version: int
    dataset_fingerprint: str
    test_used_for_selection: bool = False
    schema_version: int = EVALUATION_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        metrics = dict(self.validation_metrics)
        if self.selected_model not in metrics:
            raise ValueError("selected model must have validation metrics")
        if self.test_used_for_selection:
            raise ValueError("test data cannot be used for model selection")
        object.__setattr__(
            self, "validation_metrics", MappingProxyType(dict(sorted(metrics.items())))
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "selected_model": self.selected_model,
            "validation_metrics": {
                name: asdict(metrics) for name, metrics in self.validation_metrics.items()
            },
            "selected_test_metrics": asdict(self.selected_test_metrics),
            "trivial_test_metrics": asdict(self.trivial_test_metrics),
            "test_rmse_interval": asdict(self.test_rmse_interval),
            "split_seed": self.split_seed,
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "test_ids": list(self.test_ids),
            "feature_names": list(self.feature_names),
            "dataset_schema_version": self.dataset_schema_version,
            "dataset_fingerprint": self.dataset_fingerprint,
            "test_used_for_selection": self.test_used_for_selection,
        }


def evaluate_regression_surrogates(
    dataset: ExactPhysicsDataset,
    target_values: Mapping[str, float],
    *,
    split: DatasetSplit | None = None,
    split_config: DatasetSplitConfig | None = None,
    model_factories: Mapping[str, Callable[[], Regressor]] | None = None,
    bootstrap_seed: int = 1204,
    bootstrap_samples: int = 400,
) -> tuple[RegressionEvaluationReport, Regressor, HandcraftedFeatureSchema]:
    """Select on validation and touch test labels only for the selected/trivial models."""
    if not isinstance(dataset, ExactPhysicsDataset):
        raise TypeError("dataset must be ExactPhysicsDataset")
    selected_split = split or split_dataset(dataset, config=split_config)
    if selected_split.diagnostics.leakage_groups:
        raise ValueError("dataset split contains geometry-family leakage")
    _validate_split_coverage(dataset, selected_split)
    factories = dict(
        model_factories
        or {
            "ridge": lambda: RidgeRegressor(alpha=1e-5),
            "stump_boosting": lambda: GradientBoostingRegressor(),
        }
    )
    if not factories or any(not name for name in factories):
        raise ValueError("at least one named model factory is required")

    train_records = records_for_ids(dataset.records, selected_split.train_ids)
    validation_records = records_for_ids(dataset.records, selected_split.validation_ids)
    test_records = records_for_ids(dataset.records, selected_split.test_ids)
    if not train_records or not validation_records or not test_records:
        raise ValueError("train, validation, and test splits must all be non-empty")
    schema = HandcraftedFeatureSchema.fit(train_records)
    train_features = extract_handcrafted_features(train_records, schema=schema).values
    validation_features = extract_handcrafted_features(validation_records, schema=schema).values
    test_features = extract_handcrafted_features(test_records, schema=schema).values
    train_targets = _targets(selected_split.train_ids, target_values)
    validation_targets = _targets(selected_split.validation_ids, target_values)

    fitted: dict[str, Regressor] = {}
    validation_metrics: dict[str, RegressionMetrics] = {}
    for name, factory in sorted(factories.items()):
        model = factory()
        model.fit(train_features, train_targets)
        fitted[name] = model
        validation_metrics[name] = regression_metrics(
            validation_targets, model.predict(validation_features)
        )
    selected_name = min(
        validation_metrics,
        key=lambda name: (validation_metrics[name].rmse, validation_metrics[name].mae, name),
    )

    # Test labels are accessed only after validation selection is frozen.
    test_targets = _targets(selected_split.test_ids, target_values)
    selected_predictions = fitted[selected_name].predict(test_features)
    selected_test_metrics = regression_metrics(test_targets, selected_predictions)
    trivial = MeanRegressor().fit(train_features, train_targets)
    trivial_test_metrics = regression_metrics(test_targets, trivial.predict(test_features))
    interval = bootstrap_rmse_interval(
        test_targets,
        selected_predictions,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
    )
    report = RegressionEvaluationReport(
        selected_model=selected_name,
        validation_metrics=validation_metrics,
        selected_test_metrics=selected_test_metrics,
        trivial_test_metrics=trivial_test_metrics,
        test_rmse_interval=interval,
        split_seed=selected_split.diagnostics.seed,
        train_ids=selected_split.train_ids,
        validation_ids=selected_split.validation_ids,
        test_ids=selected_split.test_ids,
        feature_names=schema.feature_names,
        dataset_schema_version=dataset.schema_version,
        dataset_fingerprint=dataset_fingerprint(dataset),
    )
    return report, fitted[selected_name], schema


def bootstrap_rmse_interval(
    targets: np.typing.ArrayLike,
    predictions: np.typing.ArrayLike,
    *,
    seed: int,
    samples: int = 400,
    confidence: float = 0.95,
) -> MetricInterval:
    """Deterministic paired bootstrap interval for held-out RMSE."""
    truth = np.asarray(targets, dtype=float)
    estimate = np.asarray(predictions, dtype=float)
    if truth.ndim != 1 or estimate.shape != truth.shape or not len(truth):
        raise ValueError("targets/predictions must be non-empty equal-length vectors")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(estimate)):
        raise ValueError("targets/predictions must be finite")
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(truth), size=(samples, len(truth)))
    residuals = truth - estimate
    bootstrapped = np.sqrt(np.mean(np.square(residuals[indices]), axis=1))
    tail = (1.0 - confidence) / 2.0
    return MetricInterval(
        confidence=confidence,
        lower=float(np.quantile(bootstrapped, tail)),
        estimate=float(np.sqrt(np.mean(np.square(residuals)))),
        upper=float(np.quantile(bootstrapped, 1.0 - tail)),
        bootstrap_samples=samples,
    )


def save_evaluation_report(path: str | Path, report: RegressionEvaluationReport) -> Path:
    """Atomically persist the compact, versioned evaluation artifact."""
    if not isinstance(report, RegressionEvaluationReport):
        raise TypeError("report must be RegressionEvaluationReport")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def dataset_fingerprint(dataset: ExactPhysicsDataset) -> str:
    """Identify the exact label collection without copying its payload."""
    import hashlib

    digest = hashlib.sha256()
    digest.update(f"schema:{dataset.schema_version}\n".encode())
    for record_id in sorted(record.record_id for record in dataset.records):
        digest.update(record_id.encode())
        digest.update(b"\n")
    return f"exact-dataset-v1-sha256:{digest.hexdigest()}"


def _targets(record_ids: tuple[str, ...], values: Mapping[str, float]) -> np.ndarray:
    missing = [record_id for record_id in record_ids if record_id not in values]
    if missing:
        raise KeyError(f"missing target values for record identifiers: {missing}")
    result = np.asarray([values[record_id] for record_id in record_ids], dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("target values must be finite")
    return result


def _validate_split_coverage(dataset: ExactPhysicsDataset, split: DatasetSplit) -> None:
    expected = {record.record_id for record in dataset.records}
    actual = {*split.train_ids, *split.validation_ids, *split.test_ids}
    if actual != expected:
        raise ValueError("dataset split must cover every record exactly once")
    if split.diagnostics.leakage_groups:
        raise ValueError("dataset split reports geometry-family leakage")
    if dataset.schema_version != DATASET_SCHEMA_VERSION:
        raise ValueError("dataset schema is incompatible with this evaluation protocol")
