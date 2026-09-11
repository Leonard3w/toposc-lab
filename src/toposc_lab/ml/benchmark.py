"""Strict matched-budget comparison of simple models and the minimal GNN."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from toposc_lab.data import DatasetSplit, ExactPhysicsDataset
from toposc_lab.ml.evaluation import dataset_fingerprint
from toposc_lab.ml.features import (
    HandcraftedFeatureSchema,
    extract_handcrafted_features,
    records_for_ids,
)
from toposc_lab.ml.gnn import MinimalGNNRegressor
from toposc_lab.ml.graph import GraphInputSchema, graph_inputs_from_records
from toposc_lab.ml.models import (
    GradientBoostingRegressor,
    MeanRegressor,
    RegressionMetrics,
    RidgeRegressor,
    regression_metrics,
)

SURROGATE_BENCHMARK_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PairedImprovementInterval:
    confidence: float
    lower: float
    estimate: float
    upper: float
    bootstrap_samples: int


@dataclass(frozen=True, slots=True)
class SurrogateBenchmarkResult:
    selected_default: str
    selected_simple_model: str
    validation_metrics: Mapping[str, RegressionMetrics]
    test_metrics: Mapping[str, RegressionMetrics]
    gnn_validation_relative_rmse_improvement: float
    gnn_relative_rmse_improvement: float
    gnn_improvement_interval: PairedImprovementInterval
    required_relative_improvement: float
    gnn_meaningful_improvement: bool
    selection_reason: str
    train_record_count: int
    validation_record_count: int
    test_record_count: int
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    dataset_schema_version: int
    dataset_fingerprint: str
    same_split_and_data_budget: bool = True
    schema_version: int = SURROGATE_BENCHMARK_SCHEMA_VERSION


def benchmark_simple_models_vs_gnn(
    dataset: ExactPhysicsDataset,
    split: DatasetSplit,
    target_values: Mapping[str, float],
    *,
    minimum_relative_improvement: float = 0.05,
    bootstrap_seed: int = 1211,
    bootstrap_samples: int = 500,
    gnn_seed: int = 1211,
) -> SurrogateBenchmarkResult:
    """Retain a simple model unless the GNN clears both effect-size and CI gates."""
    if not 0.0 <= minimum_relative_improvement < 1.0:
        raise ValueError("minimum_relative_improvement must be in [0, 1)")
    train = records_for_ids(dataset.records, split.train_ids)
    validation = records_for_ids(dataset.records, split.validation_ids)
    test = records_for_ids(dataset.records, split.test_ids)
    if not train or not validation or not test:
        raise ValueError("benchmark requires non-empty train, validation, and test splits")
    if split.diagnostics.leakage_groups:
        raise ValueError("benchmark split contains geometry-family leakage")
    all_ids = {*split.train_ids, *split.validation_ids, *split.test_ids}
    if all_ids != {record.record_id for record in dataset.records}:
        raise ValueError("benchmark split must cover the complete dataset")
    train_targets = _targets(split.train_ids, target_values)
    validation_targets = _targets(split.validation_ids, target_values)
    test_targets = _targets(split.test_ids, target_values)

    feature_schema = HandcraftedFeatureSchema.fit(train)
    train_features = extract_handcrafted_features(train, schema=feature_schema).values
    validation_features = extract_handcrafted_features(
        validation, schema=feature_schema
    ).values
    test_features = extract_handcrafted_features(test, schema=feature_schema).values
    simple_models = {
        "ridge": RidgeRegressor(alpha=1e-5),
        "stump_boosting": GradientBoostingRegressor(),
    }
    validation_metrics: dict[str, RegressionMetrics] = {}
    for name, model in simple_models.items():
        model.fit(train_features, train_targets)
        validation_metrics[name] = regression_metrics(
            validation_targets, model.predict(validation_features)
        )
    selected_simple = min(
        simple_models,
        key=lambda name: (validation_metrics[name].rmse, name),
    )

    graph_schema = GraphInputSchema.fit(train)
    train_graphs = graph_inputs_from_records(train, schema=graph_schema)
    validation_graphs = graph_inputs_from_records(validation, schema=graph_schema)
    test_graphs = graph_inputs_from_records(test, schema=graph_schema)
    gnn = MinimalGNNRegressor(seed=gnn_seed).fit(train_graphs, train_targets)
    validation_metrics["minimal_gnn"] = regression_metrics(
        validation_targets, gnn.predict(validation_graphs)
    )

    simple_test_predictions = simple_models[selected_simple].predict(test_features)
    gnn_test_predictions = gnn.predict(test_graphs)
    trivial = MeanRegressor().fit(train_features, train_targets)
    test_metrics = {
        selected_simple: regression_metrics(test_targets, simple_test_predictions),
        "minimal_gnn": regression_metrics(test_targets, gnn_test_predictions),
        "trivial_mean": regression_metrics(test_targets, trivial.predict(test_features)),
    }
    simple_rmse = test_metrics[selected_simple].rmse
    gnn_rmse = test_metrics["minimal_gnn"].rmse
    simple_validation_rmse = validation_metrics[selected_simple].rmse
    gnn_validation_rmse = validation_metrics["minimal_gnn"].rmse
    validation_relative = (
        simple_validation_rmse - gnn_validation_rmse
    ) / max(simple_validation_rmse, np.finfo(float).eps)
    relative = (simple_rmse - gnn_rmse) / max(simple_rmse, np.finfo(float).eps)
    interval = _paired_improvement_interval(
        test_targets,
        simple_test_predictions,
        gnn_test_predictions,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
    )
    meaningful = (
        validation_relative >= minimum_relative_improvement
        and relative >= minimum_relative_improvement
        and interval.lower > 0.0
    )
    selected_default = "minimal_gnn" if meaningful else selected_simple
    reason = (
        "GNN selected: predefined relative improvement and paired bootstrap gates passed"
        if meaningful
        else "simple model retained: GNN did not pass both predefined improvement gates"
    )
    return SurrogateBenchmarkResult(
        selected_default=selected_default,
        selected_simple_model=selected_simple,
        validation_metrics=dict(sorted(validation_metrics.items())),
        test_metrics=dict(sorted(test_metrics.items())),
        gnn_validation_relative_rmse_improvement=validation_relative,
        gnn_relative_rmse_improvement=relative,
        gnn_improvement_interval=interval,
        required_relative_improvement=minimum_relative_improvement,
        gnn_meaningful_improvement=meaningful,
        selection_reason=reason,
        train_record_count=len(train),
        validation_record_count=len(validation),
        test_record_count=len(test),
        train_ids=split.train_ids,
        validation_ids=split.validation_ids,
        test_ids=split.test_ids,
        dataset_schema_version=dataset.schema_version,
        dataset_fingerprint=dataset_fingerprint(dataset),
    )


def _paired_improvement_interval(
    truth: np.ndarray,
    simple: np.ndarray,
    gnn: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> PairedImprovementInterval:
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(truth), size=(samples, len(truth)))
    simple_residuals = truth - simple
    gnn_residuals = truth - gnn
    improvements = np.sqrt(np.mean(np.square(simple_residuals[indices]), axis=1)) - np.sqrt(
        np.mean(np.square(gnn_residuals[indices]), axis=1)
    )
    estimate = float(
        np.sqrt(np.mean(np.square(simple_residuals)))
        - np.sqrt(np.mean(np.square(gnn_residuals)))
    )
    return PairedImprovementInterval(
        confidence=0.95,
        lower=float(np.quantile(improvements, 0.025)),
        estimate=estimate,
        upper=float(np.quantile(improvements, 0.975)),
        bootstrap_samples=samples,
    )


def _targets(record_ids: tuple[str, ...], values: Mapping[str, float]) -> np.ndarray:
    missing = [record_id for record_id in record_ids if record_id not in values]
    if missing:
        raise KeyError(f"missing target values for benchmark records: {missing}")
    targets = np.asarray([values[record_id] for record_id in record_ids], dtype=float)
    if not np.all(np.isfinite(targets)):
        raise ValueError("benchmark targets must be finite")
    return targets
