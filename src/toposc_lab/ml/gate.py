"""Fixed exact-physics acceptance benchmark for Surrogate Gate 12."""

from __future__ import annotations

from dataclasses import dataclass

from toposc_lab.data import DatasetSplitConfig, ExactPhysicsDataset, split_dataset
from toposc_lab.data.dataset_benchmark import _exact_kitaev_record
from toposc_lab.ml.benchmark import benchmark_simple_models_vs_gnn
from toposc_lab.ml.calibration import CalibrationMetrics, IntervalCalibrator, calibration_metrics
from toposc_lab.ml.features import (
    HandcraftedFeatureSchema,
    extract_handcrafted_features,
    records_for_ids,
)
from toposc_lab.ml.models import GradientBoostingRegressor, RidgeRegressor
from toposc_lab.ml.ood import FeatureOODDetector
from toposc_lab.ml.targets import (
    RegressionTargetDefinition,
    RegressionTargetSource,
    extract_regression_targets,
)
from toposc_lab.ml.uncertainty import (
    BootstrapUncertaintyRegressor,
    UncertaintyDiagnostics,
    uncertainty_diagnostics,
)

SURROGATE_GATE_BENCHMARK_SEED = 12_120


@dataclass(frozen=True, slots=True)
class SurrogateGateBenchmarkResult:
    seed: int
    exact_record_count: int
    selected_default: str
    selected_test_rmse: float
    trivial_test_rmse: float
    beats_trivial_predictor: bool
    gnn_meaningful_improvement: bool
    uncertainty_diagnostics: UncertaintyDiagnostics
    calibration_metrics: CalibrationMetrics
    unusual_graph_flagged_ood: bool
    exact_test_label_ids: tuple[str, ...]
    dataset_schema_version: int
    dataset_fingerprint: str
    predictions_link_exact_labels: bool

    @property
    def passed(self) -> bool:
        return (
            self.beats_trivial_predictor
            and self.unusual_graph_flagged_ood
            and self.predictions_link_exact_labels
            and self.selected_default != "minimal_gnn"
        )


def run_surrogate_gate_benchmark(
    *, seed: int = SURROGATE_GATE_BENCHMARK_SEED
) -> SurrogateGateBenchmarkResult:
    """Evaluate frozen surrogates on untouched exact Kitaev-chain labels."""
    chemical_potentials = (-3.0, -1.5, 0.0, 1.5, 3.0)
    records = tuple(
        _exact_kitaev_record(
            n_sites=n_sites,
            chemical_potential=chemical_potentials[index % len(chemical_potentials)],
            seed=seed + index,
        )
        for index, n_sites in enumerate(range(6, 36))
    )
    dataset = ExactPhysicsDataset(records)
    split = split_dataset(
        dataset,
        config=DatasetSplitConfig(
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
            seed=seed,
        ),
    )
    definition = RegressionTargetDefinition(
        name="finite_chain_minimum_absolute_energy",
        source=RegressionTargetSource.OBSERVABLE,
        result_kind="minimum_absolute_energy",
        result_version="1",
        value_key="gap",
    )
    exact_targets = extract_regression_targets(dataset, definition)
    values = exact_targets.valid_values_by_id()
    benchmark = benchmark_simple_models_vs_gnn(
        dataset,
        split,
        values,
        bootstrap_seed=seed,
        gnn_seed=seed,
    )

    train = records_for_ids(records, split.train_ids)
    validation = records_for_ids(records, split.validation_ids)
    test = records_for_ids(records, split.test_ids)
    schema = HandcraftedFeatureSchema.fit(train)
    train_features = extract_handcrafted_features(train, schema=schema)
    validation_features = extract_handcrafted_features(validation, schema=schema)
    test_features = extract_handcrafted_features(test, schema=schema)
    factory = (
        (lambda: RidgeRegressor(alpha=1e-5))
        if benchmark.selected_default == "ridge"
        else (lambda: GradientBoostingRegressor())
    )
    uncertainty_model = BootstrapUncertaintyRegressor(
        model_factory=factory,
        ensemble_size=16,
        seed=seed,
        distance_weight=0.5,
    ).fit(train_features.values, _values(split.train_ids, values))
    validation_prediction = uncertainty_model.predict_with_uncertainty(
        validation_features.values
    )
    test_prediction = uncertainty_model.predict_with_uncertainty(test_features.values)
    calibrator = IntervalCalibrator(coverage=0.9).fit(
        _values(split.validation_ids, values), validation_prediction
    )
    intervals = calibrator.predict_interval(test_prediction)
    calibration = calibration_metrics(
        _values(split.test_ids, values),
        intervals,
        inherited_failure_modes=calibrator.failure_modes,
    )
    uncertainty = uncertainty_diagnostics(_values(split.test_ids, values), test_prediction)

    unusual = _exact_kitaev_record(
        n_sites=60,
        chemical_potential=5.5,
        seed=seed + 10_000,
    )
    unusual_features = extract_handcrafted_features((unusual,), schema=schema)
    ood = FeatureOODDetector(reference_quantile=0.95).fit(train_features.values).assess(
        unusual_features.values,
        record_ids=unusual_features.record_ids,
    )
    selected_metrics = benchmark.test_metrics[benchmark.selected_default]
    trivial_metrics = benchmark.test_metrics["trivial_mean"]
    return SurrogateGateBenchmarkResult(
        seed=seed,
        exact_record_count=len(records),
        selected_default=benchmark.selected_default,
        selected_test_rmse=selected_metrics.rmse,
        trivial_test_rmse=trivial_metrics.rmse,
        beats_trivial_predictor=selected_metrics.rmse < trivial_metrics.rmse,
        gnn_meaningful_improvement=benchmark.gnn_meaningful_improvement,
        uncertainty_diagnostics=uncertainty,
        calibration_metrics=calibration,
        unusual_graph_flagged_ood=bool(ood.is_ood[0]),
        exact_test_label_ids=split.test_ids,
        dataset_schema_version=dataset.schema_version,
        dataset_fingerprint=exact_targets.dataset_fingerprint,
        predictions_link_exact_labels=(
            exact_targets.dataset_fingerprint == benchmark.dataset_fingerprint
            and set(split.test_ids).issubset(values)
        ),
    )


def _values(record_ids: tuple[str, ...], values: dict[str, float]) -> list[float]:
    return [values[record_id] for record_id in record_ids]
