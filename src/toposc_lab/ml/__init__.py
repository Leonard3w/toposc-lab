"""Machine-learning surrogates that prioritize, but never replace, exact physics."""

from toposc_lab.ml.benchmark import (
    SURROGATE_BENCHMARK_SCHEMA_VERSION,
    PairedImprovementInterval,
    SurrogateBenchmarkResult,
    benchmark_simple_models_vs_gnn,
)
from toposc_lab.ml.calibration import (
    CalibrationMetrics,
    IntervalCalibrator,
    PredictionInterval,
    calibration_metrics,
)
from toposc_lab.ml.classification import (
    BalancedLogisticClassifier,
    ClassificationMetrics,
    classification_metrics,
    select_classification_threshold,
)
from toposc_lab.ml.diagnostics import (
    ERROR_ANALYSIS_SCHEMA_VERSION,
    ErrorAnalysisReport,
    ErrorGroupSummary,
    analyze_regression_errors,
    save_error_analysis,
)
from toposc_lab.ml.evaluation import (
    EVALUATION_REPORT_SCHEMA_VERSION,
    MetricInterval,
    RegressionEvaluationReport,
    bootstrap_rmse_interval,
    dataset_fingerprint,
    evaluate_regression_surrogates,
    save_evaluation_report,
)
from toposc_lab.ml.features import (
    FEATURE_SCHEMA_VERSION,
    FeatureMatrix,
    HandcraftedFeatureSchema,
    extract_handcrafted_features,
    records_for_ids,
)
from toposc_lab.ml.gate import (
    SURROGATE_GATE_BENCHMARK_SEED,
    SurrogateGateBenchmarkResult,
    run_surrogate_gate_benchmark,
)
from toposc_lab.ml.gnn import MinimalGNNRegressor
from toposc_lab.ml.graph import (
    GRAPH_INPUT_SCHEMA_VERSION,
    GraphInput,
    GraphInputSchema,
    graph_input_from_record,
    graph_inputs_from_records,
)
from toposc_lab.ml.models import (
    GradientBoostingRegressor,
    MeanRegressor,
    RegressionMetrics,
    Regressor,
    RidgeRegressor,
    regression_metrics,
)
from toposc_lab.ml.ood import OOD_CAUTION, FeatureOODDetector, OODAssessment
from toposc_lab.ml.targets import (
    TARGET_SCHEMA_VERSION,
    RegressionTargetDefinition,
    RegressionTargets,
    RegressionTargetSource,
    extract_regression_targets,
)
from toposc_lab.ml.uncertainty import (
    BootstrapUncertaintyRegressor,
    UncertainPrediction,
    UncertaintyDiagnostics,
    uncertainty_diagnostics,
)

__all__ = [
    "ERROR_ANALYSIS_SCHEMA_VERSION",
    "EVALUATION_REPORT_SCHEMA_VERSION",
    "FEATURE_SCHEMA_VERSION",
    "GRAPH_INPUT_SCHEMA_VERSION",
    "OOD_CAUTION",
    "SURROGATE_BENCHMARK_SCHEMA_VERSION",
    "SURROGATE_GATE_BENCHMARK_SEED",
    "TARGET_SCHEMA_VERSION",
    "BalancedLogisticClassifier",
    "BootstrapUncertaintyRegressor",
    "CalibrationMetrics",
    "ClassificationMetrics",
    "ErrorAnalysisReport",
    "ErrorGroupSummary",
    "FeatureMatrix",
    "FeatureOODDetector",
    "GradientBoostingRegressor",
    "GraphInput",
    "GraphInputSchema",
    "HandcraftedFeatureSchema",
    "IntervalCalibrator",
    "MeanRegressor",
    "MetricInterval",
    "MinimalGNNRegressor",
    "OODAssessment",
    "PairedImprovementInterval",
    "PredictionInterval",
    "RegressionEvaluationReport",
    "RegressionMetrics",
    "RegressionTargetDefinition",
    "RegressionTargetSource",
    "RegressionTargets",
    "Regressor",
    "RidgeRegressor",
    "SurrogateBenchmarkResult",
    "SurrogateGateBenchmarkResult",
    "UncertainPrediction",
    "UncertaintyDiagnostics",
    "analyze_regression_errors",
    "benchmark_simple_models_vs_gnn",
    "bootstrap_rmse_interval",
    "calibration_metrics",
    "classification_metrics",
    "dataset_fingerprint",
    "evaluate_regression_surrogates",
    "extract_handcrafted_features",
    "extract_regression_targets",
    "graph_input_from_record",
    "graph_inputs_from_records",
    "records_for_ids",
    "regression_metrics",
    "run_surrogate_gate_benchmark",
    "save_error_analysis",
    "save_evaluation_report",
    "select_classification_threshold",
    "uncertainty_diagnostics",
]
