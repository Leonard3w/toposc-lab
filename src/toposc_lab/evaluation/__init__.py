"""Scientific result aggregation for geometry evaluations."""

from toposc_lab.evaluation.descriptors import evaluate_geometry_descriptors
from toposc_lab.evaluation.evaluate import (
    GeometryEvaluationConfig,
    GeometryEvaluationContext,
    GeometryEvaluationRun,
    GeometryModelAdapter,
    evaluate_geometry,
)
from toposc_lab.evaluation.eigenstates import evaluate_eigenstates
from toposc_lab.evaluation.majorana import evaluate_majorana_diagnostics
from toposc_lab.evaluation.objectives import (
    MultiObjectiveEvaluation,
    ObjectiveDirection,
    ObjectiveQuantity,
    ObjectiveScalar,
    ObjectiveSpec,
    ObjectiveValue,
    evaluate_multi_objectives,
)
from toposc_lab.evaluation.results import DescriptorScalar, GeometryEvaluation
from toposc_lab.evaluation.reproducibility import (
    GEOMETRY_ID_SCHEME,
    CodeVersionSource,
    ReproducibilityRecord,
    ReproducibilityScalar,
    ReproducibilityValue,
    create_reproducibility_record,
    exact_geometry_id,
)
from toposc_lab.evaluation.score import (
    BasicScalarScore,
    BasicScoreComponent,
    compute_basic_scalar_score,
)
from toposc_lab.evaluation.spectral import evaluate_spectrum
from toposc_lab.evaluation.topology import (
    TopologyIntegrationInput,
    integrate_topology_results,
)
from toposc_lab.evaluation.validity import (
    CandidateFailureStage,
    CandidateIssueCategory,
    CandidateIssueSeverity,
    CandidateStageFailure,
    CandidateValidityIssue,
    CandidateValidityPolicy,
    CandidateValidityReport,
    InvalidCandidateError,
    ModelGeometryRequirements,
    validate_candidate,
)

__all__ = [
    "BasicScalarScore",
    "BasicScoreComponent",
    "CandidateFailureStage",
    "CandidateIssueCategory",
    "CandidateIssueSeverity",
    "CandidateStageFailure",
    "CandidateValidityIssue",
    "CandidateValidityPolicy",
    "CandidateValidityReport",
    "DescriptorScalar",
    "GeometryEvaluation",
    "GeometryEvaluationConfig",
    "GeometryEvaluationContext",
    "GeometryEvaluationRun",
    "GeometryModelAdapter",
    "GEOMETRY_ID_SCHEME",
    "InvalidCandidateError",
    "ModelGeometryRequirements",
    "MultiObjectiveEvaluation",
    "ObjectiveDirection",
    "ObjectiveQuantity",
    "ObjectiveScalar",
    "ObjectiveSpec",
    "ObjectiveValue",
    "CodeVersionSource",
    "ReproducibilityRecord",
    "ReproducibilityScalar",
    "ReproducibilityValue",
    "TopologyIntegrationInput",
    "compute_basic_scalar_score",
    "create_reproducibility_record",
    "evaluate_geometry_descriptors",
    "evaluate_geometry",
    "evaluate_eigenstates",
    "evaluate_majorana_diagnostics",
    "evaluate_multi_objectives",
    "evaluate_spectrum",
    "exact_geometry_id",
    "integrate_topology_results",
    "validate_candidate",
]
