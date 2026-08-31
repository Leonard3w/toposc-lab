"""Scientific result aggregation for geometry evaluations."""

from toposc_lab.evaluation.descriptors import evaluate_geometry_descriptors
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
    "InvalidCandidateError",
    "ModelGeometryRequirements",
    "MultiObjectiveEvaluation",
    "ObjectiveDirection",
    "ObjectiveQuantity",
    "ObjectiveScalar",
    "ObjectiveSpec",
    "ObjectiveValue",
    "TopologyIntegrationInput",
    "compute_basic_scalar_score",
    "evaluate_geometry_descriptors",
    "evaluate_eigenstates",
    "evaluate_majorana_diagnostics",
    "evaluate_multi_objectives",
    "evaluate_spectrum",
    "integrate_topology_results",
    "validate_candidate",
]
