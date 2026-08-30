"""Scientific result aggregation for geometry evaluations."""

from toposc_lab.evaluation.eigenstates import evaluate_eigenstates
from toposc_lab.evaluation.results import DescriptorScalar, GeometryEvaluation
from toposc_lab.evaluation.spectral import evaluate_spectrum

__all__ = [
    "DescriptorScalar",
    "GeometryEvaluation",
    "evaluate_eigenstates",
    "evaluate_spectrum",
]
