"""Scientific result aggregation for geometry evaluations."""

from toposc_lab.evaluation.descriptors import evaluate_geometry_descriptors
from toposc_lab.evaluation.eigenstates import evaluate_eigenstates
from toposc_lab.evaluation.majorana import evaluate_majorana_diagnostics
from toposc_lab.evaluation.results import DescriptorScalar, GeometryEvaluation
from toposc_lab.evaluation.spectral import evaluate_spectrum
from toposc_lab.evaluation.topology import (
    TopologyIntegrationInput,
    integrate_topology_results,
)

__all__ = [
    "DescriptorScalar",
    "GeometryEvaluation",
    "TopologyIntegrationInput",
    "evaluate_geometry_descriptors",
    "evaluate_eigenstates",
    "evaluate_majorana_diagnostics",
    "evaluate_spectrum",
    "integrate_topology_results",
]
