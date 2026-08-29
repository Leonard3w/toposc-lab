"""Real-space and symmetry-aware topological analysis infrastructure."""

from toposc_lab.topology.bott import BottIndexResult, bott_index
from toposc_lab.topology.local_chern import (
    LocalChernMarkerResult,
    local_chern_marker,
)
from toposc_lab.topology.pfaffian import (
    PfaffianInvariantResult,
    one_dimensional_pfaffian_invariant,
    pfaffian,
)
from toposc_lab.topology.results import (
    InvariantGroup,
    NumericalConfidence,
    TopologyDiagnosticResult,
    TopologyMethod,
    TopologyResult,
    unify_topology_result,
)
from toposc_lab.topology.spectral_localizer import (
    SpectralLocalizerResult,
    spectral_localizer,
)
from toposc_lab.topology.symmetry import (
    AltlandZirnbauerClass,
    AntiunitarySymmetry,
    AntiunitarySymmetryKind,
    SymmetryClassification,
    classify_altland_zirnbauer,
)
from toposc_lab.topology.symmetry_validation import (
    SymmetryOperators,
    SymmetryValidationResult,
    validate_symmetry_classification,
)
from toposc_lab.topology.winding import (
    RealSpaceWindingResult,
    real_space_winding_invariant,
)

__all__ = [
    "AltlandZirnbauerClass",
    "AntiunitarySymmetry",
    "AntiunitarySymmetryKind",
    "BottIndexResult",
    "InvariantGroup",
    "LocalChernMarkerResult",
    "NumericalConfidence",
    "PfaffianInvariantResult",
    "RealSpaceWindingResult",
    "SpectralLocalizerResult",
    "SymmetryClassification",
    "SymmetryOperators",
    "SymmetryValidationResult",
    "TopologyDiagnosticResult",
    "TopologyMethod",
    "TopologyResult",
    "bott_index",
    "classify_altland_zirnbauer",
    "local_chern_marker",
    "one_dimensional_pfaffian_invariant",
    "pfaffian",
    "real_space_winding_invariant",
    "spectral_localizer",
    "unify_topology_result",
    "validate_symmetry_classification",
]
