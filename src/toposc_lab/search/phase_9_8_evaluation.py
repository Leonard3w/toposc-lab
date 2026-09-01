"""Frozen scientific screening contract for Phase 9.8.

This module deliberately contains no sampling, ranking, persistence, or search
loop.  It converts one already supplied geometry (and optionally one already
disordered Hamiltonian) into the experiment-specific geometry, topology, and
boundary evidence frozen by ``TOPOSC-P9.8-RS-001``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import TypeAlias

import numpy as np

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import BasisLayout
from toposc_lab.evaluation import (
    GeometryEvaluation,
    GeometryEvaluationConfig,
    GeometryEvaluationContext,
    GeometryEvaluationRun,
    GeometryModelAdapter,
    ModelGeometryRequirements,
    evaluate_geometry,
)
from toposc_lab.geometry import Geometry, GeometryValidationReport, validate_geometry
from toposc_lab.geometry.generators.hard_core_planar import (
    HARD_CORE_PLANAR_BOUNDARY_SHELL_THICKNESS,
    HARD_CORE_PLANAR_BOX_MAXIMUM,
    HARD_CORE_PLANAR_MAXIMUM_BOUNDARY_SITES,
    HARD_CORE_PLANAR_MAXIMUM_DEGREE,
    HARD_CORE_PLANAR_MAXIMUM_EDGE_LENGTH,
    HARD_CORE_PLANAR_MINIMUM_BOUNDARY_SITES,
    HARD_CORE_PLANAR_MINIMUM_DEGREE,
    HARD_CORE_PLANAR_MINIMUM_SEPARATION,
    HARD_CORE_PLANAR_N_EDGES,
    HARD_CORE_PLANAR_N_SITES,
)
from toposc_lab.models.chiral_p_wave import (
    ChiralPWaveModel,
    ChiralPWaveParameters,
)
from toposc_lab.hamiltonians import NambuBasis
from toposc_lab.observables.localization import LocalizationProfile
from toposc_lab.observables.majorana import MajoranaDiagnostics
from toposc_lab.robustness.disorder import DisorderParameterValue
from toposc_lab.topology import (
    BottIndexResult,
    LocalChernMarkerResult,
    SpectralLocalizerResult,
    SymmetryClassification,
    TopologyCapability,
    TopologyDispatchContext,
    TopologyDispatchDecision,
    TopologyResult,
    bott_index,
    dispatch_topology_methods,
    local_chern_marker,
    spectral_localizer,
    unify_topology_result,
)

PHASE_9_8_PROTOCOL_IDENTIFIER = "TOPOSC-P9.8-RS-001"
PHASE_9_8_PROTOCOL_COMMIT = "dc967ec2876f221d7b4f362f6224d7d3716f395e"
PHASE_9_8_EVALUATION_VERSION = 1

PHASE_9_8_MODEL_PARAMETERS = ChiralPWaveParameters(
    hopping=1.0,
    chemical_potential=2.0,
    pairing=1.0,
    chirality=1,
    plane_axes=(0, 1),
)
PHASE_9_8_MODEL_PARAMETER_SET: Mapping[str, DisorderParameterValue] = MappingProxyType(
    PHASE_9_8_MODEL_PARAMETERS.model_dump(mode="json")
)
PHASE_9_8_EVALUATION_CONFIG = GeometryEvaluationConfig(
    reference_energy=0.0,
    zero_mode_tolerance=1.0e-10,
    low_energy_count=16,
    boundary_localization_threshold=0.8,
    numerical_tolerance=1.0e-10,
    require_resolved_topology=False,
    require_topology_convergence=False,
    topology_convergence_checked=False,
)

PHASE_9_8_TOPOLOGY_CELL = (-0.5, 7.5, -0.5, 7.5)
PHASE_9_8_BOTT_PERIODS = ((7.6, 7.6), (8.0, 8.0), (8.4, 8.4))
PHASE_9_8_LOCALIZER_KAPPAS = (0.1, 0.2, 0.3)
PHASE_9_8_LOCALIZER_PROBE = (3.5, 3.5)
PHASE_9_8_LOCALIZER_PROTECTION_THRESHOLD = 0.20
PHASE_9_8_PARTICLE_HOLE_PAIR_TOLERANCE = 1.0e-8
PHASE_9_8_BOUNDARY_WEIGHT_THRESHOLD = 0.80
PHASE_9_8_POSITION_AREA_SUM_TOLERANCE = 1.0e-10
PHASE_9_8_GEOMETRY_TOLERANCE = 1.0e-12

_Edge: TypeAlias = tuple[int, int]
_Matching: TypeAlias = tuple[tuple[int, int], ...]


class Phase98GeometryApplicability(str, Enum):
    """Channel-specific geometry contract used before scientific gates."""

    CLEAN_PRIMARY = "clean_primary"
    COORDINATE_DISORDER = "coordinate_disorder"
    REMOVAL_DISORDER = "removal_disorder"


@dataclass(frozen=True, slots=True)
class Phase98ConstraintIssue:
    """One stable experiment-specific geometry failure reason."""

    code: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.isidentifier():
            raise ValueError("constraint issue code must be a Python-style identifier")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("constraint issue message must be non-empty")
        object.__setattr__(self, "message", self.message.strip())


@dataclass(frozen=True, slots=True)
class Phase98GeometryConstraintReport:
    """Measured Phase-9.8 geometry resources, separate from Phase-6 validity."""

    applicability: Phase98GeometryApplicability
    base_validation: GeometryValidationReport
    issues: tuple[Phase98ConstraintIssue, ...]
    measurements: Mapping[str, bool | int | float | None]

    def __post_init__(self) -> None:
        if not isinstance(self.applicability, Phase98GeometryApplicability):
            raise TypeError("applicability must be Phase98GeometryApplicability")
        if not isinstance(self.base_validation, GeometryValidationReport):
            raise TypeError("base_validation must be GeometryValidationReport")
        issues = tuple(self.issues)
        if any(not isinstance(issue, Phase98ConstraintIssue) for issue in issues):
            raise TypeError("issues must contain Phase98ConstraintIssue values")
        if len({issue.code for issue in issues}) != len(issues):
            raise ValueError("constraint issue codes must be unique")
        measurements: dict[str, bool | int | float | None] = {}
        for name, value in self.measurements.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("measurement names must be Python-style identifiers")
            if value is None or isinstance(value, (bool, int)):
                measurements[name] = value
            elif isinstance(value, Real) and np.isfinite(float(value)):
                measurements[name] = float(value)
            else:
                raise ValueError(f"measurement {name!r} must be a finite scalar")
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "measurements", MappingProxyType(measurements))

    @property
    def is_applicable(self) -> bool:
        return self.base_validation.is_valid and not self.issues


@dataclass(frozen=True, slots=True)
class Phase98TopologyInputs:
    """Deterministic topology inputs in Hamiltonian and unique-position order."""

    basis_coordinates: np.ndarray
    position_areas: np.ndarray | None
    bulk_masks: tuple[np.ndarray, ...]
    bott_periods: tuple[tuple[float, float], ...]
    localizer_probe: tuple[float, float]
    localizer_kappas: tuple[float, ...]
    fermi_energy: float = 0.0

    def __post_init__(self) -> None:
        basis_coordinates = _immutable_float_array(
            self.basis_coordinates,
            name="basis_coordinates",
            dimensions=2,
        )
        if basis_coordinates.shape[1] != 2:
            raise ValueError("basis_coordinates must contain two spatial columns")
        position_areas = self.position_areas
        if position_areas is not None:
            position_areas = _immutable_float_array(
                position_areas,
                name="position_areas",
                dimensions=1,
            )
            if np.any(position_areas <= 0.0):
                raise ValueError("position_areas must be positive")
        masks = tuple(_immutable_boolean_array(mask, name="bulk_mask") for mask in self.bulk_masks)
        if position_areas is not None and any(
            mask.shape != position_areas.shape for mask in masks
        ):
            raise ValueError("bulk masks and position areas must have equal length")
        if any(not np.any(mask) for mask in masks):
            raise ValueError("every supplied bulk mask must select at least one site")
        periods = tuple(
            (float(period[0]), float(period[1])) for period in self.bott_periods
        )
        if not periods or any(
            len(period) != 2 or not all(np.isfinite(value) and value > 0.0 for value in period)
            for period in periods
        ):
            raise ValueError("bott_periods must contain positive finite pairs")
        probe = (float(self.localizer_probe[0]), float(self.localizer_probe[1]))
        if not all(np.isfinite(value) for value in probe):
            raise ValueError("localizer_probe must be finite")
        kappas = tuple(float(value) for value in self.localizer_kappas)
        if not kappas or any(not np.isfinite(value) or value <= 0.0 for value in kappas):
            raise ValueError("localizer_kappas must be positive and finite")
        fermi_energy = float(self.fermi_energy)
        if not np.isfinite(fermi_energy):
            raise ValueError("fermi_energy must be finite")
        object.__setattr__(self, "basis_coordinates", basis_coordinates)
        object.__setattr__(self, "position_areas", position_areas)
        object.__setattr__(self, "bulk_masks", masks)
        object.__setattr__(self, "bott_periods", periods)
        object.__setattr__(self, "localizer_probe", probe)
        object.__setattr__(self, "localizer_kappas", kappas)
        object.__setattr__(self, "fermi_energy", fermi_energy)


@dataclass(frozen=True, slots=True)
class Phase98TopologyConvergenceBundle:
    """All specialized grid results plus method-specific representatives."""

    bott: tuple[BottIndexResult, ...]
    local_chern: tuple[LocalChernMarkerResult, ...]
    localizer: tuple[SpectralLocalizerResult, ...]
    representatives: tuple[TopologyResult, ...]

    def __post_init__(self) -> None:
        bott = tuple(self.bott)
        local_chern = tuple(self.local_chern)
        localizer = tuple(self.localizer)
        representatives = tuple(self.representatives)
        if any(not isinstance(item, BottIndexResult) for item in bott):
            raise TypeError("bott must contain BottIndexResult values")
        if any(not isinstance(item, LocalChernMarkerResult) for item in local_chern):
            raise TypeError("local_chern must contain LocalChernMarkerResult values")
        if any(not isinstance(item, SpectralLocalizerResult) for item in localizer):
            raise TypeError("localizer must contain SpectralLocalizerResult values")
        if any(not isinstance(item, TopologyResult) for item in representatives):
            raise TypeError("representatives must contain TopologyResult values")
        methods = tuple(item.method for item in representatives)
        if len(set(methods)) != len(methods):
            raise ValueError("representatives must contain at most one result per method")
        object.__setattr__(self, "bott", bott)
        object.__setattr__(self, "local_chern", local_chern)
        object.__setattr__(self, "localizer", localizer)
        object.__setattr__(self, "representatives", representatives)

    @property
    def localizer_protection_proxy(self) -> float:
        if not self.localizer:
            raise ValueError("localizer protection proxy requires a localizer grid")
        return min(result.localizer_gap for result in self.localizer)


@dataclass(frozen=True, slots=True)
class Phase98BoundaryStateRecord:
    """All retained diagnostics for one of the eight boundary-gate states."""

    state_index: int
    energy: float
    ipr: float
    localization: LocalizationProfile
    majorana: MajoranaDiagnostics

    def __post_init__(self) -> None:
        if isinstance(self.state_index, bool) or not isinstance(self.state_index, int):
            raise TypeError("state_index must be an integer")
        if self.state_index < 0:
            raise ValueError("state_index must be non-negative")
        for name in ("energy", "ipr"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or (name == "ipr" and value < 0.0):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not isinstance(self.localization, LocalizationProfile):
            raise TypeError("localization must be LocalizationProfile")
        if not isinstance(self.majorana, MajoranaDiagnostics):
            raise TypeError("majorana must be MajoranaDiagnostics")

    @property
    def boundary_weight(self) -> float:
        return self.localization.edge_weight


@dataclass(frozen=True, slots=True)
class Phase98BoundarySignature:
    """Deterministic PH pairing and explicit-boundary screening evidence."""

    states: tuple[Phase98BoundaryStateRecord, ...]
    particle_hole_pairs: _Matching
    pairing_cost: float
    maximum_pair_residual: float
    boundary_localized_count: int
    minimum_boundary_weight_first_four: float
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        states = tuple(self.states)
        if len(states) != 8:
            raise ValueError("boundary signature requires exactly eight states")
        if any(not isinstance(item, Phase98BoundaryStateRecord) for item in states):
            raise TypeError("states must contain Phase98BoundaryStateRecord values")
        pairs = tuple(tuple(pair) for pair in self.particle_hole_pairs)
        expected_indices = sorted(state.state_index for state in states)
        if (
            len(pairs) != 4
            or any(len(pair) != 2 for pair in pairs)
            or sorted(state for pair in pairs for state in pair) != expected_indices
        ):
            raise ValueError("particle_hole_pairs must perfectly match eight state indices")
        for name in (
            "pairing_cost",
            "maximum_pair_residual",
            "minimum_boundary_weight_first_four",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if not 0 <= self.boundary_localized_count <= 8:
            raise ValueError("boundary_localized_count must lie between zero and eight")
        reasons = tuple(self.reasons)
        if any(not isinstance(reason, str) or not reason.strip() for reason in reasons):
            raise ValueError("boundary reasons must be non-empty strings")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "particle_hole_pairs", pairs)
        object.__setattr__(self, "reasons", reasons)

    @property
    def passes(self) -> bool:
        return not self.reasons


@dataclass(frozen=True, slots=True)
class Phase98ScientificEvaluation:
    """One pipeline run plus separate Phase-9.8 gate evidence."""

    run: GeometryEvaluationRun
    geometry_constraints: Phase98GeometryConstraintReport
    topology_grid: Phase98TopologyConvergenceBundle | None
    boundary_signature: Phase98BoundarySignature | None
    gate_reasons: tuple[str, ...]
    evaluation_version: int = field(default=PHASE_9_8_EVALUATION_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.run, GeometryEvaluationRun):
            raise TypeError("run must be GeometryEvaluationRun")
        if not isinstance(self.geometry_constraints, Phase98GeometryConstraintReport):
            raise TypeError("geometry_constraints must be Phase98GeometryConstraintReport")
        if self.topology_grid is not None and not isinstance(
            self.topology_grid, Phase98TopologyConvergenceBundle
        ):
            raise TypeError("topology_grid must be Phase98TopologyConvergenceBundle or None")
        if self.boundary_signature is not None and not isinstance(
            self.boundary_signature, Phase98BoundarySignature
        ):
            raise TypeError("boundary_signature must be Phase98BoundarySignature or None")
        reasons = tuple(self.gate_reasons)
        if any(not isinstance(reason, str) or not reason.strip() for reason in reasons):
            raise ValueError("gate_reasons must contain non-empty strings")
        object.__setattr__(self, "gate_reasons", reasons)

    @property
    def clean_eligible(self) -> bool:
        return self.run.is_valid and not self.gate_reasons

    @property
    def localizer_protection_proxy(self) -> float:
        if self.topology_grid is None:
            raise ValueError("evaluation has no completed topology grid")
        return self.topology_grid.localizer_protection_proxy

    @property
    def minimum_boundary_weight_first_four(self) -> float:
        if self.boundary_signature is None:
            raise ValueError("evaluation has no completed boundary signature")
        return self.boundary_signature.minimum_boundary_weight_first_four


class _FrozenHamiltonianChiralPWaveModel(BaseModel):
    """Evaluate an explicit matrix while retaining the frozen model contract."""

    def __init__(self, geometry: Geometry, hamiltonian: np.ndarray) -> None:
        self.geometry = geometry
        matrix = np.asarray(hamiltonian, dtype=complex).copy()
        matrix.setflags(write=False)
        self._hamiltonian = matrix
        self.params = PHASE_9_8_MODEL_PARAMETERS

    @property
    def model_name(self) -> str:
        return "ChiralPWaveModel"

    @property
    def basis_layout(self) -> BasisLayout:
        return ChiralPWaveModel(self.geometry, self.params).basis_layout

    @property
    def nambu_basis(self) -> NambuBasis:
        return ChiralPWaveModel(self.geometry, self.params).nambu_basis

    def hamiltonian(self) -> np.ndarray:
        return self._hamiltonian.copy()


def validate_phase_9_8_geometry(
    geometry: Geometry,
    *,
    applicability: Phase98GeometryApplicability,
) -> Phase98GeometryConstraintReport:
    """Evaluate one frozen clean or channel-specific geometry contract."""
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be Geometry")
    if not isinstance(applicability, Phase98GeometryApplicability):
        raise TypeError("applicability must be Phase98GeometryApplicability")
    base = validate_geometry(geometry, require_connected=True)
    issues: list[Phase98ConstraintIssue] = []
    coordinates = geometry.coordinates
    minimum_separation: float | None = None
    maximum_edge_length: float | None = None
    crossing_count: int | None = None
    if coordinates is None or coordinates.shape != (geometry.n_sites, 2):
        _issue(issues, "missing_two_dimensional_coordinates", "explicit (n_sites, 2) coordinates are required")
    else:
        minimum_separation = (
            _minimum_separation(coordinates) if geometry.n_sites >= 2 else None
        )
        maximum_edge_length = _maximum_straight_edge_length(geometry)
        crossing_count = _straight_edge_crossing_count(geometry)
        if crossing_count:
            _issue(issues, "straight_edge_crossing", "straight edges cross away from shared endpoints")
        if not _displacements_match_coordinates(geometry):
            _issue(issues, "edge_displacement_mismatch", "edge displacements must equal target minus source coordinates")

    degrees = tuple(len(geometry.neighbors(site)) for site in range(geometry.n_sites))
    minimum_degree = min(degrees)
    maximum_degree = max(degrees)
    if applicability is Phase98GeometryApplicability.CLEAN_PRIMARY:
        if geometry.n_sites != HARD_CORE_PLANAR_N_SITES:
            _issue(issues, "site_count", "clean primary geometries require exactly 64 sites")
        if geometry.n_edges != HARD_CORE_PLANAR_N_EDGES:
            _issue(issues, "edge_count", "clean primary geometries require exactly 112 edges")
        if coordinates is not None and coordinates.shape == (geometry.n_sites, 2):
            minima = np.min(coordinates, axis=0)
            maxima = np.max(coordinates, axis=0)
            if not (
                np.allclose(minima, (0.0, 0.0), rtol=0.0, atol=PHASE_9_8_GEOMETRY_TOLERANCE)
                and np.allclose(
                    maxima,
                    (HARD_CORE_PLANAR_BOX_MAXIMUM, HARD_CORE_PLANAR_BOX_MAXIMUM),
                    rtol=0.0,
                    atol=PHASE_9_8_GEOMETRY_TOLERANCE,
                )
            ):
                _issue(issues, "bounding_box", "clean primary coordinates must span exactly [0, 7] x [0, 7]")
            if minimum_separation is not None and minimum_separation < HARD_CORE_PLANAR_MINIMUM_SEPARATION:
                _issue(issues, "minimum_site_separation", "clean minimum site separation is below 0.55")
            expected_boundary = _shell_boundary_sites(coordinates)
            if geometry.boundary_sites != expected_boundary:
                _issue(issues, "outer_boundary_membership", "boundary sites do not match the frozen coordinate shell")
        if maximum_edge_length is not None and maximum_edge_length > HARD_CORE_PLANAR_MAXIMUM_EDGE_LENGTH:
            _issue(issues, "maximum_edge_length", "clean maximum edge length exceeds 1.75")
        if minimum_degree < HARD_CORE_PLANAR_MINIMUM_DEGREE or maximum_degree > HARD_CORE_PLANAR_MAXIMUM_DEGREE:
            _issue(issues, "degree_range", "clean site degrees must lie between 2 and 4")
        if not (
            HARD_CORE_PLANAR_MINIMUM_BOUNDARY_SITES
            <= len(geometry.boundary_sites)
            <= HARD_CORE_PLANAR_MAXIMUM_BOUNDARY_SITES
        ):
            _issue(issues, "boundary_site_count", "clean outer boundary requires 24 through 32 sites")
        if len(geometry.boundary_components) != 1:
            _issue(issues, "boundary_component_count", "clean primary geometry requires one outer boundary component")
        elif (
            geometry.boundary_components[0].kind != "outer"
            or geometry.boundary_components[0].component_index != 0
            or geometry.boundary_components[0].sites != geometry.boundary_sites
        ):
            _issue(issues, "boundary_component_contract", "the sole boundary component must be outer component zero")
        if any(component.kind == "hole" for component in geometry.boundary_components):
            _issue(issues, "hole_boundary", "clean primary geometry declares no physical hole boundary")
        if any(edge.source > edge.target for edge in geometry.edges):
            _issue(issues, "edge_orientation", "clean edges must be stored lower-index source to higher-index target")
    elif applicability is Phase98GeometryApplicability.COORDINATE_DISORDER:
        if minimum_separation is not None and minimum_separation < 0.45:
            _issue(issues, "minimum_site_separation", "coordinate-disorder minimum separation is below 0.45")
        if maximum_edge_length is not None and maximum_edge_length > 1.95:
            _issue(issues, "maximum_edge_length", "coordinate-disorder maximum edge length exceeds 1.95")
        if not geometry.boundary_sites:
            _issue(issues, "empty_boundary", "coordinate disorder must retain an explicit boundary")
    else:
        if not geometry.boundary_sites:
            _issue(issues, "empty_boundary", "removal disorder must retain a nonempty explicit boundary")

    measurements: dict[str, bool | int | float | None] = {
        "site_count": geometry.n_sites,
        "edge_count": geometry.n_edges,
        "minimum_degree": minimum_degree,
        "maximum_degree": maximum_degree,
        "boundary_site_count": len(geometry.boundary_sites),
        "minimum_site_separation": minimum_separation,
        "maximum_edge_length": maximum_edge_length,
        "straight_edge_crossing_count": crossing_count,
        "base_valid": base.is_valid,
    }
    return Phase98GeometryConstraintReport(
        applicability=applicability,
        base_validation=base,
        issues=tuple(issues),
        measurements=measurements,
    )


def build_phase_9_8_primary_topology_inputs(geometry: Geometry) -> Phase98TopologyInputs:
    """Build clipped Voronoi areas and graph-distance bulk masks deterministically."""
    coordinates = _two_dimensional_coordinates(geometry)
    site_areas = _clipped_voronoi_areas(coordinates, cell=PHASE_9_8_TOPOLOGY_CELL)
    if not np.all(site_areas > 0.0):
        raise ValueError("clipped Voronoi areas must all be positive")
    if not np.isclose(
        float(np.sum(site_areas)),
        64.0,
        rtol=0.0,
        atol=PHASE_9_8_POSITION_AREA_SUM_TOLERANCE,
    ):
        raise ValueError("clipped Voronoi areas must sum to 64 within 1.0e-10")
    distances = _boundary_graph_distances(geometry)
    unique_order = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
    return Phase98TopologyInputs(
        basis_coordinates=np.tile(coordinates, (2, 1)),
        position_areas=site_areas[unique_order],
        bulk_masks=(
            (distances[unique_order] >= 2),
            (distances[unique_order] >= 3),
        ),
        bott_periods=PHASE_9_8_BOTT_PERIODS,
        localizer_probe=PHASE_9_8_LOCALIZER_PROBE,
        localizer_kappas=PHASE_9_8_LOCALIZER_KAPPAS,
    )


def build_phase_9_8_ammann_beenker_topology_inputs(
    geometry: Geometry,
) -> Phase98TopologyInputs:
    """Build the accepted native Ammann--Beenker descriptive inputs."""
    coordinates = _two_dimensional_coordinates(geometry)
    site_areas = np.zeros(geometry.n_sites, dtype=float)
    for face in geometry.faces:
        polygon = coordinates[np.asarray(face.sites, dtype=np.intp)]
        area = _polygon_area(polygon)
        for site in face.sites:
            site_areas[site] += area / 4.0
    distances = _boundary_graph_distances(geometry)
    order = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
    masks = ((distances[order] >= 2), (distances[order] >= 3))
    if any(np.any(site_areas[order][mask] <= 0.0) for mask in masks):
        raise ValueError("every selected Ammann--Beenker bulk site needs positive tile area")
    if np.any(site_areas <= 0.0):
        raise ValueError("local Chern implementation requires positive area at every site")
    return Phase98TopologyInputs(
        basis_coordinates=np.tile(coordinates, (2, 1)),
        position_areas=site_areas[order],
        bulk_masks=masks,
        bott_periods=PHASE_9_8_BOTT_PERIODS,
        localizer_probe=(0.0, 0.0),
        localizer_kappas=PHASE_9_8_LOCALIZER_KAPPAS,
    )


def build_phase_9_8_sierpinski_topology_inputs(geometry: Geometry) -> Phase98TopologyInputs:
    """Build native descriptive inputs without fabricating a local-Chern bulk."""
    coordinates = _two_dimensional_coordinates(geometry)
    return Phase98TopologyInputs(
        basis_coordinates=np.tile(coordinates, (2, 1)),
        position_areas=np.ones(geometry.n_sites, dtype=float),
        bulk_masks=(),
        bott_periods=((8.55, 8.55), (9.0, 9.0), (9.45, 9.45)),
        localizer_probe=(4.5, 4.5),
        localizer_kappas=PHASE_9_8_LOCALIZER_KAPPAS,
    )


def phase_9_8_topology_dispatch(*, include_local_chern: bool = True) -> TopologyDispatchDecision:
    """Return the declared class-D 2D applicability decision."""
    capabilities = {
        TopologyCapability.BULK_GAP_EVIDENCE,
        TopologyCapability.BASIS_COORDINATES,
        TopologyCapability.COORDINATE_PERIODS,
        TopologyCapability.LOCALIZER_PROBE,
    }
    if include_local_chern:
        capabilities.update(
            {TopologyCapability.BULK_MASK, TopologyCapability.POSITION_AREAS}
        )
    classification = _class_d_classification()
    return dispatch_topology_methods(
        TopologyDispatchContext(
            physical_dimension=2,
            embedding_dimension=2,
            classification=classification,
            capabilities=frozenset(capabilities),
        )
    )


def evaluate_phase_9_8_topology(
    hamiltonian: np.ndarray,
    inputs: Phase98TopologyInputs,
) -> Phase98TopologyConvergenceBundle:
    """Evaluate every retained topology-grid member and unify representatives."""
    if not isinstance(inputs, Phase98TopologyInputs):
        raise TypeError("inputs must be Phase98TopologyInputs")
    classification = _class_d_classification()
    bott_results = tuple(
        bott_index(
            hamiltonian,
            inputs.basis_coordinates,
            periods,
            classification,
            fermi_energy=inputs.fermi_energy,
            tolerance=1.0e-10,
            quantization_tolerance=1.0e-6,
        )
        for periods in inputs.bott_periods
    )
    local_chern_results: tuple[LocalChernMarkerResult, ...] = ()
    if inputs.bulk_masks:
        if inputs.position_areas is None:
            raise ValueError("local Chern grid requires position areas")
        local_chern_results = tuple(
            local_chern_marker(
                hamiltonian,
                inputs.basis_coordinates,
                inputs.position_areas,
                mask,
                classification,
                fermi_energy=inputs.fermi_energy,
                tolerance=1.0e-10,
                quantization_tolerance=5.0e-3,
            )
            for mask in inputs.bulk_masks
        )
    localizer_results = tuple(
        spectral_localizer(
            hamiltonian,
            inputs.basis_coordinates,
            inputs.localizer_probe,
            classification,
            energy=inputs.fermi_energy,
            kappa=kappa,
            tolerance=1.0e-10,
        )
        for kappa in inputs.localizer_kappas
    )
    bott_converged = _all_equal_resolved(tuple(item.bott_index for item in bott_results))
    local_chern_converged = _all_equal_resolved(
        tuple(item.chern_number for item in local_chern_results)
    )
    localizer_converged = _all_equal_resolved(
        tuple(item.local_chern_number for item in localizer_results)
    )
    representatives: list[TopologyResult] = [
        unify_topology_result(bott_results[len(bott_results) // 2], convergence_checked=bott_converged)
    ]
    if local_chern_results:
        representatives.append(
            unify_topology_result(
                local_chern_results[0],
                convergence_checked=local_chern_converged,
            )
        )
    representatives.append(
        unify_topology_result(
            localizer_results[len(localizer_results) // 2],
            convergence_checked=localizer_converged,
        )
    )
    return Phase98TopologyConvergenceBundle(
        bott=bott_results,
        local_chern=local_chern_results,
        localizer=localizer_results,
        representatives=tuple(representatives),
    )


def evaluate_phase_9_8_primary_geometry(
    geometry: Geometry,
    *,
    code_version: str,
    applicability: Phase98GeometryApplicability = Phase98GeometryApplicability.CLEAN_PRIMARY,
    hamiltonian: np.ndarray | None = None,
    evaluation_seed: int | None = None,
) -> Phase98ScientificEvaluation:
    """Run the frozen primary-stratum model and all clean scientific gates."""
    constraints = validate_phase_9_8_geometry(geometry, applicability=applicability)
    if not constraints.is_applicable:
        raise ValueError("geometry does not satisfy its Phase-9.8 applicability contract")
    inputs = build_phase_9_8_primary_topology_inputs(geometry)
    bundles: list[Phase98TopologyConvergenceBundle] = []

    def topology_hook(context: GeometryEvaluationContext) -> tuple[TopologyResult, ...]:
        bundle = evaluate_phase_9_8_topology(context.hamiltonian, inputs)
        bundles.append(bundle)
        return bundle.representatives

    adapter = _phase_9_8_model_adapter(geometry, hamiltonian=hamiltonian)
    run = evaluate_geometry(
        geometry,
        adapter=adapter,
        config=PHASE_9_8_EVALUATION_CONFIG,
        topology_hook=topology_hook,
        topology_dispatch=phase_9_8_topology_dispatch(include_local_chern=True),
        seed=evaluation_seed,
        code_version=code_version,
    )
    topology_grid = bundles[0] if len(bundles) == 1 else None
    boundary = None
    if run.is_valid and run.evaluation is not None:
        boundary = build_phase_9_8_boundary_signature(run.evaluation)
    gate_reasons = _primary_gate_reasons(
        run=run,
        constraints=constraints,
        topology_grid=topology_grid,
        boundary=boundary,
    )
    return Phase98ScientificEvaluation(
        run=run,
        geometry_constraints=constraints,
        topology_grid=topology_grid,
        boundary_signature=boundary,
        gate_reasons=gate_reasons,
    )


def evaluate_phase_9_8_descriptive_geometry(
    geometry: Geometry,
    *,
    inputs: Phase98TopologyInputs,
    code_version: str,
) -> tuple[GeometryEvaluationRun, Phase98TopologyConvergenceBundle | None]:
    """Evaluate an unmatched native reference without assigning clean eligibility."""
    bundles: list[Phase98TopologyConvergenceBundle] = []

    def topology_hook(context: GeometryEvaluationContext) -> tuple[TopologyResult, ...]:
        bundle = evaluate_phase_9_8_topology(context.hamiltonian, inputs)
        bundles.append(bundle)
        return bundle.representatives

    include_local_chern = bool(inputs.bulk_masks)
    run = evaluate_geometry(
        geometry,
        adapter=_phase_9_8_model_adapter(geometry, hamiltonian=None),
        config=PHASE_9_8_EVALUATION_CONFIG,
        topology_hook=topology_hook,
        topology_dispatch=phase_9_8_topology_dispatch(
            include_local_chern=include_local_chern
        ),
        seed=None,
        code_version=code_version,
    )
    return run, bundles[0] if len(bundles) == 1 else None


def build_phase_9_8_boundary_signature(
    evaluation: GeometryEvaluation,
) -> Phase98BoundarySignature:
    """Pair the eight closest retained states by the frozen exact algorithm."""
    ordered_indices = tuple(
        sorted(
            evaluation.low_energy_states,
            key=lambda state: (abs(evaluation.low_energy_states[state]), state),
        )[:8]
    )
    if len(ordered_indices) != 8:
        raise ValueError("boundary gate requires at least eight retained states")
    records = tuple(
        Phase98BoundaryStateRecord(
            state_index=state,
            energy=evaluation.low_energy_states[state],
            ipr=evaluation.ipr[state],
            localization=evaluation.localization[state],
            majorana=evaluation.majorana_metrics[state],
        )
        for state in ordered_indices
    )
    matching, cost = _minimum_particle_hole_matching(
        tuple(record.energy for record in records),
        state_indices=tuple(record.state_index for record in records),
    )
    records_by_index = {record.state_index: record for record in records}
    residuals = tuple(
        abs(records_by_index[first].energy + records_by_index[second].energy)
        for first, second in matching
    )
    maximum_residual = max(residuals)
    boundary_count = sum(
        record.boundary_weight >= PHASE_9_8_BOUNDARY_WEIGHT_THRESHOLD
        for record in records
    )
    reasons: list[str] = []
    if maximum_residual > PHASE_9_8_PARTICLE_HOLE_PAIR_TOLERANCE:
        reasons.append("particle_hole_pair_residual_exceeds_1e_8")
    if boundary_count < 4:
        reasons.append("fewer_than_four_of_eight_states_have_boundary_weight_0_8")
    return Phase98BoundarySignature(
        states=records,
        particle_hole_pairs=matching,
        pairing_cost=cost,
        maximum_pair_residual=maximum_residual,
        boundary_localized_count=boundary_count,
        minimum_boundary_weight_first_four=min(
            record.boundary_weight for record in records[:4]
        ),
        reasons=tuple(reasons),
    )


def _phase_9_8_model_adapter(
    geometry: Geometry,
    *,
    hamiltonian: np.ndarray | None,
) -> GeometryModelAdapter:
    def model_factory(candidate: Geometry) -> BaseModel:
        if hamiltonian is None:
            return ChiralPWaveModel(candidate, PHASE_9_8_MODEL_PARAMETERS)
        return _FrozenHamiltonianChiralPWaveModel(candidate, hamiltonian)

    def nambu_basis_resolver(model: BaseModel) -> NambuBasis:
        if isinstance(model, ChiralPWaveModel):
            return model.nambu_basis
        if isinstance(model, _FrozenHamiltonianChiralPWaveModel):
            return model.nambu_basis
        raise TypeError("Phase-9.8 evaluation requires a chiral-p-wave model")

    return GeometryModelAdapter(
        model_factory=model_factory,
        requirements=ModelGeometryRequirements(
            require_connected=True,
            require_edges=True,
            require_boundary_sites=True,
            required_spatial_axes=(0, 1),
        ),
        nambu_basis_resolver=nambu_basis_resolver,
    )


def _primary_gate_reasons(
    *,
    run: GeometryEvaluationRun,
    constraints: Phase98GeometryConstraintReport,
    topology_grid: Phase98TopologyConvergenceBundle | None,
    boundary: Phase98BoundarySignature | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not constraints.is_applicable:
        reasons.extend(f"geometry:{issue.code}" for issue in constraints.issues)
    if not run.is_valid:
        reasons.append("pipeline_invalid")
        return tuple(reasons)
    if topology_grid is None:
        reasons.append("topology_grid_unavailable")
    else:
        if len(topology_grid.bott) != 3 or len(topology_grid.local_chern) != 2 or len(topology_grid.localizer) != 3:
            reasons.append("topology_grid_shape")
        values = (
            tuple(item.bott_index for item in topology_grid.bott)
            + tuple(item.chern_number for item in topology_grid.local_chern)
            + tuple(item.local_chern_number for item in topology_grid.localizer)
        )
        if any(value is None for value in values):
            reasons.append("topology_unresolved")
        elif any(abs(value) != 1 for value in values if value is not None):
            reasons.append("topology_magnitude_not_one")
        elif len(set(values)) != 1:
            reasons.append("topology_signed_disagreement")
        if any(not result.confidence.convergence_checked for result in topology_grid.representatives):
            reasons.append("topology_not_converged")
        if topology_grid.localizer_protection_proxy < PHASE_9_8_LOCALIZER_PROTECTION_THRESHOLD:
            reasons.append("localizer_protection_proxy_below_0_20")
    if boundary is None:
        reasons.append("boundary_signature_unavailable")
    else:
        reasons.extend(f"boundary:{reason}" for reason in boundary.reasons)
    return tuple(reasons)


def _class_d_classification() -> SymmetryClassification:
    return SymmetryClassification.from_signature(
        time_reversal_square=None,
        particle_hole_square=1,
        chiral_symmetry=False,
    )


def _all_equal_resolved(values: tuple[int | None, ...]) -> bool:
    return bool(values) and all(value is not None for value in values) and len(set(values)) == 1


def _minimum_particle_hole_matching(
    energies: tuple[float, ...],
    *,
    state_indices: tuple[int, ...],
) -> tuple[_Matching, float]:
    if len(energies) != 8:
        raise ValueError("particle-hole matching requires exactly eight energies")
    if len(state_indices) != 8 or len(set(state_indices)) != 8:
        raise ValueError("particle-hole matching requires eight unique state indices")
    best_matching: _Matching | None = None
    best_cost: float | None = None

    def visit(remaining: tuple[int, ...], pairs: _Matching) -> None:
        nonlocal best_matching, best_cost
        if not remaining:
            canonical_slots: _Matching = tuple(
                sorted(
                    (min(first, second), max(first, second))
                    for first, second in pairs
                )
            )
            canonical_states: _Matching = tuple(
                sorted(
                    (
                        min(state_indices[first], state_indices[second]),
                        max(state_indices[first], state_indices[second]),
                    )
                    for first, second in canonical_slots
                )
            )
            cost = sum(
                abs(energies[first] + energies[second])
                for first, second in canonical_slots
            )
            if (
                best_cost is None
                or cost < best_cost
                or (
                    cost == best_cost
                    and best_matching is not None
                    and canonical_states < best_matching
                )
            ):
                best_cost = cost
                best_matching = canonical_states
            return
        first = remaining[0]
        for position in range(1, len(remaining)):
            second = remaining[position]
            visit(
                remaining[1:position] + remaining[position + 1 :],
                pairs + ((first, second),),
            )

    visit(tuple(range(8)), ())
    assert best_matching is not None and best_cost is not None
    return best_matching, best_cost


def _clipped_voronoi_areas(
    coordinates: np.ndarray,
    *,
    cell: tuple[float, float, float, float],
) -> np.ndarray:
    minimum_x, maximum_x, minimum_y, maximum_y = cell
    if not (minimum_x < maximum_x and minimum_y < maximum_y):
        raise ValueError("Voronoi clipping cell must have positive area")
    areas = np.empty(coordinates.shape[0], dtype=float)
    initial = np.asarray(
        [
            (minimum_x, minimum_y),
            (maximum_x, minimum_y),
            (maximum_x, maximum_y),
            (minimum_x, maximum_y),
        ],
        dtype=float,
    )
    for site, point in enumerate(coordinates):
        polygon = initial.copy()
        for competitor, other in enumerate(coordinates):
            if competitor == site:
                continue
            direction = other - point
            if float(np.linalg.norm(direction)) <= PHASE_9_8_GEOMETRY_TOLERANCE:
                raise ValueError("Voronoi construction rejects duplicate positions")
            offset = (float(np.dot(other, other)) - float(np.dot(point, point))) / 2.0
            polygon = _clip_polygon_to_half_plane(
                polygon,
                direction=direction,
                offset=offset,
            )
            if polygon.size == 0:
                raise ValueError("a clipped Voronoi cell is empty")
        areas[site] = _polygon_area(polygon)
    return areas


def _clip_polygon_to_half_plane(
    polygon: np.ndarray,
    *,
    direction: np.ndarray,
    offset: float,
) -> np.ndarray:
    output: list[np.ndarray] = []
    for index, end in enumerate(polygon):
        start = polygon[index - 1]
        start_value = float(np.dot(start, direction) - offset)
        end_value = float(np.dot(end, direction) - offset)
        start_inside = start_value <= PHASE_9_8_GEOMETRY_TOLERANCE
        end_inside = end_value <= PHASE_9_8_GEOMETRY_TOLERANCE
        if start_inside != end_inside:
            denominator = start_value - end_value
            if abs(denominator) <= PHASE_9_8_GEOMETRY_TOLERANCE:
                raise ValueError("Voronoi clipping encountered an unresolved degeneracy")
            fraction = start_value / denominator
            output.append(start + fraction * (end - start))
        if end_inside:
            output.append(end.copy())
    if not output:
        return np.empty((0, 2), dtype=float)
    return np.asarray(output, dtype=float)


def _boundary_graph_distances(geometry: Geometry) -> np.ndarray:
    if not geometry.boundary_sites:
        raise ValueError("bulk masks require explicit boundary sites")
    distances = np.full(geometry.n_sites, geometry.n_sites + 1, dtype=np.int64)
    queue: deque[int] = deque(sorted(geometry.boundary_sites))
    for site in queue:
        distances[site] = 0
    while queue:
        site = queue.popleft()
        for neighbor in geometry.neighbors(site):
            if distances[neighbor] > distances[site] + 1:
                distances[neighbor] = distances[site] + 1
                queue.append(neighbor)
    return distances


def _minimum_separation(coordinates: np.ndarray) -> float:
    if coordinates.shape[0] < 2:
        raise ValueError("minimum separation requires at least two sites")
    differences = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distances = np.linalg.norm(differences, axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances))


def _maximum_straight_edge_length(geometry: Geometry) -> float:
    coordinates = _two_dimensional_coordinates(geometry)
    if not geometry.edges:
        return 0.0
    return max(
        float(np.linalg.norm(coordinates[edge.target] - coordinates[edge.source]))
        for edge in geometry.edges
    )


def _straight_edge_crossing_count(geometry: Geometry) -> int:
    coordinates = _two_dimensional_coordinates(geometry)
    edges = tuple((edge.source, edge.target) for edge in geometry.edges)
    return sum(
        _segments_intersect(
            coordinates[first[0]],
            coordinates[first[1]],
            coordinates[second[0]],
            coordinates[second[1]],
        )
        for first_index, first in enumerate(edges)
        for second in edges[first_index + 1 :]
        if not set(first).intersection(second)
    )


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    tolerance = PHASE_9_8_GEOMETRY_TOLERANCE

    def orientation(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
        first = end - start
        second = point - start
        return float(first[0] * second[1] - first[1] * second[0])

    first_a = orientation(first_start, first_end, second_start)
    first_b = orientation(first_start, first_end, second_end)
    second_a = orientation(second_start, second_end, first_start)
    second_b = orientation(second_start, second_end, first_end)
    if first_a * first_b < -(tolerance**2) and second_a * second_b < -(tolerance**2):
        return True

    def on_segment(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> bool:
        return bool(
            np.all(point >= np.minimum(start, end) - tolerance)
            and np.all(point <= np.maximum(start, end) + tolerance)
        )

    return (
        (abs(first_a) <= tolerance and on_segment(first_start, first_end, second_start))
        or (abs(first_b) <= tolerance and on_segment(first_start, first_end, second_end))
        or (abs(second_a) <= tolerance and on_segment(second_start, second_end, first_start))
        or (abs(second_b) <= tolerance and on_segment(second_start, second_end, first_end))
    )


def _displacements_match_coordinates(geometry: Geometry) -> bool:
    coordinates = _two_dimensional_coordinates(geometry)
    return all(
        edge.displacement is not None
        and np.allclose(
            np.asarray(edge.displacement, dtype=float),
            coordinates[edge.target] - coordinates[edge.source],
            rtol=0.0,
            atol=PHASE_9_8_GEOMETRY_TOLERANCE,
        )
        for edge in geometry.edges
    )


def _shell_boundary_sites(coordinates: np.ndarray) -> frozenset[int]:
    return frozenset(
        site
        for site, (x_coordinate, y_coordinate) in enumerate(coordinates)
        if min(
            x_coordinate,
            y_coordinate,
            HARD_CORE_PLANAR_BOX_MAXIMUM - x_coordinate,
            HARD_CORE_PLANAR_BOX_MAXIMUM - y_coordinate,
        )
        <= HARD_CORE_PLANAR_BOUNDARY_SHELL_THICKNESS
    )


def _two_dimensional_coordinates(geometry: Geometry) -> np.ndarray:
    coordinates = geometry.coordinates
    if coordinates is None or coordinates.shape != (geometry.n_sites, 2):
        raise ValueError("Phase-9.8 topology requires explicit (n_sites, 2) coordinates")
    return np.asarray(coordinates, dtype=float)


def _polygon_area(polygon: np.ndarray) -> float:
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
        raise ValueError("polygon must contain at least three two-dimensional vertices")
    x_coordinates = polygon[:, 0]
    y_coordinates = polygon[:, 1]
    return abs(
        float(
            np.dot(x_coordinates, np.roll(y_coordinates, -1))
            - np.dot(y_coordinates, np.roll(x_coordinates, -1))
        )
    ) / 2.0


def _issue(
    issues: list[Phase98ConstraintIssue],
    code: str,
    message: str,
) -> None:
    if code not in {issue.code for issue in issues}:
        issues.append(Phase98ConstraintIssue(code, message))


def _immutable_float_array(
    values: np.ndarray,
    *,
    name: str,
    dimensions: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=float).copy()
    if array.ndim != dimensions or array.size == 0:
        raise ValueError(f"{name} must be a nonempty {dimensions}-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _immutable_boolean_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.bool_ or array.ndim != 1 or array.size == 0:
        raise TypeError(f"{name} must be a nonempty one-dimensional boolean array")
    result = array.copy()
    result.setflags(write=False)
    return result
