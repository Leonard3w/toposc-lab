"""Structured invalid-candidate detection for geometry evaluation inputs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral, Real

import numpy as np

from toposc_lab.core.results import BasisLayout, SimulationResult
from toposc_lab.evaluation.results import GeometryEvaluation
from toposc_lab.geometry.base import Geometry
from toposc_lab.geometry.validation import validate_geometry


class CandidateIssueSeverity(str, Enum):
    """Whether an issue excludes the candidate or only records a limitation."""

    WARNING = "warning"
    ERROR = "error"


class CandidateIssueCategory(str, Enum):
    """Stable ownership categories for candidate-validity issues."""

    GEOMETRY = "geometry"
    MODEL_GEOMETRY = "model_geometry"
    NUMERICAL = "numerical"
    EVALUATION = "evaluation"
    EXECUTION = "execution"


class CandidateFailureStage(str, Enum):
    """Pipeline stage at which a serializable failure was observed."""

    MODEL_CONSTRUCTION = "model_construction"
    HAMILTONIAN_CONSTRUCTION = "hamiltonian_construction"
    SOLVER = "solver"
    EVALUATION = "evaluation"
    TOPOLOGY = "topology"


@dataclass(frozen=True, slots=True)
class CandidateStageFailure:
    """Serializable failure information without retaining an exception object."""

    stage: CandidateFailureStage
    error_type: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.stage, CandidateFailureStage):
            raise TypeError("stage must be CandidateFailureStage")
        if not isinstance(self.error_type, str) or not self.error_type.isidentifier():
            raise ValueError("error_type must be a Python-style identifier")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("failure message must be a non-empty string")
        object.__setattr__(self, "message", self.message.strip())

    @classmethod
    def from_exception(
        cls,
        stage: CandidateFailureStage,
        error: Exception,
    ) -> CandidateStageFailure:
        """Create a stable failure record from a caught ordinary exception."""
        if not isinstance(stage, CandidateFailureStage):
            raise TypeError("stage must be CandidateFailureStage")
        if not isinstance(error, Exception):
            raise TypeError("error must be an Exception")
        message = str(error).strip() or "The stage raised an exception without a message."
        return cls(
            stage=stage,
            error_type=type(error).__name__,
            message=message,
        )


@dataclass(frozen=True, slots=True)
class CandidateValidityIssue:
    """One machine-readable reason affecting candidate acceptance."""

    code: str
    severity: CandidateIssueSeverity
    category: CandidateIssueCategory
    message: str
    path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.isidentifier():
            raise ValueError("candidate issue code must be a Python-style identifier")
        if not isinstance(self.severity, CandidateIssueSeverity):
            raise TypeError("severity must be CandidateIssueSeverity")
        if not isinstance(self.category, CandidateIssueCategory):
            raise TypeError("category must be CandidateIssueCategory")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("candidate issue message must be a non-empty string")
        if self.path is not None and (
            not isinstance(self.path, str) or not self.path.strip()
        ):
            raise ValueError("candidate issue path must be non-empty or None")
        object.__setattr__(self, "message", self.message.strip())
        if self.path is not None:
            object.__setattr__(self, "path", self.path.strip())


@dataclass(frozen=True, slots=True)
class ModelGeometryRequirements:
    """Explicit physical requirements supplied by a model adapter or caller."""

    require_connected: bool = False
    require_edges: bool = False
    require_boundary_sites: bool = False
    required_spatial_axes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("require_connected", self.require_connected),
            ("require_edges", self.require_edges),
            ("require_boundary_sites", self.require_boundary_sites),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")
        axes = tuple(
            _nonnegative_integer(axis, name="required spatial axis")
            for axis in self.required_spatial_axes
        )
        if len(set(axes)) != len(axes):
            raise ValueError("required_spatial_axes must not contain duplicates")
        object.__setattr__(self, "required_spatial_axes", axes)


@dataclass(frozen=True, slots=True)
class CandidateValidityPolicy:
    """Explicit thresholds and completeness requirements for one validity pass."""

    model_geometry: ModelGeometryRequirements = ModelGeometryRequirements()
    numerical_tolerance: float = 1.0e-10
    require_resolved_topology: bool = False
    require_topology_convergence: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model_geometry, ModelGeometryRequirements):
            raise TypeError("model_geometry must be ModelGeometryRequirements")
        tolerance = _positive_finite_real(
            self.numerical_tolerance,
            name="numerical_tolerance",
        )
        if not isinstance(self.require_resolved_topology, bool):
            raise TypeError("require_resolved_topology must be a boolean")
        if not isinstance(self.require_topology_convergence, bool):
            raise TypeError("require_topology_convergence must be a boolean")
        object.__setattr__(self, "numerical_tolerance", tolerance)


@dataclass(frozen=True, slots=True)
class CandidateValidityReport:
    """Complete structured result used to accept or filter one candidate."""

    issues: tuple[CandidateValidityIssue, ...]

    def __post_init__(self) -> None:
        issues = tuple(self.issues)
        if not all(isinstance(issue, CandidateValidityIssue) for issue in issues):
            raise TypeError("issues must contain only CandidateValidityIssue objects")
        object.__setattr__(self, "issues", issues)

    @property
    def errors(self) -> tuple[CandidateValidityIssue, ...]:
        """All issues that make the candidate invalid."""
        return tuple(
            issue
            for issue in self.issues
            if issue.severity is CandidateIssueSeverity.ERROR
        )

    @property
    def warnings(self) -> tuple[CandidateValidityIssue, ...]:
        """All non-fatal limitations retained for downstream decisions."""
        return tuple(
            issue
            for issue in self.issues
            if issue.severity is CandidateIssueSeverity.WARNING
        )

    @property
    def is_valid(self) -> bool:
        """Whether the candidate has no error-level issue."""
        return not self.errors

    def raise_for_errors(self) -> None:
        """Raise ``InvalidCandidateError`` when the candidate is invalid."""
        if self.errors:
            raise InvalidCandidateError(self)


class InvalidCandidateError(ValueError):
    """Error raised when a caller explicitly rejects an invalid candidate."""

    def __init__(self, report: CandidateValidityReport) -> None:
        if not isinstance(report, CandidateValidityReport):
            raise TypeError("report must be CandidateValidityReport")
        self.report = report
        details = "; ".join(
            f"{issue.code} at {issue.path or issue.category.value}: {issue.message}"
            for issue in report.errors
        )
        super().__init__(f"candidate validation failed: {details}")


def validate_candidate(
    geometry: Geometry,
    *,
    policy: CandidateValidityPolicy | None = None,
    basis_layout: BasisLayout | None = None,
    hamiltonian: np.ndarray | None = None,
    simulation_result: SimulationResult | None = None,
    evaluation: GeometryEvaluation | None = None,
    failures: Iterable[CandidateStageFailure] = (),
) -> CandidateValidityReport:
    """Detect explicit structural, compatibility, and numerical failures.

    The function reports rather than raises for candidate-level invalidity.
    Type errors in the API itself still raise. Model-specific physics enters
    only through ``policy.model_geometry``; no requirement is inferred from a
    model name, coordinate dimension, or warning string.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be a Geometry")
    if policy is None:
        policy = CandidateValidityPolicy()
    elif not isinstance(policy, CandidateValidityPolicy):
        raise TypeError("policy must be CandidateValidityPolicy or None")
    if basis_layout is not None and not isinstance(basis_layout, BasisLayout):
        raise TypeError("basis_layout must be BasisLayout or None")
    if simulation_result is not None and not isinstance(
        simulation_result,
        SimulationResult,
    ):
        raise TypeError("simulation_result must be SimulationResult or None")
    if evaluation is not None and not isinstance(evaluation, GeometryEvaluation):
        raise TypeError("evaluation must be GeometryEvaluation or None")
    if isinstance(failures, (str, bytes)) or not isinstance(failures, Iterable):
        raise TypeError("failures must be an iterable of CandidateStageFailure")
    prepared_failures = tuple(failures)
    if any(not isinstance(failure, CandidateStageFailure) for failure in prepared_failures):
        raise TypeError("failures must contain only CandidateStageFailure objects")

    geometry_report = validate_geometry(
        geometry,
        require_connected=policy.model_geometry.require_connected,
    )
    issues = [
        CandidateValidityIssue(
            code=issue.code,
            severity=CandidateIssueSeverity(issue.severity),
            category=CandidateIssueCategory.GEOMETRY,
            message=issue.message,
            path=issue.path,
        )
        for issue in geometry_report.issues
    ]

    if geometry_report.is_valid:
        _check_model_geometry_requirements(
            geometry,
            requirements=policy.model_geometry,
            issues=issues,
        )
    _check_basis_compatibility(
        geometry,
        basis_layout=basis_layout,
        simulation_result=simulation_result,
        issues=issues,
    )
    matrix = _check_hamiltonian(
        hamiltonian,
        expected_layout=(
            basis_layout
            if basis_layout is not None
            else simulation_result.basis_layout
            if simulation_result is not None
            else None
        ),
        tolerance=policy.numerical_tolerance,
        issues=issues,
    )
    if simulation_result is not None:
        _check_simulation_result(
            simulation_result,
            hamiltonian=matrix,
            tolerance=policy.numerical_tolerance,
            issues=issues,
        )
    if evaluation is not None and simulation_result is not None:
        _check_evaluation_spectrum_consistency(
            evaluation,
            simulation_result=simulation_result,
            tolerance=policy.numerical_tolerance,
            issues=issues,
        )
    _check_topology_policy(evaluation, policy=policy, issues=issues)
    _record_stage_failures(prepared_failures, issues=issues)
    return CandidateValidityReport(issues=tuple(issues))


def _check_model_geometry_requirements(
    geometry: Geometry,
    *,
    requirements: ModelGeometryRequirements,
    issues: list[CandidateValidityIssue],
) -> None:
    if requirements.require_edges and not geometry.edges:
        issues.append(
            _error(
                "missing_required_edges",
                CandidateIssueCategory.MODEL_GEOMETRY,
                "the model requires at least one geometry edge",
                "geometry.edges",
            )
        )
    if requirements.require_boundary_sites and not geometry.boundary_sites:
        issues.append(
            _error(
                "missing_required_boundary",
                CandidateIssueCategory.MODEL_GEOMETRY,
                "the model requires explicitly declared boundary sites",
                "geometry.boundary_sites",
            )
        )
    if requirements.required_spatial_axes:
        dimension = geometry.embedding_dimension
        maximum_axis = max(requirements.required_spatial_axes)
        if dimension is None or maximum_axis >= dimension:
            issues.append(
                _error(
                    "missing_required_spatial_axes",
                    CandidateIssueCategory.MODEL_GEOMETRY,
                    "geometry embedding does not provide required spatial axes "
                    f"{requirements.required_spatial_axes}",
                    "geometry.embedding_dimension",
                )
            )
            return
        for edge_index, edge in enumerate(geometry.edges):
            try:
                geometry.direction(edge.source, edge.target)
            except ValueError as error:
                issues.append(
                    _error(
                        "invalid_required_spatial_direction",
                        CandidateIssueCategory.MODEL_GEOMETRY,
                        str(error),
                        f"geometry.edges[{edge_index}]",
                    )
                )


def _check_basis_compatibility(
    geometry: Geometry,
    *,
    basis_layout: BasisLayout | None,
    simulation_result: SimulationResult | None,
    issues: list[CandidateValidityIssue],
) -> None:
    if basis_layout is not None and basis_layout.n_sites != geometry.n_sites:
        issues.append(
            _error(
                "basis_geometry_site_mismatch",
                CandidateIssueCategory.MODEL_GEOMETRY,
                f"basis has {basis_layout.n_sites} sites but geometry has "
                f"{geometry.n_sites}",
                "basis_layout.spatial_shape",
            )
        )
    if simulation_result is not None:
        result_layout = simulation_result.basis_layout
        if result_layout.n_sites != geometry.n_sites and result_layout != basis_layout:
            issues.append(
                _error(
                    "result_geometry_site_mismatch",
                    CandidateIssueCategory.MODEL_GEOMETRY,
                    f"simulation result has {result_layout.n_sites} sites but geometry "
                    f"has {geometry.n_sites}",
                    "simulation_result.basis_layout",
                )
            )
        if basis_layout is not None and result_layout != basis_layout:
            issues.append(
                _error(
                    "result_basis_mismatch",
                    CandidateIssueCategory.MODEL_GEOMETRY,
                    "simulation-result basis does not match the supplied model basis",
                    "simulation_result.basis_layout",
                )
            )


def _check_hamiltonian(
    hamiltonian: np.ndarray | None,
    *,
    expected_layout: BasisLayout | None,
    tolerance: float,
    issues: list[CandidateValidityIssue],
) -> np.ndarray | None:
    if hamiltonian is None:
        return None
    try:
        matrix = np.asarray(hamiltonian, dtype=complex)
    except (TypeError, ValueError, OverflowError) as error:
        issues.append(
            _error(
                "invalid_hamiltonian_array",
                CandidateIssueCategory.NUMERICAL,
                f"Hamiltonian cannot be converted to a complex array: {error}",
                "hamiltonian",
            )
        )
        return None
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        issues.append(
            _error(
                "invalid_hamiltonian_shape",
                CandidateIssueCategory.NUMERICAL,
                "Hamiltonian must be a non-empty square matrix",
                "hamiltonian",
            )
        )
        return None
    if not np.all(np.isfinite(matrix)):
        issues.append(
            _error(
                "nonfinite_hamiltonian",
                CandidateIssueCategory.NUMERICAL,
                "Hamiltonian contains non-finite values",
                "hamiltonian",
            )
        )
        return None
    if expected_layout is not None and matrix.shape != (
        expected_layout.dimension,
        expected_layout.dimension,
    ):
        issues.append(
            _error(
                "hamiltonian_basis_dimension_mismatch",
                CandidateIssueCategory.MODEL_GEOMETRY,
                f"Hamiltonian shape {matrix.shape} does not match basis dimension "
                f"{expected_layout.dimension}",
                "hamiltonian",
            )
        )
    scale = max(1.0, _maximum_abs(matrix))
    residual = _maximum_abs(matrix - matrix.conj().T) / scale
    if residual > tolerance:
        issues.append(
            _error(
                "nonhermitian_hamiltonian",
                CandidateIssueCategory.NUMERICAL,
                f"normalized Hermiticity residual {residual:.17g} exceeds tolerance "
                f"{tolerance:.17g}",
                "hamiltonian",
            )
        )
    return matrix


def _check_simulation_result(
    result: SimulationResult,
    *,
    hamiltonian: np.ndarray | None,
    tolerance: float,
    issues: list[CandidateValidityIssue],
) -> None:
    energies = np.asarray(result.eigenvalues)
    vectors = np.asarray(result.eigenvectors)
    if energies.ndim != 1 or vectors.ndim != 2:
        issues.append(
            _error(
                "invalid_eigensystem_shape",
                CandidateIssueCategory.NUMERICAL,
                "eigenvalues and eigenvectors have invalid dimensions",
                "simulation_result",
            )
        )
        return
    if energies.size == 0:
        issues.append(
            _error(
                "empty_eigensystem",
                CandidateIssueCategory.NUMERICAL,
                "simulation result contains no eigenstates",
                "simulation_result.eigenvalues",
            )
        )
        return
    if vectors.shape != (result.dimension, energies.size):
        issues.append(
            _error(
                "invalid_eigensystem_shape",
                CandidateIssueCategory.NUMERICAL,
                "eigenvector shape does not match basis dimension and eigenvalue count",
                "simulation_result.eigenvectors",
            )
        )
        return
    if energies.size > result.dimension:
        issues.append(
            _error(
                "too_many_eigenstates",
                CandidateIssueCategory.NUMERICAL,
                "simulation result contains more eigenstates than the basis dimension",
                "simulation_result.eigenvectors",
            )
        )
        return
    if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(vectors)):
        issues.append(
            _error(
                "nonfinite_eigensystem",
                CandidateIssueCategory.NUMERICAL,
                "simulation result contains non-finite eigenvalues or eigenvectors",
                "simulation_result",
            )
        )
        return

    gram = vectors.conj().T @ vectors
    orthonormality_residual = _maximum_abs(gram - np.eye(energies.size))
    if orthonormality_residual > tolerance:
        issues.append(
            _error(
                "nonorthonormal_eigenvectors",
                CandidateIssueCategory.NUMERICAL,
                f"orthonormality residual {orthonormality_residual:.17g} exceeds "
                f"tolerance {tolerance:.17g}",
                "simulation_result.eigenvectors",
            )
        )

    if hamiltonian is None or hamiltonian.shape != (result.dimension, result.dimension):
        return
    scale = max(1.0, _maximum_abs(hamiltonian), _maximum_abs(energies))
    residual = _maximum_abs(
        hamiltonian @ vectors - vectors * energies[np.newaxis, :]
    ) / scale
    if residual > tolerance:
        issues.append(
            _error(
                "eigenpair_residual_exceeded",
                CandidateIssueCategory.NUMERICAL,
                f"normalized eigenpair residual {residual:.17g} exceeds tolerance "
                f"{tolerance:.17g}",
                "simulation_result",
            )
        )


def _check_evaluation_spectrum_consistency(
    evaluation: GeometryEvaluation,
    *,
    simulation_result: SimulationResult,
    tolerance: float,
    issues: list[CandidateValidityIssue],
) -> None:
    for state_index, energy in evaluation.low_energy_states.items():
        if state_index >= simulation_result.n_states:
            issues.append(
                _error(
                    "evaluation_state_out_of_range",
                    CandidateIssueCategory.EVALUATION,
                    f"evaluation state {state_index} is absent from the simulation result",
                    f"evaluation.low_energy_states[{state_index}]",
                )
            )
            continue
        expected = float(simulation_result.eigenvalues[state_index])
        scale = max(1.0, abs(expected), abs(energy))
        residual = abs(energy - expected) / scale
        if residual > tolerance:
            issues.append(
                _error(
                    "evaluation_energy_mismatch",
                    CandidateIssueCategory.EVALUATION,
                    f"normalized energy mismatch {residual:.17g} exceeds tolerance "
                    f"{tolerance:.17g}",
                    f"evaluation.low_energy_states[{state_index}]",
                )
            )


def _check_topology_policy(
    evaluation: GeometryEvaluation | None,
    *,
    policy: CandidateValidityPolicy,
    issues: list[CandidateValidityIssue],
) -> None:
    if not (
        policy.require_resolved_topology or policy.require_topology_convergence
    ):
        return
    if evaluation is None or not evaluation.topology:
        issues.append(
            _error(
                "missing_required_topology",
                CandidateIssueCategory.EVALUATION,
                "validity policy requires at least one topology result",
                "evaluation.topology",
            )
        )
        return
    for result in evaluation.topology:
        if policy.require_resolved_topology and not result.confidence.is_resolved:
            issues.append(
                _error(
                    "unresolved_topology",
                    CandidateIssueCategory.NUMERICAL,
                    f"topology method {result.method.value} is numerically unresolved",
                    f"evaluation.topology.{result.method.value}",
                )
            )
        if (
            policy.require_topology_convergence
            and not result.confidence.convergence_checked
        ):
            issues.append(
                _error(
                    "unchecked_topology_convergence",
                    CandidateIssueCategory.NUMERICAL,
                    f"topology method {result.method.value} has no convergence check",
                    f"evaluation.topology.{result.method.value}",
                )
            )


def _record_stage_failures(
    failures: tuple[CandidateStageFailure, ...],
    *,
    issues: list[CandidateValidityIssue],
) -> None:
    for failure in failures:
        issues.append(
            _error(
                f"{failure.stage.value}_failure",
                CandidateIssueCategory.EXECUTION,
                f"{failure.error_type}: {failure.message}",
                failure.stage.value,
            )
        )


def _error(
    code: str,
    category: CandidateIssueCategory,
    message: str,
    path: str,
) -> CandidateValidityIssue:
    return CandidateValidityIssue(
        code=code,
        severity=CandidateIssueSeverity.ERROR,
        category=category,
        message=message,
        path=path,
    )


def _maximum_abs(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def _positive_finite_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result
