from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from toposc_lab.core.results import BasisLayout, SimulationResult
from toposc_lab.evaluation import (
    CandidateFailureStage,
    CandidateIssueCategory,
    CandidateIssueSeverity,
    CandidateStageFailure,
    CandidateValidityIssue,
    CandidateValidityPolicy,
    CandidateValidityReport,
    GeometryEvaluation,
    InvalidCandidateError,
    ModelGeometryRequirements,
    validate_candidate,
)
from toposc_lab.geometry import Geometry, GeometryEdge, chain, square
from toposc_lab.topology import NumericalConfidence, TopologyMethod, TopologyResult


def _basis(n_sites: int = 2) -> BasisLayout:
    return BasisLayout(
        spatial_shape=(n_sites,),
        components_per_site=1,
        component_labels=("orbital",),
    )


def _simulation_result(
    *,
    eigenvalues: np.ndarray | None = None,
    eigenvectors: np.ndarray | None = None,
    basis_layout: BasisLayout | None = None,
) -> SimulationResult:
    layout = _basis() if basis_layout is None else basis_layout
    values = np.array([0.0, 1.0]) if eigenvalues is None else eigenvalues
    vectors = np.eye(2) if eigenvectors is None else eigenvectors
    return SimulationResult(
        model_name="TestModel",
        eigenvalues=values,
        eigenvectors=vectors,
        basis_layout=layout,
    )


def _topology_result(
    *,
    resolved: bool,
    convergence_checked: bool,
) -> TopologyResult:
    return TopologyResult(
        invariant_value=-1 if resolved else None,
        is_topological=True if resolved else None,
        invariant_group="Z2",
        method=TopologyMethod.PFAFFIAN_1D,
        applicability_assumptions=("The method-specific assumptions hold.",),
        confidence=NumericalConfidence(
            is_resolved=resolved,
            is_quantized=True if resolved else None,
            minimum_gap=0.4,
            gap_kind="endpoint_energy_gap",
            quantization_error=0.0 if resolved else None,
            maximum_residual=1.0e-14,
            convergence_checked=convergence_checked,
        ),
        warnings=(),
    )


def test_consistent_candidate_has_clean_report() -> None:
    result = validate_candidate(
        chain(2),
        basis_layout=_basis(),
        hamiltonian=np.diag([0.0, 1.0]),
        simulation_result=_simulation_result(),
        evaluation=GeometryEvaluation(low_energy_states={0: 0.0, 1: 1.0}),
    )

    assert result.is_valid
    assert result.issues == ()
    assert result.errors == ()
    assert result.warnings == ()
    result.raise_for_errors()


def test_geometry_warning_is_preserved_without_invalidating_candidate() -> None:
    report = validate_candidate(Geometry(n_sites=2))

    assert report.is_valid
    assert tuple(issue.code for issue in report.warnings) == (
        "disconnected_components",
    )
    assert report.warnings[0].category is CandidateIssueCategory.GEOMETRY


def test_connectivity_is_invalid_only_when_explicitly_required() -> None:
    report = validate_candidate(
        Geometry(n_sites=2),
        policy=CandidateValidityPolicy(
            model_geometry=ModelGeometryRequirements(require_connected=True)
        ),
    )

    assert not report.is_valid
    assert tuple(issue.code for issue in report.errors) == (
        "disconnected_components",
    )


def test_defensively_invalid_geometry_is_converted_to_candidate_error() -> None:
    geometry = chain(2)
    object.__setattr__(geometry.edges[0], "target", 3)

    report = validate_candidate(geometry)

    assert not report.is_valid
    assert "invalid_site_index" in {issue.code for issue in report.errors}


@pytest.mark.parametrize(
    ("geometry", "requirements", "code"),
    [
        (
            Geometry(n_sites=1),
            ModelGeometryRequirements(require_edges=True),
            "missing_required_edges",
        ),
        (
            Geometry(n_sites=2, edges=(GeometryEdge(0, 1),)),
            ModelGeometryRequirements(require_boundary_sites=True),
            "missing_required_boundary",
        ),
        (
            chain(2),
            ModelGeometryRequirements(required_spatial_axes=(0, 1)),
            "missing_required_spatial_axes",
        ),
    ],
)
def test_explicit_model_geometry_requirements_detect_incompatibility(
    geometry: Geometry,
    requirements: ModelGeometryRequirements,
    code: str,
) -> None:
    report = validate_candidate(
        geometry,
        policy=CandidateValidityPolicy(model_geometry=requirements),
    )

    assert code in {issue.code for issue in report.errors}
    assert all(
        issue.category is CandidateIssueCategory.MODEL_GEOMETRY
        for issue in report.errors
    )


def test_required_spatial_axes_accept_embedded_geometry_with_directions() -> None:
    report = validate_candidate(
        square(2, 2),
        policy=CandidateValidityPolicy(
            model_geometry=ModelGeometryRequirements(
                require_edges=True,
                required_spatial_axes=(0, 1),
            )
        ),
    )

    assert report.is_valid


def test_zero_edge_displacement_is_an_invalid_required_direction() -> None:
    geometry = Geometry(
        n_sites=2,
        edges=(GeometryEdge(0, 1, displacement=(0.0, 0.0)),),
        embedding_dimension=2,
    )
    report = validate_candidate(
        geometry,
        policy=CandidateValidityPolicy(
            model_geometry=ModelGeometryRequirements(required_spatial_axes=(0, 1))
        ),
    )

    assert "invalid_required_spatial_direction" in {
        issue.code for issue in report.errors
    }


def test_basis_and_geometry_site_mismatch_is_invalid() -> None:
    report = validate_candidate(chain(2), basis_layout=_basis(3))

    issue = report.errors[0]
    assert issue.code == "basis_geometry_site_mismatch"
    assert issue.category is CandidateIssueCategory.MODEL_GEOMETRY


def test_simulation_result_and_supplied_basis_must_agree() -> None:
    layout = _basis(3)
    simulation = SimulationResult(
        model_name="OtherModel",
        eigenvalues=np.arange(3.0),
        eigenvectors=np.eye(3),
        basis_layout=layout,
    )

    report = validate_candidate(
        chain(2),
        basis_layout=_basis(2),
        simulation_result=simulation,
    )

    assert {issue.code for issue in report.errors} == {
        "result_geometry_site_mismatch",
        "result_basis_mismatch",
    }


@pytest.mark.parametrize(
    ("hamiltonian", "code"),
    [
        (np.ones(2), "invalid_hamiltonian_shape"),
        (np.ones((2, 3)), "invalid_hamiltonian_shape"),
        (np.empty((0, 0)), "invalid_hamiltonian_shape"),
        (np.array([[0.0, np.nan], [np.nan, 0.0]]), "nonfinite_hamiltonian"),
        (np.array([[0.0, 1.0], [0.0, 0.0]]), "nonhermitian_hamiltonian"),
        (np.eye(3), "hamiltonian_basis_dimension_mismatch"),
    ],
)
def test_invalid_hamiltonian_is_reported(
    hamiltonian: np.ndarray,
    code: str,
) -> None:
    report = validate_candidate(
        chain(2),
        basis_layout=_basis(),
        hamiltonian=hamiltonian,
    )

    assert code in {issue.code for issue in report.errors}


def test_nonorthonormal_eigenvectors_are_reported() -> None:
    simulation = _simulation_result(
        eigenvalues=np.array([0.0, 0.0]),
        eigenvectors=np.array([[1.0, 1.0], [0.0, 0.0]]),
    )

    report = validate_candidate(chain(2), simulation_result=simulation)

    assert "nonorthonormal_eigenvectors" in {
        issue.code for issue in report.errors
    }


def test_inaccurate_eigenpair_is_reported_against_hamiltonian() -> None:
    simulation = _simulation_result(eigenvalues=np.array([0.0, 2.0]))

    report = validate_candidate(
        chain(2),
        hamiltonian=np.diag([0.0, 1.0]),
        simulation_result=simulation,
    )

    assert "eigenpair_residual_exceeded" in {
        issue.code for issue in report.errors
    }


@pytest.mark.parametrize(
    ("low_energy_states", "code"),
    [
        ({2: 0.0}, "evaluation_state_out_of_range"),
        ({1: 0.5}, "evaluation_energy_mismatch"),
    ],
)
def test_evaluation_spectrum_must_match_simulation_result(
    low_energy_states: dict[int, float],
    code: str,
) -> None:
    report = validate_candidate(
        chain(2),
        simulation_result=_simulation_result(),
        evaluation=GeometryEvaluation(low_energy_states=low_energy_states),
    )

    assert code in {issue.code for issue in report.errors}


def test_recorded_stage_failure_invalidates_candidate_without_exception_object() -> None:
    failure = CandidateStageFailure.from_exception(
        CandidateFailureStage.SOLVER,
        np.linalg.LinAlgError("eigensolver did not converge"),
    )

    report = validate_candidate(chain(2), failures=(failure,))

    issue = report.errors[0]
    assert issue.code == "solver_failure"
    assert issue.category is CandidateIssueCategory.EXECUTION
    assert "LinAlgError" in issue.message
    assert not hasattr(failure, "exception")


def test_topology_is_not_a_default_validity_requirement() -> None:
    evaluation = GeometryEvaluation(topology=(_topology_result(
        resolved=False,
        convergence_checked=False,
    ),))

    assert validate_candidate(chain(2), evaluation=evaluation).is_valid


def test_explicit_topology_policy_detects_unresolved_and_unchecked_results() -> None:
    evaluation = GeometryEvaluation(topology=(_topology_result(
        resolved=False,
        convergence_checked=False,
    ),))
    policy = CandidateValidityPolicy(
        require_resolved_topology=True,
        require_topology_convergence=True,
    )

    report = validate_candidate(chain(2), policy=policy, evaluation=evaluation)

    assert {issue.code for issue in report.errors} == {
        "unresolved_topology",
        "unchecked_topology_convergence",
    }


def test_required_topology_must_be_present() -> None:
    report = validate_candidate(
        chain(2),
        policy=CandidateValidityPolicy(require_resolved_topology=True),
    )

    assert tuple(issue.code for issue in report.errors) == (
        "missing_required_topology",
    )


def test_report_is_immutable_and_can_raise_with_structured_result() -> None:
    issue = CandidateValidityIssue(
        code="test_failure",
        severity=CandidateIssueSeverity.ERROR,
        category=CandidateIssueCategory.NUMERICAL,
        message="A numerical test failed.",
        path="solver",
    )
    report = CandidateValidityReport((issue,))

    with pytest.raises(FrozenInstanceError):
        report.issues = ()  # type: ignore[misc]
    with pytest.raises(InvalidCandidateError, match="test_failure") as caught:
        report.raise_for_errors()
    assert caught.value.report is report


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"numerical_tolerance": 0.0}, ValueError, "finite and positive"),
        ({"numerical_tolerance": np.inf}, ValueError, "finite and positive"),
        ({"require_resolved_topology": 1}, TypeError, "must be a boolean"),
        ({"model_geometry": object()}, TypeError, "ModelGeometryRequirements"),
    ],
)
def test_policy_rejects_invalid_configuration(
    kwargs: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        CandidateValidityPolicy(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("failures", ["solver", (object(),)])
def test_failure_collection_is_typed(failures: object) -> None:
    with pytest.raises(TypeError, match="failures"):
        validate_candidate(
            chain(2),
            failures=failures,  # type: ignore[arg-type]
        )
