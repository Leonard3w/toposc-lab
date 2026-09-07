from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.search import (
    GeometryGenome,
    InvalidGeometryMutationError,
    MutationValidityIssue,
    MutationValidityPolicy,
    MutationValidityReport,
    geometry_to_genome,
    remove_edge_mutation,
    validate_geometry_mutation,
)


def _square_genome() -> GeometryGenome:
    return geometry_to_genome(
        Geometry(
            n_sites=4,
            edges=(
                GeometryEdge(0, 1),
                GeometryEdge(1, 2),
                GeometryEdge(2, 3),
                GeometryEdge(3, 0),
            ),
            coordinates=np.asarray(
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
            ),
            boundary_sites=frozenset(range(4)),
        )
    )


def _issue_codes(report: MutationValidityReport) -> tuple[str, ...]:
    return tuple(issue.code for issue in report.issues)


def test_default_policy_accepts_a_base_valid_fixed_dimension_candidate() -> None:
    source = _square_genome()
    candidate = remove_edge_mutation(source, 0)

    report = validate_geometry_mutation(source, candidate)

    assert report.is_valid
    assert report.issues == ()
    assert report.measurements == {
        "site_count": 4,
        "edge_count": 3,
        "boundary_site_count": 4,
        "minimum_degree": 1,
        "maximum_degree": 2,
        "is_connected": True,
        "minimum_site_separation": None,
        "maximum_edge_length": None,
        "straight_edge_crossing_count": None,
    }
    report.raise_for_errors()


def test_default_policy_does_not_silently_require_connectivity() -> None:
    source = _square_genome()
    candidate = geometry_to_genome(
        Geometry(
            n_sites=4,
            edges=(GeometryEdge(0, 1), GeometryEdge(2, 3)),
            coordinates=source.coordinates,
            boundary_sites=source.boundary_sites,
        )
    )

    report = validate_geometry_mutation(source, candidate)

    assert report.is_valid
    assert report.measurements["is_connected"] is False


def test_connectivity_is_rejected_only_when_requested() -> None:
    source = _square_genome()
    candidate = geometry_to_genome(
        Geometry(
            n_sites=4,
            edges=(GeometryEdge(0, 1), GeometryEdge(2, 3)),
            coordinates=source.coordinates,
            boundary_sites=source.boundary_sites,
        )
    )

    report = validate_geometry_mutation(
        source,
        candidate,
        policy=MutationValidityPolicy(require_connected=True),
    )

    assert _issue_codes(report) == ("disconnected_candidate",)


def test_embedding_dimension_change_is_always_rejected() -> None:
    source = _square_genome()
    assert source.coordinates is not None
    candidate = geometry_to_genome(
        Geometry(
            n_sites=4,
            edges=source.edges,
            coordinates=np.column_stack((source.coordinates, np.zeros(4))),
            boundary_sites=source.boundary_sites,
        )
    )

    report = validate_geometry_mutation(source, candidate)

    assert _issue_codes(report) == ("embedding_dimension_changed",)
    assert report.issues[0].path == "candidate.embedding_dimension"


def test_invalid_source_and_candidate_are_reported_without_repair() -> None:
    valid = _square_genome()
    invalid_source = replace(valid, n_sites=2)
    invalid_candidate = replace(valid, coordinates=np.zeros((3, 2)))

    report = validate_geometry_mutation(invalid_source, invalid_candidate)

    assert _issue_codes(report) == (
        "source_invalid_geometry_representation",
        "candidate_invalid_geometry_representation",
    )
    assert report.measurements == {}


def test_count_degree_and_boundary_budgets_have_separate_codes() -> None:
    source = _square_genome()
    candidate = geometry_to_genome(
        Geometry(
            n_sites=4,
            edges=(GeometryEdge(0, 1), GeometryEdge(0, 2), GeometryEdge(0, 3)),
            coordinates=source.coordinates,
            boundary_sites=frozenset({0}),
        )
    )
    policy = MutationValidityPolicy(
        minimum_site_count=5,
        maximum_edge_count=2,
        minimum_boundary_site_count=2,
        minimum_degree=2,
        maximum_degree=2,
    )

    report = validate_geometry_mutation(source, candidate, policy=policy)

    assert _issue_codes(report) == (
        "site_count_below_minimum",
        "edge_count_above_maximum",
        "boundary_site_count_below_minimum",
        "minimum_degree_below_minimum",
        "maximum_degree_above_maximum",
    )


def test_required_coordinates_and_embedding_dimension_are_explicit() -> None:
    source = geometry_to_genome(
        Geometry(n_sites=2, edges=(GeometryEdge(0, 1, displacement=(1.0,)),))
    )
    candidate = source
    policy = MutationValidityPolicy(
        require_coordinates=True,
        required_embedding_dimension=2,
    )

    report = validate_geometry_mutation(source, candidate, policy=policy)

    assert _issue_codes(report) == (
        "required_embedding_dimension",
        "missing_required_coordinates",
    )


def test_coordinate_bounds_and_minimum_separation_are_checked_with_tolerance() -> None:
    source = _square_genome()
    candidate = geometry_to_genome(
        Geometry(
            n_sites=3,
            edges=(GeometryEdge(0, 1), GeometryEdge(1, 2)),
            coordinates=np.asarray(((-0.1, 0.0), (0.0, 0.0), (1.1, 1.0))),
        )
    )
    policy = MutationValidityPolicy(
        coordinate_lower_bounds=(0.0, 0.0),
        coordinate_upper_bounds=(1.0, 1.0),
        minimum_site_separation=0.2,
    )

    report = validate_geometry_mutation(source, candidate, policy=policy)

    assert _issue_codes(report) == (
        "coordinate_below_lower_bound",
        "coordinate_above_upper_bound",
        "minimum_site_separation",
    )
    assert report.measurements["minimum_site_separation"] == pytest.approx(0.1)


def test_coordinate_bound_axis_count_must_match_candidate() -> None:
    source = _square_genome()
    policy = MutationValidityPolicy(coordinate_lower_bounds=(0.0, 0.0, 0.0))

    report = validate_geometry_mutation(source, source, policy=policy)

    assert _issue_codes(report) == ("coordinate_lower_bounds_dimension",)


def test_maximum_edge_length_uses_explicit_physical_displacement() -> None:
    source = geometry_to_genome(
        Geometry(
            n_sites=2,
            edges=(GeometryEdge(0, 1, displacement=(0.5, 0.0)),),
            coordinates=np.asarray(((0.0, 0.0), (10.0, 0.0))),
        )
    )

    report = validate_geometry_mutation(
        source,
        source,
        policy=MutationValidityPolicy(maximum_edge_length=0.75),
    )

    assert report.is_valid
    assert report.measurements["maximum_edge_length"] == pytest.approx(0.5)


def test_unavailable_and_excessive_edge_lengths_are_distinguished() -> None:
    abstract = geometry_to_genome(Geometry(n_sites=2, edges=(GeometryEdge(0, 1),)))
    unavailable = validate_geometry_mutation(
        abstract,
        abstract,
        policy=MutationValidityPolicy(maximum_edge_length=1.0),
    )
    embedded = geometry_to_genome(
        Geometry(
            n_sites=2,
            edges=(GeometryEdge(0, 1),),
            coordinates=np.asarray(((0.0,), (2.0,))),
        )
    )
    excessive = validate_geometry_mutation(
        embedded,
        embedded,
        policy=MutationValidityPolicy(maximum_edge_length=1.0),
    )

    assert _issue_codes(unavailable) == ("edge_length_unavailable",)
    assert _issue_codes(excessive) == ("maximum_edge_length",)


def test_straight_edge_crossings_are_optional_and_measured_in_two_dimensions() -> None:
    crossing = geometry_to_genome(
        Geometry(
            n_sites=4,
            edges=(GeometryEdge(0, 2), GeometryEdge(1, 3)),
            coordinates=np.asarray(
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
            ),
        )
    )

    default_report = validate_geometry_mutation(crossing, crossing)
    planar_report = validate_geometry_mutation(
        crossing,
        crossing,
        policy=MutationValidityPolicy(forbid_straight_edge_crossings=True),
    )

    assert default_report.is_valid
    assert default_report.measurements["straight_edge_crossing_count"] is None
    assert _issue_codes(planar_report) == ("straight_edge_crossing",)
    assert planar_report.measurements["straight_edge_crossing_count"] == 1


def test_crossing_policy_rejects_dimensions_where_the_check_is_undefined() -> None:
    geometry = geometry_to_genome(
        Geometry(
            n_sites=2,
            edges=(GeometryEdge(0, 1),),
            coordinates=np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
        )
    )

    report = validate_geometry_mutation(
        geometry,
        geometry,
        policy=MutationValidityPolicy(forbid_straight_edge_crossings=True),
    )

    assert _issue_codes(report) == ("straight_edge_crossing_check_unavailable",)


def test_validity_report_is_immutable_and_can_raise_explicitly() -> None:
    issue = MutationValidityIssue("example_issue", "example failure", "candidate")
    report = MutationValidityReport(
        issues=(issue,),
        measurements={"site_count": 2},
    )

    assert not report.is_valid
    assert report.measurements["site_count"] == 2
    with pytest.raises(TypeError):
        report.measurements["site_count"] = 3  # type: ignore[index]
    with pytest.raises(InvalidGeometryMutationError, match="example_issue") as caught:
        report.raise_for_errors()
    assert caught.value.report is report


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    (
        ({"require_connected": 1}, TypeError),
        ({"minimum_site_count": True}, TypeError),
        ({"minimum_site_count": -1}, ValueError),
        ({"minimum_site_count": 3, "maximum_site_count": 2}, ValueError),
        ({"required_embedding_dimension": 0}, ValueError),
        ({"coordinate_lower_bounds": ()}, ValueError),
        (
            {
                "coordinate_lower_bounds": (0.0,),
                "coordinate_upper_bounds": (1.0, 2.0),
            },
            ValueError,
        ),
        (
            {
                "coordinate_lower_bounds": (2.0,),
                "coordinate_upper_bounds": (1.0,),
            },
            ValueError,
        ),
        ({"minimum_site_separation": -0.1}, ValueError),
        ({"maximum_edge_length": np.inf}, ValueError),
        ({"numerical_tolerance": 0.0}, ValueError),
    ),
)
def test_policy_rejects_malformed_constraints(
    kwargs: dict[str, object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        MutationValidityPolicy(**kwargs)  # type: ignore[arg-type]


def test_validator_requires_domain_objects_and_policy() -> None:
    genome = _square_genome()

    with pytest.raises(TypeError, match="source must be"):
        validate_geometry_mutation(object(), genome)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="candidate must be"):
        validate_geometry_mutation(genome, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="policy must be"):
        validate_geometry_mutation(genome, genome, policy=object())  # type: ignore[arg-type]
