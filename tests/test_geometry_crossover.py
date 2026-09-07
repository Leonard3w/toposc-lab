from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from toposc_lab.evaluation import exact_geometry_id
from toposc_lab.geometry import Geometry, GeometryEdge, GeometryFace
from toposc_lab.search import (
    FIXED_SCAFFOLD_EDGE_CROSSOVER_RNG_ALGORITHM,
    FIXED_SCAFFOLD_EDGE_CROSSOVER_VERSION,
    CrossoverCompatibilityIssue,
    CrossoverCompatibilityReport,
    CrossoverParent,
    EdgeLocusDecision,
    GeometryCrossoverResult,
    GeometryGenome,
    IncompatibleCrossoverParentsError,
    MutationValidityIssue,
    MutationValidityPolicy,
    MutationValidityReport,
    fixed_scaffold_edge_crossover,
    geometry_to_genome,
    validate_fixed_scaffold_crossover_parents,
)


def _edge_locus(edge: GeometryEdge) -> tuple[int, int]:
    return tuple(sorted((edge.source, edge.target)))  # type: ignore[return-value]


def _edge_map(genome: GeometryGenome) -> dict[tuple[int, int], GeometryEdge]:
    return {_edge_locus(edge): edge for edge in genome.edges}


def _parent_pair() -> tuple[GeometryGenome, GeometryGenome]:
    coordinates = np.asarray(
        ((-0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)),
        dtype=float,
    )
    shared_metadata: dict[str, Any] = {
        "family": "fixed_scaffold",
        "weights": np.asarray((1.0, 2.0), dtype=np.float64),
    }
    first = geometry_to_genome(
        Geometry(
            n_sites=4,
            edges=(
                GeometryEdge(0, 1, edge_type="shared", metadata={"role": "base"}),
                GeometryEdge(
                    2,
                    1,
                    edge_type="first_only",
                    boundary_crossing=True,
                    metadata={"coupling": 1.5},
                ),
                GeometryEdge(3, 2, edge_type="first_variant"),
            ),
            coordinates=coordinates,
            boundary_sites=frozenset({0, 3}),
            site_types=("A", "B", "B", "A"),
            metadata=shared_metadata,
        )
    )
    second = geometry_to_genome(
        Geometry(
            n_sites=4,
            edges=(
                GeometryEdge(0, 1, edge_type="shared", metadata={"role": "base"}),
                GeometryEdge(3, 0, edge_type="second_only"),
                GeometryEdge(2, 3, edge_type="second_variant"),
            ),
            coordinates=coordinates.copy(),
            boundary_sites=frozenset({0, 3}),
            site_types=("A", "B", "B", "A"),
            metadata={
                "family": "fixed_scaffold",
                "weights": np.asarray((1.0, 2.0), dtype=np.float64),
            },
        )
    )
    return first, second


def _face_parent_pair() -> tuple[GeometryGenome, GeometryGenome]:
    coordinates = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        dtype=float,
    )
    face = GeometryFace((0, 1, 2, 3), face_type="cell", metadata={"area": 1.0})
    boundary = (
        GeometryEdge(0, 1),
        GeometryEdge(1, 2),
        GeometryEdge(2, 3),
        GeometryEdge(3, 0),
    )
    first = geometry_to_genome(
        Geometry(
            n_sites=4,
            edges=(*boundary, GeometryEdge(0, 2, edge_type="first_diagonal")),
            coordinates=coordinates,
            faces=(face,),
        )
    )
    second = geometry_to_genome(
        Geometry(
            n_sites=4,
            edges=(*boundary, GeometryEdge(1, 3, edge_type="second_diagonal")),
            coordinates=coordinates,
            faces=(face,),
        )
    )
    return first, second


def test_fixed_scaffold_crossover_is_seeded_complementary_and_auditable() -> None:
    first_parent, second_parent = _parent_pair()
    compatibility = validate_fixed_scaffold_crossover_parents(
        first_parent,
        second_parent,
    )

    first = fixed_scaffold_edge_crossover(first_parent, second_parent, seed=123)
    second = fixed_scaffold_edge_crossover(first_parent, second_parent, seed=123)

    assert compatibility.is_compatible
    assert compatibility.issues == ()
    assert first.first_parent is first_parent
    assert first.second_parent is second_parent
    assert first.seed == 123
    assert first.version == FIXED_SCAFFOLD_EDGE_CROSSOVER_VERSION
    assert first.rng_algorithm == FIXED_SCAFFOLD_EDGE_CROSSOVER_RNG_ALGORITHM
    assert first.decisions == second.decisions
    assert tuple(
        (decision.edge_locus, decision.first_candidate_source)
        for decision in first.decisions
    ) == (
        ((0, 3), CrossoverParent.FIRST),
        ((1, 2), CrossoverParent.SECOND),
        ((2, 3), CrossoverParent.SECOND),
    )
    assert tuple(exact_geometry_id(item.to_geometry()) for item in first.candidates) == tuple(
        exact_geometry_id(item.to_geometry()) for item in second.candidates
    )
    assert first.validity_reports == (first.first_validity, first.second_validity)
    assert first.all_candidates_valid


def test_every_differing_locus_is_inherited_from_complementary_parents() -> None:
    first_parent, second_parent = _parent_pair()
    result = fixed_scaffold_edge_crossover(first_parent, second_parent, seed=123)
    first_edges = _edge_map(first_parent)
    second_edges = _edge_map(second_parent)
    first_candidate_edges = _edge_map(result.first_candidate)
    second_candidate_edges = _edge_map(result.second_candidate)

    for decision in result.decisions:
        locus = decision.edge_locus
        if decision.first_candidate_source is CrossoverParent.FIRST:
            expected_first = first_edges.get(locus)
            expected_second = second_edges.get(locus)
        else:
            expected_first = second_edges.get(locus)
            expected_second = first_edges.get(locus)
        assert first_candidate_edges.get(locus) is expected_first
        assert second_candidate_edges.get(locus) is expected_second
        assert decision.second_candidate_source is not decision.first_candidate_source

    shared_locus = (0, 1)
    assert first_candidate_edges[shared_locus] is first_edges[shared_locus]
    assert second_candidate_edges[shared_locus] is second_edges[shared_locus]


def test_inherited_edges_keep_orientation_attributes_and_canonical_locus_order() -> None:
    first_parent, second_parent = _parent_pair()
    result = fixed_scaffold_edge_crossover(first_parent, second_parent, seed=123)

    assert tuple(_edge_locus(edge) for edge in result.first_candidate.edges) == (
        (0, 1),
        (2, 3),
    )
    inherited = result.first_candidate.edges[1]
    assert inherited is second_parent.edges[2]
    assert (inherited.source, inherited.target) == (2, 3)
    assert inherited.edge_type == "second_variant"

    second_edges = _edge_map(result.second_candidate)
    first_only = second_edges[(1, 2)]
    assert first_only is first_parent.edges[1]
    assert (first_only.source, first_only.target) == (2, 1)
    assert first_only.boundary_crossing
    assert first_only.metadata["coupling"] == 1.5


def test_nonedge_scaffold_is_preserved_exactly_without_parent_alignment() -> None:
    first_parent, second_parent = _parent_pair()
    result = fixed_scaffold_edge_crossover(first_parent, second_parent, seed=5)

    for candidate, source_parent in zip(
        result.candidates,
        (first_parent, second_parent),
        strict=True,
    ):
        assert candidate.n_sites == source_parent.n_sites
        assert candidate.embedding_dimension == source_parent.embedding_dimension
        assert candidate.coordinates is not None and source_parent.coordinates is not None
        assert candidate.coordinates.tobytes() == source_parent.coordinates.tobytes()
        assert candidate.boundary_sites == source_parent.boundary_sites
        assert candidate.site_types == source_parent.site_types
        assert candidate.metadata["family"] == source_parent.metadata["family"]
        assert np.array_equal(
            candidate.metadata["weights"],
            source_parent.metadata["weights"],
        )


def test_shared_faces_remain_valid_when_only_extra_edge_loci_differ() -> None:
    first_parent, second_parent = _face_parent_pair()

    result = fixed_scaffold_edge_crossover(first_parent, second_parent, seed=9)

    assert result.all_candidates_valid
    assert result.first_candidate.faces == first_parent.faces
    assert result.second_candidate.faces == second_parent.faces
    assert result.first_candidate.to_geometry().n_faces == 1
    assert result.second_candidate.to_geometry().n_faces == 1


def test_policy_invalid_candidates_are_reported_without_repair_or_retry() -> None:
    first_parent, second_parent = _parent_pair()
    policy = MutationValidityPolicy(require_connected=True)

    result = fixed_scaffold_edge_crossover(
        first_parent,
        second_parent,
        seed=1,
        validity_policy=policy,
    )

    assert not result.first_validity.is_valid
    assert tuple(issue.code for issue in result.first_validity.issues) == (
        "disconnected_candidate",
    )
    assert result.second_validity.is_valid
    assert not result.all_candidates_valid
    assert len(result.decisions) == 3


def test_abstract_parents_are_rejected_without_inventing_site_alignment() -> None:
    first = geometry_to_genome(
        Geometry(n_sites=3, edges=(GeometryEdge(0, 1), GeometryEdge(1, 2)))
    )
    second = geometry_to_genome(
        Geometry(n_sites=3, edges=(GeometryEdge(0, 1), GeometryEdge(0, 2)))
    )

    report = validate_fixed_scaffold_crossover_parents(first, second)

    assert not report.is_compatible
    assert tuple(issue.code for issue in report.issues) == (
        "missing_first_parent_coordinates",
        "missing_second_parent_coordinates",
    )
    with pytest.raises(IncompatibleCrossoverParentsError) as captured:
        fixed_scaffold_edge_crossover(first, second, seed=0)
    assert captured.value.report == report


def test_identical_edge_alleles_are_rejected_as_noop_crossover() -> None:
    first, _ = _parent_pair()
    second = replace(first)

    report = validate_fixed_scaffold_crossover_parents(first, second)

    assert tuple(issue.code for issue in report.issues) == (
        "no_differing_edge_locus",
    )


def test_site_count_and_coordinate_mismatches_are_not_aligned_implicitly() -> None:
    first, _ = _parent_pair()
    second = geometry_to_genome(
        Geometry(
            n_sites=5,
            edges=(GeometryEdge(0, 1), GeometryEdge(1, 2)),
            coordinates=np.asarray(
                ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0))
            ),
            boundary_sites=frozenset({0, 4}),
            site_types=("A", "B", "B", "B", "A"),
            metadata={"family": "fixed_scaffold"},
        )
    )

    codes = tuple(
        issue.code
        for issue in validate_fixed_scaffold_crossover_parents(first, second).issues
    )

    assert "site_count_mismatch" in codes
    assert "coordinates_mismatch" in codes
    assert "boundary_sites_mismatch" in codes
    assert "site_types_mismatch" in codes
    assert "metadata_mismatch" in codes


def test_coordinate_compatibility_is_bit_exact_including_signed_zero() -> None:
    first, second = _parent_pair()
    assert second.coordinates is not None
    coordinates = second.coordinates.copy()
    coordinates[0, 0] = 0.0
    changed = replace(second, coordinates=coordinates)

    report = validate_fixed_scaffold_crossover_parents(first, changed)

    assert "coordinates_mismatch" in tuple(issue.code for issue in report.issues)


def test_metadata_and_face_metadata_must_match_exactly() -> None:
    first, second = _parent_pair()
    changed_metadata = replace(
        second,
        metadata={
            "family": "fixed_scaffold",
            "weights": np.asarray((1.0, 3.0), dtype=np.float64),
        },
    )
    assert "metadata_mismatch" in tuple(
        issue.code
        for issue in validate_fixed_scaffold_crossover_parents(
            first,
            changed_metadata,
        ).issues
    )

    face_first, face_second = _face_parent_pair()
    changed_face = replace(
        face_second,
        faces=(GeometryFace((0, 1, 2, 3), face_type="cell", metadata={"area": 2.0}),),
    )
    assert "faces_mismatch" in tuple(
        issue.code
        for issue in validate_fixed_scaffold_crossover_parents(
            face_first,
            changed_face,
        ).issues
    )


def test_invalid_parent_is_reported_before_crossover() -> None:
    first, second = _parent_pair()
    invalid = replace(first, n_sites=0)

    report = validate_fixed_scaffold_crossover_parents(invalid, second)

    assert tuple(issue.code for issue in report.issues) == ("invalid_first_parent",)


@pytest.mark.parametrize("seed", (-1, True, 1.5))
def test_crossover_rejects_invalid_seed(seed: object) -> None:
    first, second = _parent_pair()

    with pytest.raises((TypeError, ValueError)):
        fixed_scaffold_edge_crossover(
            first,
            second,
            seed=seed,  # type: ignore[arg-type]
        )


def test_crossover_api_rejects_wrong_parent_and_policy_types() -> None:
    first, second = _parent_pair()

    with pytest.raises(TypeError, match="first_parent"):
        fixed_scaffold_edge_crossover(object(), second, seed=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="second_parent"):
        fixed_scaffold_edge_crossover(first, object(), seed=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="validity_policy"):
        fixed_scaffold_edge_crossover(
            first,
            second,
            seed=0,
            validity_policy=object(),  # type: ignore[arg-type]
        )


def test_edge_locus_decision_validates_and_exposes_complement() -> None:
    decision = EdgeLocusDecision((1, 3), CrossoverParent.FIRST)

    assert decision.second_candidate_source is CrossoverParent.SECOND
    with pytest.raises(ValueError, match="strictly ascending"):
        EdgeLocusDecision((3, 1), CrossoverParent.FIRST)
    with pytest.raises(TypeError, match="pair"):
        EdgeLocusDecision((1, 2, 3), CrossoverParent.FIRST)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CrossoverParent"):
        EdgeLocusDecision((1, 3), "first")  # type: ignore[arg-type]


def test_result_rejects_tampered_decisions_candidates_and_validity() -> None:
    first_parent, second_parent = _parent_pair()
    policy = MutationValidityPolicy(require_connected=True)
    result = fixed_scaffold_edge_crossover(
        first_parent,
        second_parent,
        seed=1,
        validity_policy=policy,
    )
    first_decision = result.decisions[0]
    tampered_decision = EdgeLocusDecision(
        first_decision.edge_locus,
        first_decision.second_candidate_source,
    )

    with pytest.raises(ValueError, match="decisions do not match"):
        replace(
            result,
            decisions=(tampered_decision, *result.decisions[1:]),
        )
    with pytest.raises(ValueError, match="first_candidate"):
        replace(result, first_candidate=first_parent)
    with pytest.raises(ValueError, match="validity reports"):
        replace(result, first_validity=MutationValidityReport())


def test_result_constructor_rejects_incompatible_parent_replacement() -> None:
    first_parent, second_parent = _parent_pair()
    result = fixed_scaffold_edge_crossover(first_parent, second_parent, seed=4)

    with pytest.raises(IncompatibleCrossoverParentsError):
        replace(result, second_parent=first_parent)


def test_compatibility_value_objects_validate_their_contracts() -> None:
    issue = CrossoverCompatibilityIssue("different_sites", "  mismatch  ", " parents ")
    report = CrossoverCompatibilityReport((issue,))

    assert issue.message == "mismatch"
    assert issue.path == "parents"
    assert not report.is_compatible
    with pytest.raises(IncompatibleCrossoverParentsError):
        report.raise_for_errors()
    with pytest.raises(ValueError, match="requires compatibility issues"):
        IncompatibleCrossoverParentsError(CrossoverCompatibilityReport())
    with pytest.raises(TypeError, match="CrossoverCompatibilityIssue"):
        CrossoverCompatibilityReport((object(),))  # type: ignore[arg-type]


def test_result_contains_no_population_fitness_or_generation_output() -> None:
    first, second = _parent_pair()
    result = fixed_scaffold_edge_crossover(first, second, seed=0)

    assert isinstance(result, GeometryCrossoverResult)
    assert not hasattr(result, "population")
    assert not hasattr(result, "fitness")
    assert not hasattr(result, "generation_index")
    assert not hasattr(result, "offspring_population")
    assert all(
        isinstance(issue, MutationValidityIssue)
        for report in result.validity_reports
        for issue in report.issues
    )
