from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.search import (
    NOVELTY_SCORE_VERSION,
    InvalidNoveltyDistanceError,
    NoveltyComparisonSource,
    NoveltyDistanceEvaluationError,
    NoveltyReferenceCohort,
    NoveltyScoreConfig,
    PopulationNoveltyReport,
    create_initial_population,
    evaluate_population_novelty,
    geometry_to_genome,
)
from toposc_lab.search.geometry_genome import GeometryGenome


def _genome(n_sites: int) -> GeometryGenome:
    return geometry_to_genome(
        Geometry(
            n_sites=n_sites,
            edges=tuple(GeometryEdge(index, index + 1) for index in range(n_sites - 1)),
        )
    )


def _population(sizes: tuple[int, ...] = (2, 4, 8)):
    return create_initial_population(tuple(_genome(size) for size in sizes))


def _site_count_distance(first: GeometryGenome, second: GeometryGenome) -> float:
    return float(abs(first.n_sites - second.n_sites))


def test_scores_are_mean_k_nearest_distances_in_population_order() -> None:
    population = _population()
    report = evaluate_population_novelty(
        population,
        config=NoveltyScoreConfig(neighbor_count=2),
        distance=_site_count_distance,
        distance_identifier="tests.site-count-distance.v1",
    )

    assert report.population is population
    assert report.scores == (4.0, 3.0, 5.0)
    assert tuple(member.population_member for member in report.members) == (
        population.members
    )
    assert report.version == NOVELTY_SCORE_VERSION
    assert report.reference_cohort is None


def test_complete_comparison_ledger_excludes_only_own_population_slot() -> None:
    genome = _genome(3)
    population = create_initial_population((genome, genome, _genome(5)))
    report = evaluate_population_novelty(
        population,
        config=NoveltyScoreConfig(neighbor_count=1),
        distance=_site_count_distance,
        distance_identifier="tests.site-count-distance.v1",
    )

    first = report.members[0]
    assert tuple(item.source_index for item in first.comparisons) == (1, 2)
    assert tuple(item.distance for item in first.comparisons) == (0.0, 2.0)
    assert first.score == 0.0
    assert first.nearest_neighbors == (first.comparisons[0],)
    assert first.geometry_id == first.comparisons[0].geometry_id


def test_explicit_reference_cohort_supplements_population_comparisons() -> None:
    cohort = NoveltyReferenceCohort(
        identifier="tests.prior-comparisons.v1",
        genomes=(_genome(3),),
    )
    report = evaluate_population_novelty(
        _population(),
        config=NoveltyScoreConfig(neighbor_count=1),
        distance=_site_count_distance,
        distance_identifier="tests.site-count-distance.v1",
        reference_cohort=cohort,
    )

    assert report.reference_cohort is cohort
    assert report.scores == (1.0, 1.0, 4.0)
    first = report.members[0]
    assert tuple(item.source for item in first.comparisons) == (
        NoveltyComparisonSource.POPULATION,
        NoveltyComparisonSource.POPULATION,
        NoveltyComparisonSource.REFERENCE_COHORT,
    )
    assert first.comparisons[-1].genome is cohort.genomes[0]


def test_equal_distance_ties_have_stable_source_and_index_order() -> None:
    cohort = NoveltyReferenceCohort("tests.tie.v1", (_genome(1),))
    report = evaluate_population_novelty(
        _population((1, 3, 5)),
        config=NoveltyScoreConfig(neighbor_count=2),
        distance=_site_count_distance,
        distance_identifier="tests.site-count-distance.v1",
        reference_cohort=cohort,
    )

    middle = report.members[1]
    assert tuple(
        (item.source, item.source_index) for item in middle.nearest_neighbors
    ) == (
        (NoveltyComparisonSource.POPULATION, 0),
        (NoveltyComparisonSource.POPULATION, 2),
    )


def test_distance_is_called_once_per_ordered_candidate_comparison() -> None:
    calls: list[tuple[int, int]] = []

    def distance(first: GeometryGenome, second: GeometryGenome) -> float:
        calls.append((first.n_sites, second.n_sites))
        return _site_count_distance(first, second)

    evaluate_population_novelty(
        _population(),
        config=NoveltyScoreConfig(1),
        distance=distance,
        distance_identifier="tests.counted-distance.v1",
        reference_cohort=NoveltyReferenceCohort("tests.references.v1", (_genome(3),)),
    )

    assert calls == [
        (2, 4),
        (2, 8),
        (2, 3),
        (4, 2),
        (4, 8),
        (4, 3),
        (8, 2),
        (8, 4),
        (8, 3),
    ]


def test_too_few_comparisons_is_rejected_before_distance_runs() -> None:
    calls = 0

    def distance(first: GeometryGenome, second: GeometryGenome) -> float:
        nonlocal calls
        calls += 1
        return _site_count_distance(first, second)

    with pytest.raises(ValueError, match="exceeds the available"):
        evaluate_population_novelty(
            _population((2,)),
            config=NoveltyScoreConfig(1),
            distance=distance,
            distance_identifier="tests.must-not-run.v1",
        )
    with pytest.raises(ValueError, match="exceeds the available"):
        evaluate_population_novelty(
            _population((2, 3)),
            config=NoveltyScoreConfig(2),
            distance=distance,
            distance_identifier="tests.must-not-run.v1",
        )
    assert calls == 0


def test_callback_failure_is_bound_to_exact_pair_without_retry() -> None:
    calls: list[tuple[int, int]] = []

    def failing(first: GeometryGenome, second: GeometryGenome) -> float:
        calls.append((first.n_sites, second.n_sites))
        if (first.n_sites, second.n_sites) == (4, 2):
            raise RuntimeError("declared distance failure")
        return _site_count_distance(first, second)

    with pytest.raises(NoveltyDistanceEvaluationError) as caught:
        evaluate_population_novelty(
            _population(),
            config=NoveltyScoreConfig(1),
            distance=failing,
            distance_identifier="tests.failing-distance.v1",
        )

    assert calls == [(2, 4), (2, 8), (4, 2)]
    assert caught.value.generation_index == 0
    assert caught.value.member_index == 1
    assert caught.value.source is NoveltyComparisonSource.POPULATION
    assert caught.value.source_index == 0
    assert caught.value.error_type == "RuntimeError"
    assert caught.value.error_message == "declared distance failure"


def test_keyboard_interrupt_is_not_converted_to_callback_failure() -> None:
    def interrupted(first: GeometryGenome, second: GeometryGenome) -> float:
        del first, second
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        evaluate_population_novelty(
            _population((2, 3)),
            config=NoveltyScoreConfig(1),
            distance=interrupted,
            distance_identifier="tests.interrupted.v1",
        )


@pytest.mark.parametrize("value", (-1.0, float("nan"), float("inf"), True, "far"))
def test_invalid_distance_values_retain_pair_context(value: object) -> None:
    with pytest.raises(InvalidNoveltyDistanceError) as caught:
        evaluate_population_novelty(
            _population((2, 3)),
            config=NoveltyScoreConfig(1),
            distance=lambda first, second: cast(float, value),
            distance_identifier="tests.invalid-distance.v1",
        )

    assert caught.value.member_index == 0
    assert caught.value.source_index == 1
    assert caught.value.value is value


def test_report_rejects_changed_nearest_neighbors_score_and_source_layout() -> None:
    report = evaluate_population_novelty(
        _population(),
        config=NoveltyScoreConfig(1),
        distance=_site_count_distance,
        distance_identifier="tests.site-count-distance.v1",
    )
    first = report.members[0]

    with pytest.raises(ValueError, match="k-nearest"):
        replace(report, members=(replace(first, nearest_neighbors=(first.comparisons[1],)), *report.members[1:]))
    with pytest.raises(ValueError, match="mean k-nearest"):
        replace(report, members=(replace(first, score=99.0), *report.members[1:]))
    with pytest.raises(ValueError, match="source order"):
        replace(
            report,
            members=(
                replace(first, comparisons=tuple(reversed(first.comparisons))),
                *report.members[1:],
            ),
        )


def test_report_constructor_rejects_k_larger_than_retained_comparison_set() -> None:
    report = evaluate_population_novelty(
        _population((2, 3)),
        config=NoveltyScoreConfig(1),
        distance=_site_count_distance,
        distance_identifier="tests.site-count-distance.v1",
    )

    with pytest.raises(ValueError, match="exceeds the retained"):
        replace(report, config=NoveltyScoreConfig(2))


def test_report_score_lookup_uses_exact_member_index() -> None:
    report = evaluate_population_novelty(
        _population(),
        config=NoveltyScoreConfig(1),
        distance=_site_count_distance,
        distance_identifier="tests.site-count-distance.v1",
    )

    assert report.score_for_member(1) == 2.0
    with pytest.raises(IndexError, match="outside"):
        report.score_for_member(3)


@pytest.mark.parametrize("neighbor_count", (0, -1, True, 1.5))
def test_neighbor_count_must_be_a_positive_integer(neighbor_count: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        NoveltyScoreConfig(cast(int, neighbor_count))


def test_reference_cohort_requires_identifier_and_genomes() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        NoveltyReferenceCohort(" ", (_genome(2),))
    with pytest.raises(ValueError, match="must not be empty"):
        NoveltyReferenceCohort("tests.empty.v1", ())
    with pytest.raises(TypeError, match="GeometryGenome"):
        NoveltyReferenceCohort("tests.wrong.v1", cast(tuple[GeometryGenome, ...], ("x",)))


def test_novelty_remains_separate_from_fitness_selection_and_diversity() -> None:
    report = evaluate_population_novelty(
        _population(),
        config=NoveltyScoreConfig(1),
        distance=_site_count_distance,
        distance_identifier="tests.site-count-distance.v1",
    )

    warning_text = " ".join(report.warnings).lower()
    assert "fitness" in warning_text
    assert "selection" in warning_text
    assert "archive" in warning_text
    assert not hasattr(report, "fitness")
    assert not hasattr(report, "archive")
    assert not hasattr(report, "family_counts")


def test_distance_identifier_is_required_and_normalized() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        evaluate_population_novelty(
            _population((2, 3)),
            config=NoveltyScoreConfig(1),
            distance=_site_count_distance,
            distance_identifier=" ",
        )
    report = evaluate_population_novelty(
        _population((2, 3)),
        config=NoveltyScoreConfig(1),
        distance=_site_count_distance,
        distance_identifier=" tests.site-count-distance.v1 ",
    )
    assert report.distance_identifier == "tests.site-count-distance.v1"


def test_report_type_is_public_and_immutable() -> None:
    report = evaluate_population_novelty(
        _population((2, 3)),
        config=NoveltyScoreConfig(1),
        distance=_site_count_distance,
        distance_identifier="tests.site-count-distance.v1",
    )

    assert isinstance(report, PopulationNoveltyReport)
    with pytest.raises((AttributeError, TypeError)):
        report.distance_identifier = "changed"  # type: ignore[misc]
