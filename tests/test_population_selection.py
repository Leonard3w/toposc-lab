from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import BasisLayout
from toposc_lab.evaluation import (
    BasicScalarScore,
    BasicScoreComponent,
    GeometryEvaluationRun,
    GeometryModelAdapter,
    MultiObjectiveEvaluation,
    ObjectiveDirection,
    ObjectiveQuantity,
    ObjectiveSpec,
    evaluate_geometry,
)
from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.search import (
    TOURNAMENT_SELECTION_RNG_ALGORITHM,
    TOURNAMENT_SELECTION_VERSION,
    GeometryGenome,
    InitialPopulation,
    InitialPopulationMember,
    MultiObjectiveFitnessDefinition,
    PopulationFitnessResult,
    PopulationSelectionResult,
    ScalarFitnessDefinition,
    TournamentSelectionConfig,
    TournamentSelectionRecord,
    create_initial_population,
    evaluate_population_fitness,
    geometry_to_genome,
    select_population_members,
)


class _SelectionModel(BaseModel):
    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry

    @property
    def parameters(self) -> dict[str, Any]:
        return {"energy_scale": float(self.geometry.n_sites)}

    @property
    def basis_layout(self) -> BasisLayout:
        return BasisLayout(
            spatial_shape=(self.geometry.n_sites,),
            components_per_site=1,
            ordering="site_major",
            component_labels=("orbital",),
        )

    def hamiltonian(self) -> np.ndarray:
        negative_count = self.geometry.n_sites // 2
        positive_count = self.geometry.n_sites - negative_count
        scale = float(self.geometry.n_sites)
        eigenvalues = np.concatenate(
            (
                -scale * np.arange(1, negative_count + 1, dtype=float),
                scale * np.arange(1, positive_count + 1, dtype=float),
            )
        )
        return np.diag(eigenvalues).astype(complex)


def _genome(n_sites: int, *, edge_count: int | None = None) -> GeometryGenome:
    if edge_count is None:
        edge_count = n_sites - 1
    possible_edges = tuple(
        GeometryEdge(source, target)
        for source in range(n_sites)
        for target in range(source + 1, n_sites)
    )
    return geometry_to_genome(
        Geometry(n_sites=n_sites, edges=possible_edges[:edge_count])
    )


def _population(
    genomes: tuple[GeometryGenome, ...] | None = None,
) -> InitialPopulation:
    if genomes is None:
        genomes = (_genome(2), _genome(3), _genome(4), _genome(5))
    return create_initial_population(genomes)


def _evaluate_member(member: InitialPopulationMember) -> GeometryEvaluationRun:
    return evaluate_geometry(
        member.genome.to_geometry(),
        adapter=GeometryModelAdapter(model_factory=_SelectionModel),
        seed=member.member_index,
        code_version="phase-10.12-test",
    )


def _scalar_result(
    *,
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE,
    genomes: tuple[GeometryGenome, ...] | None = None,
    failing_indices: frozenset[int] = frozenset(),
) -> PopulationFitnessResult:
    population = _population(genomes)

    def evaluator(member: InitialPopulationMember) -> GeometryEvaluationRun:
        if member.member_index in failing_indices:
            raise RuntimeError(f"failed member {member.member_index}")
        return _evaluate_member(member)

    return evaluate_population_fitness(
        population,
        definition=ScalarFitnessDefinition(
            weights={BasicScoreComponent.NORMALIZED_GAP: 1.0},
            direction=direction,
        ),
        evaluator=evaluator,
    )


def _multi_objective_result(
    *,
    edge_direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE,
) -> PopulationFitnessResult:
    definition = MultiObjectiveFitnessDefinition(
        objectives=(
            ObjectiveSpec(
                "site_count",
                ObjectiveQuantity.GEOMETRY_DESCRIPTOR,
                ObjectiveDirection.MAXIMIZE,
                descriptor_name="site_count",
            ),
            ObjectiveSpec(
                "edge_count",
                ObjectiveQuantity.GEOMETRY_DESCRIPTOR,
                edge_direction,
                descriptor_name="edge_count",
            ),
        )
    )
    population = _population(
        (_genome(3, edge_count=2), _genome(4, edge_count=1), _genome(2, edge_count=1))
    )
    return evaluate_population_fitness(
        population,
        definition=definition,
        evaluator=_evaluate_member,
    )


def _record_signature(
    result: PopulationSelectionResult,
) -> tuple[tuple[tuple[int, ...], int], ...]:
    return tuple(
        (record.contestant_member_indices, record.winner_member_index)
        for record in result.records
    )


def test_seeded_tournaments_are_reproducible_and_auditable() -> None:
    source = _scalar_result()
    config = TournamentSelectionConfig(selection_count=6, tournament_size=2)

    first = select_population_members(source, config=config, seed=123)
    second = select_population_members(source, config=config, seed=123)

    assert _record_signature(first) == _record_signature(second)
    assert _record_signature(first) == (
        ((0, 2), 2),
        ((3, 0), 3),
        ((3, 0), 3),
        ((0, 1), 1),
        ((3, 1), 3),
        ((0, 3), 3),
    )
    assert first.source is source
    assert first.config is config
    assert first.seed == 123
    assert first.rng_algorithm == TOURNAMENT_SELECTION_RNG_ALGORITHM
    assert first.version == TOURNAMENT_SELECTION_VERSION
    assert tuple(record.selection_index for record in first.records) == tuple(range(6))
    assert first.selected_members == tuple(record.winner for record in first.records)
    assert all(
        len(set(record.contestant_member_indices)) == config.tournament_size
        for record in first.records
    )


@pytest.mark.parametrize(
    "direction",
    (ObjectiveDirection.MAXIMIZE, ObjectiveDirection.MINIMIZE),
)
def test_scalar_selection_honors_explicit_direction(
    direction: ObjectiveDirection,
) -> None:
    source = _scalar_result(direction=direction)
    result = select_population_members(
        source,
        config=TournamentSelectionConfig(selection_count=1, tournament_size=4),
        seed=5,
    )
    values = {
        member.member_index: member.fitness.value
        for member in source.available_members
        if isinstance(member.fitness, BasicScalarScore)
    }
    expected = (
        max(values, key=values.__getitem__)
        if direction is ObjectiveDirection.MAXIMIZE
        else min(values, key=values.__getitem__)
    )

    assert result.records[0].winner_member_index == expected


def test_exact_scalar_ties_are_seeded_and_not_order_biased() -> None:
    duplicate = _genome(3)
    source = _scalar_result(genomes=(duplicate, duplicate))
    config = TournamentSelectionConfig(selection_count=20, tournament_size=2)

    first = select_population_members(source, config=config, seed=71)
    second = select_population_members(source, config=config, seed=71)

    assert _record_signature(first) == _record_signature(second)
    assert {record.winner_member_index for record in first.records} == {0, 1}


def test_multi_objective_selection_uses_pareto_dominance_without_scalarizing() -> None:
    source = _multi_objective_result()
    assert all(
        isinstance(member.fitness, MultiObjectiveEvaluation)
        for member in source.available_members
    )
    result = select_population_members(
        source,
        config=TournamentSelectionConfig(selection_count=20, tournament_size=3),
        seed=11,
    )

    assert {record.winner_member_index for record in result.records} == {0, 1}
    assert all(record.winner_member_index != 2 for record in result.records)


def test_pareto_dominance_honors_each_objective_direction() -> None:
    source = _multi_objective_result(edge_direction=ObjectiveDirection.MINIMIZE)
    result = select_population_members(
        source,
        config=TournamentSelectionConfig(selection_count=8, tournament_size=3),
        seed=13,
    )

    assert {record.winner_member_index for record in result.records} == {1}


def test_failed_members_stay_in_source_but_are_never_contestants() -> None:
    source = _scalar_result(failing_indices=frozenset({1, 3}))
    original_members = source.members

    result = select_population_members(
        source,
        config=TournamentSelectionConfig(selection_count=12, tournament_size=2),
        seed=19,
    )

    assert source.members is original_members
    assert tuple(member.member_index for member in source.available_members) == (0, 2)
    assert all(
        set(record.contestant_member_indices) == {0, 2} for record in result.records
    )


def test_winners_may_repeat_across_independent_tournaments() -> None:
    source = _scalar_result()
    result = select_population_members(
        source,
        config=TournamentSelectionConfig(selection_count=10, tournament_size=1),
        seed=29,
    )

    winner_indices = tuple(record.winner_member_index for record in result.records)
    assert len(winner_indices) > len(set(winner_indices))


def test_tournament_selection_does_not_guarantee_global_best_survival() -> None:
    source = _scalar_result()
    fitness_values = {
        member.member_index: member.fitness.value
        for member in source.available_members
        if isinstance(member.fitness, BasicScalarScore)
    }
    global_best = max(fitness_values, key=fitness_values.__getitem__)
    result = select_population_members(
        source,
        config=TournamentSelectionConfig(selection_count=1, tournament_size=1),
        seed=1,
    )

    assert result.records[0].winner_member_index != global_best


@pytest.mark.parametrize(
    ("selection_count", "tournament_size", "error_type"),
    (
        (0, 1, ValueError),
        (1, 0, ValueError),
        (-1, 1, ValueError),
        (True, 1, TypeError),
        (1, 1.5, TypeError),
    ),
)
def test_config_rejects_invalid_counts(
    selection_count: object,
    tournament_size: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        TournamentSelectionConfig(
            selection_count=selection_count,  # type: ignore[arg-type]
            tournament_size=tournament_size,  # type: ignore[arg-type]
        )


def test_selection_rejects_an_empty_eligible_pool() -> None:
    source = _scalar_result(failing_indices=frozenset({0, 1, 2, 3}))

    with pytest.raises(ValueError, match="at least one available"):
        select_population_members(
            source,
            config=TournamentSelectionConfig(selection_count=1, tournament_size=1),
            seed=0,
        )


def test_selection_rejects_tournament_larger_than_eligible_pool() -> None:
    source = _scalar_result(failing_indices=frozenset({1, 3}))

    with pytest.raises(ValueError, match="must not exceed"):
        select_population_members(
            source,
            config=TournamentSelectionConfig(selection_count=1, tournament_size=3),
            seed=0,
        )


@pytest.mark.parametrize("seed", (-1, True, 1.5))
def test_selection_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        select_population_members(
            _scalar_result(),
            config=TournamentSelectionConfig(selection_count=1, tournament_size=1),
            seed=seed,  # type: ignore[arg-type]
        )


def test_selection_api_rejects_wrong_source_or_config_type() -> None:
    source = _scalar_result()
    config = TournamentSelectionConfig(selection_count=1, tournament_size=1)

    with pytest.raises(TypeError, match="source must be"):
        select_population_members(object(), config=config, seed=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config must be"):
        select_population_members(source, config=object(), seed=0)  # type: ignore[arg-type]


def test_record_rejects_duplicate_unavailable_or_external_winner() -> None:
    source = _scalar_result(failing_indices=frozenset({2}))
    available = source.available_members
    unavailable = source.members[2]

    with pytest.raises(ValueError, match="duplicate contestants"):
        TournamentSelectionRecord(0, (available[0], available[0]), available[0])
    with pytest.raises(ValueError, match="available fitness"):
        TournamentSelectionRecord(0, (unavailable,), unavailable)
    with pytest.raises(ValueError, match="exact tournament contestants"):
        TournamentSelectionRecord(0, (available[0],), available[1])


def test_result_rejects_a_tampered_seeded_ledger() -> None:
    source = _scalar_result()
    config = TournamentSelectionConfig(selection_count=2, tournament_size=2)
    valid = select_population_members(source, config=config, seed=31)
    first = valid.records[0]
    wrong_winner = next(
        contestant for contestant in first.contestants if contestant is not first.winner
    )
    tampered = (
        TournamentSelectionRecord(
            selection_index=0,
            contestants=first.contestants,
            winner=wrong_winner,
        ),
        valid.records[1],
    )

    with pytest.raises(ValueError, match="winner does not match"):
        PopulationSelectionResult(source, config, 31, tampered)


def test_result_requires_exact_record_count_and_order() -> None:
    source = _scalar_result()
    config = TournamentSelectionConfig(selection_count=2, tournament_size=2)
    valid = select_population_members(source, config=config, seed=37)

    with pytest.raises(ValueError, match="exactly config.selection_count"):
        PopulationSelectionResult(source, config, 37, valid.records[:1])
    reordered = (
        TournamentSelectionRecord(
            1,
            valid.records[0].contestants,
            valid.records[0].winner,
        ),
        TournamentSelectionRecord(
            0,
            valid.records[1].contestants,
            valid.records[1].winner,
        ),
    )
    with pytest.raises(ValueError, match="selection indices"):
        PopulationSelectionResult(source, config, 37, reordered)
