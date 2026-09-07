from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import BasisLayout
from toposc_lab.evaluation import (
    BasicScoreComponent,
    GeometryEvaluationRun,
    GeometryModelAdapter,
    ObjectiveDirection,
    ObjectiveQuantity,
    ObjectiveSpec,
    evaluate_geometry,
)
from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.search import (
    POPULATION_ELITISM_VERSION,
    EliteTier,
    ElitismConfig,
    GeometryGenome,
    InitialPopulationMember,
    MultiObjectiveFitnessDefinition,
    PopulationElitismResult,
    PopulationFitnessResult,
    ScalarFitnessDefinition,
    create_initial_population,
    evaluate_population_fitness,
    geometry_to_genome,
    identify_population_elites,
)


class _ElitismModel(BaseModel):
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


def _evaluate_member(member: InitialPopulationMember) -> GeometryEvaluationRun:
    return evaluate_geometry(
        member.genome.to_geometry(),
        adapter=GeometryModelAdapter(model_factory=_ElitismModel),
        seed=member.member_index,
        code_version="phase-10.13-test",
    )


def _scalar_result(
    *,
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE,
    failing_indices: frozenset[int] = frozenset(),
) -> PopulationFitnessResult:
    population = create_initial_population(
        (_genome(2), _genome(3), _genome(3), _genome(4))
    )

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
    population = create_initial_population(
        (
            _genome(4, edge_count=1),
            _genome(3, edge_count=2),
            _genome(3, edge_count=1),
            _genome(2, edge_count=1),
        )
    )
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
    return evaluate_population_fitness(
        population,
        definition=definition,
        evaluator=_evaluate_member,
    )


def _tier_indices(result: PopulationElitismResult) -> tuple[tuple[int, ...], ...]:
    return tuple(tier.member_indices for tier in result.tiers)


def test_scalar_elitism_retains_complete_cutoff_tier_in_fitness_order() -> None:
    source = _scalar_result()
    config = ElitismConfig(minimum_elite_count=2)

    result = identify_population_elites(source, config=config)

    assert result.source is source
    assert result.config is config
    assert result.version == POPULATION_ELITISM_VERSION
    assert _tier_indices(result) == ((3,), (1, 2))
    assert tuple(member.member_index for member in result.elite_members) == (3, 1, 2)
    assert result.elite_count == 3
    assert result.cutoff_was_expanded


def test_scalar_elitism_honors_minimize_direction() -> None:
    result = identify_population_elites(
        _scalar_result(direction=ObjectiveDirection.MINIMIZE),
        config=ElitismConfig(minimum_elite_count=2),
    )

    assert _tier_indices(result) == ((0,), (1, 2))


def test_exact_scalar_ties_are_not_split_at_cutoff() -> None:
    result = identify_population_elites(
        _scalar_result(),
        config=ElitismConfig(minimum_elite_count=2),
    )

    assert result.tiers[-1].member_indices == (1, 2)
    assert result.elite_count > result.config.minimum_elite_count


def test_exact_minimum_has_no_cutoff_expansion() -> None:
    result = identify_population_elites(
        _scalar_result(),
        config=ElitismConfig(minimum_elite_count=1),
    )

    assert _tier_indices(result) == ((3,),)
    assert result.elite_count == 1
    assert not result.cutoff_was_expanded


def test_multi_objective_elitism_retains_complete_pareto_fronts() -> None:
    source = _multi_objective_result()

    first_front = identify_population_elites(
        source,
        config=ElitismConfig(minimum_elite_count=1),
    )
    through_second_front = identify_population_elites(
        source,
        config=ElitismConfig(minimum_elite_count=3),
    )

    assert _tier_indices(first_front) == ((0, 1),)
    assert first_front.cutoff_was_expanded
    assert _tier_indices(through_second_front) == ((0, 1), (2,))
    assert not through_second_front.cutoff_was_expanded


def test_pareto_fronts_honor_each_objective_direction() -> None:
    result = identify_population_elites(
        _multi_objective_result(edge_direction=ObjectiveDirection.MINIMIZE),
        config=ElitismConfig(minimum_elite_count=1),
    )

    assert _tier_indices(result) == ((0,),)


def test_failed_members_remain_in_source_but_cannot_be_elites() -> None:
    source = _scalar_result(failing_indices=frozenset({3}))
    original_members = source.members

    result = identify_population_elites(
        source,
        config=ElitismConfig(minimum_elite_count=1),
    )

    assert source.members is original_members
    assert tuple(member.member_index for member in source.available_members) == (0, 1, 2)
    assert _tier_indices(result) == ((1, 2),)
    assert all(member is not source.members[3] for member in result.elite_members)


@pytest.mark.parametrize(
    ("minimum", "error_type"),
    ((0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)),
)
def test_config_rejects_invalid_minimum(
    minimum: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        ElitismConfig(minimum_elite_count=minimum)  # type: ignore[arg-type]


def test_elitism_rejects_an_empty_available_pool() -> None:
    source = _scalar_result(failing_indices=frozenset({0, 1, 2, 3}))

    with pytest.raises(ValueError, match="at least one available"):
        identify_population_elites(
            source,
            config=ElitismConfig(minimum_elite_count=1),
        )


def test_minimum_must_not_exceed_available_member_count() -> None:
    source = _scalar_result(failing_indices=frozenset({3}))

    with pytest.raises(ValueError, match="must not exceed"):
        identify_population_elites(
            source,
            config=ElitismConfig(minimum_elite_count=4),
        )


def test_elitism_api_rejects_wrong_source_or_config_type() -> None:
    source = _scalar_result()
    config = ElitismConfig(minimum_elite_count=1)

    with pytest.raises(TypeError, match="source must be"):
        identify_population_elites(object(), config=config)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="config must be"):
        identify_population_elites(source, config=object())  # type: ignore[arg-type]


def test_elite_tier_rejects_duplicate_unavailable_or_empty_members() -> None:
    source = _scalar_result(failing_indices=frozenset({2}))
    available = source.available_members
    unavailable = source.members[2]

    with pytest.raises(ValueError, match="at least one"):
        EliteTier(0, ())
    with pytest.raises(ValueError, match="duplicate members"):
        EliteTier(0, (available[0], available[0]))
    with pytest.raises(ValueError, match="available fitness"):
        EliteTier(0, (unavailable,))


def test_result_rejects_missing_reordered_or_tampered_tiers() -> None:
    source = _scalar_result()
    config = ElitismConfig(minimum_elite_count=2)
    valid = identify_population_elites(source, config=config)

    with pytest.raises(ValueError, match="every selected fitness tier"):
        PopulationElitismResult(source, config, valid.tiers[:1])

    wrong_index = (
        EliteTier(1, valid.tiers[0].members),
        valid.tiers[1],
    )
    with pytest.raises(ValueError, match="tier indices"):
        PopulationElitismResult(source, config, wrong_index)

    tampered = (
        valid.tiers[0],
        EliteTier(1, tuple(reversed(valid.tiers[1].members))),
    )
    with pytest.raises(ValueError, match="do not match"):
        PopulationElitismResult(source, config, tampered)


def test_result_contains_no_population_or_reproduction_output() -> None:
    result = identify_population_elites(
        _scalar_result(),
        config=ElitismConfig(minimum_elite_count=1),
    )

    assert not hasattr(result, "population")
    assert not hasattr(result, "offspring")
    assert not hasattr(result, "seed")
