from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from toposc_lab.geometry import Geometry, GeometryEdge, geometry_to_bytes
from toposc_lab.search import (
    INITIAL_POPULATION_VERSION,
    GeometryGenome,
    GeometrySamplingRecipe,
    InitialPopulation,
    InitialPopulationError,
    InitialPopulationMember,
    MutationValidityPolicy,
    MutationValidityReport,
    RandomGeometrySamplingConfig,
    create_initial_population,
    geometry_to_genome,
    sample_random_geometries,
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


def _path_genome() -> GeometryGenome:
    return geometry_to_genome(
        Geometry(
            n_sites=4,
            edges=(GeometryEdge(0, 1), GeometryEdge(1, 2), GeometryEdge(2, 3)),
            coordinates=np.asarray(
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
            ),
            boundary_sites=frozenset(range(4)),
        )
    )


def test_initial_population_preserves_exact_order_and_genome_objects() -> None:
    first = _square_genome()
    second = _path_genome()

    population = create_initial_population((first, second))

    assert population.population_size == 2
    assert population.genomes == (first, second)
    assert population.members[0].genome is first
    assert population.members[1].genome is second
    assert tuple(member.member_index for member in population.members) == (0, 1)
    assert all(member.validity.is_valid for member in population.members)


def test_population_records_generation_zero_version_policy_and_dimension() -> None:
    policy = MutationValidityPolicy(require_connected=True, maximum_degree=2)

    population = create_initial_population((_square_genome(),), validity_policy=policy)

    assert population.generation_index == 0
    assert population.population_version == INITIAL_POPULATION_VERSION
    assert population.validity_policy is policy
    assert population.embedding_dimension == 2


def test_default_population_allows_abstract_disconnected_genomes() -> None:
    first = geometry_to_genome(Geometry(n_sites=2))
    second = geometry_to_genome(Geometry(n_sites=3, edges=(GeometryEdge(0, 1),)))

    population = create_initial_population((first, second))

    assert population.embedding_dimension is None
    assert population.population_size == 2
    assert all(member.validity.is_valid for member in population.members)


def test_population_is_independent_of_the_mutable_input_container() -> None:
    first = _square_genome()
    supplied = [first]

    population = create_initial_population(supplied)
    supplied.append(_path_genome())

    assert population.genomes == (first,)


def test_duplicate_genomes_are_retained_in_order() -> None:
    genome = _square_genome()

    population = create_initial_population((genome, genome, genome))

    assert population.population_size == 3
    assert all(member.genome is genome for member in population.members)


def test_population_rejects_mixed_embedding_dimensions() -> None:
    two_dimensional = _square_genome()
    one_dimensional = geometry_to_genome(
        Geometry(
            n_sites=4,
            edges=two_dimensional.edges,
            coordinates=np.arange(4, dtype=float).reshape(4, 1),
            boundary_sites=two_dimensional.boundary_sites,
        )
    )

    with pytest.raises(InitialPopulationError, match="candidate 1") as caught:
        create_initial_population((two_dimensional, one_dimensional))

    assert caught.value.candidate_index == 1
    assert tuple(issue.code for issue in caught.value.report.issues) == (
        "embedding_dimension_changed",
    )


def test_first_policy_invalid_candidate_stops_without_partial_population() -> None:
    policy = MutationValidityPolicy(minimum_site_count=5)

    with pytest.raises(InitialPopulationError, match="candidate 0") as caught:
        create_initial_population((_square_genome(), _path_genome()), validity_policy=policy)

    assert caught.value.candidate_index == 0
    assert tuple(issue.code for issue in caught.value.report.issues) == (
        "site_count_below_minimum",
    )


def test_later_policy_invalid_candidate_retains_its_original_index() -> None:
    policy = MutationValidityPolicy(minimum_edge_count=4)

    with pytest.raises(InitialPopulationError, match="candidate 1") as caught:
        create_initial_population((_square_genome(), _path_genome()), validity_policy=policy)

    assert caught.value.candidate_index == 1
    assert tuple(issue.code for issue in caught.value.report.issues) == (
        "edge_count_below_minimum",
    )


def test_representation_invalid_candidate_is_reported_at_its_input_index() -> None:
    valid = _square_genome()
    invalid = replace(valid, n_sites=2)

    with pytest.raises(InitialPopulationError, match="candidate 1") as caught:
        create_initial_population((valid, invalid))

    assert tuple(issue.code for issue in caught.value.report.issues) == (
        "candidate_invalid_geometry_representation",
    )


def test_existing_random_sampler_owns_seed_and_order_provenance() -> None:
    config = RandomGeometrySamplingConfig(
        recipes=(
            GeometrySamplingRecipe(
                "random_graph",
                {"n_sites": 5, "edge_probability": 0.5},
            ),
        ),
        sample_count=3,
    )
    first_sampling = sample_random_geometries(config, seed=20260907)
    second_sampling = sample_random_geometries(config, seed=20260907)

    first = create_initial_population(
        tuple(geometry_to_genome(sample.geometry) for sample in first_sampling.samples)
    )
    second = create_initial_population(
        tuple(geometry_to_genome(sample.geometry) for sample in second_sampling.samples)
    )

    assert tuple(geometry_to_bytes(genome.to_geometry()) for genome in first.genomes) == tuple(
        geometry_to_bytes(genome.to_geometry()) for genome in second.genomes
    )
    assert tuple(
        genome.metadata["generation"]["seed"] for genome in first.genomes
    ) == tuple(sample.generation_request.seed for sample in first_sampling.samples)


def test_member_requires_a_valid_report_and_domain_genome() -> None:
    genome = _square_genome()
    valid_report = validate_geometry_mutation(genome, genome)
    invalid_report = validate_geometry_mutation(
        genome,
        _path_genome(),
        policy=MutationValidityPolicy(minimum_edge_count=4),
    )

    with pytest.raises(TypeError, match="GeometryGenome"):
        InitialPopulationMember(0, object(), valid_report)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="MutationValidityReport"):
        InitialPopulationMember(0, genome, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="valid report"):
        InitialPopulationMember(0, genome, invalid_report)


def test_population_constructor_cross_checks_order_and_dimension() -> None:
    genome = _square_genome()
    report = validate_geometry_mutation(genome, genome)
    member = InitialPopulationMember(1, genome, report)

    with pytest.raises(ValueError, match="indices"):
        InitialPopulation((member,), MutationValidityPolicy(), 2)

    member = InitialPopulationMember(0, genome, report)
    with pytest.raises(ValueError, match="embedding_dimension"):
        InitialPopulation((member,), MutationValidityPolicy(), 1)


def test_population_constructor_rejects_a_report_from_another_policy() -> None:
    genome = _square_genome()
    default_report = validate_geometry_mutation(genome, genome)
    member = InitialPopulationMember(0, genome, default_report)

    with pytest.raises(ValueError, match="validity reports"):
        InitialPopulation(
            (member,),
            MutationValidityPolicy(maximum_edge_length=2.0),
            2,
        )


def test_population_and_error_constructors_require_consistent_values() -> None:
    genome = _square_genome()
    valid_report = MutationValidityReport()

    with pytest.raises(ValueError, match="at least one"):
        InitialPopulation((), MutationValidityPolicy(), 2)
    with pytest.raises(TypeError, match="validity_policy"):
        InitialPopulation(
            (InitialPopulationMember(0, genome, valid_report),),
            object(),  # type: ignore[arg-type]
            2,
        )
    with pytest.raises(ValueError, match="invalid report"):
        InitialPopulationError(candidate_index=0, report=valid_report)


def test_factory_rejects_empty_malformed_inputs_and_policy() -> None:
    genome = _square_genome()

    with pytest.raises(ValueError, match="at least one"):
        create_initial_population(())
    with pytest.raises(TypeError, match="iterable"):
        create_initial_population(3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="only GeometryGenome"):
        create_initial_population((genome, object()))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="validity_policy"):
        create_initial_population((genome,), validity_policy=object())  # type: ignore[arg-type]
