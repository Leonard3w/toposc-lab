from __future__ import annotations

from typing import Any

import pytest

from toposc_lab.geometry import (
    FunctionGeometryGenerator,
    Geometry,
    GeometryEdge,
    GeometryGeneratorRegistry,
    geometry_to_bytes,
)
from toposc_lab.search import (
    RANDOM_GEOMETRY_SAMPLER_RNG_ALGORITHM,
    RANDOM_GEOMETRY_SAMPLER_VERSION,
    GeometrySamplingFailureStage,
    GeometrySamplingRecipe,
    RandomGeometrySamplingConfig,
    RandomGeometrySamplingError,
    sample_random_geometries,
)


def _mixed_config(*, sample_count: int = 8) -> RandomGeometrySamplingConfig:
    return RandomGeometrySamplingConfig(
        recipes=(
            GeometrySamplingRecipe(
                "square",
                {"n_x": 2, "n_y": 2, "spacing": 0.5},
            ),
            GeometrySamplingRecipe(
                "random_graph",
                {"n_sites": 6, "edge_probability": 0.35},
            ),
            GeometrySamplingRecipe(
                "scale_free_graph",
                {"n_sites": 8, "attachments_per_site": 2},
            ),
        ),
        sample_count=sample_count,
    )


def test_fixed_pcg64_sampling_order_and_derived_seeds() -> None:
    result = sample_random_geometries(_mixed_config(sample_count=6), seed=20260901)

    assert tuple(sample.recipe_index for sample in result.samples) == (2, 1, 2, 0, 1, 0)
    assert tuple(sample.generation_request.seed for sample in result.samples) == (
        10743510066261649426,
        6534260988641411628,
        2496215248607298552,
        None,
        7797986099714834301,
        None,
    )


def test_same_configuration_and_seed_reproduce_exact_ordered_geometries() -> None:
    config = _mixed_config()

    first = sample_random_geometries(config, seed=123456)
    second = sample_random_geometries(config, seed=123456)

    assert first.seed == second.seed == 123456
    assert first.rng_algorithm == second.rng_algorithm
    assert first.sampler_version == second.sampler_version
    assert tuple(sample.recipe_index for sample in first.samples) == tuple(
        sample.recipe_index for sample in second.samples
    )
    assert tuple(sample.generation_request for sample in first.samples) == tuple(
        sample.generation_request for sample in second.samples
    )
    assert tuple(geometry_to_bytes(sample.geometry) for sample in first.samples) == tuple(
        geometry_to_bytes(sample.geometry) for sample in second.samples
    )


def test_result_records_version_rng_and_strict_sample_order() -> None:
    result = sample_random_geometries(_mixed_config(sample_count=4), seed=9)

    assert result.rng_algorithm == RANDOM_GEOMETRY_SAMPLER_RNG_ALGORITHM
    assert result.sampler_version == RANDOM_GEOMETRY_SAMPLER_VERSION
    assert tuple(sample.sample_index for sample in result.samples) == (0, 1, 2, 3)


def test_stochastic_generators_receive_only_derived_explicit_seeds() -> None:
    config = RandomGeometrySamplingConfig(
        recipes=(
            GeometrySamplingRecipe(
                "random_graph",
                {"n_sites": 8, "edge_probability": 0.25},
            ),
        ),
        sample_count=4,
    )

    result = sample_random_geometries(config, seed=42)

    seeds = tuple(sample.generation_request.seed for sample in result.samples)
    assert all(seed is not None for seed in seeds)
    assert len(set(seeds)) == len(seeds)
    for sample in result.samples:
        assert sample.geometry.metadata["generation"]["seed"] == (
            sample.generation_request.seed
        )


def test_deterministic_generators_never_receive_a_seed() -> None:
    config = RandomGeometrySamplingConfig(
        recipes=(GeometrySamplingRecipe("chain", {"n_sites": 4}),),
        sample_count=3,
    )

    result = sample_random_geometries(config, seed=42)

    assert all(sample.generation_request.seed is None for sample in result.samples)
    assert all(
        sample.geometry.metadata["generation"]["seed"] is None
        for sample in result.samples
    )
    assert len({geometry_to_bytes(sample.geometry) for sample in result.samples}) == 1


def test_disconnected_geometry_is_retained_with_base_validation_warning() -> None:
    config = RandomGeometrySamplingConfig(
        recipes=(
            GeometrySamplingRecipe(
                "random_graph",
                {"n_sites": 5, "edge_probability": 0.0},
            ),
        ),
        sample_count=1,
    )

    sample = sample_random_geometries(config, seed=7).samples[0]

    assert sample.validation.is_valid
    assert not sample.validation.require_connected
    assert tuple(issue.code for issue in sample.validation.warnings) == (
        "disconnected_components",
    )


def test_sampler_preserves_oriented_edges_without_inventing_geometry_data() -> None:
    def oriented_abstract_graph() -> Geometry:
        return Geometry(
            n_sites=2,
            edges=(GeometryEdge(1, 0, edge_type="oriented_reference"),),
        )

    registry = GeometryGeneratorRegistry(
        (
            FunctionGeometryGenerator(
                key="oriented_abstract_graph",
                builder=oriented_abstract_graph,
            ),
        )
    )
    config = RandomGeometrySamplingConfig(
        recipes=(GeometrySamplingRecipe("oriented_abstract_graph"),),
        sample_count=1,
    )

    geometry = sample_random_geometries(config, seed=3, registry=registry).samples[
        0
    ].geometry

    assert tuple((edge.source, edge.target) for edge in geometry.edges) == ((1, 0),)
    assert geometry.coordinates is None
    assert geometry.embedding_dimension is None
    assert geometry.boundary_sites == frozenset()
    assert geometry.boundary_components == ()


def test_recipe_parameters_are_normalized_immutable_and_seed_is_reserved() -> None:
    source: dict[str, Any] = {"options": {"boundary": "open"}}
    recipe = GeometrySamplingRecipe("custom", source)
    source["options"] = {"boundary": "periodic"}

    assert recipe.parameters["options"] == {"boundary": "open"}
    with pytest.raises(TypeError):
        recipe.parameters["new"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="seed belongs to the random geometry sampler"):
        GeometrySamplingRecipe("random_graph", {"seed": 4})


@pytest.mark.parametrize("sample_count", (0, -1, -10))
def test_config_rejects_nonpositive_sample_count(sample_count: int) -> None:
    with pytest.raises(ValueError, match="sample_count must be positive"):
        RandomGeometrySamplingConfig(
            recipes=(GeometrySamplingRecipe("chain", {"n_sites": 2}),),
            sample_count=sample_count,
        )


@pytest.mark.parametrize("sample_count", (True, 1.5, "2"))
def test_config_rejects_noninteger_sample_count(sample_count: object) -> None:
    with pytest.raises(TypeError, match="sample_count must be an integer"):
        RandomGeometrySamplingConfig(
            recipes=(GeometrySamplingRecipe("chain", {"n_sites": 2}),),
            sample_count=sample_count,  # type: ignore[arg-type]
        )


def test_config_rejects_empty_or_invalid_recipe_collections() -> None:
    with pytest.raises(ValueError, match="at least one"):
        RandomGeometrySamplingConfig(recipes=(), sample_count=1)
    with pytest.raises(TypeError, match="GeometrySamplingRecipe"):
        RandomGeometrySamplingConfig(
            recipes=("chain",),  # type: ignore[arg-type]
            sample_count=1,
        )


@pytest.mark.parametrize("seed", (-1, -10))
def test_sampler_rejects_negative_root_seed(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be nonnegative"):
        sample_random_geometries(_mixed_config(sample_count=1), seed=seed)


@pytest.mark.parametrize("seed", (True, 1.5, "1"))
def test_sampler_rejects_noninteger_root_seed(seed: object) -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        sample_random_geometries(
            _mixed_config(sample_count=1),
            seed=seed,  # type: ignore[arg-type]
        )


def test_all_registry_keys_are_preflighted_before_any_generation() -> None:
    calls: list[str] = []

    def tracked_geometry() -> Geometry:
        calls.append("called")
        return Geometry(n_sites=1)

    registry = GeometryGeneratorRegistry(
        (FunctionGeometryGenerator(key="tracked", builder=tracked_geometry),)
    )
    config = RandomGeometrySamplingConfig(
        recipes=(
            GeometrySamplingRecipe("tracked"),
            GeometrySamplingRecipe("missing"),
        ),
        sample_count=5,
    )

    with pytest.raises(ValueError, match="unknown geometry generator key: 'missing'"):
        sample_random_geometries(config, seed=1, registry=registry)

    assert calls == []


def test_generation_failure_stops_without_retry_and_reports_context() -> None:
    calls = 0

    def failing_generator() -> Geometry:
        nonlocal calls
        calls += 1
        raise ValueError("invalid recipe value")

    registry = GeometryGeneratorRegistry(
        (FunctionGeometryGenerator(key="failing", builder=failing_generator),)
    )
    config = RandomGeometrySamplingConfig(
        recipes=(GeometrySamplingRecipe("failing"),),
        sample_count=4,
    )

    with pytest.raises(RandomGeometrySamplingError) as captured:
        sample_random_geometries(config, seed=8, registry=registry)

    error = captured.value
    assert calls == 1
    assert error.stage is GeometrySamplingFailureStage.GENERATION
    assert error.sample_index == 0
    assert error.recipe_index == 0
    assert error.generator_key == "failing"
    assert error.error_type == "ValueError"
    assert error.detail == "invalid recipe value"
    assert isinstance(error.__cause__, ValueError)


def test_invalid_generator_provenance_is_an_infrastructure_failure() -> None:
    class MissingProvenanceGenerator:
        key = "missing_provenance"
        version = 1
        stochastic = False

        def generate(self, request: object) -> Geometry:
            return Geometry(n_sites=1)

    registry = GeometryGeneratorRegistry((MissingProvenanceGenerator(),))
    config = RandomGeometrySamplingConfig(
        recipes=(GeometrySamplingRecipe("missing_provenance"),),
        sample_count=1,
    )

    with pytest.raises(RandomGeometrySamplingError) as captured:
        sample_random_geometries(config, seed=0, registry=registry)

    assert captured.value.stage is GeometrySamplingFailureStage.GENERATION
    assert captured.value.error_type == "ValueError"
    assert "generation provenance" in captured.value.detail


def test_base_geometry_validation_failure_has_separate_stage() -> None:
    def invalid_metadata_geometry() -> Geometry:
        return Geometry(n_sites=1, metadata={"unsupported": object()})

    registry = GeometryGeneratorRegistry(
        (
            FunctionGeometryGenerator(
                key="invalid_metadata_geometry",
                builder=invalid_metadata_geometry,
            ),
        )
    )
    config = RandomGeometrySamplingConfig(
        recipes=(GeometrySamplingRecipe("invalid_metadata_geometry"),),
        sample_count=2,
    )

    with pytest.raises(RandomGeometrySamplingError) as captured:
        sample_random_geometries(config, seed=5, registry=registry)

    error = captured.value
    assert error.stage is GeometrySamplingFailureStage.GEOMETRY_VALIDATION
    assert error.sample_index == 0
    assert error.error_type == "GeometryValidationError"
    assert "invalid_metadata_type" in error.detail
