from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    CHAIN_GENERATOR,
    FunctionGeometryGenerator,
    Geometry,
    GeometryGenerationRequest,
    GeometryGenerator,
    GeometryGeneratorRegistry,
    chain,
    square,
)


def test_builtin_generators_implement_common_protocol_in_stable_order() -> None:
    generators = BUILTIN_GEOMETRY_GENERATORS.generators()

    assert tuple(generator.key for generator in generators) == (
        "body_centered_cubic",
        "cayley_tree",
        "chain",
        "cubic",
        "fibonacci_chain",
        "honeycomb",
        "irregular_cluster",
        "kagome",
        "menger_sponge",
        "random_graph",
        "random_regular_graph",
        "ring",
        "scale_free_graph",
        "sierpinski_carpet",
        "sierpinski_gasket",
        "small_world_network",
        "square",
        "tree",
        "triangular",
    )
    assert all(isinstance(generator, GeometryGenerator) for generator in generators)


def test_registry_generates_geometry_with_complete_provenance() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "square",
        parameters={"n_x": 2, "n_y": 3, "spacing": 0.5},
    )
    direct_geometry = square(2, 3, spacing=0.5)

    assert isinstance(geometry, Geometry)
    assert geometry.coordinates is not None
    assert direct_geometry.coordinates is not None
    assert geometry.edges == direct_geometry.edges
    assert np.array_equal(geometry.coordinates, direct_geometry.coordinates)
    assert geometry.boundary_sites == direct_geometry.boundary_sites
    assert geometry.metadata["generator"] == "square"
    assert geometry.metadata["generation"] == {
        "generator_key": "square",
        "generator_version": 1,
        "parameters": {"n_x": 2, "n_y": 3, "spacing": 0.5},
        "seed": None,
    }


def test_existing_generator_functions_remain_backward_compatible() -> None:
    geometry = chain(3, spacing=0.25)

    assert "generation" not in geometry.metadata
    assert geometry.metadata == {
        "generator": "chain",
        "boundary_condition": "open",
        "spacing": 0.25,
    }


def test_generation_request_is_defensive_and_deeply_immutable() -> None:
    source: dict[str, Any] = {
        "labels": ["A", "B"],
        "options": {"boundary": "open"},
    }
    request = GeometryGenerationRequest(parameters=source)

    source["labels"].append("C")
    cast(dict[str, str], source["options"])["boundary"] = "periodic"

    assert request.parameters["labels"] == ("A", "B")
    options = cast(Mapping[str, Any], request.parameters["options"])
    assert options["boundary"] == "open"
    with pytest.raises(TypeError):
        options["boundary"] = "periodic"  # type: ignore[index]


@pytest.mark.parametrize(
    "parameters",
    (
        {"invalid-name": 1},
        {"value": np.nan},
        {"value": object()},
    ),
)
def test_generation_request_rejects_nonportable_parameters(
    parameters: Mapping[str, Any],
) -> None:
    with pytest.raises(ValueError):
        GeometryGenerationRequest(parameters=parameters)


@pytest.mark.parametrize("seed", (True, 1.5, "4"))
def test_generation_request_rejects_non_integer_seed(seed: object) -> None:
    with pytest.raises(TypeError, match="integer or None"):
        GeometryGenerationRequest(seed=seed)  # type: ignore[arg-type]


def test_generation_request_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        GeometryGenerationRequest(seed=-1)


def test_deterministic_generator_rejects_seed() -> None:
    with pytest.raises(ValueError, match="do not accept a seed"):
        CHAIN_GENERATOR.generate(
            GeometryGenerationRequest(parameters={"n_sites": 3}, seed=7)
        )


def test_seed_is_reserved_for_the_common_request_contract() -> None:
    with pytest.raises(ValueError, match="belongs in GeometryGenerationRequest.seed"):
        CHAIN_GENERATOR.generate(
            GeometryGenerationRequest(parameters={"n_sites": 3, "seed": 7})
        )


def test_stochastic_generator_requires_seed_and_is_reproducible() -> None:
    def random_points(*, n_sites: int, seed: int) -> Geometry:
        coordinates = np.random.default_rng(seed).random((n_sites, 2))
        return Geometry(n_sites=n_sites, coordinates=coordinates)

    generator = FunctionGeometryGenerator(
        key="random_points",
        builder=random_points,
        stochastic=True,
    )

    with pytest.raises(ValueError, match="require an explicit seed"):
        generator.generate(GeometryGenerationRequest(parameters={"n_sites": 4}))

    request = GeometryGenerationRequest(parameters={"n_sites": 4}, seed=42)
    first = generator.generate(request)
    second = generator.generate(request)

    assert first.coordinates is not None
    assert second.coordinates is not None
    assert np.array_equal(first.coordinates, second.coordinates)
    assert first.metadata["generation"]["seed"] == 42


def test_function_generator_validates_builder_result() -> None:
    generator = FunctionGeometryGenerator(
        key="invalid_result",
        builder=lambda: "not a geometry",  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(TypeError, match="must return Geometry"):
        generator.generate(GeometryGenerationRequest())


def test_registry_rejects_duplicates_and_reports_unknown_keys() -> None:
    registry = GeometryGeneratorRegistry((CHAIN_GENERATOR,))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(CHAIN_GENERATOR)
    with pytest.raises(ValueError, match="unknown geometry generator"):
        registry.get("missing")


@pytest.mark.parametrize("key", ("", "Square", "square-lattice", "2d_square"))
def test_function_generator_rejects_unstable_keys(key: str) -> None:
    with pytest.raises(ValueError, match="generator key"):
        FunctionGeometryGenerator(key=key, builder=lambda: Geometry(n_sites=1))
