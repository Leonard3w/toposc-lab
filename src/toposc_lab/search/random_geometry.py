"""Neutral, reproducible random sampling of registered geometries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from typing import Any

import numpy as np

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    Geometry,
    GeometryGenerationRequest,
    GeometryGenerator,
    GeometryGeneratorRegistry,
    GeometryValidationReport,
    validate_geometry,
)

RANDOM_GEOMETRY_SAMPLER_VERSION = 1
RANDOM_GEOMETRY_SAMPLER_RNG_ALGORITHM = "numpy.random.PCG64"


@dataclass(frozen=True, slots=True)
class GeometrySamplingRecipe:
    """One explicit registered generator and its fixed geometry parameters.

    A recipe never contains a seed. The sampler supplies a derived seed through
    ``GeometryGenerationRequest.seed`` exactly when the selected generator is
    stochastic.
    """

    generator_key: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.generator_key, str) or not self.generator_key:
            raise ValueError("generator_key must be a non-empty string")
        request = GeometryGenerationRequest(parameters=self.parameters)
        if "seed" in request.parameters:
            raise ValueError(
                "seed belongs to the random geometry sampler, not recipe parameters"
            )
        object.__setattr__(self, "parameters", request.parameters)


@dataclass(frozen=True, slots=True)
class RandomGeometrySamplingConfig:
    """Ordered finite recipe space and requested number of geometry samples."""

    recipes: tuple[GeometrySamplingRecipe, ...]
    sample_count: int

    def __post_init__(self) -> None:
        if isinstance(self.recipes, (str, bytes, bytearray)):
            raise TypeError("recipes must be an iterable of GeometrySamplingRecipe")
        try:
            recipes = tuple(self.recipes)
        except TypeError as error:
            raise TypeError(
                "recipes must be an iterable of GeometrySamplingRecipe"
            ) from error
        if not recipes:
            raise ValueError("recipes must contain at least one geometry recipe")
        if not all(isinstance(recipe, GeometrySamplingRecipe) for recipe in recipes):
            raise TypeError("recipes must contain only GeometrySamplingRecipe values")
        sample_count = _positive_integer(self.sample_count, name="sample_count")
        object.__setattr__(self, "recipes", recipes)
        object.__setattr__(self, "sample_count", sample_count)


class GeometrySamplingFailureStage(str, Enum):
    """Sampler-owned execution stage that failed before scientific evaluation."""

    GENERATION = "generation"
    GEOMETRY_VALIDATION = "geometry_validation"


class RandomGeometrySamplingError(RuntimeError):
    """Context-rich infrastructure failure for one selected sampling slot."""

    def __init__(
        self,
        *,
        stage: GeometrySamplingFailureStage,
        sample_index: int,
        recipe_index: int,
        generator_key: str,
        error: Exception,
    ) -> None:
        if not isinstance(stage, GeometrySamplingFailureStage):
            raise TypeError("stage must be GeometrySamplingFailureStage")
        if not isinstance(error, Exception):
            raise TypeError("error must be an Exception")
        self.stage = stage
        self.sample_index = _nonnegative_integer(sample_index, name="sample_index")
        self.recipe_index = _nonnegative_integer(recipe_index, name="recipe_index")
        self.generator_key = generator_key
        self.error_type = type(error).__name__
        self.detail = str(error).strip() or "The stage raised without a message."
        super().__init__(
            f"random geometry sampling failed during {stage.value} at sample "
            f"{self.sample_index} using recipe {self.recipe_index} "
            f"({generator_key!r}): {self.error_type}: {self.detail}"
        )


@dataclass(frozen=True, slots=True)
class RandomGeometrySample:
    """One ordered geometry draw with generation and base-validation evidence."""

    sample_index: int
    recipe_index: int
    generation_request: GeometryGenerationRequest
    geometry: Geometry
    validation: GeometryValidationReport

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_index",
            _nonnegative_integer(self.sample_index, name="sample_index"),
        )
        object.__setattr__(
            self,
            "recipe_index",
            _nonnegative_integer(self.recipe_index, name="recipe_index"),
        )
        if not isinstance(self.generation_request, GeometryGenerationRequest):
            raise TypeError("generation_request must be GeometryGenerationRequest")
        if not isinstance(self.geometry, Geometry):
            raise TypeError("geometry must be Geometry")
        if not isinstance(self.validation, GeometryValidationReport):
            raise TypeError("validation must be GeometryValidationReport")
        if self.validation.require_connected:
            raise ValueError("sampling validation must not impose connectivity")
        if not self.validation.is_valid:
            raise ValueError("a successful sample requires valid base geometry")


@dataclass(frozen=True, slots=True)
class RandomGeometrySamplingResult:
    """Complete ordered output of one versioned sampler execution."""

    config: RandomGeometrySamplingConfig
    seed: int
    samples: tuple[RandomGeometrySample, ...]
    rng_algorithm: str = field(
        default=RANDOM_GEOMETRY_SAMPLER_RNG_ALGORITHM,
        init=False,
    )
    sampler_version: int = field(
        default=RANDOM_GEOMETRY_SAMPLER_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.config, RandomGeometrySamplingConfig):
            raise TypeError("config must be RandomGeometrySamplingConfig")
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, name="seed"))
        if isinstance(self.samples, (str, bytes, bytearray)):
            raise TypeError("samples must be an iterable of RandomGeometrySample")
        try:
            samples = tuple(self.samples)
        except TypeError as error:
            raise TypeError(
                "samples must be an iterable of RandomGeometrySample"
            ) from error
        if not all(isinstance(sample, RandomGeometrySample) for sample in samples):
            raise TypeError("samples must contain only RandomGeometrySample values")
        if len(samples) != self.config.sample_count:
            raise ValueError("samples must contain exactly config.sample_count values")
        for sample_index, sample in enumerate(samples):
            if sample.sample_index != sample_index:
                raise ValueError("sample indices must match output order exactly")
            if sample.recipe_index >= len(self.config.recipes):
                raise ValueError("sample recipe_index is outside the configured recipes")
            recipe = self.config.recipes[sample.recipe_index]
            if sample.generation_request.parameters != recipe.parameters:
                raise ValueError("sample parameters must match the selected recipe")
        object.__setattr__(self, "samples", samples)


def sample_random_geometries(
    config: RandomGeometrySamplingConfig,
    *,
    seed: int,
    registry: GeometryGeneratorRegistry = BUILTIN_GEOMETRY_GENERATORS,
) -> RandomGeometrySamplingResult:
    """Sample geometries sequentially from uniformly selected recipes.

    Configuration and registry errors are rejected before the random stream is
    created. For every output slot, the sampler first selects one recipe with
    ``Generator.integers`` and then draws one raw PCG64 word. The raw word is
    supplied as the explicit generation seed only for stochastic generators;
    deterministic generators receive ``None``. Generation and model-independent
    geometry-validation failures stop at the first affected slot without retry,
    filtering, or a partial result.
    """
    if not isinstance(config, RandomGeometrySamplingConfig):
        raise TypeError("config must be RandomGeometrySamplingConfig")
    prepared_seed = _nonnegative_integer(seed, name="seed")
    if not isinstance(registry, GeometryGeneratorRegistry):
        raise TypeError("registry must be GeometryGeneratorRegistry")

    generators = tuple(
        registry.get(recipe.generator_key) for recipe in config.recipes
    )
    random_number_generator = np.random.Generator(np.random.PCG64(prepared_seed))
    samples: list[RandomGeometrySample] = []
    for sample_index in range(config.sample_count):
        recipe_index = int(random_number_generator.integers(len(config.recipes)))
        generated_seed = int(random_number_generator.bit_generator.random_raw())
        recipe = config.recipes[recipe_index]
        generator = generators[recipe_index]
        request = GeometryGenerationRequest(
            parameters=recipe.parameters,
            seed=generated_seed if generator.stochastic else None,
        )
        try:
            geometry = generator.generate(request)
            _validate_generation_provenance(
                geometry,
                generator=generator,
                request=request,
            )
        except Exception as error:
            raise RandomGeometrySamplingError(
                stage=GeometrySamplingFailureStage.GENERATION,
                sample_index=sample_index,
                recipe_index=recipe_index,
                generator_key=recipe.generator_key,
                error=error,
            ) from error

        try:
            validation = validate_geometry(geometry, require_connected=False)
            validation.raise_for_errors()
        except Exception as error:
            raise RandomGeometrySamplingError(
                stage=GeometrySamplingFailureStage.GEOMETRY_VALIDATION,
                sample_index=sample_index,
                recipe_index=recipe_index,
                generator_key=recipe.generator_key,
                error=error,
            ) from error
        samples.append(
            RandomGeometrySample(
                sample_index=sample_index,
                recipe_index=recipe_index,
                generation_request=request,
                geometry=geometry,
                validation=validation,
            )
        )

    return RandomGeometrySamplingResult(
        config=config,
        seed=prepared_seed,
        samples=tuple(samples),
    )


def _validate_generation_provenance(
    geometry: object,
    *,
    generator: GeometryGenerator,
    request: GeometryGenerationRequest,
) -> None:
    if not isinstance(geometry, Geometry):
        raise TypeError("registered geometry generators must return Geometry")
    expected = {
        "generator_key": generator.key,
        "generator_version": generator.version,
        "parameters": dict(request.parameters),
        "seed": request.seed,
    }
    if geometry.metadata.get("generation") != expected:
        raise ValueError(
            "registered geometry generator output has missing or inconsistent "
            "generation provenance"
        )


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result
