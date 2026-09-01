"""Neutral, reproducible random sampling of explicit model-parameter sets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import TypeAlias

import numpy as np

from toposc_lab.robustness.disorder import (
    DisorderParameterValue,
    ModelParameterSet,
    exact_model_parameter_set_id,
)

RANDOM_MODEL_PARAMETER_SAMPLER_VERSION = 1
RANDOM_MODEL_PARAMETER_SAMPLER_RNG_ALGORITHM = "numpy.random.PCG64"

_MINIMUM_INT64 = -(2**63)
_MAXIMUM_INT64_EXCLUSIVE = 2**63


@dataclass(frozen=True, slots=True)
class UniformFloatDistribution:
    """Continuous uniform distribution over finite ``[low, high)`` bounds."""

    low: float
    high: float

    def __post_init__(self) -> None:
        low = _finite_real(self.low, name="low")
        high = _finite_real(self.high, name="high")
        if not low < high:
            raise ValueError("low must be smaller than high")
        if not math.isfinite(high - low):
            raise ValueError("the interval width must be finite")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)


@dataclass(frozen=True, slots=True)
class UniformIntegerDistribution:
    """Discrete uniform distribution over ``[low, high)`` integer bounds."""

    low: int
    high: int

    def __post_init__(self) -> None:
        low = _integer(self.low, name="low")
        high = _integer(self.high, name="high")
        if not low < high:
            raise ValueError("low must be smaller than high")
        if low < _MINIMUM_INT64 or high > _MAXIMUM_INT64_EXCLUSIVE:
            raise ValueError("integer bounds must fit the NumPy int64 sampling domain")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)


@dataclass(frozen=True, slots=True)
class CategoricalDistribution:
    """Uniform choice among explicit, exactly typed parameter values."""

    choices: tuple[DisorderParameterValue, ...]

    def __post_init__(self) -> None:
        if isinstance(self.choices, (str, bytes, bytearray)):
            raise TypeError("choices must be an iterable of parameter values")
        try:
            source_choices = tuple(self.choices)
        except TypeError as error:
            raise TypeError(
                "choices must be an iterable of parameter values"
            ) from error
        if not source_choices:
            raise ValueError("choices must contain at least one parameter value")
        choices = tuple(
            _freeze_parameter_value(choice, name=f"choices[{index}]")
            for index, choice in enumerate(source_choices)
        )
        choice_ids = tuple(
            exact_model_parameter_set_id({"choice": choice}) for choice in choices
        )
        if len(set(choice_ids)) != len(choice_ids):
            raise ValueError("choices must not contain exact duplicate values")
        object.__setattr__(self, "choices", choices)


ModelParameterDistribution: TypeAlias = (
    UniformFloatDistribution
    | UniformIntegerDistribution
    | CategoricalDistribution
)
_DISTRIBUTION_TYPES = (
    UniformFloatDistribution,
    UniformIntegerDistribution,
    CategoricalDistribution,
)


@dataclass(frozen=True, slots=True)
class RandomModelParameterSamplingConfig:
    """Fixed parameter template, explicit search dimensions, and sample count."""

    base_parameters: ModelParameterSet
    distributions: Mapping[str, ModelParameterDistribution]
    sample_count: int

    def __post_init__(self) -> None:
        base_parameters = _freeze_parameter_mapping(
            self.base_parameters,
            name="base_parameters",
        )
        distributions = _freeze_distributions(
            self.distributions,
            base_parameters=base_parameters,
        )
        sample_count = _positive_integer(self.sample_count, name="sample_count")
        object.__setattr__(self, "base_parameters", base_parameters)
        object.__setattr__(self, "distributions", distributions)
        object.__setattr__(self, "sample_count", sample_count)


class RandomModelParameterSamplingError(RuntimeError):
    """Context-rich infrastructure failure for one distribution draw."""

    def __init__(
        self,
        *,
        sample_index: int,
        parameter_name: str,
        distribution_name: str,
        error: Exception,
    ) -> None:
        if not isinstance(error, Exception):
            raise TypeError("error must be an Exception")
        self.sample_index = _nonnegative_integer(sample_index, name="sample_index")
        if not isinstance(parameter_name, str) or not parameter_name.isidentifier():
            raise ValueError("parameter_name must be a Python-style identifier")
        if not isinstance(distribution_name, str) or not distribution_name:
            raise ValueError("distribution_name must be a non-empty string")
        self.parameter_name = parameter_name
        self.distribution_name = distribution_name
        self.error_type = type(error).__name__
        self.detail = str(error).strip() or "The distribution raised without a message."
        super().__init__(
            f"random model-parameter sampling failed at sample {self.sample_index} "
            f"for parameter {parameter_name!r} using {distribution_name}: "
            f"{self.error_type}: {self.detail}"
        )


@dataclass(frozen=True, slots=True)
class RandomModelParameterSample:
    """One ordered, deeply immutable sampled model-parameter mapping."""

    sample_index: int
    parameters: ModelParameterSet

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_index",
            _nonnegative_integer(self.sample_index, name="sample_index"),
        )
        parameters = _freeze_parameter_mapping(self.parameters, name="parameters")
        object.__setattr__(self, "parameters", parameters)


@dataclass(frozen=True, slots=True)
class RandomModelParameterSamplingResult:
    """Complete ordered output of one versioned parameter-sampler execution."""

    config: RandomModelParameterSamplingConfig
    seed: int
    samples: tuple[RandomModelParameterSample, ...]
    rng_algorithm: str = field(
        default=RANDOM_MODEL_PARAMETER_SAMPLER_RNG_ALGORITHM,
        init=False,
    )
    sampler_version: int = field(
        default=RANDOM_MODEL_PARAMETER_SAMPLER_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.config, RandomModelParameterSamplingConfig):
            raise TypeError("config must be RandomModelParameterSamplingConfig")
        object.__setattr__(self, "seed", _nonnegative_integer(self.seed, name="seed"))
        if isinstance(self.samples, (str, bytes, bytearray)):
            raise TypeError("samples must be an iterable of parameter samples")
        try:
            samples = tuple(self.samples)
        except TypeError as error:
            raise TypeError(
                "samples must be an iterable of parameter samples"
            ) from error
        if not all(isinstance(sample, RandomModelParameterSample) for sample in samples):
            raise TypeError(
                "samples must contain only RandomModelParameterSample values"
            )
        if len(samples) != self.config.sample_count:
            raise ValueError("samples must contain exactly config.sample_count values")
        expected_keys = tuple(self.config.base_parameters)
        for sample_index, sample in enumerate(samples):
            if sample.sample_index != sample_index:
                raise ValueError("sample indices must match output order exactly")
            if tuple(sample.parameters) != expected_keys:
                raise ValueError("sample parameter keys must match base_parameters")
        object.__setattr__(self, "samples", samples)


def sample_random_model_parameters(
    config: RandomModelParameterSamplingConfig,
    *,
    seed: int,
) -> RandomModelParameterSamplingResult:
    """Draw parameter mappings in sample-index and parameter-name order.

    The function creates one local PCG64 stream. Each configured distribution
    consumes exactly one high-level NumPy draw per sample, in lexicographic
    parameter-name order. It performs no clipping, retry, model construction,
    model-specific validation, geometry coupling, or scientific evaluation.
    """
    if not isinstance(config, RandomModelParameterSamplingConfig):
        raise TypeError("config must be RandomModelParameterSamplingConfig")
    prepared_seed = _nonnegative_integer(seed, name="seed")
    random_number_generator = np.random.Generator(np.random.PCG64(prepared_seed))

    samples: list[RandomModelParameterSample] = []
    for sample_index in range(config.sample_count):
        parameters = dict(config.base_parameters)
        for parameter_name, distribution in config.distributions.items():
            try:
                parameters[parameter_name] = _sample_distribution(
                    distribution,
                    rng=random_number_generator,
                )
            except Exception as error:
                raise RandomModelParameterSamplingError(
                    sample_index=sample_index,
                    parameter_name=parameter_name,
                    distribution_name=type(distribution).__name__,
                    error=error,
                ) from error
        samples.append(
            RandomModelParameterSample(
                sample_index=sample_index,
                parameters=parameters,
            )
        )

    return RandomModelParameterSamplingResult(
        config=config,
        seed=prepared_seed,
        samples=tuple(samples),
    )


def _sample_distribution(
    distribution: ModelParameterDistribution,
    *,
    rng: np.random.Generator,
) -> DisorderParameterValue:
    if isinstance(distribution, UniformFloatDistribution):
        value = float(rng.uniform(distribution.low, distribution.high))
        if not math.isfinite(value):
            raise ValueError("uniform float sampling produced a non-finite value")
        return value
    if isinstance(distribution, UniformIntegerDistribution):
        return int(rng.integers(distribution.low, distribution.high, dtype=np.int64))
    choice_index = int(rng.integers(len(distribution.choices)))
    return distribution.choices[choice_index]


def _freeze_distributions(
    distributions: Mapping[str, ModelParameterDistribution],
    *,
    base_parameters: ModelParameterSet,
) -> Mapping[str, ModelParameterDistribution]:
    if not isinstance(distributions, Mapping):
        raise TypeError("distributions must be a mapping")
    if not distributions:
        raise ValueError("distributions must contain at least one parameter")
    if any(
        not isinstance(name, str) or not name.isidentifier()
        for name in distributions
    ):
        raise ValueError(
            "distribution keys must be non-empty Python-style parameter names"
        )
    unknown_names = sorted(set(distributions) - set(base_parameters))
    if unknown_names:
        raise ValueError(
            f"distributions refer to unknown model parameters: {unknown_names!r}"
        )
    prepared: dict[str, ModelParameterDistribution] = {}
    for name in sorted(distributions):
        distribution = distributions[name]
        if not isinstance(distribution, _DISTRIBUTION_TYPES):
            raise TypeError(
                f"distribution for {name!r} has unsupported type "
                f"{type(distribution).__name__}"
            )
        prepared[name] = distribution
    return MappingProxyType(prepared)


def _freeze_parameter_mapping(
    values: Mapping[str, DisorderParameterValue],
    *,
    name: str,
) -> ModelParameterSet:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) or not key.isidentifier() for key in values):
        raise ValueError(
            f"{name} keys must be non-empty Python-style parameter names"
        )
    prepared: dict[str, DisorderParameterValue] = {}
    for key in sorted(values):
        prepared[key] = _freeze_parameter_value(
            values[key],
            name=f"{name}[{key!r}]",
        )
    return MappingProxyType(prepared)


def _freeze_parameter_value(
    value: object,
    *,
    name: str,
) -> DisorderParameterValue:
    if value is None or isinstance(value, str) or isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result
    if isinstance(value, Mapping):
        return _freeze_parameter_mapping(value, name=name)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_parameter_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{name} has unsupported type {type(value).__name__}")


def _finite_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _positive_integer(value: int, *, name: str) -> int:
    result = _integer(value, name=name)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: int, *, name: str) -> int:
    result = _integer(value, name=name)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result
