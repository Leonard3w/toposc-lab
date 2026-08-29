"""Common protocol and registry for reproducible geometry generators."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from numbers import Integral
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from toposc_lab.geometry.base import Geometry

_GENERATOR_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class GeometryGenerationRequest:
    """Validated parameters and random seed for one geometry generation.

    Parameters are normalized through JSON so requests can later be serialized
    and compared without relying on arbitrary Python objects. Stochastic generators
    receive their seed separately instead of hiding it in a parameter dictionary.
    """

    parameters: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None

    def __post_init__(self) -> None:
        try:
            parameters = json.loads(
                json.dumps(dict(self.parameters), allow_nan=False, sort_keys=True)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("generator parameters must be JSON compatible") from error

        if not isinstance(parameters, dict):
            raise TypeError("parameters must be a mapping")
        if not all(isinstance(name, str) and name.isidentifier() for name in parameters):
            raise ValueError(
                "generator parameter names must be non-empty Python-style identifiers"
            )

        seed = self.seed
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise TypeError("seed must be an integer or None")
            seed = int(seed)
            if seed < 0:
                raise ValueError("seed must be nonnegative")

        object.__setattr__(self, "parameters", _freeze_json_mapping(parameters))
        object.__setattr__(self, "seed", seed)


@runtime_checkable
class GeometryGenerator(Protocol):
    """Structural interface shared by all geometry generators."""

    @property
    def key(self) -> str:
        """Stable technical key used for discovery and persistence."""
        ...

    @property
    def version(self) -> int:
        """Positive version of the generator algorithm."""
        ...

    @property
    def stochastic(self) -> bool:
        """Whether generation requires an explicit random seed."""
        ...

    def generate(self, request: GeometryGenerationRequest) -> Geometry:
        """Generate one immutable geometry and attach its provenance."""
        ...


@dataclass(frozen=True, slots=True)
class FunctionGeometryGenerator:
    """Adapt a validated geometry-building function to ``GeometryGenerator``."""

    key: str
    builder: Callable[..., Geometry] = field(repr=False, compare=False)
    version: int = 1
    stochastic: bool = False
    seed_parameter: str = "seed"

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or _GENERATOR_KEY_PATTERN.fullmatch(self.key) is None:
            raise ValueError(
                "generator key must start with a lowercase letter and contain only "
                "lowercase letters, digits, and underscores"
            )
        if isinstance(self.version, bool) or not isinstance(self.version, Integral):
            raise TypeError("generator version must be an integer")
        version = int(self.version)
        if version < 1:
            raise ValueError("generator version must be at least one")
        if not isinstance(self.stochastic, bool):
            raise TypeError("stochastic must be a boolean")
        if not callable(self.builder):
            raise TypeError("builder must be callable")
        if not isinstance(self.seed_parameter, str) or not self.seed_parameter.isidentifier():
            raise ValueError("seed_parameter must be a Python-style identifier")

        object.__setattr__(self, "version", version)

    def generate(self, request: GeometryGenerationRequest) -> Geometry:
        """Build a geometry with deterministic, JSON-safe provenance metadata."""
        if not isinstance(request, GeometryGenerationRequest):
            raise TypeError("request must be a GeometryGenerationRequest")
        if self.seed_parameter in request.parameters:
            raise ValueError(
                f"{self.seed_parameter!r} belongs in GeometryGenerationRequest.seed"
            )
        if self.stochastic and request.seed is None:
            raise ValueError("stochastic generators require an explicit seed")
        if not self.stochastic and request.seed is not None:
            raise ValueError("deterministic generators do not accept a seed")

        keyword_arguments = dict(request.parameters)
        if self.stochastic:
            keyword_arguments[self.seed_parameter] = request.seed

        geometry = self.builder(**keyword_arguments)
        if not isinstance(geometry, Geometry):
            raise TypeError("geometry generator builders must return Geometry")

        metadata = dict(geometry.metadata)
        metadata["generation"] = {
            "generator_key": self.key,
            "generator_version": self.version,
            "parameters": dict(request.parameters),
            "seed": request.seed,
        }
        return replace(geometry, metadata=metadata)


class GeometryGeneratorRegistry:
    """Ordered registry of uniquely keyed geometry generators."""

    def __init__(self, generators: Iterable[GeometryGenerator] = ()) -> None:
        self._generators: dict[str, GeometryGenerator] = {}
        for generator in generators:
            self.register(generator)

    def register(self, generator: GeometryGenerator) -> None:
        """Register a generator while rejecting invalid or duplicate keys."""
        if not isinstance(generator, GeometryGenerator):
            raise TypeError("generator must implement GeometryGenerator")
        if (
            not isinstance(generator.key, str)
            or _GENERATOR_KEY_PATTERN.fullmatch(generator.key) is None
        ):
            raise ValueError("generator has an invalid key")
        if isinstance(generator.version, bool) or not isinstance(generator.version, Integral):
            raise TypeError("generator version must be an integer")
        if int(generator.version) < 1:
            raise ValueError("generator version must be at least one")
        if not isinstance(generator.stochastic, bool):
            raise TypeError("generator stochastic flag must be a boolean")
        if generator.key in self._generators:
            raise ValueError(f"a geometry generator is already registered as {generator.key!r}")
        self._generators[generator.key] = generator

    def get(self, key: str) -> GeometryGenerator:
        """Return a generator through its stable technical key."""
        try:
            return self._generators[key]
        except KeyError as error:
            raise ValueError(f"unknown geometry generator key: {key!r}") from error

    def generators(self) -> tuple[GeometryGenerator, ...]:
        """Return all registered generators in registration order."""
        return tuple(self._generators.values())

    def generate(
        self,
        key: str,
        *,
        parameters: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> Geometry:
        """Generate a geometry by key using the common request contract."""
        request = GeometryGenerationRequest(
            parameters={} if parameters is None else parameters,
            seed=seed,
        )
        return self.get(key).generate(request)


def _freeze_json_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {name: _freeze_json_value(value) for name, value in values.items()}
    )


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value
