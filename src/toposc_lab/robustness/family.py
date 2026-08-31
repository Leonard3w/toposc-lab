"""Explicit cross-size geometry-family provenance for robustness scaling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
import re
from types import MappingProxyType

from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.geometry.base import Geometry
from toposc_lab.geometry.generators.protocol import GeometryGenerationRequest
from toposc_lab.robustness.finite_size import (
    FiniteSizeRobustnessPoint,
    FiniteSizeScalingResult,
)

GEOMETRY_FAMILY_CONTRACT_VERSION = 1

_TECHNICAL_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_GENERATION_METADATA_KEYS = frozenset(
    {"generator_key", "generator_version", "parameters", "seed"}
)


class GeometryFamilySeedPolicy(str, Enum):
    """Declared relationship between geometry-generation seeds across sizes."""

    NO_SEED = "no_seed"
    COMMON_SEED = "common_seed"
    DISTINCT_PER_SIZE = "distinct_per_size"


@dataclass(frozen=True, slots=True)
class GeometryFamilySpec:
    """Explicit construction contract for one cross-size geometry family."""

    family_key: str
    family_version: int
    description: str
    size_key: str
    generator_key: str
    generator_version: int
    varying_parameter_keys: tuple[str, ...]
    seed_policy: GeometryFamilySeedPolicy

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family_key",
            _technical_key(self.family_key, name="family_key"),
        )
        object.__setattr__(
            self,
            "family_version",
            _positive_integer(self.family_version, name="family_version"),
        )
        object.__setattr__(
            self,
            "description",
            _description(self.description, name="description"),
        )
        object.__setattr__(
            self,
            "size_key",
            _identifier(self.size_key, name="size_key"),
        )
        object.__setattr__(
            self,
            "generator_key",
            _technical_key(self.generator_key, name="generator_key"),
        )
        object.__setattr__(
            self,
            "generator_version",
            _positive_integer(self.generator_version, name="generator_version"),
        )
        if isinstance(self.varying_parameter_keys, (str, bytes, bytearray)):
            raise TypeError("varying_parameter_keys must be an iterable of strings")
        keys = tuple(self.varying_parameter_keys)
        if not keys:
            raise ValueError("varying_parameter_keys must not be empty")
        if any(not isinstance(key, str) or not key.isidentifier() for key in keys):
            raise ValueError(
                "varying_parameter_keys must contain Python-style identifiers"
            )
        if len(set(keys)) != len(keys):
            raise ValueError("varying_parameter_keys must not contain duplicates")
        if not isinstance(self.seed_policy, GeometryFamilySeedPolicy):
            raise TypeError("seed_policy must be GeometryFamilySeedPolicy")
        object.__setattr__(self, "varying_parameter_keys", keys)


@dataclass(frozen=True, slots=True)
class GeometryFamilyMember:
    """One exact geometry snapshot bound to one finite-size scaling point."""

    geometry: Geometry = field(repr=False, compare=False)
    scaling_point: FiniteSizeRobustnessPoint
    geometry_id: str = field(init=False)
    generator_key: str = field(init=False)
    generator_version: int = field(init=False)
    generation_request: GeometryGenerationRequest = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, Geometry):
            raise TypeError("geometry must be Geometry")
        if not isinstance(self.scaling_point, FiniteSizeRobustnessPoint):
            raise TypeError("scaling_point must be FiniteSizeRobustnessPoint")
        generator_key, generator_version, request = _generation_record(self.geometry)
        object.__setattr__(self, "geometry_id", exact_geometry_id(self.geometry))
        object.__setattr__(self, "generator_key", generator_key)
        object.__setattr__(self, "generator_version", generator_version)
        object.__setattr__(self, "generation_request", request)


@dataclass(frozen=True, slots=True)
class CrossSizeGeometryFamily:
    """Validated construction provenance bound to one finite-size scaling fit."""

    spec: GeometryFamilySpec
    scaling: FiniteSizeScalingResult
    members: tuple[GeometryFamilyMember, ...]
    contract_version: int = field(
        default=GEOMETRY_FAMILY_CONTRACT_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.spec, GeometryFamilySpec):
            raise TypeError("spec must be GeometryFamilySpec")
        if not isinstance(self.scaling, FiniteSizeScalingResult):
            raise TypeError("scaling must be FiniteSizeScalingResult")
        members = tuple(self.members)
        if any(not isinstance(member, GeometryFamilyMember) for member in members):
            raise TypeError("members must contain only GeometryFamilyMember values")
        if len(members) != len(self.scaling.points):
            raise ValueError("members must contain one geometry per scaling point")
        if self.spec.size_key != self.scaling.spec.size_key:
            raise ValueError("family and scaling size keys must match")

        for member, scaling_point in zip(
            members,
            self.scaling.points,
            strict=True,
        ):
            if member.scaling_point != scaling_point:
                raise ValueError("members must match scaling points in exact order")
            if (
                member.generator_key != self.spec.generator_key
                or member.generator_version != self.spec.generator_version
            ):
                raise ValueError(
                    "all members must match the declared generator key and version"
                )

        geometry_ids = tuple(member.geometry_id for member in members)
        if len(set(geometry_ids)) != len(geometry_ids):
            raise ValueError("family members must have distinct exact geometry snapshots")
        _validate_generation_parameters(members, spec=self.spec)
        _validate_generation_seeds(members, policy=self.spec.seed_policy)
        object.__setattr__(self, "members", members)

    @property
    def system_sizes(self) -> tuple[float, ...]:
        """Scaling sizes in the exact family-member order."""
        return tuple(member.scaling_point.system_size for member in self.members)

    @property
    def geometry_ids(self) -> tuple[str, ...]:
        """Exact representation-sensitive snapshot IDs for family members."""
        return tuple(member.geometry_id for member in self.members)

    @property
    def site_counts(self) -> tuple[int, ...]:
        """Raw site counts for auditing, without treating them as the size variable."""
        return tuple(member.geometry.n_sites for member in self.members)

    @property
    def generation_requests(self) -> tuple[GeometryGenerationRequest, ...]:
        """Phase-6 generation requests retained in size order."""
        return tuple(member.generation_request for member in self.members)

    @property
    def generation_seeds(self) -> tuple[int | None, ...]:
        """Geometry-generation seeds, distinct from disorder ensemble seeds."""
        return tuple(request.seed for request in self.generation_requests)

    @property
    def fixed_generation_parameters(self) -> Mapping[str, object]:
        """Generator parameters held fixed across the declared family."""
        parameters = self.generation_requests[0].parameters
        return MappingProxyType(
            {
                key: value
                for key, value in parameters.items()
                if key not in self.spec.varying_parameter_keys
            }
        )

    @property
    def varying_generation_parameters(self) -> tuple[Mapping[str, object], ...]:
        """Declared size-varying generator parameters for every member."""
        return tuple(
            MappingProxyType(
                {
                    key: request.parameters[key]
                    for key in self.spec.varying_parameter_keys
                }
            )
            for request in self.generation_requests
        )


def create_cross_size_geometry_family(
    spec: GeometryFamilySpec,
    *,
    scaling: FiniteSizeScalingResult,
    geometries: tuple[Geometry, ...],
) -> CrossSizeGeometryFamily:
    """Bind ordered protocol-generated geometries to an existing scaling fit."""
    if not isinstance(spec, GeometryFamilySpec):
        raise TypeError("spec must be GeometryFamilySpec")
    if not isinstance(scaling, FiniteSizeScalingResult):
        raise TypeError("scaling must be FiniteSizeScalingResult")
    geometries = tuple(geometries)
    if len(geometries) != len(scaling.points):
        raise ValueError("geometries must contain one value per scaling point")
    if any(not isinstance(geometry, Geometry) for geometry in geometries):
        raise TypeError("geometries must contain only Geometry values")
    members = tuple(
        GeometryFamilyMember(
            geometry=geometry,
            scaling_point=point,
        )
        for geometry, point in zip(geometries, scaling.points, strict=True)
    )
    return CrossSizeGeometryFamily(
        spec=spec,
        scaling=scaling,
        members=members,
    )


def _generation_record(
    geometry: Geometry,
) -> tuple[str, int, GeometryGenerationRequest]:
    generation = geometry.metadata.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError(
            "geometry family members require Phase-6 generation metadata"
        )
    if set(generation) != _GENERATION_METADATA_KEYS:
        raise ValueError("geometry generation metadata has an invalid schema")
    generator_key = _technical_key(
        generation["generator_key"],
        name="generation generator_key",
    )
    generator_version = _positive_integer(
        generation["generator_version"],
        name="generation generator_version",
    )
    parameters = generation["parameters"]
    if not isinstance(parameters, Mapping):
        raise TypeError("generation parameters must be a mapping")
    request = GeometryGenerationRequest(
        parameters=_thaw_json_mapping(parameters),
        seed=generation["seed"],
    )
    return generator_key, generator_version, request


def _validate_generation_parameters(
    members: tuple[GeometryFamilyMember, ...],
    *,
    spec: GeometryFamilySpec,
) -> None:
    varying_signatures: list[tuple[object, ...]] = []
    reference_fixed: dict[str, object] | None = None
    for member in members:
        parameters = member.generation_request.parameters
        missing = tuple(
            key for key in spec.varying_parameter_keys if key not in parameters
        )
        if missing:
            raise ValueError(
                "every member must contain all declared varying generator parameters"
            )
        fixed = {
            key: value
            for key, value in parameters.items()
            if key not in spec.varying_parameter_keys
        }
        if reference_fixed is None:
            reference_fixed = fixed
        elif fixed != reference_fixed:
            raise ValueError(
                "non-varying generator parameters must be identical across the family"
            )
        signature = tuple(
            parameters[key] for key in spec.varying_parameter_keys
        )
        if any(signature == previous for previous in varying_signatures):
            raise ValueError(
                "declared varying generator parameters must identify each size uniquely"
            )
        varying_signatures.append(signature)


def _validate_generation_seeds(
    members: tuple[GeometryFamilyMember, ...],
    *,
    policy: GeometryFamilySeedPolicy,
) -> None:
    seeds = tuple(member.generation_request.seed for member in members)
    if policy is GeometryFamilySeedPolicy.NO_SEED:
        valid = all(seed is None for seed in seeds)
    elif policy is GeometryFamilySeedPolicy.COMMON_SEED:
        valid = seeds[0] is not None and all(seed == seeds[0] for seed in seeds)
    elif policy is GeometryFamilySeedPolicy.DISTINCT_PER_SIZE:
        valid = all(seed is not None for seed in seeds) and len(set(seeds)) == len(seeds)
    else:
        raise AssertionError(f"unsupported geometry family seed policy: {policy!r}")
    if not valid:
        raise ValueError("geometry-generation seeds do not match the declared seed policy")


def _thaw_json_mapping(values: Mapping[object, object]) -> dict[str, object]:
    prepared: dict[str, object] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise ValueError("generation parameter names must be strings")
        prepared[key] = _thaw_json_value(value)
    return prepared


def _thaw_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _thaw_json_mapping(value)
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _technical_key(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _TECHNICAL_KEY_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{name} must start with a lowercase letter and contain only lowercase "
            "letters, digits, and underscores"
        )
    return value


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"{name} must be a Python-style identifier")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be at least one")
    return result


def _description(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()
