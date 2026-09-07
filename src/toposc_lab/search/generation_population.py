"""Validated immutable geometry populations after generation zero."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Integral

from toposc_lab.search.geometry_genome import GeometryGenome
from toposc_lab.search.mutation_validity import (
    MutationValidityPolicy,
    MutationValidityReport,
    validate_geometry_mutation,
)

GENERATION_POPULATION_VERSION = 1


@dataclass(frozen=True, slots=True)
class GenerationPopulationMember:
    """One ordered member of a validated population after generation zero."""

    generation_index: int
    member_index: int
    genome: GeometryGenome
    validation_source: GeometryGenome
    validity: MutationValidityReport

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "generation_index",
            _positive_integer(self.generation_index, name="generation_index"),
        )
        object.__setattr__(
            self,
            "member_index",
            _nonnegative_integer(self.member_index, name="member_index"),
        )
        if not isinstance(self.genome, GeometryGenome):
            raise TypeError("genome must be a GeometryGenome instance")
        if not isinstance(self.validation_source, GeometryGenome):
            raise TypeError("validation_source must be a GeometryGenome instance")
        if not isinstance(self.validity, MutationValidityReport):
            raise TypeError("validity must be a MutationValidityReport")
        if not self.validity.is_valid:
            raise ValueError("a generation population member requires a valid report")


@dataclass(frozen=True, slots=True)
class GenerationPopulation:
    """One immutable ordered geometry population after generation zero."""

    generation_index: int
    members: tuple[GenerationPopulationMember, ...]
    validity_policy: MutationValidityPolicy
    embedding_dimension: int | None
    population_version: int = field(
        default=GENERATION_POPULATION_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        generation_index = _positive_integer(
            self.generation_index,
            name="generation_index",
        )
        if isinstance(self.members, (str, bytes, bytearray)) or not isinstance(
            self.members,
            Iterable,
        ):
            raise TypeError("members must be an iterable of GenerationPopulationMember")
        members = tuple(self.members)
        if not members:
            raise ValueError("a generation population must contain at least one member")
        if not all(isinstance(member, GenerationPopulationMember) for member in members):
            raise TypeError(
                "members must contain only GenerationPopulationMember values"
            )
        if not isinstance(self.validity_policy, MutationValidityPolicy):
            raise TypeError("validity_policy must be a MutationValidityPolicy")
        dimension = _embedding_dimension(self.embedding_dimension)
        for member_index, member in enumerate(members):
            if member.generation_index != generation_index:
                raise ValueError(
                    "member generation indices must match the population generation"
                )
            if member.member_index != member_index:
                raise ValueError("member indices must match population order exactly")
            if member.genome.embedding_dimension != dimension:
                raise ValueError(
                    "every generation population member must share embedding_dimension"
                )
            expected_validity = validate_geometry_mutation(
                member.validation_source,
                member.genome,
                policy=self.validity_policy,
            )
            if member.validity != expected_validity:
                raise ValueError(
                    "member validity reports must match their validation source and "
                    "population policy"
                )
        object.__setattr__(self, "generation_index", generation_index)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "embedding_dimension", dimension)

    @property
    def population_size(self) -> int:
        """Number of ordered genomes in this generation."""
        return len(self.members)

    @property
    def genomes(self) -> tuple[GeometryGenome, ...]:
        """Ordered immutable genome references without validity wrappers."""
        return tuple(member.genome for member in self.members)


def _embedding_dimension(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("embedding_dimension must be an integer or None")
    result = int(value)
    if result < 1:
        raise ValueError("embedding_dimension must be positive")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    result = _nonnegative_integer(value, name=name)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result
