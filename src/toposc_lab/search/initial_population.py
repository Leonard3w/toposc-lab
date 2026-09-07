"""Deterministic assembly of validated generation-zero genome populations."""

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

INITIAL_POPULATION_VERSION = 1


@dataclass(frozen=True, slots=True)
class InitialPopulationMember:
    """One ordered, policy-valid member of an initial population."""

    member_index: int
    genome: GeometryGenome
    validity: MutationValidityReport

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "member_index",
            _nonnegative_integer(self.member_index, name="member_index"),
        )
        if not isinstance(self.genome, GeometryGenome):
            raise TypeError("genome must be a GeometryGenome instance")
        if not isinstance(self.validity, MutationValidityReport):
            raise TypeError("validity must be a MutationValidityReport")
        if not self.validity.is_valid:
            raise ValueError("an initial population member requires a valid report")


@dataclass(frozen=True, slots=True)
class InitialPopulation:
    """Immutable ordered geometry genomes at generation zero."""

    members: tuple[InitialPopulationMember, ...]
    validity_policy: MutationValidityPolicy
    embedding_dimension: int | None
    generation_index: int = field(default=0, init=False)
    population_version: int = field(
        default=INITIAL_POPULATION_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        if isinstance(self.members, (str, bytes, bytearray)):
            raise TypeError("members must be an iterable of InitialPopulationMember")
        try:
            members = tuple(self.members)
        except TypeError as error:
            raise TypeError(
                "members must be an iterable of InitialPopulationMember"
            ) from error
        if not members:
            raise ValueError("an initial population must contain at least one member")
        if not all(isinstance(member, InitialPopulationMember) for member in members):
            raise TypeError("members must contain only InitialPopulationMember values")
        if not isinstance(self.validity_policy, MutationValidityPolicy):
            raise TypeError("validity_policy must be a MutationValidityPolicy")
        dimension = _embedding_dimension(self.embedding_dimension)
        anchor = members[0].genome
        for member_index, member in enumerate(members):
            if member.member_index != member_index:
                raise ValueError("member indices must match population order exactly")
            if member.genome.embedding_dimension != dimension:
                raise ValueError(
                    "every initial population member must share embedding_dimension"
                )
            expected_validity = validate_geometry_mutation(
                anchor,
                member.genome,
                policy=self.validity_policy,
            )
            if member.validity != expected_validity:
                raise ValueError(
                    "member validity reports must match the population validity policy"
                )
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "embedding_dimension", dimension)

    @property
    def population_size(self) -> int:
        """Number of ordered genomes in generation zero."""
        return len(self.members)

    @property
    def genomes(self) -> tuple[GeometryGenome, ...]:
        """Ordered immutable genome references without validity wrappers."""
        return tuple(member.genome for member in self.members)


class InitialPopulationError(ValueError):
    """Raised when one supplied candidate cannot enter the initial population."""

    def __init__(
        self,
        *,
        candidate_index: int,
        report: MutationValidityReport,
    ) -> None:
        if not isinstance(report, MutationValidityReport):
            raise TypeError("report must be a MutationValidityReport")
        if report.is_valid:
            raise ValueError("an initial population error requires an invalid report")
        self.candidate_index = _nonnegative_integer(
            candidate_index,
            name="candidate_index",
        )
        self.report = report
        details = "; ".join(
            f"{issue.code} at {issue.path or 'candidate'}: {issue.message}"
            for issue in report.issues
        )
        super().__init__(
            f"initial population candidate {self.candidate_index} is invalid: {details}"
        )


def create_initial_population(
    genomes: Iterable[GeometryGenome],
    *,
    validity_policy: MutationValidityPolicy | None = None,
) -> InitialPopulation:
    """Assemble a non-empty ordered population without sampling or filtering.

    The first supplied genome anchors the embedding dimension. Every candidate
    is checked against that anchor through the Phase-10.9 validity contract.
    The first invalid candidate raises with its original index and full report;
    no partial population is returned.
    """
    if isinstance(genomes, (str, bytes, bytearray)) or not isinstance(genomes, Iterable):
        raise TypeError("genomes must be an iterable of GeometryGenome instances")
    candidates = tuple(genomes)
    if not candidates:
        raise ValueError("genomes must contain at least one candidate")
    if not all(isinstance(genome, GeometryGenome) for genome in candidates):
        raise TypeError("genomes must contain only GeometryGenome instances")
    if validity_policy is None:
        validity_policy = MutationValidityPolicy()
    elif not isinstance(validity_policy, MutationValidityPolicy):
        raise TypeError("validity_policy must be a MutationValidityPolicy or None")

    anchor = candidates[0]
    members: list[InitialPopulationMember] = []
    for candidate_index, candidate in enumerate(candidates):
        report = validate_geometry_mutation(
            anchor,
            candidate,
            policy=validity_policy,
        )
        if not report.is_valid:
            raise InitialPopulationError(
                candidate_index=candidate_index,
                report=report,
            )
        members.append(
            InitialPopulationMember(
                member_index=candidate_index,
                genome=candidate,
                validity=report,
            )
        )

    return InitialPopulation(
        members=tuple(members),
        validity_policy=validity_policy,
        embedding_dimension=anchor.embedding_dimension,
    )


def _embedding_dimension(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("embedding_dimension must be an integer or None")
    result = int(value)
    if result < 1:
        raise ValueError("embedding_dimension must be positive")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result
