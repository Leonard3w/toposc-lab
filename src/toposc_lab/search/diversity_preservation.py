"""Explicit geometry-family occupancy constraints for search populations."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import TypeAlias

from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.search.generation_population import (
    GenerationPopulation,
    GenerationPopulationMember,
)
from toposc_lab.search.geometry_genome import GeometryGenome
from toposc_lab.search.initial_population import (
    InitialPopulation,
    InitialPopulationMember,
)

DIVERSITY_PRESERVATION_VERSION = 1

_TECHNICAL_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_DIVERSITY_WARNINGS = (
    (
        "Family labels are caller-declared search categories; they are not inferred "
        "physical equivalence classes or scientific discoveries."
    ),
    (
        "The family occupancy policy is a hard generation constraint and does not "
        "modify fitness, rank, Pareto dominance, or elite tiers."
    ),
    (
        "No graph, coordinate, genotype, phenotype, objective-space, or novelty "
        "distance is defined by this result."
    ),
    (
        "Violating populations are reported without filtering, repair, replacement, "
        "or retry."
    ),
)

SearchPopulation: TypeAlias = InitialPopulation | GenerationPopulation
SearchPopulationMember: TypeAlias = (
    InitialPopulationMember | GenerationPopulationMember
)


@dataclass(frozen=True, slots=True)
class GeometrySearchFamily:
    """One explicit versioned search-family label without inferred equivalence."""

    family_key: str
    family_version: int

    def __post_init__(self) -> None:
        if not isinstance(self.family_key, str) or (
            _TECHNICAL_KEY_PATTERN.fullmatch(self.family_key) is None
        ):
            raise ValueError(
                "family_key must start with a lowercase letter and contain only "
                "lowercase letters, digits, and underscores"
            )
        object.__setattr__(
            self,
            "family_version",
            _positive_integer(self.family_version, name="family_version"),
        )


@dataclass(frozen=True, slots=True)
class DiversityPreservationPolicy:
    """Explicit absolute occupancy constraints for one fixed-size population."""

    minimum_distinct_families: int | None = None
    maximum_members_per_family: int | None = None

    def __post_init__(self) -> None:
        minimum = _optional_positive_integer(
            self.minimum_distinct_families,
            name="minimum_distinct_families",
        )
        maximum = _optional_positive_integer(
            self.maximum_members_per_family,
            name="maximum_members_per_family",
        )
        if minimum is None and maximum is None:
            raise ValueError("at least one diversity occupancy constraint is required")
        object.__setattr__(self, "minimum_distinct_families", minimum)
        object.__setattr__(self, "maximum_members_per_family", maximum)


@dataclass(frozen=True, slots=True)
class PopulationFamilyMember:
    """One exact population member and its caller-declared search family."""

    population_member: SearchPopulationMember
    family: GeometrySearchFamily
    geometry_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.population_member,
            (InitialPopulationMember, GenerationPopulationMember),
        ):
            raise TypeError("population_member must be a supported population member")
        if not isinstance(self.family, GeometrySearchFamily):
            raise TypeError("family must be a GeometrySearchFamily")
        object.__setattr__(
            self,
            "geometry_id",
            exact_geometry_id(self.population_member.genome.to_geometry()),
        )

    @property
    def generation_index(self) -> int:
        """Generation containing the classified member."""
        return self.population_member.generation_index

    @property
    def member_index(self) -> int:
        """Member index retained from the classified population."""
        return self.population_member.member_index


@dataclass(frozen=True, slots=True)
class DiversityPolicyViolation:
    """One stable population-level family occupancy violation."""

    code: str
    message: str
    family: GeometrySearchFamily | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.isidentifier():
            raise ValueError("violation code must be a Python identifier")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("violation message must be non-empty")
        if self.family is not None and not isinstance(
            self.family,
            GeometrySearchFamily,
        ):
            raise TypeError("family must be GeometrySearchFamily or None")
        object.__setattr__(self, "message", self.message.strip())


@dataclass(frozen=True, slots=True)
class PopulationDiversityReport:
    """Ordered family ledger and hard-policy outcome for one population."""

    population: SearchPopulation
    policy: DiversityPreservationPolicy
    classifier_identifier: str
    members: tuple[PopulationFamilyMember, ...]
    violations: tuple[DiversityPolicyViolation, ...]
    version: int = field(default=DIVERSITY_PRESERVATION_VERSION, init=False)
    warnings: tuple[str, ...] = field(default=_DIVERSITY_WARNINGS, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.population, (InitialPopulation, GenerationPopulation)):
            raise TypeError("population must be a supported search population")
        if not isinstance(self.policy, DiversityPreservationPolicy):
            raise TypeError("policy must be a DiversityPreservationPolicy")
        classifier_identifier = _nonempty_string(
            self.classifier_identifier,
            name="classifier_identifier",
        )
        if isinstance(self.members, (str, bytes, bytearray)) or not isinstance(
            self.members,
            Iterable,
        ):
            raise TypeError("members must be an iterable of PopulationFamilyMember")
        members = tuple(self.members)
        if len(members) != self.population.population_size:
            raise ValueError("members must classify every population member")
        if not all(isinstance(member, PopulationFamilyMember) for member in members):
            raise TypeError("members must contain only PopulationFamilyMember values")
        for classified, population_member in zip(
            members,
            self.population.members,
            strict=True,
        ):
            if classified.population_member is not population_member:
                raise ValueError(
                    "classified members must preserve population order and identity"
                )
        _validate_consistent_family_labels(members)
        if isinstance(self.violations, (str, bytes, bytearray)) or not isinstance(
            self.violations,
            Iterable,
        ):
            raise TypeError("violations must be an iterable of policy violations")
        violations = tuple(self.violations)
        if not all(
            isinstance(violation, DiversityPolicyViolation)
            for violation in violations
        ):
            raise TypeError(
                "violations must contain only DiversityPolicyViolation values"
            )
        expected_violations = _policy_violations(members, policy=self.policy)
        if violations != expected_violations:
            raise ValueError("violations do not match the family occupancy policy")
        object.__setattr__(self, "classifier_identifier", classifier_identifier)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "violations", violations)

    @property
    def satisfies_policy(self) -> bool:
        """Whether every explicit family occupancy constraint is satisfied."""
        return not self.violations

    @property
    def family_counts(self) -> Mapping[GeometrySearchFamily, int]:
        """Family counts in stable order of first population occurrence."""
        return MappingProxyType(_family_counts(self.members))

    @property
    def distinct_family_count(self) -> int:
        """Number of explicit family labels present in the population."""
        return len(self.family_counts)

    def family_for_member(self, member_index: int) -> GeometrySearchFamily:
        """Return the family at one exact population member index."""
        prepared_index = _nonnegative_integer(member_index, name="member_index")
        if prepared_index >= len(self.members):
            raise IndexError("member_index is outside the population")
        return self.members[prepared_index].family

    def raise_for_violations(self) -> None:
        """Raise with this complete report when the hard policy is violated."""
        if self.violations:
            raise DiversityPreservationError(self)


class DiversityPreservationError(ValueError):
    """Raised with the complete classified population when occupancy fails."""

    def __init__(self, report: PopulationDiversityReport) -> None:
        if not isinstance(report, PopulationDiversityReport):
            raise TypeError("report must be a PopulationDiversityReport")
        if report.satisfies_policy:
            raise ValueError("a diversity-preservation error requires violations")
        self.report = report
        details = "; ".join(
            f"{violation.code}"
            + (
                f" for {violation.family.family_key}@{violation.family.family_version}"
                if violation.family is not None
                else ""
            )
            + f": {violation.message}"
            for violation in report.violations
        )
        super().__init__(
            f"generation {report.population.generation_index} violates diversity "
            f"preservation: {details}"
        )


class DiversityClassificationError(RuntimeError):
    """Raised when the caller's family classifier fails for one member."""

    def __init__(
        self,
        *,
        classifier_identifier: str,
        generation_index: int,
        member_index: int,
        error: Exception,
    ) -> None:
        self.classifier_identifier = _nonempty_string(
            classifier_identifier,
            name="classifier_identifier",
        )
        self.generation_index = _nonnegative_integer(
            generation_index,
            name="generation_index",
        )
        self.member_index = _nonnegative_integer(member_index, name="member_index")
        if not isinstance(error, Exception):
            raise TypeError("error must be an Exception")
        self.error_type = type(error).__name__
        self.error_message = str(error).strip() or "The classifier raised without a message."
        super().__init__(
            f"diversity classifier {self.classifier_identifier!r} failed for "
            f"generation {self.generation_index}, member {self.member_index}: "
            f"{self.error_type}: {self.error_message}"
        )


class InconsistentGeometryFamilyError(ValueError):
    """Raised when one exact geometry snapshot receives different family labels."""

    def __init__(
        self,
        *,
        geometry_id: str,
        first_family: GeometrySearchFamily,
        second_family: GeometrySearchFamily,
    ) -> None:
        if not isinstance(geometry_id, str) or not geometry_id:
            raise ValueError("geometry_id must be non-empty")
        if not isinstance(first_family, GeometrySearchFamily) or not isinstance(
            second_family,
            GeometrySearchFamily,
        ):
            raise TypeError("family values must be GeometrySearchFamily")
        if first_family == second_family:
            raise ValueError("an inconsistent-family error requires different labels")
        self.geometry_id = geometry_id
        self.first_family = first_family
        self.second_family = second_family
        super().__init__(
            "one exact geometry snapshot received inconsistent search-family labels: "
            f"{first_family.family_key}@{first_family.family_version} versus "
            f"{second_family.family_key}@{second_family.family_version}"
        )


DiversityFamilyClassifier: TypeAlias = Callable[
    [GeometryGenome], GeometrySearchFamily
]


def assess_population_diversity(
    population: SearchPopulation,
    *,
    policy: DiversityPreservationPolicy,
    classifier: DiversityFamilyClassifier,
    classifier_identifier: str,
) -> PopulationDiversityReport:
    """Classify once per member and evaluate explicit family occupancy limits."""
    if not isinstance(population, (InitialPopulation, GenerationPopulation)):
        raise TypeError("population must be a supported search population")
    if not isinstance(policy, DiversityPreservationPolicy):
        raise TypeError("policy must be a DiversityPreservationPolicy")
    if not callable(classifier):
        raise TypeError("classifier must be callable")
    prepared_identifier = _nonempty_string(
        classifier_identifier,
        name="classifier_identifier",
    )

    members: list[PopulationFamilyMember] = []
    for population_member in population.members:
        try:
            family = classifier(population_member.genome)
        except Exception as error:
            raise DiversityClassificationError(
                classifier_identifier=prepared_identifier,
                generation_index=population.generation_index,
                member_index=population_member.member_index,
                error=error,
            ) from error
        if not isinstance(family, GeometrySearchFamily):
            raise TypeError("classifier must return GeometrySearchFamily")
        members.append(
            PopulationFamilyMember(
                population_member=population_member,
                family=family,
            )
        )
    classified = tuple(members)
    return PopulationDiversityReport(
        population=population,
        policy=policy,
        classifier_identifier=prepared_identifier,
        members=classified,
        violations=_policy_violations(classified, policy=policy),
    )


def validate_diversity_history(
    reports: Iterable[PopulationDiversityReport],
) -> None:
    """Require stable labels for repeated exact snapshots across generations."""
    if isinstance(reports, (str, bytes, bytearray)) or not isinstance(
        reports,
        Iterable,
    ):
        raise TypeError("reports must be an iterable of PopulationDiversityReport")
    prepared = tuple(reports)
    if not all(isinstance(report, PopulationDiversityReport) for report in prepared):
        raise TypeError("reports must contain only PopulationDiversityReport values")
    labels_by_geometry_id: dict[str, GeometrySearchFamily] = {}
    for report in prepared:
        for member in report.members:
            previous = labels_by_geometry_id.setdefault(member.geometry_id, member.family)
            if previous != member.family:
                raise InconsistentGeometryFamilyError(
                    geometry_id=member.geometry_id,
                    first_family=previous,
                    second_family=member.family,
                )


def _validate_consistent_family_labels(
    members: tuple[PopulationFamilyMember, ...],
) -> None:
    labels_by_geometry_id: dict[str, GeometrySearchFamily] = {}
    for member in members:
        previous = labels_by_geometry_id.setdefault(member.geometry_id, member.family)
        if previous != member.family:
            raise InconsistentGeometryFamilyError(
                geometry_id=member.geometry_id,
                first_family=previous,
                second_family=member.family,
            )


def _policy_violations(
    members: tuple[PopulationFamilyMember, ...],
    *,
    policy: DiversityPreservationPolicy,
) -> tuple[DiversityPolicyViolation, ...]:
    counts = _family_counts(members)
    violations: list[DiversityPolicyViolation] = []
    minimum = policy.minimum_distinct_families
    if minimum is not None and len(counts) < minimum:
        violations.append(
            DiversityPolicyViolation(
                code="minimum_distinct_families_not_met",
                message=(
                    f"population contains {len(counts)} distinct families; "
                    f"policy requires at least {minimum}"
                ),
            )
        )
    maximum = policy.maximum_members_per_family
    if maximum is not None:
        violations.extend(
            DiversityPolicyViolation(
                code="family_capacity_exceeded",
                message=(
                    f"family contains {count} members; policy permits at most {maximum}"
                ),
                family=family,
            )
            for family, count in counts.items()
            if count > maximum
        )
    return tuple(violations)


def _family_counts(
    members: tuple[PopulationFamilyMember, ...],
) -> dict[GeometrySearchFamily, int]:
    counts: dict[GeometrySearchFamily, int] = {}
    for member in members:
        counts[member.family] = counts.get(member.family, 0) + 1
    return counts


def _optional_positive_integer(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, name=name)


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


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()
