"""Deterministic, policy-explicit ranking of completed Phase-9 batch members."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import TypeAlias

from toposc_lab.evaluation import ObjectiveDirection
from toposc_lab.search.batch_evaluation import (
    BatchEvaluationMember,
    BatchEvaluationResult,
)

CANDIDATE_RANKING_VERSION = 1

RankingScalar: TypeAlias = bool | int | float
CandidateRankingValueFactory: TypeAlias = Callable[
    [BatchEvaluationMember],
    Mapping[str, object],
]

_RANKING_WARNINGS = (
    "Candidate ranks are an engineering ordering, not a physical observable, "
    "topological invariant, or scientific discovery.",
    "Ranks are comparable only under the same ordered criteria and identical "
    "value-construction conventions.",
    "A top-ranked candidate still requires the research charter's independent "
    "eligibility, reference, robustness, finite-size, and leakage controls.",
)


class CandidateRankingValueKind(str, Enum):
    """Declared scalar type for one ranking criterion."""

    BOOLEAN = "boolean"
    INTEGER = "integer"
    REAL = "real"


class UnrankedCandidateReason(str, Enum):
    """Why a retained Phase-9.3 member was not supplied to the ranker."""

    INVALID_EVALUATION = "invalid_evaluation"
    EVALUATION_CALLBACK_FAILURE = "evaluation_callback_failure"


@dataclass(frozen=True, slots=True)
class CandidateRankingCriterion:
    """One named, typed criterion and its explicit optimization preference."""

    name: str
    direction: ObjectiveDirection
    value_kind: CandidateRankingValueKind

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("criterion name must be a Python-style identifier")
        if not isinstance(self.direction, ObjectiveDirection):
            raise TypeError("direction must be ObjectiveDirection")
        if not isinstance(self.value_kind, CandidateRankingValueKind):
            raise TypeError("value_kind must be CandidateRankingValueKind")


@dataclass(frozen=True, slots=True)
class CandidateRankingConfig:
    """Nonempty lexicographic criterion order with no implicit defaults."""

    criteria: tuple[CandidateRankingCriterion, ...]

    def __post_init__(self) -> None:
        if isinstance(self.criteria, (str, bytes, bytearray)) or not isinstance(
            self.criteria,
            Iterable,
        ):
            raise TypeError("criteria must be an iterable of ranking criteria")
        criteria = tuple(self.criteria)
        if not criteria:
            raise ValueError("criteria must not be empty")
        if not all(isinstance(item, CandidateRankingCriterion) for item in criteria):
            raise TypeError("criteria must contain only CandidateRankingCriterion values")
        names = tuple(item.name for item in criteria)
        if len(set(names)) != len(names):
            raise ValueError("criterion names must be unique")
        object.__setattr__(self, "criteria", criteria)


@dataclass(frozen=True, slots=True)
class CandidateRankingEntry:
    """One retained batch member with either a rank or an unranked reason."""

    member: BatchEvaluationMember
    values: Mapping[str, RankingScalar]
    rank: int | None
    unranked_reason: UnrankedCandidateReason | None

    def __post_init__(self) -> None:
        if not isinstance(self.member, BatchEvaluationMember):
            raise TypeError("member must be BatchEvaluationMember")
        if not isinstance(self.values, Mapping):
            raise TypeError("values must be a mapping")
        source_values = dict(self.values)
        if any(not isinstance(name, str) or not name.isidentifier() for name in source_values):
            raise ValueError("ranking value names must be Python-style identifiers")
        values: dict[str, RankingScalar] = {}
        for name, value in source_values.items():
            if isinstance(value, bool):
                values[name] = value
                continue
            if isinstance(value, Integral):
                values[name] = int(value)
                continue
            if isinstance(value, Real) and math.isfinite(float(value)):
                values[name] = float(value)
                continue
            raise ValueError(f"ranking value {name!r} must be a finite scalar")

        rank = self.rank
        if rank is not None:
            if isinstance(rank, bool) or not isinstance(rank, Integral):
                raise TypeError("rank must be an integer or None")
            rank = int(rank)
            if rank < 1:
                raise ValueError("rank must be positive")
        if self.unranked_reason is not None and not isinstance(
            self.unranked_reason,
            UnrankedCandidateReason,
        ):
            raise TypeError("unranked_reason must be UnrankedCandidateReason or None")
        if (rank is None) == (self.unranked_reason is None):
            raise ValueError("an entry requires exactly one rank or unranked reason")
        if rank is None and values:
            raise ValueError("an unranked entry must not contain ranking values")
        if rank is not None and not values:
            raise ValueError("a ranked entry requires ranking values")

        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(self, "rank", rank)

    @property
    def candidate_index(self) -> int:
        """Original Phase-9.3 request index retained across both result views."""
        return self.member.candidate_index


@dataclass(frozen=True, slots=True)
class CandidateRankingResult:
    """Complete batch-order ledger plus deterministic rank-order view."""

    source: BatchEvaluationResult
    config: CandidateRankingConfig
    entries: tuple[CandidateRankingEntry, ...]
    ranked_entries: tuple[CandidateRankingEntry, ...]
    version: int = field(default=CANDIDATE_RANKING_VERSION, init=False)
    warnings: tuple[str, ...] = field(default=_RANKING_WARNINGS, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, BatchEvaluationResult):
            raise TypeError("source must be BatchEvaluationResult")
        if not isinstance(self.config, CandidateRankingConfig):
            raise TypeError("config must be CandidateRankingConfig")
        entries = tuple(self.entries)
        ranked_entries = tuple(self.ranked_entries)
        if not all(isinstance(item, CandidateRankingEntry) for item in entries):
            raise TypeError("entries must contain only CandidateRankingEntry values")
        if not all(isinstance(item, CandidateRankingEntry) for item in ranked_entries):
            raise TypeError("ranked_entries must contain only CandidateRankingEntry values")
        if len(entries) != len(self.source.members):
            raise ValueError("entries must retain every source batch member")
        for entry, member in zip(entries, self.source.members, strict=True):
            if entry.member is not member:
                raise ValueError("entries must preserve source batch order and identity")
            _validate_entry_against_member(entry, config=self.config)

        expected_ranked = tuple(entry for entry in entries if entry.rank is not None)
        if len(ranked_entries) != len(expected_ranked) or {
            id(entry) for entry in ranked_entries
        } != {id(entry) for entry in expected_ranked}:
            raise ValueError("ranked_entries must contain every ranked entry exactly once")
        expected_order = _sort_ranked_entries(expected_ranked, criteria=self.config.criteria)
        if tuple(map(id, ranked_entries)) != tuple(map(id, expected_order)):
            raise ValueError("ranked_entries do not follow the configured ordering")
        _validate_competition_ranks(
            ranked_entries,
            criteria=self.config.criteria,
        )

        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "ranked_entries", ranked_entries)


def rank_evaluated_candidates(
    result: BatchEvaluationResult,
    *,
    config: CandidateRankingConfig,
    value_factory: CandidateRankingValueFactory,
) -> CandidateRankingResult:
    """Rank valid evaluations and retain all other batch members as unranked.

    The value factory is called exactly once for each valid evaluation, in batch
    order. Missing, additional, non-finite, or incorrectly typed values are API
    errors. Factory exceptions propagate and no partial ranking is returned.
    """
    if not isinstance(result, BatchEvaluationResult):
        raise TypeError("result must be BatchEvaluationResult")
    if not isinstance(config, CandidateRankingConfig):
        raise TypeError("config must be CandidateRankingConfig")
    if not callable(value_factory):
        raise TypeError("value_factory must be callable")

    rankable: list[tuple[BatchEvaluationMember, Mapping[str, RankingScalar]]] = []
    unranked: dict[int, UnrankedCandidateReason] = {}
    for member in result.members:
        if member.failure is not None:
            unranked[member.candidate_index] = UnrankedCandidateReason.EVALUATION_CALLBACK_FAILURE
            continue
        evaluation = member.evaluation
        assert evaluation is not None
        if not evaluation.is_valid:
            unranked[member.candidate_index] = UnrankedCandidateReason.INVALID_EVALUATION
            continue
        if evaluation.evaluation is None:
            raise ValueError("a valid evaluation run has no scientific evaluation")
        raw_values = value_factory(member)
        values = _prepare_ranking_values(raw_values, criteria=config.criteria)
        rankable.append((member, values))

    ordered_rankable = _sort_rankable_values(rankable, criteria=config.criteria)
    ranks = _competition_ranks(ordered_rankable, criteria=config.criteria)
    values_by_index = {member.candidate_index: values for member, values in ordered_rankable}
    entries = tuple(
        CandidateRankingEntry(
            member=member,
            values=values_by_index.get(member.candidate_index, {}),
            rank=ranks.get(member.candidate_index),
            unranked_reason=unranked.get(member.candidate_index),
        )
        for member in result.members
    )
    entry_by_index = {entry.candidate_index: entry for entry in entries}
    ranked_entries = tuple(entry_by_index[member.candidate_index] for member, _ in ordered_rankable)
    return CandidateRankingResult(
        source=result,
        config=config,
        entries=entries,
        ranked_entries=ranked_entries,
    )


def _prepare_ranking_values(
    values: Mapping[str, object],
    *,
    criteria: tuple[CandidateRankingCriterion, ...],
) -> Mapping[str, RankingScalar]:
    if not isinstance(values, Mapping):
        raise TypeError("value_factory must return a mapping")
    expected_names = {criterion.name for criterion in criteria}
    actual_names = set(values)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        additional = sorted(actual_names - expected_names)
        raise ValueError(
            "ranking values must match configured criteria exactly; "
            f"missing={missing!r}, additional={additional!r}"
        )
    prepared = {
        criterion.name: _prepare_ranking_value(
            values[criterion.name],
            criterion=criterion,
        )
        for criterion in criteria
    }
    return MappingProxyType(prepared)


def _prepare_ranking_value(
    value: object,
    *,
    criterion: CandidateRankingCriterion,
) -> RankingScalar:
    path = f"ranking value {criterion.name!r}"
    if criterion.value_kind is CandidateRankingValueKind.BOOLEAN:
        if not isinstance(value, bool):
            raise TypeError(f"{path} must be a boolean")
        return value
    if criterion.value_kind is CandidateRankingValueKind.INTEGER:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{path} must be an integer")
        return int(value)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{path} must be a real number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{path} must be finite")
    return numeric_value


def _sort_rankable_values(
    values: Iterable[tuple[BatchEvaluationMember, Mapping[str, RankingScalar]]],
    *,
    criteria: tuple[CandidateRankingCriterion, ...],
) -> tuple[tuple[BatchEvaluationMember, Mapping[str, RankingScalar]], ...]:
    ordered = list(values)
    ordered.sort(key=lambda item: item[0].candidate_index)
    for criterion in reversed(criteria):
        criterion_name = criterion.name
        ordered.sort(
            key=lambda item: item[1][criterion_name],
            reverse=criterion.direction is ObjectiveDirection.MAXIMIZE,
        )
    return tuple(ordered)


def _competition_ranks(
    ordered: tuple[tuple[BatchEvaluationMember, Mapping[str, RankingScalar]], ...],
    *,
    criteria: tuple[CandidateRankingCriterion, ...],
) -> dict[int, int]:
    ranks: dict[int, int] = {}
    previous_values: tuple[RankingScalar, ...] | None = None
    current_rank = 0
    for position, (member, values) in enumerate(ordered, start=1):
        candidate_values = tuple(values[item.name] for item in criteria)
        if previous_values is None or candidate_values != previous_values:
            current_rank = position
        ranks[member.candidate_index] = current_rank
        previous_values = candidate_values
    return ranks


def _sort_ranked_entries(
    entries: Iterable[CandidateRankingEntry],
    *,
    criteria: tuple[CandidateRankingCriterion, ...],
) -> tuple[CandidateRankingEntry, ...]:
    ordered = list(entries)
    ordered.sort(key=lambda entry: entry.candidate_index)
    for criterion in reversed(criteria):
        criterion_name = criterion.name
        ordered.sort(
            key=lambda entry: entry.values[criterion_name],
            reverse=criterion.direction is ObjectiveDirection.MAXIMIZE,
        )
    return tuple(ordered)


def _validate_entry_against_member(
    entry: CandidateRankingEntry,
    *,
    config: CandidateRankingConfig,
) -> None:
    member = entry.member
    if member.failure is not None:
        if entry.unranked_reason is not (UnrankedCandidateReason.EVALUATION_CALLBACK_FAILURE):
            raise ValueError("callback failures must remain explicitly unranked")
        return
    evaluation = member.evaluation
    assert evaluation is not None
    if not evaluation.is_valid:
        if entry.unranked_reason is not UnrankedCandidateReason.INVALID_EVALUATION:
            raise ValueError("invalid evaluations must remain explicitly unranked")
        return
    if entry.rank is None:
        raise ValueError("valid evaluations must be ranked")
    _prepare_ranking_values(entry.values, criteria=config.criteria)


def _validate_competition_ranks(
    entries: tuple[CandidateRankingEntry, ...],
    *,
    criteria: tuple[CandidateRankingCriterion, ...],
) -> None:
    ordered = tuple((entry.member, entry.values) for entry in entries)
    expected = _competition_ranks(ordered, criteria=criteria)
    if any(entry.rank != expected[entry.candidate_index] for entry in entries):
        raise ValueError("rank values do not follow competition-ranking semantics")
