"""Auditable statistics over explicitly supplied, already-ranked search trials."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Integral, Real
from statistics import NormalDist
from typing import TypeAlias

from toposc_lab.search.candidate_ranking import (
    CandidateRankingEntry,
    CandidateRankingResult,
    UnrankedCandidateReason,
)

BASELINE_STATISTICS_VERSION = 1

BaselineSuccessPredicate: TypeAlias = Callable[[CandidateRankingEntry], bool]


class BaselineUncertaintyMethod(str, Enum):
    """Supported interval construction for the search-trial success fraction."""

    WILSON_SCORE = "wilson_score"


@dataclass(frozen=True, slots=True)
class BaselineSuccessCriterion:
    """Explicit caller-owned definition of a strong ranked candidate."""

    key: str
    description: str
    predicate: BaselineSuccessPredicate = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.isidentifier():
            raise ValueError("criterion key must be a Python-style identifier")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("criterion description must be a non-empty string")
        if not callable(self.predicate):
            raise TypeError("predicate must be callable")
        object.__setattr__(self, "description", self.description.strip())

    def evaluate(self, entry: CandidateRankingEntry) -> bool:
        """Evaluate one ranked candidate and require a strict boolean decision."""
        if not isinstance(entry, CandidateRankingEntry):
            raise TypeError("entry must be CandidateRankingEntry")
        if entry.rank is None:
            raise ValueError("baseline criterion requires a ranked entry")
        result = self.predicate(entry)
        if not isinstance(result, bool):
            raise TypeError("baseline success predicate must return a boolean")
        return result


@dataclass(frozen=True, slots=True)
class SearchBaselineTrial:
    """One explicitly identified, already-ranked search attempt."""

    trial_key: str
    ranking: CandidateRankingResult

    def __post_init__(self) -> None:
        if not isinstance(self.trial_key, str) or not self.trial_key.isidentifier():
            raise ValueError("trial_key must be a Python-style identifier")
        if not isinstance(self.ranking, CandidateRankingResult):
            raise TypeError("ranking must be CandidateRankingResult")


@dataclass(frozen=True, slots=True)
class SearchBaselineTrialOutcome:
    """Per-candidate decisions and failure classifications for one trial."""

    trial: SearchBaselineTrial
    criterion: BaselineSuccessCriterion
    candidate_successes: tuple[bool, ...]
    invalid_candidate_indices: tuple[int, ...]
    callback_failure_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.trial, SearchBaselineTrial):
            raise TypeError("trial must be SearchBaselineTrial")
        if not isinstance(self.criterion, BaselineSuccessCriterion):
            raise TypeError("criterion must be BaselineSuccessCriterion")
        successes = tuple(self.candidate_successes)
        if any(not isinstance(success, bool) for success in successes):
            raise TypeError("candidate_successes must contain only booleans")
        if len(successes) != len(self.trial.ranking.entries):
            raise ValueError("candidate_successes must contain one decision per batch member")
        invalid_indices = _ordered_indices(
            self.invalid_candidate_indices,
            name="invalid_candidate_indices",
            upper_bound=len(successes),
        )
        callback_indices = _ordered_indices(
            self.callback_failure_indices,
            name="callback_failure_indices",
            upper_bound=len(successes),
        )
        if set(invalid_indices).intersection(callback_indices):
            raise ValueError("invalid and callback-failure indices must be disjoint")

        expected_invalid = tuple(
            entry.candidate_index
            for entry in self.trial.ranking.entries
            if entry.unranked_reason is UnrankedCandidateReason.INVALID_EVALUATION
        )
        expected_callbacks = tuple(
            entry.candidate_index
            for entry in self.trial.ranking.entries
            if entry.unranked_reason is UnrankedCandidateReason.EVALUATION_CALLBACK_FAILURE
        )
        if invalid_indices != expected_invalid:
            raise ValueError("invalid indices must match the ranking result exactly")
        if callback_indices != expected_callbacks:
            raise ValueError("callback-failure indices must match the ranking result exactly")
        unranked_indices = set(invalid_indices).union(callback_indices)
        if any(successes[index] for index in unranked_indices):
            raise ValueError("an unranked candidate cannot be marked successful")

        object.__setattr__(self, "candidate_successes", successes)
        object.__setattr__(self, "invalid_candidate_indices", invalid_indices)
        object.__setattr__(self, "callback_failure_indices", callback_indices)

    @property
    def is_successful(self) -> bool:
        """Whether this search attempt found at least one strong candidate."""
        return any(self.candidate_successes)

    @property
    def successful_candidate_indices(self) -> tuple[int, ...]:
        """Original batch indices satisfying the explicit criterion."""
        return tuple(index for index, success in enumerate(self.candidate_successes) if success)

    @property
    def successful_candidate_count(self) -> int:
        """Number of candidates satisfying the explicit criterion."""
        return sum(self.candidate_successes)

    @property
    def candidate_count(self) -> int:
        """All requested candidates, including invalid and failed members."""
        return len(self.candidate_successes)


@dataclass(frozen=True, slots=True)
class SearchBaselineStatistics:
    """Trial-level hit rate plus audit counts for one explicit criterion."""

    criterion: BaselineSuccessCriterion
    outcomes: tuple[SearchBaselineTrialOutcome, ...]
    confidence_level: float
    uncertainty_method: BaselineUncertaintyMethod = BaselineUncertaintyMethod.WILSON_SCORE
    statistics_version: int = field(
        default=BASELINE_STATISTICS_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.criterion, BaselineSuccessCriterion):
            raise TypeError("criterion must be BaselineSuccessCriterion")
        outcomes = tuple(self.outcomes)
        if not outcomes:
            raise ValueError("outcomes must not be empty")
        if not all(isinstance(item, SearchBaselineTrialOutcome) for item in outcomes):
            raise TypeError("outcomes must contain only SearchBaselineTrialOutcome values")
        trial_keys = tuple(item.trial.trial_key for item in outcomes)
        if len(set(trial_keys)) != len(trial_keys):
            raise ValueError("trial keys must be unique")
        if any(item.criterion is not self.criterion for item in outcomes):
            raise ValueError("outcomes must reference the exact baseline criterion")
        reference_config = outcomes[0].trial.ranking.config
        candidate_count = outcomes[0].candidate_count
        for outcome in outcomes[1:]:
            if outcome.trial.ranking.config != reference_config:
                raise ValueError("all baseline trials must use the same ranking configuration")
            if outcome.candidate_count != candidate_count:
                raise ValueError("all baseline trials must use the same candidate count")
        if not isinstance(self.uncertainty_method, BaselineUncertaintyMethod):
            raise TypeError("uncertainty_method must be BaselineUncertaintyMethod")

        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(
            self,
            "confidence_level",
            _confidence_level(self.confidence_level),
        )

    @property
    def trial_count(self) -> int:
        """Number of explicitly supplied search attempts."""
        return len(self.outcomes)

    @property
    def successful_trial_count(self) -> int:
        """Attempts finding at least one candidate satisfying the criterion."""
        return sum(outcome.is_successful for outcome in self.outcomes)

    @property
    def trial_successes(self) -> tuple[bool, ...]:
        """Ordered per-trial hit decisions."""
        return tuple(outcome.is_successful for outcome in self.outcomes)

    @property
    def trial_success_fraction(self) -> float:
        """Observed fraction of attempts finding at least one strong candidate."""
        return self.successful_trial_count / self.trial_count

    @property
    def trial_success_standard_error(self) -> float:
        """Plug-in binomial standard error of the observed trial hit rate."""
        probability = self.trial_success_fraction
        return math.sqrt(probability * (1.0 - probability) / self.trial_count)

    @property
    def critical_value(self) -> float:
        """Two-sided standard-normal critical value for the confidence level."""
        lower_tail_probability = (1.0 - self.confidence_level) / 2.0
        return -NormalDist().inv_cdf(lower_tail_probability)

    @property
    def trial_success_confidence_interval(self) -> tuple[float, float]:
        """Two-sided Wilson interval for the search-trial success fraction."""
        probability = self.trial_success_fraction
        sample_size = self.trial_count
        critical_value = self.critical_value
        squared_critical_value = critical_value * critical_value
        denominator = 1.0 + squared_critical_value / sample_size
        center = (probability + squared_critical_value / (2.0 * sample_size)) / denominator
        half_width = (
            critical_value
            * math.sqrt(
                probability * (1.0 - probability) / sample_size
                + squared_critical_value / (4.0 * sample_size * sample_size)
            )
            / denominator
        )
        return max(0.0, center - half_width), min(1.0, center + half_width)

    @property
    def candidates_per_trial(self) -> int:
        """Explicit common search budget inferred from each source batch."""
        return self.outcomes[0].candidate_count

    @property
    def total_candidate_count(self) -> int:
        """All candidates across every trial, including invalid and failed ones."""
        return self.trial_count * self.candidates_per_trial

    @property
    def successful_candidate_count(self) -> int:
        """All candidates satisfying the explicit criterion."""
        return sum(item.successful_candidate_count for item in self.outcomes)

    @property
    def candidate_success_fraction(self) -> float:
        """Descriptive candidate hit fraction without independence assumptions."""
        return self.successful_candidate_count / self.total_candidate_count

    @property
    def invalid_candidate_count(self) -> int:
        """Invalid evaluation runs retained in the candidate denominator."""
        return sum(len(item.invalid_candidate_indices) for item in self.outcomes)

    @property
    def callback_failure_count(self) -> int:
        """Evaluation callback failures retained in the candidate denominator."""
        return sum(len(item.callback_failure_indices) for item in self.outcomes)

    @property
    def warnings(self) -> tuple[str, ...]:
        """Interpretation limits that remain attached to every estimate."""
        warnings = [
            "The success criterion must be frozen before outcomes are inspected; "
            "the statistics do not define a strong candidate.",
            "The Wilson interval has a repeated-trial interpretation only when "
            "trials are representative independent attempts under one frozen search "
            "protocol.",
            "The candidate success fraction is descriptive because candidates within "
            "and across trials may not be independent.",
            "A baseline hit rate is not evidence of topology, robustness, novelty, "
            "causal geometry advantage, or scientific discovery.",
        ]
        if self.invalid_candidate_count or self.callback_failure_count:
            warnings.append(
                "Invalid candidates and callback failures remain unsuccessful in the "
                "denominator and are reported separately."
            )
        return tuple(warnings)


def compute_search_baseline_statistics(
    trials: Iterable[SearchBaselineTrial],
    *,
    criterion: BaselineSuccessCriterion,
    confidence_level: float,
    uncertainty_method: BaselineUncertaintyMethod = (BaselineUncertaintyMethod.WILSON_SCORE),
) -> SearchBaselineStatistics:
    """Evaluate every rankable candidate and summarize per-trial search hits."""
    if isinstance(trials, (str, bytes, bytearray)) or not isinstance(
        trials,
        Iterable,
    ):
        raise TypeError("trials must be an iterable of SearchBaselineTrial values")
    prepared_trials = tuple(trials)
    if not prepared_trials:
        raise ValueError("trials must not be empty")
    if not all(isinstance(item, SearchBaselineTrial) for item in prepared_trials):
        raise TypeError("trials must contain only SearchBaselineTrial values")
    if not isinstance(criterion, BaselineSuccessCriterion):
        raise TypeError("criterion must be BaselineSuccessCriterion")
    prepared_confidence_level = _confidence_level(confidence_level)
    if not isinstance(uncertainty_method, BaselineUncertaintyMethod):
        raise TypeError("uncertainty_method must be BaselineUncertaintyMethod")

    _validate_trial_contracts(prepared_trials)
    outcomes = tuple(_evaluate_trial(trial, criterion=criterion) for trial in prepared_trials)
    return SearchBaselineStatistics(
        criterion=criterion,
        outcomes=outcomes,
        confidence_level=prepared_confidence_level,
        uncertainty_method=uncertainty_method,
    )


def _evaluate_trial(
    trial: SearchBaselineTrial,
    *,
    criterion: BaselineSuccessCriterion,
) -> SearchBaselineTrialOutcome:
    successes: list[bool] = []
    invalid_indices: list[int] = []
    callback_indices: list[int] = []
    for entry in trial.ranking.entries:
        if entry.unranked_reason is UnrankedCandidateReason.INVALID_EVALUATION:
            successes.append(False)
            invalid_indices.append(entry.candidate_index)
        elif entry.unranked_reason is UnrankedCandidateReason.EVALUATION_CALLBACK_FAILURE:
            successes.append(False)
            callback_indices.append(entry.candidate_index)
        else:
            successes.append(criterion.evaluate(entry))
    return SearchBaselineTrialOutcome(
        trial=trial,
        criterion=criterion,
        candidate_successes=tuple(successes),
        invalid_candidate_indices=tuple(invalid_indices),
        callback_failure_indices=tuple(callback_indices),
    )


def _validate_trial_contracts(trials: tuple[SearchBaselineTrial, ...]) -> None:
    trial_keys = tuple(trial.trial_key for trial in trials)
    if len(set(trial_keys)) != len(trial_keys):
        raise ValueError("trial keys must be unique")
    reference_config = trials[0].ranking.config
    candidate_count = len(trials[0].ranking.entries)
    for trial in trials[1:]:
        if trial.ranking.config != reference_config:
            raise ValueError("all baseline trials must use the same ranking configuration")
        if len(trial.ranking.entries) != candidate_count:
            raise ValueError("all baseline trials must use the same candidate count")


def _ordered_indices(
    values: tuple[int, ...],
    *,
    name: str,
    upper_bound: int,
) -> tuple[int, ...]:
    indices = tuple(_nonnegative_integer(value, name=name) for value in values)
    if indices != tuple(sorted(set(indices))):
        raise ValueError(f"{name} must be unique and ascending")
    if any(index >= upper_bound for index in indices):
        raise ValueError(f"{name} contains an index outside the trial")
    return indices


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must contain only integers")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must contain only non-negative integers")
    return result


def _confidence_level(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("confidence_level must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError("confidence_level must be finite and strictly between 0 and 1")
    return result
