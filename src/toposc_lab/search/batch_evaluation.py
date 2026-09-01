"""Sequential, failure-aware batch execution of explicit evaluation candidates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from toposc_lab.evaluation import GeometryEvaluationRun
from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.geometry import Geometry
from toposc_lab.robustness.disorder import (
    ModelParameterSet,
    exact_model_parameter_set_id,
)
from toposc_lab.search.random_model_parameters import (
    _freeze_parameter_mapping,
    _nonnegative_integer,
)

BATCH_EVALUATION_EXECUTION_VERSION = 1


@dataclass(frozen=True, slots=True)
class BatchEvaluationCandidate:
    """One explicit geometry/parameter pair and optional evaluation seed."""

    geometry: Geometry
    model_parameters: ModelParameterSet
    evaluation_seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, Geometry):
            raise TypeError("geometry must be Geometry")
        parameters = _freeze_parameter_mapping(
            self.model_parameters,
            name="model_parameters",
        )
        seed = self.evaluation_seed
        if seed is not None:
            seed = _nonnegative_integer(seed, name="evaluation_seed")
        object.__setattr__(self, "model_parameters", parameters)
        object.__setattr__(self, "evaluation_seed", seed)


@dataclass(frozen=True, slots=True)
class BatchEvaluationRequest:
    """Nonempty ordered candidates for one sequential batch execution."""

    candidates: tuple[BatchEvaluationCandidate, ...]

    def __post_init__(self) -> None:
        if isinstance(self.candidates, (str, bytes, bytearray)):
            raise TypeError("candidates must be an iterable of batch candidates")
        try:
            candidates = tuple(self.candidates)
        except TypeError as error:
            raise TypeError(
                "candidates must be an iterable of batch candidates"
            ) from error
        if not candidates:
            raise ValueError("candidates must contain at least one candidate")
        if not all(
            isinstance(candidate, BatchEvaluationCandidate)
            for candidate in candidates
        ):
            raise TypeError(
                "candidates must contain only BatchEvaluationCandidate values"
            )
        object.__setattr__(self, "candidates", candidates)


class BatchEvaluationFailureStage(str, Enum):
    """Batch-owned callback stage that raised an ordinary exception."""

    EVALUATION_CALLBACK = "evaluation_callback"


@dataclass(frozen=True, slots=True)
class BatchEvaluationFailure:
    """Serializable callback failure without exception object or traceback."""

    stage: BatchEvaluationFailureStage
    error_type: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.stage, BatchEvaluationFailureStage):
            raise TypeError("stage must be BatchEvaluationFailureStage")
        if not isinstance(self.error_type, str) or not self.error_type.isidentifier():
            raise ValueError("error_type must be a Python-style identifier")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("failure message must be a non-empty string")
        object.__setattr__(self, "message", self.message.strip())

    @classmethod
    def from_exception(cls, error: Exception) -> BatchEvaluationFailure:
        """Create stable failure data from one callback exception."""
        if not isinstance(error, Exception):
            raise TypeError("error must be an Exception")
        message = str(error).strip() or "The callback raised without a message."
        return cls(
            stage=BatchEvaluationFailureStage.EVALUATION_CALLBACK,
            error_type=type(error).__name__,
            message=message,
        )


@dataclass(frozen=True, slots=True)
class BatchEvaluationMember:
    """One retained candidate with either an evaluation run or callback failure."""

    candidate_index: int
    candidate: BatchEvaluationCandidate
    evaluation: GeometryEvaluationRun | None
    failure: BatchEvaluationFailure | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_index",
            _nonnegative_integer(self.candidate_index, name="candidate_index"),
        )
        if not isinstance(self.candidate, BatchEvaluationCandidate):
            raise TypeError("candidate must be BatchEvaluationCandidate")
        if self.evaluation is not None and not isinstance(
            self.evaluation,
            GeometryEvaluationRun,
        ):
            raise TypeError("evaluation must be GeometryEvaluationRun or None")
        if self.failure is not None and not isinstance(
            self.failure,
            BatchEvaluationFailure,
        ):
            raise TypeError("failure must be BatchEvaluationFailure or None")
        if (self.evaluation is None) == (self.failure is None):
            raise ValueError(
                "a batch member requires exactly one evaluation or failure"
            )


@dataclass(frozen=True, slots=True)
class BatchEvaluationResult:
    """Ordered batch members without persistence, aggregation, or ranking."""

    request: BatchEvaluationRequest
    members: tuple[BatchEvaluationMember, ...]
    execution_version: int = field(
        default=BATCH_EVALUATION_EXECUTION_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.request, BatchEvaluationRequest):
            raise TypeError("request must be BatchEvaluationRequest")
        if isinstance(self.members, (str, bytes, bytearray)):
            raise TypeError("members must be an iterable of batch members")
        try:
            members = tuple(self.members)
        except TypeError as error:
            raise TypeError("members must be an iterable of batch members") from error
        if not all(isinstance(member, BatchEvaluationMember) for member in members):
            raise TypeError("members must contain only BatchEvaluationMember values")
        if len(members) != len(self.request.candidates):
            raise ValueError("members must contain one record per requested candidate")
        for candidate_index, (member, candidate) in enumerate(
            zip(members, self.request.candidates, strict=True)
        ):
            if member.candidate_index != candidate_index:
                raise ValueError("member indices must match request order exactly")
            if member.candidate is not candidate:
                raise ValueError("members must retain the exact requested candidates")
        object.__setattr__(self, "members", members)


CandidateEvaluator: TypeAlias = Callable[
    [BatchEvaluationCandidate],
    GeometryEvaluationRun,
]


def execute_evaluation_batch(
    request: BatchEvaluationRequest,
    *,
    evaluator: CandidateEvaluator,
) -> BatchEvaluationResult:
    """Evaluate every candidate sequentially while retaining callback failures.

    Returned invalid ``GeometryEvaluationRun`` values are completed results and
    remain evaluations. Ordinary callback exceptions become member failures and
    later candidates continue. Wrong return types or reproducibility records that
    do not match their requested candidate are API-contract violations and raise.
    """
    if not isinstance(request, BatchEvaluationRequest):
        raise TypeError("request must be BatchEvaluationRequest")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")

    members: list[BatchEvaluationMember] = []
    for candidate_index, candidate in enumerate(request.candidates):
        try:
            evaluation = evaluator(candidate)
        except Exception as error:
            members.append(
                BatchEvaluationMember(
                    candidate_index=candidate_index,
                    candidate=candidate,
                    evaluation=None,
                    failure=BatchEvaluationFailure.from_exception(error),
                )
            )
            continue
        if not isinstance(evaluation, GeometryEvaluationRun):
            raise TypeError("evaluator must return GeometryEvaluationRun")
        _validate_evaluation_correspondence(
            evaluation,
            candidate=candidate,
        )
        members.append(
            BatchEvaluationMember(
                candidate_index=candidate_index,
                candidate=candidate,
                evaluation=evaluation,
                failure=None,
            )
        )

    return BatchEvaluationResult(request=request, members=tuple(members))


def _validate_evaluation_correspondence(
    evaluation: GeometryEvaluationRun,
    *,
    candidate: BatchEvaluationCandidate,
) -> None:
    reproducibility = evaluation.reproducibility
    if reproducibility is None:
        return
    expected_geometry_id = exact_geometry_id(candidate.geometry)
    if reproducibility.geometry_id != expected_geometry_id:
        raise ValueError(
            "evaluation reproducibility geometry_id does not match the candidate"
        )
    if reproducibility.seed != candidate.evaluation_seed:
        raise ValueError(
            "evaluation reproducibility seed does not match evaluation_seed"
        )
    if exact_model_parameter_set_id(reproducibility.model_parameters) != (
        exact_model_parameter_set_id(candidate.model_parameters)
    ):
        raise ValueError(
            "evaluation reproducibility model_parameters do not match the candidate"
        )
