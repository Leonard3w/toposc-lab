"""Exact-evaluator boundary and validation before labels enter training/storage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from toposc_lab.active_learning.pool import Candidate
from toposc_lab.data import DatasetRecord, validate_dataset_record

ExactEvaluator = Callable[[Candidate, int], DatasetRecord]


@dataclass(frozen=True, slots=True)
class Verification:
    candidate_id: str
    seed: int
    record: DatasetRecord | None
    failure: str | None
    exact_attempts: int = 1


def verify_exact(candidate: Candidate, *, seed: int, evaluator: ExactEvaluator) -> Verification:
    """Every selection invokes the configured exact evaluator; never accepts predictions.

    The evaluator owns model compatibility and solver semantics. This boundary
    additionally verifies identity, provenance and dataset scientific contracts.
    Ordinary failures remain auditable attempts and are never training labels.
    """
    try:
        record = evaluator(candidate, seed)
        validate_dataset_record(record).raise_for_errors()
        if Candidate(record.geometry, record.model).candidate_id != candidate.candidate_id:
            raise ValueError("exact evaluator returned a different candidate")
        if record.provenance.seed != seed:
            raise ValueError("exact evaluator returned the wrong seed")
        return Verification(candidate.candidate_id, seed, record, None)
    except Exception as error:  # noqa: BLE001 -- isolate arbitrary user-supplied exact solvers
        return Verification(candidate.candidate_id, seed, None, f"{type(error).__name__}: {error}")
