"""Versioned single-writer checkpoints with write-ahead exact-attempt accounting."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from toposc_lab.active_learning.cycle import CycleConfig, run_cycle
from toposc_lab.active_learning.exact import ExactEvaluator, verify_exact
from toposc_lab.active_learning.pool import Candidate, CandidateSpace
from toposc_lab.data import (
    DatasetRecord,
    ExactPhysicsDataset,
    record_from_dict,
    record_to_dict,
    validate_dataset,
)
from toposc_lab.ml import dataset_fingerprint


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    cycle: CycleConfig
    seed: int
    cycles: int
    exact_budget: int
    evaluator_identifier: str
    code_version: str

    def __post_init__(self) -> None:
        if self.seed < 0 or self.cycles < 1 or self.exact_budget < 1:
            raise ValueError("invalid campaign seed/cycles/budget")
        if self.cycles * self.cycle.acquisition.batch_size > self.exact_budget:
            raise ValueError("configured cycles exceed acquisition exact budget")
        if not self.evaluator_identifier or not self.code_version:
            raise ValueError("explicit evaluator and source version required")


@dataclass(frozen=True, slots=True)
class CampaignResult:
    dataset: ExactPhysicsDataset
    completed_cycles: int
    exact_attempts: int
    history: tuple[dict[str, Any], ...]
    checkpoint_path: Path


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish a checksummed, fsynced snapshot. Temporary files are never read."""
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))
    envelope = json.dumps(
        {"sha256": hashlib.sha256(encoded.encode()).hexdigest(), "payload": payload},
        sort_keys=True,
        allow_nan=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(envelope)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_snapshot(path: Path) -> dict[str, Any]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope["payload"]
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))
    if hashlib.sha256(encoded.encode()).hexdigest() != envelope["sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    return payload


def run_campaign(
    space: CandidateSpace,
    initial: ExactPhysicsDataset,
    *,
    config: CampaignConfig,
    evaluator: ExactEvaluator,
    checkpoint_path: str | Path,
    reserved_records: tuple[DatasetRecord, ...] = (),
    stop_after: int | None = None,
) -> CampaignResult:
    """Resume committed cycles and completed exact attempts without duplicate labels.

    One process owns a checkpoint. An interrupted attempt without a durable result
    is charged and marked failed on resume, never silently repeated. Initial exact
    labels are an explicit additional cost, not included in ``exact_budget``.
    """
    path = Path(checkpoint_path)
    validate_dataset(initial).raise_for_errors()
    contract = {
        "config": asdict(config),
        "initial": dataset_fingerprint(initial),
        "reserved": dataset_fingerprint(ExactPhysicsDataset(reserved_records)),
        "space": [c.candidate_id for c in space.candidates],
        "constraints": [space.minimum_sites, space.maximum_sites, space.require_connected],
    }
    contract = json.loads(json.dumps(contract))
    if path.exists():
        state = read_snapshot(path)
        if state.get("schema_version") != 1 or state["contract"] != contract:
            raise ValueError("checkpoint configuration/data/source contract mismatch")
    else:
        state = {
            "schema_version": 1,
            "contract": contract,
            "completed_cycles": 0,
            "exact_attempts": 0,
            "history": [],
            "pending": {},
            "records": [record_to_dict(r) for r in initial.records],
        }
        atomic_json(path, state)
    dataset = ExactPhysicsDataset(tuple(record_from_dict(r) for r in state["records"]))
    validate_dataset(dataset).raise_for_errors()
    limit = config.cycles if stop_after is None else min(config.cycles, stop_after)
    for cycle_index in range(state["completed_cycles"], limit):

        def journaled(candidate: Candidate, exact_seed: int) -> DatasetRecord:
            key = candidate.candidate_id
            if key not in state["pending"]:
                if state["exact_attempts"] >= config.exact_budget:
                    raise RuntimeError("exact simulation budget exhausted")
                state["exact_attempts"] += 1
                state["pending"][key] = {"seed": exact_seed, "status": "started"}
                atomic_json(path, state)
                result = verify_exact(candidate, seed=exact_seed, evaluator=evaluator)
                state["pending"][key] = {
                    "seed": exact_seed,
                    "status": "complete",
                    "failure": result.failure,
                    "record": None if result.record is None else record_to_dict(result.record),
                }
                atomic_json(path, state)
            saved = state["pending"][key]
            if saved["seed"] != exact_seed:
                raise ValueError("pending exact seed mismatch")
            if saved["status"] == "started":
                raise RuntimeError("interrupted exact attempt; budget charged, no label")
            if saved["record"] is None:
                raise RuntimeError(saved["failure"])
            return record_from_dict(saved["record"])

        attempted = frozenset(
            v["candidate_id"] for h in state["history"] for v in h["verifications"]
        )
        result = run_cycle(
            space,
            dataset,
            config=config.cycle,
            seed=config.seed,
            cycle_index=cycle_index,
            evaluator=journaled,
            reserved_records=reserved_records,
            attempted_ids=attempted,
        )
        history = {
            "cycle_index": cycle_index,
            "pool_ids": result.pool_ids,
            "predictions": [asdict(p) for p in result.predictions],
            "selected": [asdict(s) for s in result.selected],
            "verifications": [
                {
                    "candidate_id": v.candidate_id,
                    "seed": v.seed,
                    "record_id": None if v.record is None else v.record.record_id,
                    "failure": v.failure,
                }
                for v in result.verifications
            ],
            "training_fingerprint": result.surrogate.training_fingerprint,
            "prediction_failure": result.prediction_failure,
        }
        state["history"].append(json.loads(json.dumps(history)))
        state["records"] = [record_to_dict(r) for r in result.dataset.records]
        state["completed_cycles"] = cycle_index + 1
        state["pending"] = {}
        atomic_json(path, state)
        dataset = result.dataset
    return CampaignResult(
        dataset, state["completed_cycles"], state["exact_attempts"], tuple(state["history"]), path
    )
