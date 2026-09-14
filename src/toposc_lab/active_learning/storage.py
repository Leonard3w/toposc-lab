"""Single-writer atomic exact-label updates with simulation-identity deduplication."""

from __future__ import annotations

from pathlib import Path

from toposc_lab.active_learning.exact import Verification
from toposc_lab.active_learning.pool import Candidate
from toposc_lab.data import ExactPhysicsDataset, load_dataset, save_dataset, validate_dataset_record


def append_verified(path: str | Path, verification: Verification) -> bool:
    """Reject failures and repeated simulations even when timestamps/seeds differ."""
    if verification.record is None or verification.failure is not None:
        raise ValueError("only successful exact verification can be stored")
    record = verification.record
    validate_dataset_record(record).raise_for_errors()
    if (
        Candidate(record.geometry, record.model).candidate_id != verification.candidate_id
        or record.provenance.seed != verification.seed
    ):
        raise ValueError("verification identity mismatch")
    destination = Path(path)
    dataset = load_dataset(destination) if destination.exists() else ExactPhysicsDataset(())
    if any(
        Candidate(r.geometry, r.model).candidate_id == verification.candidate_id
        for r in dataset.records
    ):
        return False
    save_dataset(
        destination, ExactPhysicsDataset((*dataset.records, record)), overwrite=destination.exists()
    )
    return True
