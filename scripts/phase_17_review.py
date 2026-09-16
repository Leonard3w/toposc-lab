"""Read-only audit of persisted workbench evidence; no numerical physics calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from toposc_lab.data import record_from_dict, validate_dataset_record
    from toposc_lab.discovery.storage import read_json
    from toposc_lab.research.engine import ResearchEngine
    from toposc_lab.research.provenance import verify
    from toposc_lab.research.service import ResearchService
    from toposc_lab.research.space import geometry_from_payload
    from toposc_lab.research.storage import ResearchStore, atomic_text

    store = ResearchStore(args.directory)
    store.integrity_check()
    snapshot = ResearchService.snapshot(args.directory)
    verify(args.directory, snapshot["manifest"])
    before = len(store.attempts())
    engine = ResearchEngine(args.directory)
    stages = store.all("exact_result")
    for stage in stages:
        if "dataset_record" in stage:
            validate_dataset_record(record_from_dict(stage["dataset_record"])).raise_for_errors()
    checked = 0
    complete = {}
    for candidate in snapshot["candidates"]:
        if not candidate.get("observed"):
            continue
        assert not engine.space.validate(geometry_from_payload(candidate["geometry"]))
        summary = engine.evaluator.summarize(candidate["exact_results"])
        assert summary["score"] == candidate["score"]
        assert summary["validation_state"] == candidate["validation_state"]
        assert not summary["phase_claim"] and not summary["majorana_claim"]
        assert summary["critical_disorder_strength"]["value"] is None
        assert candidate["score"] is None or candidate["origin"] == "exact"
        prediction = candidate.get("prediction")
        if prediction:
            assert candidate["id"] not in prediction["training_ids"]
            for identity in prediction["training_ids"]:
                training = store.get("candidate", identity)
                assert training["exact_evaluation_count"] < candidate["exact_evaluation_count"]
        complete[candidate["id"]] = candidate
        checked += 1
    for elite in snapshot["archive"]:
        assert elite["id"] in complete
        assert elite["score"] == complete[elite["id"]]["score"]
        assert elite["origin"] == "exact"
    checkpoints = list((args.directory / "checkpoints").glob("*.json"))
    for path in checkpoints:
        assert read_json(path) == store.get("checkpoint", path.stem)
    assert (args.directory / "final_report.md").read_text(encoding="utf-8") == snapshot["report"]
    assert len(store.attempts()) == before
    report = {"gate": "PASS", "directory": str(args.directory.resolve()),
              "checked_stage_records": len(stages), "checked_complete_candidates": checked,
              "checked_archive_cells": len(snapshot["archive"]), "checked_checkpoints": len(checkpoints),
              "charged_attempts": before, "additional_exact_calls": 0,
              "source_sha256": snapshot["manifest"]["source_sha256"],
              "checks": ["SQLite integrity and payload checksums", "Existing dataset validators",
                         "Geometry constraints", "Pure summary regeneration", "No training-label leakage",
                         "Archive exact provenance", "Checkpoint envelopes", "Final report export"]}
    atomic_text(args.output, json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
