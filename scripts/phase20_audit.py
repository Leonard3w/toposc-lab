"""Read-only compatibility audit; no numerical evaluations or historical writes."""
from __future__ import annotations

import argparse
import json
import tempfile
from hashlib import sha256
from pathlib import Path

from toposc_lab.research.config import ExperimentConfig
from toposc_lab.research.service import ResearchService
from toposc_lab.research.storage import ResearchStore
from toposc_lab.research.studio_config import preview_config


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(directory: Path) -> dict:
    store = ResearchStore(directory)
    before = file_digest(store.path)
    counts = {}
    with store.connect(readonly=True) as db:
        db.execute("BEGIN")
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"SQLite integrity failure: {directory}")
        for row in db.execute("SELECT kind,id,payload,checksum FROM objects"):
            if sha256(row["payload"].encode()).hexdigest() != row["checksum"]:
                raise ValueError(f"Payload checksum failure: {directory}/{row['kind']}/{row['id']}")
            counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    config, manifest = store.get("config"), store.get("manifest")
    config_matches = ExperimentConfig.from_dict(config).fingerprint == manifest["config_sha256"]
    if not config_matches:
        raise ValueError(f"Historical configuration hash changed: {directory}")
    snapshot = ResearchService.snapshot(directory, lightweight=True)
    settings = store.get("validation_settings")
    cohort = store.get("validation_cohort")
    fresh_seed = max(settings["seeds"]) + 1
    with tempfile.TemporaryDirectory(prefix="toposc-phase20-read-audit-") as temporary:
        destination = Path(temporary) / "follow-up"
        following = ResearchService.follow_up(
            directory, [cohort["candidates"][0]["id"]], widths=settings["widths"][:1],
            seeds=[fresh_seed, fresh_seed + 1], output_directory=str(destination),
        )
        preview = preview_config(following)
        cloned = ResearchService.clone_config(directory, output_directory=str(destination))
        cloned_preview = preview_config(cloned)
        if preview["errors"] or cloned_preview["errors"] or destination.exists():
            raise ValueError("Read-only follow-up/clone configuration audit failed")
    after = file_digest(store.path)
    if before != after:
        raise ValueError(f"Historical database changed during read audit: {directory}")
    return {
        "directory": str(directory), "database_sha256": before,
        "database_unchanged_by_read": before == after, "sqlite_integrity": integrity,
        "all_payload_checksums_valid": True, "object_counts": counts,
        "config_sha256": manifest["config_sha256"], "legacy_config_hash_preserved": config_matches,
        "browser_candidate_count": len(snapshot["candidates"]),
        "browser_exact_attempts": snapshot["state"]["exact_evaluations"],
        "browser_completed_evaluations": snapshot["state"]["completed_evaluations"],
        "followup_planned_evaluations": preview["planned_evaluations"],
        "clone_candidate_count": len(cloned["space"]["candidates"]),
        "followup_and_clone_validated_without_start": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/decisions/phase20_compatibility_audit.json"))
    args = parser.parse_args()
    report = {
        "new_exact_evaluations": 0,
        "scope": "Historical checksums and read adapters; no re-evaluation of stored eigensystems.",
        "source_baseline_commit": "cf3806a",
        "generator_archive_comparison": {
            "seed": 190001, "all_equal_to_HEAD_before_phase20": True,
            "hard_core_planar_graph": "10d32f2a28383d292af5201bb3abd74e44dc0d5dad6430b95cf4415d3117b8a2",
            "hard_core_planar_reference": "ecae2159667b120f3d4c8653fb7864b531f06486229971c44246816a95ab1147",
            "constrained_random_embedded_graph": "a587a878b08c602e37c3a9c2ff8924152b73a4c3deb00a6c7ba538fb8986eb78",
            "method": "Manual comparison to git show HEAD:.../hard_core_planar.py on 2026-09-27; not rerun by this script.",
        },
        "studies": [audit(Path(path)) for path in (
            "results/phase18-confirmation", "results/phase19-exploration-v2")],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2) + "\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output), "studies":len(report["studies"]),
                      "exact_records":sum(s["object_counts"].get("exact_result",0) for s in report["studies"])}))


if __name__ == "__main__":
    main()
