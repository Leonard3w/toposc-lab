"""Read-only artifact, dataset-feasibility and independent-reproduction audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from toposc_lab.data import load_dataset, validate_dataset_record
from toposc_lab.generative.benchmark import scientific_record, write_json
from toposc_lab.generative.generators import METHODS
from toposc_lab.generative.physics import candidate
from toposc_lab.generative.space import GeometrySearchSpace, structural_distance


def feasibility(root: Path) -> dict:
    records = []
    files = []
    for arm in ("active", "random", "evolutionary"):
        for p in sorted((root / "results/phase13r-gate").glob(f"*/{arm}.json")):
            files.append(p)
            records.extend(load_dataset(p).records)
    if not files:
        raise ValueError("Phase-13R evidence not found")
    return {
        "scope": "Phase-13R datasets; supplementary Phase-11/12 and Phase-10 source/report audit",
        "files": len(files),
        "exact_records": len(records),
        "unique_geometry_ids": len({r.geometry.exact_id for r in records}),
        "family_fingerprints": len({r.geometry.family_fingerprint for r in records}),
        "models": dict(Counter(r.model.model_name for r in records)),
        "sizes": sorted({r.geometry.to_geometry().n_sites for r in records}),
        "compatible_36_site_planar_records": sum(
            r.geometry.to_geometry().n_sites == 36
            and r.geometry.to_geometry().embedding_dimension == 2
            and r.model.model_name == "chiral_p_wave"
            for r in records
        ),
        "decision": "NO-GO learned graph model; no compatible exact training corpus established",
        "supplementary": [
            "Phase 11 uses three Kitaev references; Phase 12 uses 30 clean chains.",
            "Phase-10 primary search uses 64-site/112-edge graphs, outside this 36-site stratum.",
            "Constructive REINFORCE demo rewards are geometric, not exact physics labels.",
            "Final Phase-14 gate labels cannot select or train a generator architecture.",
            "Historical information transfer is unvalidated, not declared impossible.",
        ],
        "inputs": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        },
    }


def audit(directory: Path, repeat: Path | None = None) -> dict:
    report = json.loads((directory / "report.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    inventory = json.loads((directory / "inventory.json").read_text())
    for relative, expected in inventory.items():
        assert hashlib.sha256((directory / relative).read_bytes()).hexdigest() == expected, relative
    entries = [
        json.loads(line) for line in (directory / "exact-attempts.jsonl").read_text().splitlines()
    ]
    starts = [e for e in entries if e["status"] == "started"]
    ends = [e for e in entries if e["status"] == "complete"]
    assert len(entries) == 2 * len(starts) == 2 * len(ends)
    assert len(ends) == report["exact_calls"] == manifest["expected_exact_calls"]
    assert [e["call"] for e in starts] == list(range(1, len(starts) + 1))
    assert [e["call"] for e in starts] == [e["call"] for e in ends]
    for entry in ends:
        (record,) = load_dataset(directory / entry["path"]).records
        validate_dataset_record(record).raise_for_errors()
        assert record.record_id == entry["record_id"]
        assert (
            candidate(
                record.geometry.to_geometry(), record.model.parameters["chemical_potential"]
            ).candidate_id
            == entry["candidate_id"]
        )
        assert record.provenance.seed == entry["seed"]
        assert record.provenance.runtime["source_sha256"] == manifest["source_sha256"]
        assert record.provenance.runtime["protocol_sha256"] == manifest["protocol_sha256"]
    ood_exploitation = 0
    prediction_count = 0
    for trial in report["trials"]:
        warm_edges = None
        for method in METHODS:
            arm = directory / str(trial["seed"]) / method
            records = [load_dataset(p).records[0] for p in sorted(arm.glob("search-*.json"))]
            assert len(records) == report["budget"]
            geometries = [r.geometry.to_geometry() for r in records]
            for i, g in enumerate(geometries):
                assert not GeometrySearchSpace().reasons(g)
                assert all(structural_distance(g, old) > 0.06 for old in geometries[:i])
            warm = [r.geometry.exact_id for r in records[:4]]
            if warm_edges is None:
                warm_edges = warm
            assert warm == warm_edges
            for batch in range(1, report["budget"] // 4):
                assessment = json.loads((arm / f"selected-ood-{batch}.json").read_text())
                assert assessment["stage"] == "before_exact"
                assert assessment["reference_record_ids"] == [r.record_id for r in records[:4]]
                assert assessment["candidate_ids"] == [
                    candidate(g).candidate_id for g in geometries[batch * 4 : (batch + 1) * 4]
                ]
                assert (
                    assessment["is_ood"] == trial[method]["ood_flags"][batch * 4 : (batch + 1) * 4]
                )
            assert trial[method]["qualities"] == [
                r.observables[0].values["quality"] for r in records
            ]
            winner = max(records, key=lambda r: r.observables[0].values["quality"])
            confirmation = load_dataset(arm / "confirmation-0.json").records[0]
            assert scientific_record(winner) == scientific_record(confirmation)
            proposals = json.loads((arm / "proposal-audit.json").read_text())
            events = proposals["events"]
            assert len(events) == trial[method]["proposal_attempts"]
            assert (
                sum(e["status"] == "invalid" for e in events) == trial[method]["invalid_proposals"]
            )
            assert (
                sum(e["status"] == "duplicate" for e in events)
                == trial[method]["duplicate_proposals"]
            )
            labels = {r.record_id: i for i, r in enumerate(records)}
            positions = {candidate(g).candidate_id: i for i, g in enumerate(geometries)}
            for p in proposals["predictions"]:
                prediction_count += 1
                assert p["kind"] == "predicted"
                assert set(p["training_record_ids"]) <= labels.keys()
                if p["strategy"]:
                    position = positions[p["candidate_id"]]
                    assert max(labels[r] for r in p["training_record_ids"]) < (position // 4) * 4
                ood_exploitation += p["is_ood"] and p["strategy"] == "exploitation"
    assert ood_exploitation == 0
    reproduced = 0
    if repeat:
        other = json.loads((repeat / "manifest.json").read_text())
        assert other["source_sha256"] == manifest["source_sha256"]
        assert other["protocol_sha256"] == manifest["protocol_sha256"]
        assert other["timestamp_utc"] != manifest["timestamp_utc"]
        for method in METHODS:
            for p in sorted((repeat / "14101" / method).glob("*.json")):
                if p.name.startswith(("search-", "confirmation-")):
                    (first,) = load_dataset(directory / "14101" / method / p.name).records
                    (second,) = load_dataset(p).records
                    assert scientific_record(first) == scientific_record(second)
                    reproduced += 1
    return {
        "passed": True,
        "inventory_files": len(inventory),
        "exact_records": len(ends),
        "prediction_count": prediction_count,
        "ood_exploitation": ood_exploitation,
        "reproduced_records": reproduced,
        "source_sha256": manifest["source_sha256"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--repeat", type=Path)
    parser.add_argument("--feasibility", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = feasibility(Path.cwd()) if args.feasibility else audit(args.directory, args.repeat)
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "inputs"}, indent=2))
