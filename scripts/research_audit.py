"""Read-only audit of an interrupted research run; never calls an exact evaluator.

Example: python -B scripts/research_audit.py EXPERIMENT --output results/run-audit
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from toposc_lab.research.config import ExperimentConfig
from toposc_lab.research.descriptors import compute_descriptors
from toposc_lab.research.physics import create_evaluator, provenance_from_manifest
from toposc_lab.research.reporting import search_assessment, structural_comparison
from toposc_lab.research.space import SPACE_REGISTRY, geometry_from_payload
from toposc_lab.research.storage import ResearchStore


def audit(directory: Path, output: Path) -> dict:
    directory, output = directory.resolve(), output.resolve()
    if output == directory or output.is_relative_to(directory):
        raise ValueError("Audit output must be outside the archived experiment")
    store = ResearchStore(directory)
    rows, stages, checksums = [], {}, []
    states: Counter = Counter()
    with store.connect(readonly=True) as db:
        db.execute("BEGIN")

        def get(kind):
            return store.decode(db.execute(
                "SELECT * FROM objects WHERE kind=? AND id='current'", (kind,)).fetchone())

        config, manifest, state = get("config"), get("manifest"), get("state")
        protocol = ExperimentConfig(**config).physics_protocol()
        evaluator = create_evaluator(protocol, provenance=provenance_from_manifest(manifest, protocol))
        space = SPACE_REGISTRY[config["geometry_space"]](**config["space"])
        attempts = [dict(row) for row in db.execute("SELECT * FROM attempts ORDER BY number")]
        for row in db.execute("SELECT * FROM objects WHERE kind='candidate' ORDER BY rowid"):
            candidate = store.decode(row)
            checksums.append([row["id"], row["checksum"]])
            states[candidate["validation_state"]] += 1
            if not candidate.get("observed") or candidate.get("score") is None:
                continue
            geometry = geometry_from_payload(candidate["geometry"])
            if space.validate(geometry):
                raise ValueError(f"Invalid stored geometry: {candidate['id']}")
            computed = compute_descriptors(geometry)
            if computed.keys() != candidate["descriptors"].keys() or any(
                not np.isclose(value, candidate["descriptors"][key], rtol=1e-10, atol=1e-12)
                for key, value in computed.items()
            ):
                raise ValueError(f"Descriptor mismatch: {candidate['id']}")
            results = candidate["exact_results"]
            journal = {r["id"].removeprefix(candidate["id"] + ":"): store.decode(r)
                       for r in db.execute(
                           "SELECT * FROM objects WHERE kind='exact_result' AND id GLOB ?",
                           (candidate["id"] + ":*",))}
            if journal != results:
                raise ValueError(f"Stage journal mismatch: {candidate['id']}")
            # Summarization reuses persisted evidence and performs no diagonalization.
            summary = evaluator.summarize(results)
            if (summary["score"] != candidate["score"] or not summary["complete"]
                    or summary["validation_state"] != candidate["validation_state"]):
                raise ValueError(f"Stored summary mismatch: {candidate['id']}")
            stages[candidate["id"]] = [
                {"width": stage["width"], "seed": stage["seed"],
                 "quality": results[stage["key"]]["metrics"]["quality"]}
                for stage in evaluator.plan() if stage["kind"] == "disorder"]
            rows.append({key: candidate[key] for key in (
                "id", "origin", "validation_state", "baseline", "family", "score",
                "generation", "descriptors", "raw_metrics", "robustness", "geometry")}
                | {"acquisition": candidate.get("acquisition", {}),
                   "exact_evaluation_count": candidate.get("exact_evaluation_count")})
    searched = [c for c in rows if not c["baseline"]]
    best = max(searched, key=lambda c: (c["score"], c["id"]), default=None)
    comparisons = []
    if best:
        best_stages = {(s["width"], s["seed"]): s["quality"] for s in stages[best["id"]]}
        for baseline in (c for c in rows if c["baseline"]):
            differences = [{**s, "difference": best_stages[s["width"], s["seed"]] - s["quality"]}
                           for s in stages[baseline["id"]]]
            comparisons.append({"baseline": baseline["id"], "matched_differences": differences})
    snapshot = {"candidates": rows, "config": config}
    report = {
        "experiment_id": manifest["experiment_id"], "source_sha256": manifest["source_sha256"],
        "stored_state": state, "candidate_states": dict(states),
        "attempt_statuses": dict(Counter(a["status"] for a in attempts)),
        "charged_attempts": len(attempts), "exact_seconds": sum(a["seconds"] or 0 for a in attempts),
        "complete_candidates": len(rows), "complete_searched": len(searched),
        "last_exported_state": json.loads((directory / "state.json").read_text())["payload"],
        "candidate_checksum_digest": hashlib.sha256(json.dumps(checksums).encode()).hexdigest(),
        "attempt_digest": hashlib.sha256(json.dumps(attempts, sort_keys=True).encode()).hexdigest(),
        "verified": ["all candidate payload checksums", "complete geometry constraints",
                     "complete descriptor recomputation", "complete scores and validation states",
                     "complete candidate stages match checksummed exact-result journal"],
        "additional_exact_calls": 0,
        "structural_comparison": structural_comparison(snapshot),
        "search_assessment": search_assessment(snapshot),
        "paired_comparisons": comparisons,
        "scope": "Exploratory same-seed comparisons after adaptive selection; not held-out "
                 "confirmation. No extrapolation to W_c, bulk gap or Majorana modes. "
                 "A stored RUNNING state does not prove a worker is still alive.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["id", "baseline", "generation", "score", "regular_edge_distance", "long_bond_fraction",
                  "boundary_weight", "minimum_abs_energy", "robustness_success_fraction"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows({key: ({**c, **c["descriptors"], **c["raw_metrics"]}).get(key) for key in fields}
                         for c in rows)
    _plot(rows, output / "evidence.png")
    return report


def _plot(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    searched = [c for c in rows if not c["baseline"]]
    if not searched:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    scatter = axes[0].scatter([c["descriptors"]["regular_edge_distance"] for c in searched],
                             [c["score"] for c in searched],
                             c=[c["descriptors"]["long_bond_fraction"] for c in searched],
                             cmap="viridis", s=22, alpha=.8)
    fig.colorbar(scatter, ax=axes[0], label="Long-bond fraction")
    axes[0].set(xlabel="Fraction of regular edges replaced", ylabel="Mean localizer quality")
    axes[1].scatter([c["raw_metrics"]["boundary_weight"] for c in searched],
                    [c["score"] for c in searched], s=18, alpha=.6, label="Searched")
    axes[1].set(xlabel="Clean low-energy boundary weight", ylabel="Mean localizer quality")
    best = max(searched, key=lambda c: c["score"])
    for (c, label), color in zip(
        [(best, "Best searched"), *[(c, c["family"]) for c in rows if c["baseline"]]],
        ["tab:orange", "tab:green", "tab:red"],
    ):
        axes[1].scatter(c["raw_metrics"]["boundary_weight"], c["score"], marker="*", s=130, label=label, color=color)
        groups = c["robustness"]
        axes[2].plot([g["width"] for g in groups], [g["quality_mean"] for g in groups], "o-", label=label, color=color)
    axes[1].legend(fontsize=8)
    axes[2].set(xlabel="Disorder width W (W=0 repeats are deterministic)", ylabel="Mean localizer quality (stored samples)")
    axes[2].legend(fontsize=8)
    fig.suptitle("Interrupted Experiment 002: stored finite-system evidence; adaptive sample")
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.directory, args.output)
    print(json.dumps({key: result[key] for key in
                     ("complete_candidates", "charged_attempts", "additional_exact_calls")}))
