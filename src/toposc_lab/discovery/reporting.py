"""Compact reports and Pareto views derived exclusively from stored exact records."""

from pathlib import Path
from typing import Any, cast

import numpy as np

from toposc_lab.data import load_dataset, record_to_dict
from toposc_lab.discovery.config import DiscoveryConfig
from toposc_lab.discovery.storage import atomic_json, read_json
from toposc_lab.generative.space import structural_distance


def write_reports(directory: Path, config: DiscoveryConfig, completed: int) -> dict[str, Any]:
    records = load_dataset(directory / "dataset.json").records
    leaderboard: list[dict[str, Any]] = []
    report_dir = directory / "reports"
    report_dir.mkdir(exist_ok=True)
    for record in records:
        metrics = record.observables[0].values
        row: dict[str, Any] = {
            "record_id": record.record_id,
            "geometry_id": record.geometry.exact_id,
            "kind": "exact",
            "cycle": record.provenance.runtime["discovery_cycle"],
            "quality": metrics["quality"],
            "success": metrics["success"],
            "boundary_weight": metrics["boundary_weight"],
            "robustness_mean": record.robustness[0].statistics["quality_mean"],
            "topology": record.topology[0].validity.value,
            "majorana_claim": False,
            "finite_size": "unavailable",
        }
        leaderboard.append(row)
        name = record.geometry.exact_id.rsplit(":", 1)[-1]
        geometry = record.geometry.to_geometry()
        atomic_json(
            report_dir / f"{name}.json",
            {
                "record": record_to_dict(record),
                "coordinates": geometry.coordinates.tolist()
                if geometry.coordinates is not None
                else None,
                "edges": [[e.source, e.target] for e in geometry.edges],
            },
        )
        (report_dir / f"{name}.md").write_text(
            f"# Exact candidate {name}\n\n"
            f"Record: `{record.record_id}`. Cycle: {row['cycle']}.\n\n"
            f"Quality: {row['quality']:.9f}; frozen threshold: {config.success_threshold}. "
            f"Boundary weight: {row['boundary_weight']:.6f}.\n\n"
            f"Topology: {row['topology']}; center localizer evidence at three kappas. "
            "All raw indices, gaps and disagreement are retained in the JSON.\n\n"
            "Majorana: diagnostics only; no separated zero-mode claim. "
            "Finite-size/family scaling: unavailable in the fixed-size stratum.\n\n"
            f"Onsite-disorder mean exact quality: {row['robustness_mean']:.9f}; "
            f"{config.disorder_samples} realizations, width {config.onsite_width}. "
            "The JSON includes every realization and Wilson uncertainty for success.\n\n"
            f"[Complete reproducible record]({name}.json) contains the geometry archive "
            "(coordinates/edges), model, spectrum, raw diagnostics, ensemble seeds and provenance. "
            "The campaign source.zip and manifest freeze executable source and configuration.\n",
            encoding="utf-8",
        )
    objectives = ("quality", "boundary_weight", "robustness_mean")
    for row in leaderboard:
        row["pareto"] = not any(
            all(other[key] >= row[key] for key in objectives)
            and any(other[key] > row[key] for key in objectives)
            for other in leaderboard
        )
        row["alternate_quality"] = row["quality"] * row["boundary_weight"]
    leaderboard.sort(key=lambda row: (-row["quality"], row["geometry_id"]))
    atomic_json(directory / "leaderboard.json", leaderboard)
    plans = [read_json(directory / f"cycle-{i:04d}" / "plan.json") for i in range(completed)]
    attempts = [read_json(p) for p in sorted((directory / "attempts").glob("*.json"))]
    outcomes = [read_json(p) for p in sorted(directory.glob("cycle-*/candidate-*.json"))]
    distances = [
        structural_distance(a.geometry.to_geometry(), b.geometry.to_geometry())
        for i, a in enumerate(records)
        for b in records[i + 1 :]
    ]
    curves = []
    best = 0.0
    for cycle in range(completed):
        values = [
            float(cast(float, r.observables[0].values["quality"]))
            for r in records
            if r.provenance.runtime["discovery_cycle"] == cycle
        ]
        previous = best
        best = max([best, *values])
        curves.append(
            {
                "cycle": cycle,
                "best_exact": best,
                "improvement": best - previous,
                "exact_candidates": len(values),
            }
        )
    generated = sum(p["audit"]["attempted"] for p in plans)
    summary = {
        "schema_version": 1,
        "stratum": config.stratum,
        "generator": config.resolved_generator,
        "completed_cycles": completed,
        "raw_generated": generated,
        "valid_pool_candidates": sum(len(p["geometries"]) for p in plans),
        "selected_candidates": sum(len(p["selected_indices"]) for p in plans),
        "exact_base_completed": sum(
            read_json(p)["status"] == "complete" for p in directory.glob("cycle-*/base-*.json")
        ),
        "validated_dataset_records": len(records),
        "invalid": sum(p["audit"]["invalid"] for p in plans),
        "duplicate": sum(p["audit"]["duplicate"] for p in plans),
        "selected_ood": sum(
            p["common_ood"][p["candidate_ids"][i]]["is_ood"]
            for p in plans
            for i in p["selected_indices"]
        ),
        "selected_ood_reference_unavailable": sum(
            p["common_ood"][p["candidate_ids"][i]]["reason"] == "reference_unavailable"
            for p in plans
            for i in p["selected_indices"]
        ),
        "predictions": sum(len(p["predictions"]) for p in plans),
        "training_events": [p["training"] for p in plans if p["training"] is not None],
        "exact_attempts": len(attempts),
        "exact_attempt_cap": config.exact_attempt_cap,
        "failed_attempts": sum(a["status"] == "failed" for a in attempts),
        "interrupted_or_unfinalized_attempts": sum(a["status"] == "started" for a in attempts),
        "candidate_failures": [o for o in outcomes if o["status"] != "complete"],
        "recorded_exact_wall_seconds": sum(a.get("wall_seconds", 0.0) for a in attempts),
        "completed_cycle_wall_seconds": sum(
            read_json(directory / f"cycle-{i:04d}" / "commit.json")["wall_seconds_this_invocation"]
            for i in range(completed)
        ),
        "timing_caution": "Interrupted time is unavailable; measured totals are lower bounds after crashes.",
        "bdg_dimension": 72,
        "localizer_dimension": 144,
        "cubic_work_proxy_attempts": len(attempts) * (72**3 + 3 * 144**3),
        "mean_pairwise_distance": float(np.mean(distances)) if distances else None,
        "minimum_pairwise_distance": min(distances) if distances else None,
        "successes": sum(bool(r.observables[0].values["success"]) for r in records),
        "best": leaderboard[:3],
        "curves": curves,
        "behavior": "insufficient_cycles"
        if completed < 2
        else "improves"
        if any(c["improvement"] > 1e-10 for c in curves[1:])
        else "stagnates",
        "scientific_claim": "Finite localizer evidence only; no Majorana or thermodynamic claim.",
    }
    atomic_json(directory / "summary.json", summary)
    if config.plots and records:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        axes[0].plot([c["cycle"] + 1 for c in curves], [c["best_exact"] for c in curves], "o-")
        axes[0].axhline(
            config.success_threshold, color="gray", linestyle="--", label="Frozen success"
        )
        axes[0].set(xlabel="Completed cycle", ylabel="Best exact finite quality", ylim=(0, 0.22))
        axes[0].legend()
        axes[1].scatter(
            [r["quality"] for r in leaderboard],
            [r["robustness_mean"] for r in leaderboard],
            c=[r["cycle"] for r in leaderboard],
            cmap="viridis",
        )
        axes[1].set(xlabel="Clean exact quality", ylabel="Onsite-disorder mean quality")
        fig.tight_layout()
        fig.savefig(directory / "discovery.png", dpi=150)
        plt.close(fig)
    return summary
