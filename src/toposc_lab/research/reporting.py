"""Diagnostic reports derived entirely from persisted evidence."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def diagnostics(snapshot: dict[str, Any]) -> dict[str, Any]:
    candidates = snapshot.get("candidates", [])
    exact = [c for c in candidates if c.get("origin") == "exact" and c.get("score") is not None]
    validated = [c for c in exact if c.get("validation_state") in
                 ("ROBUSTNESS_VALIDATED", "FINITE_SIZE_VALIDATED")]
    best = max(exact, key=lambda c: c["score"], default=None)
    best_validated = max(validated, key=lambda c: c["score"], default=None)
    families = Counter(c.get("lineage_family", c.get("family", "unclassified")) for c in exact)
    state = snapshot.get("state", {})
    generated = max(1, state.get("generated", len(candidates)))
    reasons = Counter(c.get("rejection_reason", "") for c in candidates
                      if c.get("validation_state") == "REJECTED")
    duplicates = sum(v for k, v in reasons.items() if "duplicate" in k and "near" not in k)
    near = sum(v for k, v in reasons.items() if "near" in k)
    invalid = sum(reasons.values()) - duplicates - near
    archive = snapshot.get("archive", [])
    bins = snapshot.get("config", {}).get("search", {}).get("archive_bins", [12, 12])
    dimensions = len(snapshot.get("config", {}).get("search", {}).get("behavior_descriptors", [0, 1]))
    total_cells = bins**dimensions if isinstance(bins, int) else int(np.prod(bins))
    coverage = len(archive) / total_cells if total_cells else 0.0
    fractions = np.array(list(families.values()), float) / max(1, len(exact))
    entropy = float(-np.sum(fractions * np.log(fractions))) if len(fractions) else 0.0
    history = snapshot.get("history", [])
    anomalies = []
    if invalid / generated > 0.8:
        anomalies.append("More than 80% of proposals violate geometry constraints")
    concentration = float(max(fractions, default=0.0))
    if len(exact) >= 8 and concentration > 0.8:
        anomalies.append("Possible family concentration: more than 80% in one lineage family")
    if len(history) >= 4 and len({h.get("best_score") for h in history[-4:]}) == 1:
        anomalies.append("No best-score improvement in the last four checkpoints")
    if (duplicates + near) / generated > 0.5:
        anomalies.append("Search revisits previously proposed geometry neighborhoods")
    failures = [c["id"] for c in candidates if c.get("validation_state") == "FAILED"]
    if failures:
        anomalies.append(f"{len(failures)} candidate evaluation failures; excluded from elites")
    boundary_scores = [c["id"] for c in exact if abs(c.get("raw_metrics", {}).get(
        "quality", -1) - 0.2) < 1e-6]
    if boundary_scores:
        anomalies.append("Quality lies very close to the frozen 0.20 threshold; inspect raw results")
    calibration = snapshot.get("surrogate", {}).get("calibration", {})
    if isinstance(calibration, list):
        errors = [r["absolute_error"] for r in calibration]
        calibration = {"sample_count": len(errors),
                       "mae": float(np.mean(errors)) if errors else None,
                       "rmse": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                       "coverage_2sigma": float(np.mean([r["covered_2sigma"] for r in calibration]))
                       if errors else None,
                       "method": "prequential; heuristic coverage, no calibrated-interval claim"}
    return {"best_candidate": best["id"] if best else None,
            "best_score": best["score"] if best else None,
            "best_validated_candidate": best_validated["id"] if best_validated else None,
            "best_validated_score": best_validated["score"] if best_validated else None,
            "exact_candidate_count": len(exact), "archive_coverage": coverage,
            "occupied_cells": len(archive), "unexplored_cells": total_cells - len(archive),
            "diversity": entropy, "family_distribution": dict(families),
            "family_concentration": concentration, "duplicate_rate": duplicates / generated,
            "near_duplicate_rate": near / generated, "invalid_rate": invalid / generated,
            "surrogate_calibration": calibration,
            "surrogate_error": calibration.get("mae", calibration.get("mean_absolute_error")),
            "anomalies": anomalies, "numerical_failures": failures,
            "baseline_comparison": [{"id": c["id"], "family": c.get("family"),
                                     "score": c.get("score"), "raw_metrics": c.get("raw_metrics")}
                                    for c in candidates if c.get("baseline")],
            "exact_evaluations": state.get("exact_evaluations", 0),
            "compute_usage": {k: state.get(k) for k in
                              ("elapsed_seconds", "exact_seconds", "generated")},
            "scientific_claim": "Finite-system evidence only; hypothesis inconclusive"}


def final_report(snapshot: dict[str, Any]) -> str:
    import json

    config, state = snapshot["config"], snapshot["state"]
    summary = diagnostics(snapshot)
    exact = [c for c in snapshot["candidates"] if c.get("origin") == "exact"
             and c.get("score") is not None]
    correlations = {}
    if len(exact) >= 4:
        score = np.array([c["score"] for c in exact])
        for name in exact[0].get("descriptors", {}):
            values = np.array([c["descriptors"].get(name, np.nan) for c in exact], float)
            if np.isfinite(values).all() and np.std(values) > 1e-12 and np.std(score) > 1e-12:
                correlations[name] = float(np.corrcoef(values, score)[0, 1])
    best = sorted(exact, key=lambda c: (-c["score"], c["id"]))[:5]
    def block(value: Any) -> str:
        return "```json\n" + json.dumps(value, indent=2, allow_nan=False) + "\n```\n"
    parts = [f"# {config['name']} — research report\n",
             f"Experiment: {state.get('experiment_id')} · status: {state['status']}\n",
             "## 1. Research question\n\n" + config["question"] + "\n",
             "## 2. Frozen experiment configuration\n\n" + block(config),
             (f"## 3. Search strategy\n\nRegistered strategy: `{config['algorithm']}`. "
             "Surrogate predictions are acquisition aids and are never validation evidence.\n"),
             (f"## 4. Exact-evaluation budget\n\n{state.get('exact_evaluations', 0)} charged "
             f"attempts / {config['exact_budget']}; interrupted and failed attempts are included. "
             "Baseline, confirmation and disorder calls share this cap. Compare algorithms at "
             "matched exact-attempt counts, protocols and seeds.\n"),
             "## 5. Matched baselines\n\n" + block(summary["baseline_comparison"]) +
             "Other geometry families have no matched fixed-site adapter in this experiment; "
             "they are not silently treated as controls.\n",
             "## 6. Best observed candidates\n\n" + block([
                 {k: c.get(k) for k in ("id", "score", "validation_state", "raw_metrics")}
                 for c in best]),
             "## 7. MAP-Elites archive\n\n" + block({k: summary[k] for k in
                 ("archive_coverage", "occupied_cells", "unexplored_cells", "diversity")}),
             "## 8. Geometry–property correlations\n\n" + block(correlations) +
             "Exploratory Pearson associations within an adaptively selected, dependent sample; "
             "no causal or multiple-testing claim. Constant targets yield no correlation.\n",
             "## 9–12. Robustness, topology, Majorana and finite-size evidence\n\n" + block([
                 {"id": c["id"], "validation": c.get("validation_results"),
                  "robustness": c.get("robustness")} for c in best]) +
             "Topology is a finite center-probe localizer diagnostic. Boundary localization and "
             "self-conjugacy diagnostics do not certify chiral Majorana modes. A graph at another "
             "size is not a finite-size continuation without a declared family map.\n",
             "## 13. Surrogate performance\n\n" + block(summary["surrogate_calibration"]) +
             "Calibration is prequential: predictions saved before their exact labels arrived. "
             "Bootstrap uncertainty is a prioritization signal, not a confidence certificate.\n",
             ("## 14. Statistical uncertainty\n\nDisorder success fractions retain Wilson "
             "intervals and sample counts in candidate records. These are finite-protocol "
             "estimates; adaptive selection and shared seeds limit population inference.\n"),
             "## 15. Negative results\n\n" + block({"validated_candidates":
                 sum(c.get("validation_state") in ("ROBUSTNESS_VALIDATED", "FINITE_SIZE_VALIDATED")
                     for c in exact), "failures": summary["numerical_failures"]}),
             "## 16. Anomalies\n\n" + block(summary["anomalies"]),
             ("## 17. Limitations\n\nOnly fixed square sites and the versioned chiral-p-wave "
             "adapter are available initially. The numerical convention is reused from the "
             "validated 36-site protocol; larger systems need scientific benchmarking. One "
             "serial exact worker per experiment. No thermodynamic phase, true critical disorder "
             "W_c, separated Majorana certification or finite-size scaling has been established.\n"),
             ("## 18. Hypothesis assessment\n\n**Inconclusive.** Observed finite-system scores "
             "and explicitly passed validation stages are recorded separately. This engineering "
             "run cannot establish superior disorder-robust topological Majorana structures.\n"),
             ("## 19. Recommended next experiment\n\nInspect the saved anomalies and exact "
             "candidate evidence, then predeclare a matched-budget, multiple-search-seed "
             "comparison. Establish a family extension and appropriate Majorana validation "
             "before testing the full hypothesis. No large experiment is launched automatically.\n")]
    return "\n".join(parts)
