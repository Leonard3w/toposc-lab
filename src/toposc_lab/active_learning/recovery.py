"""Frozen Phase-13R recovery configuration and paired, multiplicity-aware gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from toposc_lab.active_learning.acquisition import AcquisitionConfig
from toposc_lab.active_learning.benchmark import (
    reference_training_config,
    run_discovery_benchmark,
    source_provenance,
)
from toposc_lab.active_learning.cycle import CycleConfig

RECOVERY_SEEDS = tuple(range(13301, 13321))


def recovery_config() -> CycleConfig:
    return CycleConfig(
        replace(reference_training_config(), flag_extrapolation=True),
        AcquisitionConfig(scaling="target_scale", diversity_radius_fraction=0.5),
        40,
    )


def paired_statistics(differences: list[float]) -> dict:
    """Positive means better AL. Seed is the sampling unit, never candidate/iteration.

    The exact sign-flip test assumes exchangeability of paired differences under
    the null. Report the paired bootstrap CI alongside it; neither addresses
    generalization to an untested geometry family.
    """
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or not 1 <= len(d) <= 20 or not np.all(np.isfinite(d)):
        raise ValueError("paired statistics require 1..20 finite seed differences")
    rng = np.random.default_rng(13300)
    boot = rng.choice(d, size=(50000, len(d))).mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    observed = float(d.sum())
    extreme = 0
    total = 2 ** len(d)
    for start in range(0, total, 8192):
        assignments = np.arange(start, min(start + 8192, total), dtype=np.uint32)
        signs = (
            2 * ((assignments[:, None] >> np.arange(len(d), dtype=np.uint32)) & 1).astype(float) - 1
        )
        extreme += int(np.sum(signs @ d >= observed - 1e-12))
    p_value = extreme / total
    wins = int(np.sum(d > 1e-12))
    return {
        "differences": d.tolist(),
        "mean": float(d.mean()),
        "ci95": [float(low), float(high)],
        "wins": wins,
        "ties": int(np.sum(np.abs(d) <= 1e-12)),
        "losses": int(np.sum(d < -1e-12)),
        "sign_flip_p_one_sided": p_value,
        "bonferroni_p_four_tests": min(1.0, 4 * p_value),
        "passes": bool(len(d) == 20 and low > 1e-12 and p_value <= 0.0125 and wins >= 16),
    }


def recovery_statistics(report: dict) -> dict:
    trials = report["trials"]
    quality = {}
    efficiency = {}
    for baseline in ("random", "evolutionary"):
        quality[baseline] = paired_statistics(
            [t["arms"]["active"]["best_score"] - t["arms"][baseline]["best_score"] for t in trials]
        )
        # Non-hits are right-censored at budget+1; success rate is also retained.
        efficiency[baseline] = paired_statistics(
            [
                (t["arms"][baseline]["first_hit_0_02"] or 25)
                - (t["arms"]["active"]["first_hit_0_02"] or 25)
                for t in trials
            ]
        )
    quality_pass = all(r["passes"] for r in quality.values())
    efficiency_pass = all(r["passes"] for r in efficiency.values())
    confirmed = all(r["agrees"] for t in trials for r in t["revalidation"].values())
    valid = (
        tuple(t["seed"] for t in trials) == RECOVERY_SEEDS
        and all(a["exact_simulations"] == 24 for t in trials for a in t["arms"].values())
        and all(t["arms"]["active"]["unique_candidates"] == 24 for t in trials)
        and not any(t["failures"] for t in trials)
    )
    return {
        "quality": quality,
        "sample_efficiency": efficiency,
        "success_rates_0_02": {
            a: sum(t["arms"][a]["first_hit_0_02"] is not None for t in trials) / len(trials)
            for a in ("active", "random", "evolutionary")
        },
        "quality_advantage": quality_pass,
        "sample_efficiency_advantage": efficiency_pass,
        "winners_confirmed": confirmed,
        "budget_seed_validity": valid,
        "empirical_gate_passed": bool((quality_pass or efficiency_pass) and confirmed and valid),
        "full_suite_and_artifact_audit_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_discovery_benchmark(
        args.output,
        provenance=source_provenance(Path.cwd()),
        seeds=RECOVERY_SEEDS,
        cycle_config=recovery_config(),
    )
    report["legacy_gate_indicator"] = report["empirical_gate_passed"]
    report["recovery_statistics"] = recovery_statistics(report)
    report["empirical_gate_passed"] = report["recovery_statistics"]["empirical_gate_passed"]
    report["recovery_protocol"] = "phase13r.frozen.v1"
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report["recovery_statistics"], indent=2))


if __name__ == "__main__":
    main()
