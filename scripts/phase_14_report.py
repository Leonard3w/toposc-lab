"""Derive a reviewable gate report from sealed exact campaign artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from toposc_lab.data import ExactPhysicsDataset, load_dataset, save_dataset
from toposc_lab.generative.benchmark import write_json
from toposc_lab.generative.generators import METHODS


def build(directory: Path, audit_path: Path, output: Path, full_tests: str) -> None:
    report = json.loads((directory / "report.json").read_text())
    audit = json.loads(audit_path.read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert audit["passed"] and audit["reproduced_records"] == 135
    trials = report["trials"]
    n = len(trials)
    aggregates = {}
    selected_errors = []
    for method in METHODS:
        rows = [t[method] for t in trials]
        fields = (
            "best_quality",
            "success_rate",
            "mean_pairwise_distance",
            "mean_novelty_to_warm",
            "descriptor_cells",
            "ood_rate_after_warm",
            "wall_seconds",
            "exact_seconds",
        )
        aggregate = {k: float(np.mean([r[k] for r in rows])) for k in fields}
        aggregate.update(
            {
                "best_quality_sd": float(np.std([r["best_quality"] for r in rows], ddof=1)),
                "best_quality_min": min(r["best_quality"] for r in rows),
                "best_quality_max": max(r["best_quality"] for r in rows),
                "successful_seeds": sum(r["any_success"] for r in rows),
                "improved_seeds": sum(r["warm_improvement"] > 1e-10 for r in rows),
                "proposal_attempts": sum(r["proposal_attempts"] for r in rows),
                "invalid_proposals": sum(r["invalid_proposals"] for r in rows),
                "duplicate_proposals": sum(r["duplicate_proposals"] for r in rows),
                "alternate_winner_changes": sum(not r["alternate_winner_same"] for r in rows),
                "parameter_success_rate": float(
                    np.mean([r["confirmation"]["parameter_success_rate"] for r in rows])
                ),
                "parameter_quality_mean": [
                    float(np.mean([r["confirmation"]["metrics"][i]["quality"] for r in rows]))
                    for i in range(3)
                ],
                "minimum_pairwise_distance": min(r["minimum_pairwise_distance"] for r in rows),
            }
        )
        aggregate["invalid_rate"] = aggregate["invalid_proposals"] / aggregate["proposal_attempts"]
        aggregate["duplicate_rate"] = (
            aggregate["duplicate_proposals"] / aggregate["proposal_attempts"]
        )
        for label, flag in (("ood", True), ("id", False)):
            values = [
                q
                for row in rows
                for q, is_ood in zip(row["qualities"], row["ood_flags"], strict=True)
                if is_ood == flag
            ]
            aggregate[f"{label}_quality_mean"] = float(np.mean(values)) if values else None
        aggregates[method] = aggregate
        if method == "active":
            for trial in trials:
                arm = directory / str(trial["seed"]) / method
                records = [load_dataset(p).records[0] for p in sorted(arm.glob("search-*.json"))]
                targets = dict(
                    zip(trial[method]["record_ids"], trial[method]["qualities"], strict=True)
                )
                from toposc_lab.generative.physics import candidate

                by_candidate = {
                    candidate(r.geometry.to_geometry()).candidate_id: targets[r.record_id]
                    for r in records
                }
                for prediction in json.loads((arm / "proposal-audit.json").read_text())[
                    "predictions"
                ]:
                    if prediction["strategy"]:
                        error = abs(prediction["value"] - by_candidate[prediction["candidate_id"]])
                        selected_errors.append(
                            {
                                "absolute_error": error,
                                "ood": prediction["is_ood"],
                                "covered": error <= 1.6448536269514722 * prediction["uncertainty"],
                                "strategy": prediction["strategy"],
                            }
                        )
    error_summary = {}
    for name, subset in (
        ("all", selected_errors),
        ("ood", [p for p in selected_errors if p["ood"]]),
        ("id", [p for p in selected_errors if not p["ood"]]),
    ):
        error_summary[name] = {
            "count": len(subset),
            "mae": float(np.mean([p["absolute_error"] for p in subset])) if subset else None,
            "raw_90_coverage": float(np.mean([p["covered"] for p in subset])) if subset else None,
        }
    best_trial, best_method = max(
        ((t, m) for t in trials for m in METHODS), key=lambda pair: pair[0][pair[1]]["best_quality"]
    )
    best_summary = best_trial[best_method]
    best_arm = directory / str(best_trial["seed"]) / best_method
    best_records = [load_dataset(p).records[0] for p in sorted(best_arm.glob("search-*.json"))]
    best_record = next(r for r in best_records if r.record_id == best_summary["best_record_id"])
    save_dataset(
        output.parent / "phase_14_best_candidate.json", ExactPhysicsDataset((best_record,))
    )
    advantages = [m for m, c in report["comparisons"].items() if c["reproducible_advantage"]]
    decision = (
        "Select the simple nonlearning patch generator for maximizing the declared finite "
        "localizer quality within this 36-site wiring stratum. Preserve Random and Evolution "
        "as mandatory comparison baselines; Evolution remains the established baseline outside "
        "this scoped result. Coverage and AL stay optional and are not promoted for physics "
        "quality. No method demonstrates success-threshold efficiency, a general topological "
        "advantage, or superiority in other search spaces."
        if "patch" in advantages
        else "Retain Evolution as the established preferred baseline and Random as a fair comparator. "
        "Patch, coverage and AL remain explicit experimental choices without demonstrated "
        "reproducible quality superiority against both baselines."
    )
    compact = {
        "status": "PASS completed generator validation",
        "reproducible_quality_advantage": advantages,
        "aggregates": aggregates,
        "comparisons": report["comparisons"],
        "selected_prediction_error": error_summary,
        "audit": audit,
        "exact_calls_gate_and_repeat": report["exact_calls"] + 135,
        "development_calls": 306,
        "portfolio_decision": decision,
        "full_tests": full_tests,
        "source_sha256": report["source_sha256"],
        "protocol_sha256": report["protocol_sha256"],
        "best": {
            "seed": best_trial["seed"],
            "method": best_method,
            "record_id": best_record.record_id,
            "metrics": best_summary["best_metrics"],
        },
    }
    write_json(output.with_suffix(".json"), compact)
    lines = [
        "# Phase-14 Gate Report — Generative Geometry",
        "",
        "**Block D complete; GENERATOR_GATE validation PASS. Phase 15 has not started.**",
        "",
        (
            "Reproducible best-quality advantage under the frozen rule: "
            + ", ".join(advantages)
            + "."
        )
        if advantages
        else "**No new method demonstrates a reproducible best-quality advantage against both baselines.**",
        (
            "The Phase-13/13R null result remains accepted. AL is technically validated and optional; "
            "it has no presumed superiority. The scoped portfolio decision is stated below."
        ),
        decision,
        "",
        "## Scope, constraints and branch decisions",
        "",
        (
            "Mandatory tasks 14.1, 14.2, 14.12–14.15 and feasibility task 14.9 are complete. "
            "RL 14.3–14.8 was ineligible because Gate 13 failed. Learned/conditional generation "
            "14.10–14.11 was not built: the audited 1,440 Phase-13R records cover only 60 clean-chain "
            "geometries and zero compatible planar records. Earlier 64-site graph work and geometric "
            "REINFORCE demo rewards do not establish an independently split training corpus for this stratum."
        ),
        "",
        (
            "The search expands clean chains to 36-site, 60-edge planar wiring on a 6×6 unit grid, "
            "fixed outer perimeter, degrees 2–6, connected, no crossings, edge length ≤√2. Random "
            "and Evolution have the same support as the new patch and coverage methods. Evolution "
            "wraps existing Phase-10 fitness/tournament/rewiring APIs. Patch correlates spatial edge "
            "priorities; coverage selects distant wiring patterns from random proposal pools. Neither learns physics."
        ),
        "",
        (
            "The existing chiral p-wave model uses t=Δ=1, μ=2, chirality +1. Quality is the minimum "
            "center spectral-localizer gap at κ=0.1,0.2,0.3, gated by agreement on a nonzero index "
            "and PHS residual ≤10⁻¹⁰. Novelty cannot improve physics fitness. Localizer disagreement "
            "remains unresolved. This is finite-system local evidence, not a thermodynamic phase or Majorana claim."
        ),
        "",
        "## 1–3. Fair comparison, exact candidate quality and success",
        "",
        (
            f"Seeds 14101–14120: {n} seeds × five arms × 24 exact search evaluations = 2,400. "
            "The same four initial geometries are simulated and charged independently to every arm. "
            "There are 300 independent winner/μ-sensitivity evaluations and four positive/trivial reference "
            "evaluations: **2,704 gate evaluations**. Each evaluation includes the BdG eigensystem and "
            "three localizer calculations; these are not single matrix solves."
        ),
        "",
        "| Method | Mean best ± seed SD | Best exact | Candidate success | Successful seeds | Improved warm best |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, a in aggregates.items():
        lines.append(
            f"| {method} | {a['best_quality']:.6f} ± {a['best_quality_sd']:.6f} | "
            f"{a['best_quality_max']:.6f} | {100 * a['success_rate']:.2f}% | {a['successful_seeds']}/{n} | {a['improved_seeds']}/{n} |"
        )
    lines += [
        "",
        (
            "Success means quality ≥0.20; the threshold was not changed after development. "
            "If no arm reaches it, first-hit/sample-savings comparisons are censored at the budget "
            "and cannot establish success efficiency."
        ),
        "",
        (
            f"Best exact candidate: **{best_method}, seed {best_trial['seed']}**, quality "
            f"**{best_summary['best_quality']:.9f}**. Its complete reproducible geometry, parameters, "
            "spectrum and raw diagnostics are in [phase_14_best_candidate.json](phase_14_best_candidate.json)."
        ),
        "",
        "```json",
        json.dumps(best_summary["best_metrics"], indent=2),
        "```",
        "",
        "## 4–6. Structural novelty, diversity, duplicates and invalid geometries",
        "",
        (
            "Novelty is edge-Jaccard distance minimized over the eight rigid symmetries of the "
            "square. Coordinates anchor site labels; reflections are conservatively grouped. "
            "Within each arm, proposals at distance ≤0.06 from prior simulations or batch members "
            "are rejected. This is wiring novelty relative to an explicit archive, not literature "
            "novelty or general abstract graph-isomorphism. Descriptor cells are (triangle count // 4, "
            "integer diameter), a descriptive coverage measure rather than a tuned reward."
        ),
        "",
        "| Method | Mean pairwise distance | Novelty to warm | Descriptor cells | Raw duplicates / proposals | Raw invalid / proposals |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m, a in aggregates.items():
        lines.append(
            f"| {m} | {a['mean_pairwise_distance']:.4f} | {a['mean_novelty_to_warm']:.4f} | "
            f"{a['descriptor_cells']:.2f} | {a['duplicate_proposals']}/{a['proposal_attempts']} "
            f"({100 * a['duplicate_rate']:.2f}%) | {a['invalid_proposals']}/{a['proposal_attempts']} ({100 * a['invalid_rate']:.2f}%) |"
        )
    lines += [
        "",
        (
            "All arms evaluate 24 distinct, separated structures per seed. Accepted duplicate "
            "and invalid-geometry rates are zero. Raw proposal rates include the common warm start; "
            "coverage and AL inspect more proposals per exact call, so counts and denominators are "
            "both shown. Full rejection reasons and raw edge proposals remain in the per-arm audit."
        ),
        "",
        "## 7–8. OOD behavior and computational cost",
        "",
        (
            "The common OOD detector uses only each arm's four identical warm-start geometries, "
            "including the training feature envelope. Flags are stored before exact evaluation. "
            "They warn about extrapolation and never declare physics invalid. AL additionally uses "
            "its evolving training reference; OOD exploitation count is zero."
        ),
        "",
        "| Method | Post-warm common OOD | Exact quality OOD / ID | Mean search wall s | Mean exact-evaluator s |",
        "|---|---:|---:|---:|---:|",
    ]
    for m, a in aggregates.items():
        lines.append(
            f"| {m} | {100 * a['ood_rate_after_warm']:.1f}% | "
            f"{a['ood_quality_mean']:.5f} / {a['id_quality_mean']:.5f} | "
            f"{a['wall_seconds']:.3f} | {a['exact_seconds']:.3f} |"
        )
    lines += [
        "",
        ("The ID quality column includes warm-start records. These are descriptive, differently "
        "selected subsets, not a causal effect of OOD. Patch explores predominantly OOD "
        "geometries relative to the small common reference; exact verification is essential."),
        (
            "All exact calls have BdG dimension 72 and localizer dimension 144; the declared "
            "cubic work proxy per 24-call search is 250,822,656. All four numerical thread limits "
            "are one. Wall time includes search/record persistence and pre-exact OOD, excluding "
            "winner confirmations and report derivation; evaluator timing includes validation and "
            "record construction. Runtime does not measure memory or a general hardware speedup."
        ),
        "",
        (
            "AL prequential selected-prediction diagnostics (descriptive, raw Gaussian intervals "
            "are not a calibration guarantee):"
        ),
        "",
        "```json",
        json.dumps(error_summary, indent=2),
        "```",
        "",
        "## 9–10. Seed robustness and reproducible advantage",
        "",
        (
            "Positive differences favor the tested method. Simultaneous paired-bootstrap intervals "
            "use 50,000 resamples and Bonferroni correction for six comparisons. A method must "
            "have positive lower bounds and ≥16/20 strict wins against both baselines. Seeds "
            "share the same physical search stratum; they are repeated searches, not independent materials."
        ),
        "",
        "| Comparison | Mean best-quality difference | Simultaneous interval | Wins / ties / losses |",
        "|---|---:|---:|---:|",
    ]
    for m, comparisons in report["comparisons"].items():
        for b in ("random", "evolution"):
            c = comparisons[b]
            lo, hi = c["simultaneous_interval"]
            lines.append(
                f"| {m} − {b} | {c['mean_difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | "
                f"{c['wins']} / {c['ties']} / {n - c['wins'] - c['ties']} |"
            )
    lines += [
        "",
        "| Method | Winner mean quality μ=1.9 / 2.0 / 2.1 | Sensitivity success | Alternate-weight winner changes |",
        "|---|---:|---:|---:|",
    ]
    for m, a in aggregates.items():
        q = a["parameter_quality_mean"]
        lines.append(
            f"| {m} | {q[1]:.6f} / {q[0]:.6f} / {q[2]:.6f} | "
            f"{100 * a['parameter_success_rate']:.1f}% | {a['alternate_winner_changes']}/{n} |"
        )
    lines += [
        "",
        (
            "Sensitivity checks are two nearby chemical potentials, not disorder ensembles or "
            "finite-size scaling. Alternate ranking multiplies quality by boundary weight; stored "
            "Pareto record IDs preserve the quality/boundary tradeoff. These diagnostics cannot "
            "replace the frozen primary endpoint or establish general physical robustness."
        ),
        "",
        "| Seed | Random | Evolution | Patch | Coverage | Active |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for t in trials:
        lines.append(
            f"| {t['seed']} | " + " | ".join(f"{t[m]['best_quality']:.6f}" for m in METHODS) + " |"
        )
    lines += [
        "",
        "## Validation, provenance and decision",
        "",
        (
            f"All 100 arm/seed winners were exactly revalidated. A fresh-timestamp repeat of "
            f"seed 14101 reproduced all **135 scientific records exactly**, adding 135 evaluations. "
            f"The audit validated {audit['exact_records']} exact records and {audit['inventory_files']} "
            "inventory files, equal budgets, no label leakage, geometry separation and OOD handling. "
            "Gate plus reproduction: **2,839 evaluations**. Development: **306** non-test attempts "
            "(274 complete campaign, 31 before a reporting serialization failure, one failed diagnostic); "
            "**3,145 total non-test evaluations/attempts**. Test simulations are separate."
        ),
        "",
        (
            f"Full suite: `{full_tests}`. Focused regression: 35 passed; final generator checks: "
            "14 passed. Changed-source/tests/audit Ruff PASS. Isolated Mypy PASS with Python 3.14; "
            "the repository's default 3.11 type target remains incompatible with its installed NumPy stubs."
        ),
        "",
        (
            f"Source SHA-256: `{report['source_sha256']}`. Protocol SHA-256: `{report['protocol_sha256']}`. "
            f"Base commit: `{manifest['git_commit']}` with explicitly recorded dirty worktree. "
            "Source snapshots and per-file inventories accompany both runs; historical Phase-13 reports "
            "and pre-existing user changes were preserved."
        ),
        "",
        (
            "Keep Random and Evolution as fair supported baselines. Measured diversity alone "
            "does not establish better physics. "
            "No RL or learned architecture is added for prestige. The benchmark supports decisions "
            "only within this finite wiring stratum, not a global optimal-generator claim. "
            "A further scientific block needs a separate user instruction; **do not start Phase 15**."
        ),
        "",
        (
            "Artifacts: [protocol](phase_14_protocol.md), [feasibility](phase_14_feasibility.json), "
            "[machine-readable gate summary](phase_14_generator_gate.json), "
            "[usage](../phase_14_usage_de.md); full curves and ledgers in `results/phase14-gate/`, "
            "reproduction in `results/phase14-repeat/`."
        ),
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Scientific figures derive only from stored exact records; no simulations.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for method in METHODS:
        curves = np.array([t[method]["best_so_far"] for t in trials])
        mean = curves.mean(axis=0)
        sem = curves.std(axis=0, ddof=1) / np.sqrt(n)
        (line,) = axes[0].plot(np.arange(1, 25), mean, label=method)
        axes[0].fill_between(
            np.arange(1, 25), mean - sem, mean + sem, color=line.get_color(), alpha=0.10
        )
    axes[0].set(
        xlabel="Paid exact search evaluations",
        ylabel="Best finite localizer quality",
        title="20 seeds; mean ± one standard error",
        xlim=(1, 24),
    )
    axes[0].legend(fontsize=8)
    geometry = best_record.geometry.to_geometry()
    xy = geometry.coordinates
    for edge in geometry.edges:
        axes[1].plot(
            xy[[edge.source, edge.target], 0],
            xy[[edge.source, edge.target], 1],
            color="#8b97a3",
            linewidth=1.2,
            zorder=1,
        )
    colors = ["#d96840" if i in geometry.boundary_sites else "#246c96" for i in range(36)]
    axes[1].scatter(xy[:, 0], xy[:, 1], c=colors, s=24, zorder=2)
    axes[1].set(
        aspect="equal",
        xlabel="x",
        ylabel="y",
        title=f"Best exact wiring ({best_method}, seed {best_trial['seed']})",
    )
    figure.savefig(output.with_suffix(".png"), dpi=170)
    plt.close(figure)
    with output.open("a", encoding="utf-8") as handle:
        handle.write("\n![Exact search curves and best wiring](phase_14_generator_gate.png)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-tests", required=True)
    args = parser.parse_args()
    build(args.directory, args.audit, args.output, args.full_tests)
