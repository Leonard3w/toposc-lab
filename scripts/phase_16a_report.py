"""Render the Phase-16A evidence from checksummed stored results; no exact solves."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from hashlib import sha256
from pathlib import Path
from textwrap import fill
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from toposc_lab.discovery.storage import read_json
from toposc_lab.patterns.statistics import grouped_mean_interval

LEGACY = "#73578d"
FRESH = "#157c85"
REJECTED = "#b96a2b"
INK = "#26333e"
MOTIF_LABELS = {
    "triangle_count": "Triangles (K3)",
    "chordless_square_count": "Chordless squares (C4)",
    "boundary_bulk_edges": "Boundary-to-bulk edges",
    "central_diagonal_count": "Central diagonals",
}
INPUT_NAMES = (
    "analysis.json",
    "features.json",
    "experiments.json",
    "accounting.json",
    "manifest.json",
    "intervention-plan.json",
)


def strongest_associations(analysis: dict[str, Any], count: int = 6) -> list[dict[str, Any]]:
    return sorted(
        [r for r in analysis["fresh"]["associations"] if r["partial_spearman"] is not None],
        key=lambda r: (-abs(r["partial_spearman"]), r["feature"]),
    )[:count]


def interval_point(
    ax: Axes, value: float, interval: list[float] | None, y: float, color: str, marker: str = "o"
) -> None:
    if interval is not None:
        ax.plot(interval, [y, y], color=color, linewidth=1.8, solid_capstyle="round")
        ax.plot(interval, [y, y], color=color, marker="|", linestyle="none", markersize=7)
    ax.plot(value, y, marker=marker, color=color, markersize=6, linestyle="none", zorder=4)


def plot_correlations(ax: Axes, analysis: dict[str, Any]) -> None:
    selected = strongest_associations(analysis)
    legacy = {r["feature"]: r for r in analysis["legacy"]["associations"]}
    for i, fresh in enumerate(selected):
        for row, offset, color in (
            (legacy.get(fresh["feature"]), -0.15, LEGACY),
            (fresh, 0.15, FRESH),
        ):
            if row is not None and row["partial_spearman"] is not None:
                interval_point(ax, row["partial_spearman"], row.get("ci95"), i + offset, color)
    ax.axvline(0, color="#9ca7af", linewidth=0.8)
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels([fill(r["feature"].replace("_", " "), 25) for r in selected], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Adjusted Spearman correlation with exact quality")
    ax.set_title("A  Structural associations", loc="left", fontweight="bold")
    ax.plot([], [], "o", color=LEGACY, label="Legacy")
    ax.plot([], [], "o", color=FRESH, label="Fresh")
    ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=9)
    if not selected:
        ax.text(
            0.5, 0.5, "No identifiable partial associations", transform=ax.transAxes, ha="center"
        )
    ax.text(
        0,
        -0.23,
        "Six strongest fresh partials; exploratory, with 95% grouped intervals.\n"
        "Adjusted for degree, boundary degree, origin and fresh seed.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        color=INK,
    )


def plot_ablations(ax: Axes, analysis: dict[str, Any]) -> None:
    labels = []
    for i, row in enumerate(analysis["ablations"]):
        estimate = row.get("estimates", {}).get("target_control_quality")
        labels.append(
            MOTIF_LABELS.get(row["motif"], row["motif"])
            + f"\n{row['parents']} parents / {row.get('seed_count', 0)} seeds"
        )
        if estimate is None:
            ax.text(
                0.02, i, "No matched interventions", transform=ax.get_yaxis_transform(), fontsize=8
            )
            continue
        seeds = list(estimate["group_means"].values())
        offsets = np.linspace(-0.14, 0.14, len(seeds))
        ax.scatter(seeds, i + offsets, s=17, color="#aab5bd", alpha=0.9, zorder=2)
        color = FRESH if row["survived"] else INK
        interval_point(ax, estimate["mean"], estimate["ci95"], i, color, marker="D")
    ax.axvline(0, color="#9ca7af", linewidth=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.margins(x=0.15, y=0.2)
    ax.set_xlabel("Exact quality: targeted edit minus matched control")
    ax.set_title("B  Controlled motif removal", loc="left", fontweight="bold")
    ax.plot([], [], "o", color="#aab5bd", markersize=4, label="Seed means")
    ax.plot([], [], "D-", color=INK, markersize=5, label="Mean / seed 95% CI")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.text(
        0,
        -0.23,
        "Negative effects favor the predeclared beneficial-motif hypothesis.\n"
        "A confidence interval alone does not meet the survival criterion.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        color=INK,
    )


def plot_quality(ax: Axes, analysis: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    groups = []
    labels = []
    for i, (cohort, color) in enumerate((("legacy", LEGACY), ("fresh", FRESH))):
        selected = sorted((r for r in rows if r["cohort"] == cohort), key=lambda r: r["id"])
        values = np.array([r["metrics"]["quality"] for r in selected])
        groups.append(values)
        # Display jitter is deterministic and does not alter scientific measurements.
        jitter = np.random.default_rng(16690 + i).uniform(-0.16, 0.16, len(values))
        ax.scatter(i + jitter, values, s=20, color=color, alpha=0.7, edgecolors="none", zorder=3)
        success = int(np.sum(values >= analysis["success_threshold"]))
        labels.append(f"{cohort.capitalize()} (n={len(values)})\n{success} frozen successes")
    ax.boxplot(
        groups,
        positions=[0, 1],
        widths=0.5,
        showfliers=False,
        medianprops={"color": INK, "linewidth": 1.8},
        boxprops={"color": "#adb7bd"},
        whiskerprops={"color": "#adb7bd"},
        capprops={"color": "#adb7bd"},
    )
    threshold = analysis["success_threshold"]
    ax.axhline(
        threshold,
        color=REJECTED,
        linestyle="--",
        linewidth=1.2,
        label=f"Frozen success threshold {threshold:.2f}",
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9)
    maximum = max(float(values.max()) for values in groups)
    ax.set_ylim(-0.007, max(threshold, maximum) * 1.14)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Eligible minimum localizer gap")
    ax.set_title("C  Exact quality distributions", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(
        0,
        -0.25,
        "Each dot is one D4-distinct geometry in its analysis cohort.\n"
        "Zero includes ineligible or unresolved topology; high quartiles are not success.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        color=INK,
    )


def plot_simplification(ax: Axes, analysis: dict[str, Any]) -> None:
    selected = sorted(analysis["simplifications"], key=lambda r: r["seed"])
    ratios = []
    for i, row in enumerate(selected):
        ratio = row.get("retention_ratio")
        if ratio is None:
            ax.text(i, 3, "n/a", ha="center", fontsize=8, color=INK)
            continue
        value = 100 * ratio
        ratios.append(value)
        color = FRESH if row["retains_90_percent"] else REJECTED
        ax.vlines(i, 0, value, color=color, linewidth=2, alpha=0.4)
        ax.scatter(i, value, color=color, marker="D", s=42, zorder=3)
    ax.axhline(90, color=INK, linestyle="--", linewidth=1, label="90% quality retention")
    ax.axhline(100, color="#b8c0c6", linewidth=0.8)
    ax.set_xticks(range(len(selected)))
    ax.set_xticklabels([str(r["seed"]) for r in selected], rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, max([115.0, *ratios]) * 1.1)
    ax.set_xlim(-0.6, max(0.6, len(selected) - 0.4))
    ax.set_ylabel("Exact quality retained (%)")
    ax.set_xlabel("Parent generation seed")
    ax.set_title("D  Single-edge deletion retention", loc="left", fontweight="bold")
    ax.scatter([], [], color=FRESH, marker="D", s=25, label="Retains criterion")
    ax.scatter([], [], color=REJECTED, marker="D", s=25, label="Fails criterion")
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        fontsize=8,
        loc="lower right",
    )
    ax.text(
        0,
        -0.25,
        "Criterion: >=90% quality, unchanged eligibility and raw indices.\n"
        "One seeded removable edge per best fresh parent; no reoptimization.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        color=INK,
    )


def render_figure(analysis: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> None:
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": INK,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#b8c0c6",
            "figure.facecolor": "white",
        }
    ):
        fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.0))
        fig.subplots_adjust(left=0.17, right=0.98, top=0.88, bottom=0.16, hspace=0.68, wspace=0.65)
        plot_correlations(axes[0, 0], analysis)
        plot_ablations(axes[0, 1], analysis)
        plot_quality(axes[1, 0], analysis, rows)
        plot_simplification(axes[1, 1], analysis)
        fig.suptitle(
            "Phase 16A | Structural patterns and controlled ablation",
            x=0.055,
            ha="left",
            fontsize=19,
            fontweight="bold",
            y=0.975,
        )
        fig.text(
            0.055,
            0.925,
            "Stored exact evidence in the finite 36-site chiral p-wave wiring stratum",
            fontsize=12,
            color="#5d6a75",
        )
        fig.text(
            0.055,
            0.025,
            "Exploratory finite-model evidence; no design law, Majorana certification, "
            "thermodynamic phase or general robustness claim.\n"
            "Phase 16B has not started. All plotted values come from archived exact records "
            "and their frozen analysis.",
            fontsize=9,
            color="#5d6a75",
            va="bottom",
        )
        fig.savefig(path, dpi=190, facecolor="white", metadata={"Software": "TOPOSC-LAB Phase16A"})
        plt.close(fig)


def write_plain_json(path: Path, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def evidence_summary(analysis: dict[str, Any], accounting: dict[str, Any]) -> dict[str, Any]:
    retained_rows = [r for r in analysis["simplifications"] if r.get("retention_ratio") is not None]
    return {
        "cohorts": {
            cohort: {
                k: analysis[cohort][k]
                for k in ("count", "eligible_count", "success_count", "quality_mean", "quality_max")
            }
            for cohort in ("legacy", "fresh")
        },
        "strongest_fresh_partials": [
            {
                k: row.get(k)
                for k in (
                    "feature",
                    "spearman",
                    "partial_spearman",
                    "ci95",
                    "bh_q",
                    "by_q",
                    "leave_one_seed_out_sign_agreement",
                )
            }
            for row in strongest_associations(analysis)
        ],
        "importance_models": [
            {
                k: row[k]
                for k in ("model", "mean_mse", "mean_baseline_mse", "baseline_beaten_fold_fraction")
            }
            for row in analysis["fresh"]["importance"]["summaries"]
        ],
        "ablations": [
            {
                "motif": row["motif"],
                "parents": row["parents"],
                "pairs": row["pairs"],
                "seed_count": row.get("seed_count", 0),
                "holm_p": row["holm_p"],
                "survived": row["survived"],
                "status": row["status"],
                "quality_effect": row.get("estimates", {}).get("target_control_quality"),
            }
            for row in analysis["ablations"]
        ],
        "simplification": {
            "complete": sum(r["status"] == "complete" for r in analysis["simplifications"]),
            "retained": sum(
                r.get("retains_90_percent", False) for r in analysis["simplifications"]
            ),
            "retention_ratio_seed_bootstrap": grouped_mean_interval(
                np.array([r["retention_ratio"] for r in retained_rows]),
                np.array([r["seed"] for r in retained_rows]),
                16603,
            )
            if retained_rows
            else None,
        },
        "accounting": accounting,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=ROOT / "results/phase16a")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/decisions")
    args = parser.parse_args()
    campaign, destination = args.campaign.resolve(), args.output_dir.resolve()
    artifacts = {name: read_json(campaign / name) for name in INPUT_NAMES}
    input_hashes = {
        name: sha256((campaign / name).read_bytes()).hexdigest() for name in INPUT_NAMES
    }
    analysis, accounting = artifacts["analysis.json"], artifacts["accounting.json"]
    rows, experiments = artifacts["features.json"], artifacts["experiments.json"]
    plan = artifacts["intervention-plan.json"]
    if analysis["phase16b_started"] or analysis["majorana_claim"]:
        raise ValueError("unexpected claims outside the Phase-16A reporting scope")
    if accounting["attempts"] > accounting["cap"]:
        raise ValueError("campaign exceeded its exact-attempt budget")
    if len(rows) != analysis["legacy"]["count"] + analysis["fresh"]["count"]:
        raise ValueError("feature rows disagree with the analyzed cohort counts")
    if len(experiments) != sum(r["pairs"] for r in analysis["ablations"]):
        raise ValueError("stored experiments disagree with the ablation summary")
    inventory_path = campaign / "inventory.json"
    if inventory_path.exists():
        inventory = read_json(inventory_path)
        if any(inventory.get(name) != checksum for name, checksum in input_hashes.items()):
            raise ValueError("report input disagrees with the campaign inventory")
    destination.mkdir(parents=True, exist_ok=True)
    figure_path = destination / "phase_16a_patterns.png"
    render_figure(analysis, rows, figure_path)
    summary = evidence_summary(analysis, accounting)
    report = {
        "schema_version": 1,
        "block": "F",
        "phase": "16A",
        "checkpoint": "Internal Mechanism Checkpoint",
        "phase16b_started": False,
        "source_kind": "checksummed_stored_exact_results_and_frozen_analysis",
        "analysis": analysis,
        "accounting": accounting,
        "manifest": artifacts["manifest.json"],
        "availability": plan["availability"],
        "intervention_selection": plan["selection"],
        "stored_experiment_count": len(experiments),
        "summary": summary,
        "report_input_sha256": input_hashes,
        "report_driver_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "figure": {
            "filename": figure_path.name,
            "sha256": sha256(figure_path.read_bytes()).hexdigest(),
        },
        "scientific_scope": {
            "exact_physics_is_ground_truth": True,
            "causal_design_rule_claim": False,
            "majorana_claim": False,
            "thermodynamic_phase_claim": False,
            "general_robustness_claim": False,
            "hypotheses_require_independent_phase16b_validation": True,
        },
    }
    write_plain_json(destination / "phase_16a_pattern_ablation.json", report)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
