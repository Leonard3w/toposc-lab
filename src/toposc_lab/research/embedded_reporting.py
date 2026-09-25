"""Descriptive family report composed from persisted diagnostics and existing plots."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from toposc_lab.research.embedded import EmbeddedDomain
from toposc_lab.research.space import geometry_from_payload
from toposc_lab.research.storage import ResearchStore, atomic_text, dumps
from toposc_lab.research.validation_statistics import usable

METRICS = (
    "quality",
    "success",
    "minimum_abs_energy",
    "interior_localizer_gap",
    "interior_nonzero_fraction",
    "edge_weight",
    "center_weight",
    "chern_bulk_mean",
    "old_boundary_weight",
    "polarization",
    "self_conjugacy",
    "window_state_count",
)


def scalar_record(
    candidate: dict[str, Any], result: dict[str, Any], domain: EmbeddedDomain
) -> dict[str, Any]:
    row = {
        "candidate": candidate["id"],
        "family": candidate["family"],
        **result["stage"],
        "valid": usable(result),
        "error": result.get("error"),
        **dict.fromkeys(METRICS),
    }
    if result.get("status") != "completed":
        return row
    coords = np.asarray(candidate["geometry"]["coordinates"])
    points: dict[tuple[float, ...], list[dict[str, Any]]] = {}
    for p in result["spatial"]:
        if domain.distances(np.array([p["point"]]))[0] >= domain.bulk_inset:
            points.setdefault(tuple(p["point"]), []).append(p)
    valid = bool(points) and all(p["valid"] for ps in points.values() for p in ps)
    metric = result["metrics"]
    row.update(
        quality=metric["quality"] if row["valid"] else None,
        success=int(metric["success"]) if row["valid"] else None,
        minimum_abs_energy=metric["minimum_abs_energy"],
        old_boundary_weight=metric["boundary_weight"],
        numerical_residual=max(
            metric[k]
            for k in (
                "operator_phs_residual",
                "spectral_phs_residual",
                "eigensystem_residual",
                "hermiticity_residual",
            )
        ),
        bulk_gap=metric["bulk_gap"],
        bulk_gap_status=metric["bulk_gap_status"],
    )
    if valid:
        row["interior_localizer_gap"] = min(p["gap"] for ps in points.values() for p in ps)
        row["interior_nonzero_fraction"] = sum(
            len({p["index"] for p in ps}) == 1 and ps[0]["index"] not in (0, None)
            for ps in points.values()
        ) / len(points)
    window = result["boundary_window"]
    row["window_state_count"] = window["state_count"]
    center = np.all(np.abs(coords - np.asarray(domain.center)) <= 1, axis=1)
    row["center_site_count"] = int(center.sum())
    if window["window"] is not None:
        row["edge_weight"] = window["window"]["strip_weights"]["1"]
        row["center_weight"] = float(np.asarray(window["window"]["site_probability"])[center].sum())
    if result["chern_marker"]["status"] == "available":
        row["chern_bulk_mean"] = result["chern_marker"]["bulk_mean"]
    row["polarization"] = float(
        np.mean([s["polarization_norm"] for s in result["majorana"]["states"]])
    )
    row["self_conjugacy"] = float(
        np.mean([s["self_conjugacy"] for s in result["majorana"]["states"]])
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
    writer.writeheader()
    writer.writerows(
        {k: dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows
    )
    atomic_text(path, stream.getvalue())


def paired_contrast(rows: list[dict[str, Any]], identity: str, width: float) -> dict[str, Any]:
    """Compare only valid matched seeds; retain metric-specific denominators."""
    candidate = {
        r["seed"]: r
        for r in rows
        if r["candidate"] == identity
        and r["kind"] == "disorder"
        and r["width"] == width
        and r["valid"]
    }
    baseline = {
        r["seed"]: r
        for r in rows
        if r["candidate"] == "regular"
        and r["kind"] == "disorder"
        and r["width"] == width
        and r["valid"]
    }
    result = {}
    for metric in METRICS:
        seeds = sorted(
            s
            for s in candidate.keys() & baseline.keys()
            if candidate[s][metric] is not None and baseline[s][metric] is not None
        )
        values = [candidate[s][metric] - baseline[s][metric] for s in seeds]
        result[metric] = float(np.mean(values)) if values else None
        result[metric + "_paired_seeds"] = seeds
    return result


def export_study(directory: Path) -> str:
    store = ResearchStore(directory)
    cohort, settings = store.get("validation_cohort"), store.get("validation_settings")
    candidates = {c["id"]: c for c in cohort["candidates"]}
    domain = EmbeddedDomain(**cohort["domain"])
    output = directory / "reports"
    output.mkdir(exist_ok=True)
    rows, examples = [], {}
    # Stream large complete raw payloads; never retain the full dataset in memory.
    temporary = output / "realizations.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as stream, store.connect(readonly=True) as db:
        for saved in db.execute("SELECT * FROM objects WHERE kind='exact_result' ORDER BY id"):
            r = store.decode(saved)
            identity = saved["id"].split(":", 1)[0]
            stream.write(dumps({"candidate": identity, "result": r}) + "\n")
            rows.append(scalar_record(candidates[identity], r, domain))
            if r.get("status") == "completed" and (
                r["stage"]["kind"] == "clean"
                or (r["stage"]["width"] == 6 and r["stage"]["seed"] == settings["seeds"][0])
            ):
                examples[saved["id"]] = r
    temporary.replace(output / "realizations.jsonl")
    groups = []
    for c in candidates.values():
        for w in settings["widths"]:
            selected = [
                r
                for r in rows
                if r["candidate"] == c["id"] and r["kind"] == "disorder" and r["width"] == w
            ]
            entry = {
                "candidate": c["id"],
                "family": c["family"],
                "width": w,
                "present": len(selected),
                "valid": sum(r["valid"] for r in selected),
            }
            for metric in METRICS:
                values = [r[metric] for r in selected if r["valid"] and r[metric] is not None]
                entry[metric] = float(np.mean(values)) if values else None
            groups.append(entry)
    family_rows, correlations = [], []
    families = list(dict.fromkeys(c["family"] for c in candidates.values()))
    for family in families:
        for w in settings["widths"]:
            group = [g for g in groups if g["family"] == family and g["width"] == w]
            for metric in METRICS:
                values = [g[metric] for g in group if g[metric] is not None]
                family_rows.append(
                    {
                        "family": family,
                        "width": w,
                        "metric": metric,
                        "candidate_n": len(values),
                        "mean": float(np.mean(values)) if values else None,
                        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                        "quantiles": np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).tolist()
                        if values
                        else None,
                    }
                )
            for feature in next(iter(candidates.values()))["descriptors"]:
                for metric in (
                    "quality",
                    "edge_weight",
                    "interior_localizer_gap",
                    "chern_bulk_mean",
                ):
                    pairs = [
                        (candidates[g["candidate"]]["descriptors"][feature], g[metric])
                        for g in group
                        if g[metric] is not None
                    ]
                    rho = None
                    if len(pairs) > 2:
                        a, b = np.asarray(pairs).T
                        if np.ptp(a) > 0 and np.ptp(b) > 0:
                            rho = float(spearmanr(a, b).statistic)
                    correlations.append(
                        {
                            "family": family,
                            "width": w,
                            "feature": feature,
                            "metric": metric,
                            "n": len(pairs),
                            "spearman_rho": rho,
                            "scope": "descriptive; no causal or significance claim",
                        }
                    )
    contrasts = []
    for g in groups:
        contrasts.append(
            {
                "candidate": g["candidate"],
                "family": g["family"],
                "width": g["width"],
                **paired_contrast(rows, g["candidate"], g["width"]),
            }
        )
    best = []
    for family in families:
        ids = [c["id"] for c in candidates.values() if c["family"] == family]
        scores = [
            (
                float(
                    np.mean(
                        [
                            g["quality"]
                            for g in groups
                            if g["candidate"] == i and g["quality"] is not None
                        ]
                    )
                ),
                i,
            )
            for i in ids
            if any(g["quality"] is not None for g in groups if g["candidate"] == i)
        ]
        if scores:
            best.append(min(scores, key=lambda p: (-p[0], p[1]))[1])
    summary = {
        "state": store.get("state"),
        "families": family_rows,
        "examples_descriptive": best,
        "candidate_counts": dict(Counter(c["family"] for c in candidates.values())),
        "attempt_statuses": dict(Counter(a["status"] for a in store.attempts())),
        "records": len(rows),
        "invalid": sum(not r["valid"] for r in rows),
        "spatial_discordant_successes": sum(
            r["success"] == 1
            and r["interior_nonzero_fraction"] is not None
            and r["interior_nonzero_fraction"] < 1
            for r in rows
            if r["kind"] == "disorder"
        ),
        "successes": sum(r["success"] == 1 for r in rows if r["kind"] == "disorder"),
        "max_numeric_residual": max((r.get("numerical_residual", 0) for r in rows), default=None),
        "proposals": len(cohort["proposals"]),
        "rejected_proposals": sum(not p["accepted"] for p in cohort["proposals"]),
        "cohort_sha256": cohort["sha256"],
        "source_sha256": store.get("manifest")["source_sha256"],
        "scope": "Exploratory distributions; shared disorder seeds, no confirmatory family inference",
    }
    atomic_text(output / "summary.json", json.dumps(summary, indent=2))
    for filename, table in (
        ("realization_diagnostics", rows),
        ("candidate_statistics", groups),
        ("family_statistics", family_rows),
        ("feature_correlations", correlations),
        ("baseline_differences", contrasts),
        ("rejections", cohort["proposals"]),
    ):
        write_csv(output / (filename + ".csv"), table)
    if groups and best:
        figures(directory / "plots", rows, groups, candidates, examples, best, settings)
    text = (
        "# Phase 19 exploratory output\n\nState: "
        + store.get("state")["status"]
        + "\n\nAll raw realizations and descriptive family tables are in reports/. "
        "No discovery or family-superiority claim. See docs/decisions/phase19_exploratory_report_de.md.\n"
    )
    atomic_text(directory / "final_report.md", text)
    return text


def figures(
    output: Path,
    rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    candidates: dict[str, Any],
    examples: dict[str, Any],
    selected: list[str],
    settings: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from toposc_lab.visualization.geometry_plots import plot_geometry
    from toposc_lab.visualization.plots import plot_eigenvalue_spectrum

    families = list(dict.fromkeys(c["family"] for c in candidates.values()))
    output.mkdir(exist_ok=True)
    for w in settings["widths"]:
        fig, axes = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)
        for metric, ax in zip(METRICS, axes.flat, strict=True):
            values = [
                [
                    g[metric]
                    for g in groups
                    if g["family"] == f and g["width"] == w and g[metric] is not None
                ]
                for f in families
            ]
            ax.boxplot(values, tick_labels=families, showmeans=True)
            ax.set_title(metric)
            ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        fig.suptitle(f"W={w:g} | candidate means over 3 common seeds | descriptive only")
        fig.savefig(output / f"family_distributions_W{w:g}.png", dpi=140)
        plt.close(fig)
    fig, axes = plt.subplots(
        len(selected), 3, figsize=(13, 3.5 * len(selected)), squeeze=False, constrained_layout=True
    )
    for identity, axs in zip(selected, axes, strict=True):
        c = candidates[identity]
        g = geometry_from_payload(c["geometry"])
        plot_geometry(
            g, axes=axs[0], title=c["family"] + "\n" + identity[:30], show=False, site_size=15
        )
        clean = examples.get(identity + ":clean")
        if clean:
            plot_eigenvalue_spectrum(
                np.array(clean["spectrum"]), axes=axs[1], title="Clean spectrum", show=False
            )
        width_index = settings["widths"].index(6) if 6 in settings["widths"] else None
        disorder = examples.get(identity + f":disorder_{width_index}_0")
        if disorder and disorder["boundary_window"]["window"]:
            # Compose existing geometry renderer with saved per-site projector density.
            plot_geometry(
                g,
                axes=axs[2],
                title="W=6 first seed | energy-window density",
                show=False,
                site_size=8,
                show_boundary_sites=False,
            )
            density = disorder["boundary_window"]["window"]["site_probability"]
            im = axs[2].scatter(
                g.coordinates[:, 0], g.coordinates[:, 1], c=density, cmap="magma", s=35, zorder=5
            )
            fig.colorbar(im, ax=axs[2], label="Mean site probability")
    fig.suptitle("Highest mean-Q example per family, selected descriptively; not discoveries")
    fig.savefig(output / "candidate_examples.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for row_axes, metric in zip(axes, ("quality", "interior_localizer_gap"), strict=True):
        for ax, feature in zip(
            row_axes,
            ("bond_length_mean", "coordination_variance", "clustering_coefficient"),
            strict=True,
        ):
            for family in families:
                data = [
                    g
                    for g in groups
                    if g["family"] == family and g["width"] == 6 and g[metric] is not None
                ]
                ax.scatter(
                    [candidates[g["candidate"]]["descriptors"][feature] for g in data],
                    [g[metric] for g in data],
                    s=18,
                    alpha=0.7,
                    label=family,
                )
            ax.set(xlabel=feature, ylabel=metric)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("W=6 | graph features versus candidate mean diagnostics | descriptive")
    fig.savefig(output / "feature_physics.png", dpi=140)
    plt.close(fig)
