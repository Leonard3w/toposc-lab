"""Post-process the frozen confirmation; never call a solver or alter its study.

The inferential rules are frozen in phase18_confirmation_analysis_plan.md.
Streams checked raw records into compact scalar rows to bound memory use.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from toposc_lab.research.storage import ResearchStore, atomic_text, dumps
from toposc_lab.research.validation_cohort import digest, validate_cohort
from toposc_lab.research.validation_statistics import paired_interval, usable, wilson

QUANTILES = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]
DIAGNOSTICS = [
    "interior_localizer_gap_min",
    "interior_nonzero_consistent_fraction",
    "minimum_abs_energy",
    "edge_weight_1",
    "edge_weight_2",
    "edge_weight_3",
    "center_weight_16_sites",
    "interior_weight_d_ge2",
    "mean_edge_distance",
    "window_state_count",
    "chern_bulk_mean",
    "old_four_state_boundary",
    "old_four_state_self_conjugacy",
    "old_four_state_polarization",
]


def describe(values: list[float], *, alpha: float = 0.05) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite observations must be counted as invalid before statistics")
    n = len(array)
    if not n:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "sd": None,
            "sem": None,
            "mean_bootstrap_ci": None,
            "mean_t_ci": None,
            "quantiles": None,
        }
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if n > 1 else None
    sem = sd / math.sqrt(n) if sd is not None else None
    half = float(stats.t.ppf(1 - alpha / 2, n - 1) * sem) if n > 1 else None
    return {
        "n": n,
        "mean": mean,
        "median": float(np.median(array)),
        "sd": sd,
        "sem": sem,
        "mean_bootstrap_ci": paired_interval(values, alpha=alpha)["interval"],
        "mean_t_ci": [mean - half, mean + half] if half is not None else None,
        "confidence": 1 - alpha,
        "quantiles": dict(
            zip(map(str, QUANTILES), np.quantile(array, QUANTILES).tolist(), strict=True)
        ),
    }


def simultaneous_interval(values: list[float], comparisons: int = 45) -> dict[str, Any]:
    n = len(values)
    if n < 2:
        return {
            "n": n,
            "mean": float(np.mean(values)) if n else None,
            "interval": None,
            "family_comparisons": comparisons,
        }
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf(1 - 0.05 / (2 * comparisons), n - 1) * sem)
    return {
        "n": n,
        "mean": mean,
        "sem": sem,
        "interval": [mean - half, mean + half],
        "family_comparisons": comparisons,
        "family_confidence": 0.95,
        "method": "Bonferroni paired t intervals; approximate finite-sample coverage",
    }


def crossover_evidence(rows: list[dict[str, Any]], *, all_pairs_complete: bool) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: r["width"])
    positive = [
        r["width"]
        for r in rows
        if r["simultaneous"]["interval"] is not None and r["simultaneous"]["interval"][0] > 0
    ]
    negative = [
        r["width"]
        for r in rows
        if r["simultaneous"]["interval"] is not None and r["simultaneous"]["interval"][1] < 0
    ]
    pairs = [[a, b] for a in positive for b in negative if a < b]
    numerical = [
        [a["width"], b["width"]]
        for a in rows
        for b in rows
        if a["width"] < b["width"]
        and a["quality"]["mean"] is not None
        and b["quality"]["mean"] is not None
        and a["quality"]["mean"] > 0 > b["quality"]["mean"]
    ]
    return {
        "statistically_supported_within_grid": bool(pairs) and all_pairs_complete,
        "positive_widths_simultaneous": positive,
        "negative_widths_simultaneous": negative,
        "positive_then_negative_pairs": pairs,
        "numerical_sign_change_pairs": numerical,
        "complete_primary_pairs": all_pairs_complete,
        "weak_disorder_advantage_tested": False,
        "scope": "Only W=6..9; cannot establish a positive low-W side outside this grid",
    }


def extract_record(
    identity: str, result: dict[str, Any], coordinates: np.ndarray
) -> dict[str, Any]:
    stage = result["stage"]
    row: dict[str, Any] = {
        "candidate": identity,
        "width": stage["width"],
        "seed": stage["seed"],
        "kind": stage["kind"],
        "status": result.get("status"),
        "primary_valid": usable(result),
        "error": result.get("error"),
        "quality": None,
        "success": None,
    }
    row.update(dict.fromkeys(DIAGNOSTICS))
    if result.get("status") != "completed":
        return row
    metrics = result["metrics"]
    row.update(
        quality=metrics["quality"] if row["primary_valid"] else None,
        success=bool(metrics["success"]) if row["primary_valid"] else None,
        primary_invalid_reason=result.get("primary_invalid_reason"),
        raw_center_localizer_gap=metrics["localizer_gap"],
        minimum_abs_energy=metrics["minimum_abs_energy"],
        old_four_state_boundary=metrics["boundary_weight"],
        center_indices=result["indices"],
        center_eligible=bool(metrics["eligible"]),
        maximum_numeric_residual=max(
            metrics[k]
            for k in (
                "operator_phs_residual",
                "spectral_phs_residual",
                "hermiticity_residual",
                "eigensystem_residual",
            )
        ),
    )
    groups = {}
    for p in result["spatial"]:
        x, y = p["point"]
        if 2 <= x <= 7 and 2 <= y <= 7:
            groups.setdefault((x, y), []).append(p)
    spatial_valid = len(groups) == 9 and all(
        len(g) == 3 and all(p["valid"] for p in g) for g in groups.values()
    )
    row["interior_spatial_valid"] = spatial_valid
    if spatial_valid:
        row["interior_localizer_gap_min"] = min(p["gap"] for g in groups.values() for p in g)
        row["interior_nonzero_consistent_fraction"] = (
            sum(
                len({p["index"] for p in g}) == 1 and g[0]["index"] not in (None, 0)
                for g in groups.values()
            )
            / 9
        )
    boundary = result["boundary_window"]
    row["window_state_count"] = boundary["state_count"]
    window = boundary["window"]
    if window is not None:
        weights = np.asarray(window["site_probability"])
        center = np.all((coordinates >= 3) & (coordinates <= 6), axis=1)
        distance = np.asarray(boundary["distance_to_boundary"])
        if center.sum() != 16 or not np.isclose(weights.sum(), 1, atol=1e-10):
            raise ValueError("Unexpected fixed center region or unnormalized window density")
        row.update(
            edge_weight_1=window["strip_weights"]["1"],
            edge_weight_2=window["strip_weights"]["2"],
            edge_weight_3=window["strip_weights"]["3"],
            center_weight_16_sites=float(weights[center].sum()),
            interior_weight_d_ge2=float(weights[distance >= 2].sum()),
            mean_edge_distance=window["mean_boundary_distance"],
        )
    marker = result["chern_marker"]
    row["chern_status"] = marker["status"]
    if marker["status"] == "available":
        row["chern_bulk_mean"] = marker["bulk_mean"]
        row["chern_trace_residual"] = marker["trace_residual"]
    states = result["majorana"]["states"]
    if states:
        row["old_four_state_self_conjugacy"] = float(np.mean([s["self_conjugacy"] for s in states]))
        row["old_four_state_polarization"] = float(
            np.mean([s["polarization_norm"] for s in states])
        )
    return row


def read_data(directory: Path) -> dict[str, Any]:
    store = ResearchStore(directory)
    rows = []
    checksums = []
    fields = {}
    with store.connect(readonly=True) as db:
        db.execute("BEGIN")
        objects = {
            kind: store.decode(
                db.execute(
                    "SELECT * FROM objects WHERE kind=? AND id='current'", (kind,)
                ).fetchone()
            )
            for kind in ("validation_settings", "validation_cohort", "manifest", "state")
        }
        if objects["state"]["status"] != "COMPLETED":
            raise ValueError(
                "Only analyze the completed fixed-size confirmation; no interim selection"
            )
        settings = objects["validation_settings"]
        if settings["mode"] != "confirmation":
            raise ValueError("Confirmation-only analysis")
        cohort = objects["validation_cohort"]
        validate_cohort(cohort)
        coordinates = {
            c["id"]: np.asarray(c["geometry"]["coordinates"]) for c in cohort["candidates"]
        }
        for saved in db.execute("SELECT * FROM objects WHERE kind='exact_result' ORDER BY id"):
            result = store.decode(saved)
            identity = saved["id"].split(":", 1)[0]
            rows.append(extract_record(identity, result, coordinates[identity]))
            checksums.append([saved["id"], saved["checksum"]])
            if result.get("status") == "completed" and result["stage"]["kind"] == "disorder":
                key = (result["stage"]["width"], result["stage"]["seed"])
                offset = np.asarray(result["onsite_offsets"])
                if key in fields and not np.allclose(fields[key], offset, rtol=0, atol=1e-12):
                    raise ValueError("Different disorder field in a matched comparison")
                fields[key] = offset
        attempts = [dict(a) for a in db.execute("SELECT * FROM attempts ORDER BY number")]
    return {
        **objects,
        "rows": rows,
        "result_checksum_digest": digest(checksums),
        "attempt_digest": digest(attempts),
        "attempts": attempts,
        "paired_fields_checked": len(fields),
    }


def analysis(data: dict[str, Any]) -> dict[str, Any]:
    settings = data["validation_settings"]
    ids = [c["id"] for c in data["validation_cohort"]["candidates"]]
    widths, seeds = settings["widths"], settings["seeds"]
    observations = {
        (r["candidate"], r["width"], r["seed"]): r for r in data["rows"] if r["kind"] == "disorder"
    }
    cells, comparisons, correlations = [], [], []
    expected = {(i, "disorder", w, s) for i in ids for w in widths for s in seeds}
    expected.update((i, "clean", 0.0, settings["clean_seed"]) for i in ids)
    actual = {(r["candidate"], r["kind"], r["width"], r["seed"]) for r in data["rows"]}
    if actual - expected or len(actual) != len(data["rows"]):
        raise ValueError("Unexpected or duplicate realization identity")
    complete_records = actual == expected
    for identity in ids:
        for w in widths:
            all_rows = [observations.get((identity, w, seed)) for seed in seeds]
            raw = [r for r in all_rows if r is not None]
            valid = [r for r in raw if r["primary_valid"]]
            n, success = len(valid), sum(r["success"] for r in valid)
            cell = {
                "candidate": identity,
                "width": w,
                "planned": len(seeds),
                "present": len(raw),
                "missing": len(seeds) - len(raw),
                "invalid": len(raw) - n,
                "quality": describe([r["quality"] for r in valid]),
                "successes": success,
                "success_fraction": success / n if n else None,
                "success_wilson95": wilson(success, n),
                "success_bounds_all_planned": [
                    success / len(seeds),
                    (success + len(seeds) - n) / len(seeds),
                ],
                "diagnostics": {
                    k: describe([r[k] for r in valid if r[k] is not None]) for k in DIAGNOSTICS
                },
            }
            cells.append(cell)
            for metric in DIAGNOSTICS:
                pair = [(r["quality"], r[metric]) for r in valid if r[metric] is not None]
                rho = None
                if len(pair) >= 3:
                    a, b = np.asarray(pair).T
                    if np.ptp(a) > 0 and np.ptp(b) > 0:
                        rho = float(stats.spearmanr(a, b).statistic)
                correlations.append(
                    {
                        "candidate": identity,
                        "width": w,
                        "metric": metric,
                        "pairs": len(pair),
                        "spearman_rho": rho,
                        "scope": "descriptive within candidate/W; no causal interpretation",
                    }
                )
            if identity == "regular":
                continue
            pairs = [(r, observations.get(("regular", w, r["seed"]))) for r in valid]
            pairs = [(a, b) for a, b in pairs if b is not None and b["primary_valid"]]
            differences = [a["quality"] - b["quality"] for a, b in pairs]
            binary = [int(a["success"]) - int(b["success"]) for a, b in pairs]
            comparisons.append(
                {
                    "candidate": identity,
                    "width": w,
                    "included_seeds": [a["seed"] for a, _ in pairs],
                    "differences": differences,
                    "quality": describe(differences),
                    "simultaneous": simultaneous_interval(differences),
                    "success_difference": describe(binary),
                    "diagnostic_differences": {
                        k: describe(
                            [a[k] - b[k] for a, b in pairs if a[k] is not None and b[k] is not None]
                        )
                        for k in DIAGNOSTICS
                    },
                }
            )
    crossover = {}
    primary = []
    ranking = []
    for identity in ids:
        block_means = []
        deltas = []
        included = []
        for seed in seeds:
            a = [observations.get((identity, w, seed)) for w in widths]
            b = [observations.get(("regular", w, seed)) for w in widths]
            if all(r is not None and r["primary_valid"] for r in a):
                block_means.append(float(np.mean([r["quality"] for r in a])))
            if all(r is not None and r["primary_valid"] for r in a + b):
                deltas.append(
                    float(np.mean([x["quality"] - y["quality"] for x, y in zip(a, b, strict=True)]))
                )
                included.append(seed)
        ranking.append(
            {"candidate": identity, **describe(block_means), "scope": "descriptive only"}
        )
        if identity == "regular":
            continue
        contrast = describe(deltas, alpha=0.05 / 3 if identity.startswith("historical_") else 0.05)
        ci = contrast["mean_bootstrap_ci"]
        primary.append(
            {
                "candidate": identity,
                "included_seeds": included,
                **contrast,
                "precision_insufficient": ci is None or (ci[1] - ci[0]) / 2 > 0.01,
                "confirmatory_family": identity.startswith("historical_"),
            }
        )
        by_w = [r for r in comparisons if r["candidate"] == identity]
        crossover[identity] = crossover_evidence(
            by_w, all_pairs_complete=all(r["quality"]["n"] == len(seeds) for r in by_w)
        )
    ranking.sort(key=lambda r: -(r["mean"] if r["mean"] is not None else -np.inf))
    for rank, r in enumerate(ranking, 1):
        r["rank"] = rank
    # Predefined discordance counts, descriptive only; no new pass criterion.
    discordance = []
    for identity in ids:
        raw = [
            r
            for r in data["rows"]
            if r["candidate"] == identity and r["kind"] == "disorder" and r["primary_valid"]
        ]
        success_rows = [r for r in raw if r["success"]]
        spatial = [r for r in success_rows if r["interior_nonzero_consistent_fraction"] is not None]
        gaps = [r for r in success_rows if r["interior_localizer_gap_min"] is not None]
        discordance.append(
            {
                "candidate": identity,
                "successes": len(success_rows),
                "spatial_available": len(spatial),
                "center_pass_interior_not_all_nonzero": sum(
                    r["interior_nonzero_consistent_fraction"] < 1 for r in spatial
                ),
                "interior_gap_available": len(gaps),
                "center_pass_interior_gap_below_point2": sum(
                    r["interior_localizer_gap_min"] < 0.2 for r in gaps
                ),
                "scope": "Different spatial requirements; not reclassified failures",
            }
        )
    return {
        "cells": cells,
        "paired_comparisons": comparisons,
        "primary_seed_blocks": primary,
        "crossover": crossover,
        "ranking_descriptive": ranking,
        "within_cell_correlations": correlations,
        "score_diagnostic_discordance": discordance,
        "complete_records": complete_records,
        "numerical": {
            "planned": len(ids) * (1 + len(widths) * len(seeds)),
            "present": len(data["rows"]),
            "attempts": len(data["attempts"]),
            "attempt_statuses": dict(Counter(a["status"] for a in data["attempts"])),
            "invalid_primary": sum(not r["primary_valid"] for r in data["rows"]),
            "max_numeric_residual": max(
                (r.get("maximum_numeric_residual", 0) for r in data["rows"]), default=None
            ),
            "result_checksum_digest": data["result_checksum_digest"],
            "attempt_digest": data["attempt_digest"],
            "paired_fields_checked": data["paired_fields_checked"],
        },
        "source_sha256": data["manifest"]["source_sha256"],
        "cohort_sha256": data["validation_cohort"]["sha256"],
        "widths": widths,
        "seeds": seeds,
        "elapsed_seconds": data["state"]["elapsed_seconds"],
        "scope": "Finite fixed-cohort confirmation on W=6..9; no low-W confirmation or causal mechanism",
    }


def export_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    stream = io.StringIO(newline="")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    writer = csv.DictWriter(stream, keys)
    writer.writeheader()
    writer.writerows(
        {k: dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows
    )
    atomic_text(path, stream.getvalue())


def plots(output: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original_ids = list(dict.fromkeys(r["candidate"] for r in summary["cells"]))
    colors = {name: plt.get_cmap("tab10")(i / 10) for i, name in enumerate(original_ids)}
    colors["regular"] = "black"
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    output.mkdir(parents=True, exist_ok=True)

    def save(fig: Any, name: str) -> None:
        fig.savefig(output / (name + ".png"), dpi=160)
        plt.close(fig)

    for metric in ("quality", "success"):
        fig, axes = plt.subplots(2, 5, figsize=(17, 8), constrained_layout=True)
        for identity, ax in zip(original_ids, axes.ravel(), strict=True):
            cells = [r for r in summary["cells"] if r["candidate"] == identity]
            xs = [r["width"] for r in cells]
            means = [
                r["quality"]["mean"] if metric == "quality" else r["success_fraction"]
                for r in cells
            ]
            ci = [
                r["quality"]["mean_bootstrap_ci"] if metric == "quality" else r["success_wilson95"]
                for r in cells
            ]
            ax.plot(xs, means, "o-", color=colors[identity])
            ax.fill_between(
                xs,
                [c[0] if c else np.nan for c in ci],
                [c[1] if c else np.nan for c in ci],
                alpha=0.2,
                color=colors[identity],
            )
            baseline = [r for r in summary["cells"] if r["candidate"] == "regular"]
            ax.plot(
                xs,
                [
                    r["quality"]["mean"] if metric == "quality" else r["success_fraction"]
                    for r in baseline
                ],
                "--",
                color="black",
                alpha=0.65,
                label="regular mean",
            )
            ax.set(
                title=identity,
                xlabel="W",
                ylabel="Q (unchanged center score)"
                if metric == "quality"
                else "P(Q >= 0.20 | valid)",
                ylim=(-0.01, 0.45) if metric == "quality" else (-0.03, 1.03),
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        fig.suptitle(
            "Confirmation, 50 paired seeds/cell | "
            + (
                "mean Q, pointwise bootstrap 95% CI"
                if metric == "quality"
                else "success, Wilson 95% CI"
            )
        )
        save(fig, "confirmation_" + metric)
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    for identity, ax in zip([i for i in original_ids if i != "regular"], axes.ravel(), strict=True):
        entries = [r for r in summary["paired_comparisons"] if r["candidate"] == identity]
        x = [r["width"] for r in entries]
        y = np.array([r["quality"]["mean"] for r in entries])
        outer = [r["simultaneous"]["interval"] for r in entries]
        inner = [r["quality"]["mean_bootstrap_ci"] for r in entries]
        ax.fill_between(
            x,
            [c[0] for c in outer],
            [c[1] for c in outer],
            color=colors[identity],
            alpha=0.15,
            label="simultaneous 45 contrasts",
        )
        ax.fill_between(
            x,
            [c[0] for c in inner],
            [c[1] for c in inner],
            color=colors[identity],
            alpha=0.35,
            label="pointwise bootstrap 95%",
        )
        ax.plot(x, y, "o-", color=colors[identity])
        ax.axhline(0, color="black", lw=0.8)
        ax.set(title=identity, xlabel="W", ylabel="paired Q - regular Q")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6)
    fig.suptitle("Paired differences | no crossover interpolation; all geometries retained")
    save(fig, "paired_differences")

    fig, axes = plt.subplots(1, 3, figsize=(17, 7), constrained_layout=True)
    for width, ax in zip((6, 7.5, 9), axes, strict=True):
        values = [
            [
                r["quality"]
                for r in rows
                if r["candidate"] == identity and r["width"] == width and r["primary_valid"]
            ]
            for identity in original_ids
        ]
        boxes = ax.boxplot(
            values,
            orientation="horizontal",
            tick_labels=original_ids,
            patch_artist=True,
            showmeans=True,
        )
        for patch, identity in zip(boxes["boxes"], original_ids, strict=True):
            patch.set_facecolor(colors[identity])
            patch.set_alpha(0.35)
        ax.set(
            title=f"W={width:g} (preselected)",
            xlabel="Q distribution; median, IQR, whiskers, outliers",
        )
        ax.axvline(0.2, ls=":", color="gray")
    fig.suptitle("Raw distributions | fixed endpoints and midpoint | 50 realizations per cell")
    save(fig, "distributions_fixed_W")

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    rank = summary["ranking_descriptive"]
    y = np.arange(len(rank))
    means = np.array([r["mean"] for r in rank])
    errors = np.array(
        [
            [r["mean"] - r["mean_bootstrap_ci"][0] for r in rank],
            [r["mean_bootstrap_ci"][1] - r["mean"] for r in rank],
        ]
    )
    ax.barh(y, means, color=[colors[r["candidate"]] for r in rank], alpha=0.65)
    ax.errorbar(means, y, xerr=errors, fmt="none", ecolor="black", capsize=3)
    ax.set_yticks(y, [r["candidate"] for r in rank])
    ax.invert_yaxis()
    ax.set(
        xlabel="mean Q over W, 50 whole-seed blocks; pointwise 95% CI",
        title="Descriptive ranking only; no geometry selection",
    )
    save(fig, "ranking_descriptive")

    selected = [
        "interior_localizer_gap_min",
        "edge_weight_1",
        "center_weight_16_sites",
        "chern_bulk_mean",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for metric, ax in zip(selected, axes.ravel(), strict=True):
        for identity in original_ids:
            cells = [r for r in summary["cells"] if r["candidate"] == identity]
            ax.plot(
                [r["width"] for r in cells],
                [r["diagnostics"][metric]["mean"] for r in cells],
                "o-",
                color=colors[identity],
                label=identity,
                markersize=3,
            )
        ax.set(xlabel="W", ylabel=metric)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle(
        "Independent diagnostic trends | conditional on valid Q, each diagnostic n in tables"
    )
    save(fig, "physical_diagnostics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study, output = args.study.resolve(), args.output.resolve()
    if output == study or output.is_relative_to(study):
        raise ValueError("Analysis output must be separate from frozen study")
    data = read_data(study)
    summary = analysis(data)
    output.mkdir(parents=True, exist_ok=True)
    summary["analysis_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary["analysis_plan_sha256"] = hashlib.sha256(
        (study / "analysis_plan.md").read_bytes()
    ).hexdigest()
    atomic_text(output / "analysis_script.py", Path(__file__).read_text(encoding="utf-8"))
    atomic_text(
        output / "analysis_plan.md", (study / "analysis_plan.md").read_text(encoding="utf-8")
    )
    atomic_text(output / "analysis.json", json.dumps(summary, indent=2, allow_nan=False))
    export_csv(output / "realization_diagnostics.csv", data["rows"])
    export_csv(
        output / "cell_statistics.csv",
        [
            {
                "candidate": r["candidate"],
                "width": r["width"],
                **r["quality"],
                "success_fraction": r["success_fraction"],
                "success_wilson95": r["success_wilson95"],
                "invalid": r["invalid"],
                "missing": r["missing"],
            }
            for r in summary["cells"]
        ],
    )
    export_csv(output / "paired_comparisons.csv", summary["paired_comparisons"])
    export_csv(
        output / "physical_diagnostics.csv",
        [
            {"candidate": r["candidate"], "width": r["width"], "metric": k, **v}
            for r in summary["cells"]
            for k, v in r["diagnostics"].items()
        ],
    )
    export_csv(output / "within_cell_correlations.csv", summary["within_cell_correlations"])
    export_csv(output / "ranking_descriptive.csv", summary["ranking_descriptive"])
    plots(output / "plots", summary, data["rows"])
    print(
        dumps(
            {
                "records": len(data["rows"]),
                "attempts": len(data["attempts"]),
                "crossover": {
                    k: v["statistically_supported_within_grid"]
                    for k, v in summary["crossover"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
