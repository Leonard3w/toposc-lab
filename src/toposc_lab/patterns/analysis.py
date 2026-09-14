"""Stored-result-only Phase-16A analysis; no solver calls or adaptive search."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from toposc_lab.patterns.interventions import MOTIFS
from toposc_lab.patterns.statistics import (
    correlation,
    family_labels,
    grouped_mean_interval,
    holm,
    importance_stability,
    rank_associations,
    residualize,
    sign_flip_pvalue,
)

CONTROL_NAMES = tuple(f"degree_fraction_{i}" for i in range(2, 7)) + ("boundary_degree_mean",)
DIAGNOSTICS = ("quality", "localizer_gap", "minimum_abs_energy", "boundary_weight", "eligible")


def feature_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str], list[str]]:
    if not rows:
        raise ValueError("at least one geometry record is required")
    all_names = sorted(rows[0]["features"])
    if any(sorted(r["features"]) != all_names for r in rows):
        raise ValueError("geometry records must share the same feature schema")
    if any(v is not None and not np.isfinite(v) for r in rows for v in r["features"].values()):
        raise ValueError("features must be finite or explicitly unavailable")
    names = [
        k
        for k in all_names
        if all(r["features"][k] is not None for r in rows)
        and np.std([r["features"][k] for r in rows]) > 1e-12
    ]
    x = np.array([[r["features"][k] for k in names] for r in rows], dtype=float)
    return x, names, [k for k in all_names if k not in names]


def _controls(rows: list[dict[str, Any]]) -> np.ndarray:
    structural = np.array([[r["features"][k] for k in CONTROL_NAMES] for r in rows])
    origins = sorted({o for r in rows for o in r["origins"]})
    membership = np.array([[float(o in r["origins"]) for o in origins] for r in rows])
    controls = np.column_stack((structural, membership))
    if all(r.get("cohort") == "fresh" for r in rows):
        seeds = np.array([r["seed"] for r in rows])
        controls = np.column_stack((controls, seeds[:, None] == np.unique(seeds)[None, :]))
    return controls


def matched_quality_comparison(rows: list[dict[str, Any]], cutoff: float) -> dict[str, Any]:
    """Exploratory high/low labels are fixed separately from frozen success."""
    high = [i for i, r in enumerate(rows) if r["metrics"]["quality"] >= cutoff]
    low = [i for i, r in enumerate(rows) if r["metrics"]["quality"] < cutoff]
    if not high or not low:
        return {"status": "unavailable: empty group", "cutoff": cutoff}
    controls = _controls(rows)
    sd = controls.std(axis=0)
    sd[sd < 1e-12] = 1
    distances = cdist(controls[high] / sd, controls[low] / sd) / np.sqrt(controls.shape[1])
    a, b = linear_sum_assignment(distances)
    pairs = [
        {
            "high": rows[high[i]]["id"],
            "low": rows[low[j]]["id"],
            "control_rms_distance": float(distances[i, j]),
            "same_origin_membership": set(rows[high[i]]["origins"]) == set(rows[low[j]]["origins"]),
            "same_seed": rows[high[i]].get("seed") == rows[low[j]].get("seed"),
            "structural_control_differences": {
                k: rows[high[i]]["features"][k] - rows[low[j]]["features"][k] for k in CONTROL_NAMES
            },
            "quality_difference": rows[high[i]]["metrics"]["quality"]
            - rows[low[j]]["metrics"]["quality"],
            "motif_differences": {
                k: rows[high[i]]["features"][k] - rows[low[j]]["features"][k] for k in MOTIFS
            },
        }
        for i, j in zip(a, b, strict=True)
    ]
    return {
        "status": "exploratory_higher_vs_lower_quality_not_success",
        "cutoff": cutoff,
        "high_count": len(high),
        "low_count": len(low),
        "pairs": pairs,
        "mean_motif_differences": {
            k: float(np.mean([p["motif_differences"][k] for p in pairs])) for k in MOTIFS
        },
        "maximum_control_rms_distance": max(p["control_rms_distance"] for p in pairs),
        "same_seed_pair_count": sum(p["same_seed"] for p in pairs),
        "same_origin_pair_count": sum(p["same_origin_membership"] for p in pairs),
        "matching_note": "Minimum total covariate distance; imperfect matches are retained and displayed.",
        "causal_interpretation": False,
    }


def cohort_analysis(
    rows: list[dict[str, Any]], distances: np.ndarray, *, fresh: bool, cutoff: float
) -> dict[str, Any]:
    x, names, excluded = feature_matrix(rows)
    y = np.array([r["metrics"]["quality"] for r in rows])
    controls = _controls(rows)
    if fresh:
        seeds = np.array([r["seed"] for r in rows])
        strata = np.array([f"{r['seed']}-{r['generator']}" for r in rows])
        # Seed blocks, more conservative than treating independent accepted proposals as iid.
        bootstrap_groups = seeds
    else:
        strata = np.array(["/".join(sorted(r["origins"])) for r in rows])
        bootstrap_groups = family_labels(distances, 0.15)
    associations = rank_associations(
        x, y, names, controls, strata, bootstrap_groups, 16502 if fresh else 16501
    )
    if fresh:
        for row in associations:
            values = []
            j = names.index(row["feature"])
            for seed in np.unique(seeds):
                keep = seeds != seed
                c = rankdata(controls[keep], axis=0)
                values.append(
                    correlation(
                        residualize(rankdata(x[keep, j]), c), residualize(rankdata(y[keep]), c)
                    )
                )
            row["leave_one_seed_out_partial_rho"] = values
            available = [v for v in values if v is not None]
            observed = row["partial_spearman"]
            row["leave_one_seed_out_sign_agreement"] = (
                float(np.mean([np.sign(v) == np.sign(observed) for v in available]))
                if available and observed is not None
                else None
            )
    families = {}
    for radius in (0.10, 0.15, 0.20):
        labels = family_labels(distances, radius)
        families[str(radius)] = [
            {
                "family": int(g),
                "ids": [rows[i]["id"] for i in np.flatnonzero(labels == g)],
                "size": int(np.sum(labels == g)),
                "maximum_internal_distance": float(
                    distances[np.ix_(labels == g, labels == g)].max()
                ),
                "quality_mean": float(y[labels == g].mean()),
            }
            for g in np.unique(labels)
        ]
    z = (x - x.mean(axis=0)) / x.std(axis=0) if x.shape[1] else x
    feature_clusters = {}
    for k in (2, 3, 4):
        labels = (
            fcluster(linkage(z, method="ward"), k, criterion="maxclust")
            if len(rows) > 1 and x.shape[1]
            else np.ones(len(rows), dtype=int)
        )
        feature_clusters[str(k)] = [
            {
                "cluster": int(g),
                "count": int(np.sum(labels == g)),
                "ids": [rows[i]["id"] for i in np.flatnonzero(labels == g)],
                "quality_mean": float(y[labels == g].mean()),
                "motif_means": {
                    m: float(np.mean([rows[i]["features"][m] for i in np.flatnonzero(labels == g)]))
                    for m in MOTIFS
                },
            }
            for g in np.unique(labels)
        ]
    success_count = int(np.sum(y >= 0.2))
    success_comparison = None
    if 0 < success_count < len(rows):
        success_comparison = matched_quality_comparison(rows, 0.2)
        success_comparison["status"] = "descriptive_frozen_success_vs_unsuccessful"
    return {
        "count": len(rows),
        "success_count": int(np.sum(y >= 0.2)),
        "success_comparison": (
            "unavailable: zero frozen successes"
            if not success_count
            else "unavailable: zero unsuccessful candidates"
            if success_count == len(rows)
            else "available: descriptive comparison, not causal"
        ),
        "successful_vs_unsuccessful": success_comparison,
        "eligible_count": sum(bool(r["metrics"]["eligible"]) for r in rows),
        "quality_mean": float(y.mean()),
        "quality_max": float(y.max()),
        "feature_names": names,
        "excluded_constant_or_unavailable": excluded,
        "associations": sorted(associations, key=lambda r: -abs(r["partial_spearman"] or 0)),
        "secondary_descriptive_spearman": {
            d: {
                name: correlation(rankdata(x[:, j]), rankdata([r["metrics"][d] for r in rows]))
                for j, name in enumerate(names)
            }
            for d in DIAGNOSTICS[1:]
        },
        "matched_higher_quality": matched_quality_comparison(rows, cutoff),
        "structural_families": families,
        "feature_clusters": feature_clusters,
        "importance": importance_stability(x, y, names, seeds, distances) if fresh else None,
    }


def ablation_analysis(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports = []
    for motif in MOTIFS:
        selected = [r for r in experiments if r["motif"] == motif]
        parents = sorted({r["parent"] for r in selected})
        effects = []
        for parent in parents:
            members = [r for r in selected if r["parent"] == parent]
            item: dict[str, Any] = {
                "parent": parent,
                "seed": members[0]["seed"],
                "replicates": len(members),
            }
            for diagnostic in DIAGNOSTICS:
                for comparison, first, second in (
                    ("target_control", "target", "control"),
                    ("target_parent", "target", "before"),
                    ("control_parent", "control", "before"),
                ):
                    item[f"{comparison}_{diagnostic}"] = float(
                        np.mean([r[first][diagnostic] - r[second][diagnostic] for r in members])
                    )
            effects.append(item)
        report: dict[str, Any] = {
            "motif": motif,
            "parents": len(parents),
            "pairs": len(selected),
            "parent_effects": effects,
            "survived": False,
            "holm_p": None,
            "status": "unavailable",
        }
        if effects:
            seeds = np.array([p["seed"] for p in effects])
            report["estimates"] = {
                key: grouped_mean_interval(np.array([p[key] for p in effects]), seeds, 16601)
                for key in effects[0]
                if key.startswith(("target_", "control_"))
            }
            quality = np.array([p["target_control_quality"] for p in effects])
            report["parent_bootstrap"] = grouped_mean_interval(
                quality, np.arange(len(quality)), 16602
            )
            mean_by_seed = np.array([quality[seeds == s].mean() for s in np.unique(seeds)])
            report["sign_flip_p"] = sign_flip_pvalue(mean_by_seed)
            report["negative_seeds"] = int(np.sum(mean_by_seed < 0))
            report["positive_seeds"] = int(np.sum(mean_by_seed > 0))
            report["seed_count"] = len(mean_by_seed)
            report["minimum_two_sided_sign_flip_p"] = 2 / 2 ** len(mean_by_seed)
            sd = float(mean_by_seed.std(ddof=1)) if len(mean_by_seed) > 1 else 0.0
            report["seed_standardized_effect"] = float(mean_by_seed.mean() / sd) if sd > 0 else None
            report["predeclared_direction"] = (
                "negative target-minus-control: removing a beneficial motif reduces quality"
            )
            report["status"] = "inconclusive_no_controlled_benefit_established"
            report["topology_transitions"] = [
                {
                    "parent": r["parent"],
                    "target_id": r["target_id"],
                    "control_id": r["control_id"],
                    "before": r["before_indices"],
                    "target": r["target_indices"],
                    "control": r["control_indices"],
                }
                for r in selected
                if r["before_indices"] != r["target_indices"]
                or r["before_indices"] != r["control_indices"]
            ]
            report["non_target_count_differences"] = {
                k: [r["target_counts"][k] - r["control_counts"][k] for r in selected]
                for k in (*MOTIFS, "diagonal_count")
            }
        reports.append(report)
    adjusted = holm([r.get("sign_flip_p", 1.0) for r in reports])
    for report, p in zip(reports, adjusted, strict=True):
        report["holm_p"] = p
        if "estimates" in report:
            interval = report["estimates"]["target_control_quality"]["ci95"]
            report["survived"] = bool(
                interval is not None
                and interval[1] < 0
                and report["negative_seeds"] >= 7
                and p <= 0.05
            )
            report["opposite_direction_evidence"] = bool(
                interval is not None
                and interval[0] > 0
                and report["positive_seeds"] >= 7
                and p <= 0.05
            )
            if report["survived"]:
                report["status"] = "survived_limited_finite_model_ablation"
            elif report["opposite_direction_evidence"]:
                report["status"] = "failed_predeclared_benefit_direction_opposite_effect"
    return reports


def analyze(
    rows: list[dict[str, Any]],
    distances: np.ndarray,
    experiments: list[dict[str, Any]],
    simplifications: list[dict[str, Any]],
) -> dict[str, Any]:
    legacy_indices = np.array([i for i, r in enumerate(rows) if r["cohort"] == "legacy"])
    fresh_indices = np.array([i for i, r in enumerate(rows) if r["cohort"] == "fresh"])
    legacy = [rows[i] for i in legacy_indices]
    fresh = [rows[i] for i in fresh_indices]
    cutoff = float(np.quantile([r["metrics"]["quality"] for r in legacy], 0.75))
    lx, names, _ = feature_matrix(legacy)
    fx = np.array([[r["features"][k] for k in names] for r in fresh])
    standardized = (fx - lx.mean(axis=0)) / lx.std(axis=0)
    outliers = sorted(
        [
            {
                "id": r["id"],
                "seed": r["seed"],
                "quality": r["metrics"]["quality"],
                "rms_distance_from_legacy_center": float(np.sqrt(np.mean(standardized[i] ** 2))),
                "nearest_legacy_structural_distance": float(
                    distances[fresh_indices[i], legacy_indices].min()
                ),
                "largest_standardized_features": [
                    {"feature": names[j], "z": float(standardized[i, j])}
                    for j in np.argsort(-np.abs(standardized[i]))[:3]
                ],
            }
            for i, r in enumerate(fresh)
        ],
        key=lambda r: -r["rms_distance_from_legacy_center"],
    )
    result = {
        "schema_version": 1,
        "success_threshold": 0.2,
        "legacy_quartile_cutoff": cutoff,
        "legacy": cohort_analysis(
            legacy, distances[np.ix_(legacy_indices, legacy_indices)], fresh=False, cutoff=cutoff
        ),
        "fresh": cohort_analysis(
            fresh, distances[np.ix_(fresh_indices, fresh_indices)], fresh=True, cutoff=cutoff
        ),
        "outliers": outliers,
        "ablations": ablation_analysis(experiments),
        "simplifications": simplifications,
        "phase16b_started": False,
        "finite_size_evidence": "unavailable",
        "majorana_claim": False,
    }
    return result
