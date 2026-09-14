"""Exploratory rank associations and grouped intervention uncertainty."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import false_discovery_control, rankdata

from toposc_lab.ml.models import RidgeRegressor


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(values)), controls))
    residual = values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
    # Exact collinearity is scientific non-identifiability, not a tiny noisy signal.
    scale = np.maximum(1.0, np.linalg.norm(values, axis=0))
    negligible = np.linalg.norm(residual, axis=0) <= 1e-12 * scale
    return np.where(negligible, 0.0, residual)


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    a, b = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / denominator, -1, 1)) if denominator > 1e-12 else None


def family_labels(distances: np.ndarray, radius: float) -> np.ndarray:
    """Transitive single-link families; input is a square precomputed distance."""
    if not 0 <= radius <= 1:
        raise ValueError("radius must lie in [0,1]")
    _validate_distances(distances)
    if len(distances) <= 1:
        return np.ones(len(distances), dtype=int)
    return np.asarray(
        fcluster(
            linkage(squareform(distances, checks=True), method="single"),
            t=radius,
            criterion="distance",
        ),
        dtype=int,
    )


def grouped_mean_interval(
    values: np.ndarray, groups: np.ndarray, seed: int, samples: int = 4000
) -> dict[str, Any]:
    """Equal group weights; resample independent group means, never edit replicas."""
    if len(values) != len(groups) or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty values and aligned groups required")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    labels = np.unique(groups)
    means = np.array([values[groups == g].mean() for g in labels])
    result: dict[str, Any] = {
        "mean": float(means.mean()),
        "groups": len(labels),
        "group_means": {str(g): float(v) for g, v in zip(labels, means, strict=True)},
        "ci95": None,
    }
    if len(means) > 1:
        draws = np.random.default_rng(seed).choice(means, (samples, len(means))).mean(axis=1)
        result["ci95"] = np.quantile(draws, [0.025, 0.975]).tolist()
    return result


def sign_flip_pvalue(group_means: np.ndarray) -> float:
    """Exact two-sided sign-flip test at the independent group level."""
    if not 1 <= len(group_means) <= 16 or not np.isfinite(group_means).all():
        raise ValueError("one to sixteen finite group means required")
    signs = np.array(list(product((-1, 1), repeat=len(group_means))))
    null = np.abs(signs @ group_means / len(group_means))
    return float(np.mean(null >= abs(group_means.mean()) - 1e-14))


def holm(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite in [0,1]")
    order = np.argsort(values)
    adjusted = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    result = np.empty(len(values))
    result[order] = adjusted
    return result.tolist()


def rank_associations(
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    controls: np.ndarray,
    strata: np.ndarray,
    families: np.ndarray,
    seed: int,
    permutations: int = 1999,
    bootstrap_samples: int = 1000,
) -> list[dict[str, Any]]:
    """Partial Spearman via rank residuals; approximate restricted residual null.

    Permutation validity relies on exchangeable regression residuals within the
    supplied strata. Permuted target residuals are projected against the nuisance
    design again (the residual form of Freedman-Lane). Intervals refit rank
    adjustment under family resampling. Untestable variable features retain a
    placeholder p=1 in the multiple-testing family, with their result left null.
    """
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or controls.ndim != 2
        or len(controls) != len(x)
        or strata.shape != y.shape
        or families.shape != y.shape
        or len(names) != x.shape[1]
        or not len(x)
        or not all(np.isfinite(a).all() for a in (x, y, controls))
    ):
        raise ValueError("finite aligned feature, outcome, control and group arrays required")
    for count in (permutations, bootstrap_samples):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("permutation and bootstrap counts must be positive integers")
    ranks = rankdata(x, axis=0)
    target = rankdata(y)
    covariates = rankdata(controls, axis=0)
    rx, ry = residualize(ranks, covariates), residualize(target, covariates)
    rng = np.random.default_rng(seed)
    null_targets = []
    for _ in range(permutations):
        permuted = ry.copy()
        for group in np.unique(strata):
            indices = np.flatnonzero(strata == group)
            permuted[indices] = ry[rng.permutation(indices)]
        null_targets.append(permuted)
    null = residualize(np.asarray(null_targets).T, covariates).T
    null_norm = np.linalg.norm(null, axis=1)
    rows = []
    valid = []
    for j, name in enumerate(names):
        observed = correlation(rx[:, j], ry)
        row: dict[str, Any] = {
            "feature": name,
            "spearman": correlation(ranks[:, j], target),
            "partial_spearman": observed,
            "p": None,
            "ci95": None,
            "bh_q": None,
            "by_q": None,
            "status": "unidentifiable_after_adjustment" if observed is None else "exploratory",
            "multiplicity_family_size": len(names),
            "valid_bootstrap_draws": 0,
        }
        if observed is not None:
            denominator = np.linalg.norm(rx[:, j]) * null_norm
            null_rho = np.divide(
                null @ rx[:, j], denominator, out=np.zeros(len(null)), where=denominator > 1e-12
            )
            row["p"] = float(
                (1 + np.sum(np.abs(null_rho) >= abs(observed) - 1e-14)) / (permutations + 1)
            )
            valid.append(j)
        rows.append(row)
    if valid:
        ps = [row["p"] if row["p"] is not None else 1.0 for row in rows]
        for method, key in (("bh", "bh_q"), ("by", "by_q")):
            adjusted = false_discovery_control(ps, method=method)
            for j in valid:
                rows[j][key] = float(adjusted[j])
    labels = np.unique(families)
    bootstrap: list[list[float]] = [[] for _ in names]
    if len(labels) > 1:
        for _ in range(bootstrap_samples):
            chosen = rng.choice(labels, len(labels))
            indices = np.concatenate([np.flatnonzero(families == g) for g in chosen])
            cx = rankdata(x[indices], axis=0)
            cy = rankdata(y[indices])
            cc = rankdata(controls[indices], axis=0)
            bx, by = residualize(cx, cc), residualize(cy, cc)
            for j in valid:
                value = correlation(bx[:, j], by)
                if value is not None:
                    bootstrap[j].append(value)
        for j in valid:
            if len(bootstrap[j]) >= 0.9 * bootstrap_samples:
                rows[j]["ci95"] = np.quantile(bootstrap[j], [0.025, 0.975]).tolist()
            rows[j]["valid_bootstrap_draws"] = len(bootstrap[j])
    return rows


def purged_group_splits(
    groups: np.ndarray, distances: np.ndarray, radius: float = 0.15
) -> list[tuple[np.ndarray, np.ndarray]]:
    _validate_distances(distances)
    if groups.shape != (len(distances),) or not 0 <= radius <= 1:
        raise ValueError("aligned groups and a radius in [0,1] required")
    result = []
    for group in np.unique(groups):
        test = np.flatnonzero(groups == group)
        train = np.flatnonzero((groups != group) & (distances[:, test].min(axis=1) > radius))
        result.append((train, test))
    return result


def importance_stability(
    x: np.ndarray, y: np.ndarray, names: list[str], groups: np.ndarray, distances: np.ndarray
) -> dict[str, Any]:
    """Fixed ridge(alpha=10) and five-neighbor regression, no test-label fitting."""
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or len(names) != x.shape[1]
        or groups.shape != y.shape
        or distances.shape != (len(x), len(x))
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("finite aligned features, targets, groups and distances required")
    folds: list[dict[str, Any]] = []
    for fold_index, (train, test) in enumerate(purged_group_splits(groups, distances)):
        if len(train) < 8 or not x.shape[1]:
            folds.append(
                {
                    "heldout": str(groups[test[0]]),
                    "status": "insufficient_training",
                    "train_indices": train.tolist(),
                    "test_indices": test.tolist(),
                }
            )
            continue
        mean, sd = x[train].mean(axis=0), x[train].std(axis=0)
        sd[sd < 1e-12] = 1
        z, t = (x[train] - mean) / sd, (x[test] - mean) / sd
        intercept = float(y[train].mean())
        ridge = RidgeRegressor(alpha=10).fit(x[train], y[train])
        for model in ("ridge_10", "knn_5"):

            def predict(
                values: np.ndarray,
                model: str = model,
                ridge: RidgeRegressor = ridge,
                z: np.ndarray = z,
                targets: np.ndarray = y[train],
                mean: np.ndarray = mean,
                sd: np.ndarray = sd,
            ) -> np.ndarray:
                if model == "ridge_10":
                    return ridge.predict(values * sd + mean)
                nearest = np.argsort(np.sum((values[:, None] - z[None, :]) ** 2, axis=2), axis=1)[
                    :, :5
                ]
                return targets[nearest].mean(axis=1)

            predictions = predict(t)
            mse = float(np.mean((predictions - y[test]) ** 2))
            effects = []
            for repeat in (16201, 16202, 16203):
                rng = np.random.default_rng([repeat, fold_index])
                row = []
                for j in range(x.shape[1]):
                    changed = t.copy()
                    changed[:, j] = t[rng.permutation(len(test)), j]
                    row.append(float(np.mean((predict(changed) - y[test]) ** 2) - mse))
                effects.append(row)
            folds.append(
                {
                    "heldout": str(groups[test[0]]),
                    "model": model,
                    "train_indices": train.tolist(),
                    "test_indices": test.tolist(),
                    "mse": mse,
                    "mean_baseline_mse": float(np.mean((intercept - y[test]) ** 2)),
                    "training_mean": intercept,
                    "training_feature_mean": mean.tolist(),
                    "training_feature_scale": sd.tolist(),
                    "predictions": predictions.tolist(),
                    "targets": y[test].tolist(),
                    "permutation_mse_increases": effects,
                }
            )
    summaries: list[dict[str, Any]] = []
    for model in ("ridge_10", "knn_5"):
        selected = [f for f in folds if f.get("model") == model]
        if not selected:
            continue
        fold_effects = np.array([np.mean(f["permutation_mse_increases"], axis=0) for f in selected])
        ranks = rankdata(-fold_effects, axis=1, method="average")
        fold_mse = np.array([f["mse"] for f in selected])
        baseline = np.array([f["mean_baseline_mse"] for f in selected])
        summaries.append(
            {
                "model": model,
                "mean_mse": float(fold_mse.mean()),
                "mean_baseline_mse": float(baseline.mean()),
                "baseline_improvement": grouped_mean_interval(
                    baseline - fold_mse, np.arange(len(selected)), 16204
                ),
                "baseline_beaten_fold_fraction": float(np.mean(fold_mse < baseline)),
                "evaluated_fold_count": len(selected),
                "features": [
                    {
                        "feature": name,
                        "mean_mse_increase": float(fold_effects[:, j].mean()),
                        "positive_fold_fraction": float(np.mean(fold_effects[:, j] > 0)),
                        "top5_fold_fraction": float(np.mean(ranks[:, j] <= 5)),
                        "mean_rank": float(ranks[:, j].mean()),
                        "rank_sd": float(ranks[:, j].std()),
                        "fold_mean_mse_increases": fold_effects[:, j].tolist(),
                        "seed_bootstrap": grouped_mean_interval(
                            fold_effects[:, j], np.arange(len(selected)), 16205
                        ),
                    }
                    for j, name in enumerate(names)
                ],
            }
        )
    agreement = None
    if len(summaries) == 2:
        agreement = correlation(
            rankdata([f["mean_mse_increase"] for f in summaries[0]["features"]]),
            rankdata([f["mean_mse_increase"] for f in summaries[1]["features"]]),
        )
    return {
        "folds": folds,
        "summaries": summaries,
        "cross_model_importance_spearman": agreement,
        "uncertainty_note": "Seed-fold bootstrap is descriptive: training folds overlap.",
        "causal_interpretation": False,
    }


def _validate_distances(distances: np.ndarray) -> None:
    if (
        distances.ndim != 2
        or distances.shape[0] != distances.shape[1]
        or not np.isfinite(distances).all()
        or np.any((distances < 0) | (distances > 1))
        or not np.allclose(distances, distances.T, rtol=0, atol=1e-12)
        or not np.allclose(np.diag(distances), 0, rtol=0, atol=1e-12)
    ):
        raise ValueError("finite symmetric square distances in [0,1] with zero diagonal required")
