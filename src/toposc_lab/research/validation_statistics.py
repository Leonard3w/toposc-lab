"""Paired seed-block summaries with explicit missing/invalid denominators."""

from __future__ import annotations

import itertools
from typing import Any, TypeGuard

import numpy as np


def wilson(successes: int, n: int) -> list[float] | None:
    if not 0 <= successes <= n:
        raise ValueError("Invalid binomial counts")
    if n == 0:
        return None
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [float(max(0, center - half)), float(min(1, center + half))]


def paired_interval(
    values: list[float], *, alpha: float = 0.05, seed: int = 180900, resamples: int = 20000
) -> dict[str, Any]:
    data = np.asarray(values, dtype=float)
    if not np.isfinite(data).all():
        raise ValueError("Pair differences must be finite")
    result: dict[str, Any] = {
        "pairs": len(values),
        "mean": float(data.mean()) if len(data) else None,
        "interval": None,
        "confidence": 1 - alpha,
    }
    if len(data) >= 2:
        rng = np.random.default_rng(seed)
        means = data[rng.integers(0, len(data), (resamples, len(data)))].mean(axis=1)
        result["interval"] = np.quantile(means, [alpha / 2, 1 - alpha / 2]).tolist()
    result["method"] = "paired percentile bootstrap of independent seed blocks"
    result["small_sample_warning"] = len(data) < 20
    return result


def usable(result: dict[str, Any] | None) -> TypeGuard[dict[str, Any]]:
    return bool(
        result
        and result.get("status") == "completed"
        and result.get("primary_valid")
        and np.isfinite(result["metrics"]["quality"])
    )


def summarize(
    candidates: list[str],
    widths: list[float],
    seeds: list[int],
    records: dict[tuple[str, float, int], dict[str, Any]],
    *,
    confirmation: bool,
) -> dict[str, Any]:
    curves = []
    paired = []
    for identity in candidates:
        for width in widths:
            raw = [records.get((identity, width, seed)) for seed in seeds]
            valid = [r for r in raw if usable(r)]
            missing = sum(r is None for r in raw)
            invalid = len(raw) - missing - len(valid)
            success = sum(bool(r["metrics"]["success"]) for r in valid)
            q = [float(r["metrics"]["quality"]) for r in valid]
            row = {
                "candidate": identity,
                "width": width,
                "planned": len(seeds),
                "present": len(seeds) - missing,
                "valid": len(valid),
                "invalid": invalid,
                "missing": missing,
                "successes": success,
                "success_fraction_valid": success / len(valid) if valid else None,
                "wilson95_conditional_on_valid": wilson(success, len(valid)),
                "success_fraction_bounds_all_planned": [
                    success / len(seeds),
                    (success + invalid + missing) / len(seeds),
                ],
                "qualities": q,
                "quality_mean": float(np.mean(q)) if q else None,
                "quality_quantiles": np.quantile(q, [0, 0.25, 0.5, 0.75, 1]).tolist()
                if q
                else None,
            }
            curves.append(row)
            if identity == "regular":
                continue
            differences, binary, window = [], [], []
            for seed in seeds:
                a, b = records.get((identity, width, seed)), records.get(("regular", width, seed))
                if not usable(a) or not usable(b):
                    continue
                assert a is not None and b is not None
                differences.append(float(a["metrics"]["quality"] - b["metrics"]["quality"]))
                binary.append(float(a["metrics"]["success"]) - float(b["metrics"]["success"]))
                aw, bw = (
                    a.get("boundary_window", {}).get("window"),
                    b.get("boundary_window", {}).get("window"),
                )
                if aw is not None and bw is not None:
                    window.append(aw["strip_weights"]["1"] - bw["strip_weights"]["1"])
            paired.append(
                {
                    "candidate": identity,
                    "width": width,
                    "quality_difference": paired_interval(differences),
                    "success_difference": paired_interval(binary),
                    "boundary_strip1_difference": paired_interval(window),
                }
            )
    primary = []
    for identity in candidates:
        if identity == "regular":
            continue
        differences = []
        included_seeds = []
        for seed in seeds:
            block = [
                (records.get((identity, w, seed)), records.get(("regular", w, seed)))
                for w in widths
            ]
            valid_block = [(a, b) for a, b in block if usable(a) and usable(b)]
            if len(valid_block) == len(block):
                differences.append(
                    float(
                        np.mean(
                            [
                                a["metrics"]["quality"] - b["metrics"]["quality"]
                                for a, b in valid_block
                            ]
                        )
                    )
                )
                included_seeds.append(seed)
        confirmatory = identity.startswith("historical_")
        interval = paired_interval(differences, alpha=0.05 / 3 if confirmatory else 0.05)
        ci = interval["interval"]
        primary.append(
            {
                "candidate": identity,
                "included_seeds": included_seeds,
                "seed_block_differences": differences,
                **interval,
                "family": "three historical contrasts" if confirmatory else "exploratory",
                "claim_eligible": confirmation and confirmatory and len(differences) == len(seeds),
                "precision_insufficient": ci is None or (ci[1] - ci[0]) / 2 > 0.01,
            }
        )
    return {
        "curves": curves,
        "paired_by_width": paired,
        "primary_seed_block_contrasts": primary,
        "scope": "Confirmation"
        if confirmation
        else "Pilot: descriptive; not independent confirmation",
        "automatic_sample_extension": False,
        "W50": None,
        "W50_reason": "No monotonicity enforced or thermodynamic inference",
    }


def confirmation_grid(summary: dict[str, Any], pilot_widths: list[float]) -> dict[str, Any]:
    reference = [r for r in summary["curves"] if r["candidate"] == "regular"]
    by_width = {r["width"]: r for r in reference}
    rates = [by_width[w]["success_fraction_valid"] for w in pilot_widths]
    valid = all(by_width[w]["invalid"] == 0 and by_width[w]["missing"] == 0 for w in pilot_widths)
    nonmonotone = any(
        a is not None and b is not None and b > a for a, b in itertools.pairwise(rates)
    )
    if valid:
        for i, width in enumerate(pilot_widths):
            if rates[i] <= 0.5:
                low = pilot_widths[i - 1] if i else width / 5
                return {
                    "widths": np.linspace(low, width, 5).tolist(),
                    "rule": "first regular fraction <=0.5, preceding positive interval, 5 points",
                    "bracketed": True,
                    "pilot_nonmonotone": nonmonotone,
                }
    return {
        "widths": pilot_widths,
        "rule": "fallback to predeclared grid",
        "bracketed": False,
        "pilot_nonmonotone": nonmonotone,
        "open_question": "Pilot did not provide a valid sampled threshold bracket",
    }
