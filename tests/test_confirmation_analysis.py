"""Verify postprocessing inference without running new physical realizations."""

from copy import deepcopy

import numpy as np
import pytest

from scripts.phase18_confirmation_analysis import (
    DIAGNOSTICS,
    analysis,
    crossover_evidence,
    describe,
    extract_record,
    simultaneous_interval,
)


def test_paired_seed_blocks_preserve_dependence_across_widths():
    widths, seeds = [6, 6.75, 7.5, 8.25, 9], [1, 2, 3, 4]
    rows = []
    for identity in ("regular", "historical_best"):
        for width, seed, kind in [(0, 0, "clean")] + [
            (w, s, "disorder") for w in widths for s in seeds
        ]:
            q = 0.1 + seed * 0.01 + (seed * 0.02 if identity == "historical_best" else 0)
            rows.append(
                {
                    "candidate": identity,
                    "width": width,
                    "seed": seed,
                    "kind": kind,
                    "primary_valid": True,
                    "quality": q,
                    "success": q >= 0.2,
                    **dict.fromkeys(DIAGNOSTICS),
                }
            )
    # Reverse storage order: matching must be by identity, W and seed, never row order.
    data = {
        "rows": rows[::-1],
        "validation_settings": {"widths": widths, "seeds": seeds, "clean_seed": 0},
        "validation_cohort": {
            "candidates": [{"id": "regular"}, {"id": "historical_best"}],
            "sha256": "synthetic",
        },
        "manifest": {"source_sha256": "synthetic"},
        "state": {"elapsed_seconds": 0},
        "attempts": [],
        "result_checksum_digest": "synthetic",
        "attempt_digest": "synthetic",
        "paired_fields_checked": 20,
    }
    summary = analysis(data)
    primary = summary["primary_seed_blocks"][0]
    assert summary["complete_records"]
    assert primary["n"] == 4  # Not 20 pseudo-independent observations.
    assert primary["mean"] == pytest.approx(0.05)
    assert primary["sem"] == pytest.approx(np.std([0.02, 0.04, 0.06, 0.08], ddof=1) / 2)
    assert summary["paired_comparisons"][0]["differences"] == pytest.approx(
        [0.02, 0.04, 0.06, 0.08]
    )
    missing = deepcopy(data)
    missing["rows"] = [
        r
        for r in missing["rows"]
        if not (r["candidate"] == "regular" and r["width"] == 6 and r["seed"] == 1)
    ]
    summary = analysis(missing)
    assert not summary["complete_records"]
    assert summary["primary_seed_blocks"][0]["n"] == 3
    assert not summary["crossover"]["historical_best"]["complete_primary_pairs"]
    duplicate = deepcopy(data)
    duplicate["rows"].append(duplicate["rows"][0])
    with pytest.raises(ValueError, match="duplicate"):
        analysis(duplicate)


def test_sample_uncertainty_and_missing_values():
    result = describe([1.0, 2.0, 3.0])
    assert result["mean"] == result["median"] == 2
    assert result["sd"] == 1
    assert result["sem"] == pytest.approx(1 / np.sqrt(3))
    assert result["quantiles"]["0.25"] == 1.5
    assert describe([])["mean"] is None
    assert describe([0.0])["mean_bootstrap_ci"] is None
    with pytest.raises(ValueError, match="Nonfinite"):
        describe([1.0, np.nan])


def entry(width, values):
    return {
        "width": width,
        "quality": describe(values),
        "simultaneous": simultaneous_interval(values),
    }


def test_crossover_requires_two_supported_sides_and_complete_pairs():
    rows = [entry(6, [0.1] * 50), entry(9, [-0.1] * 50)]
    assert crossover_evidence(rows, all_pairs_complete=True)["statistically_supported_within_grid"]
    assert not crossover_evidence(rows, all_pairs_complete=False)[
        "statistically_supported_within_grid"
    ]
    negative_only = [entry(6, [-0.05] * 50), rows[1]]
    assert not crossover_evidence(negative_only, all_pairs_complete=True)[
        "statistically_supported_within_grid"
    ]
    reverse = [entry(6, [-0.1] * 50), entry(9, [0.1] * 50)]
    assert not crossover_evidence(reverse, all_pairs_complete=True)[
        "statistically_supported_within_grid"
    ]


def test_numeric_sign_change_does_not_imply_significant_crossover():
    rows = [entry(6, [-1, 1.02] * 25), entry(9, [-1.02, 1] * 25)]
    result = crossover_evidence(rows, all_pairs_complete=True)
    assert result["numerical_sign_change_pairs"] == [[6, 9]]
    assert not result["statistically_supported_within_grid"]
    interval = simultaneous_interval([-1, 1.02] * 25)["interval"]
    pointwise = describe([-1, 1.02] * 25)["mean_t_ci"]
    assert interval[0] < pointwise[0] < pointwise[1] < interval[1]


def test_invalid_record_is_not_physical_zero():
    result = extract_record(
        "regular",
        {
            "stage": {"width": 6, "seed": 1, "kind": "disorder"},
            "status": "failed",
            "error": "synthetic",
        },
        np.zeros((100, 2)),
    )
    assert result["quality"] is None
    assert result["success"] is None
    assert not result["primary_valid"]


def test_center_region_and_spatial_topology_remain_independent_of_score():
    coordinates = np.array([(x, y) for x in range(10) for y in range(10)])
    points = [
        {"point": [x, y], "valid": True, "index": 1, "gap": 0.3}
        for x in (2, 4.5, 7)
        for y in (2, 4.5, 7)
        for _ in range(3)
    ]
    result = {
        "stage": {"width": 6, "seed": 1, "kind": "disorder"},
        "status": "completed",
        "primary_valid": True,
        "metrics": {
            "quality": 0.3,
            "success": True,
            "localizer_gap": 0.3,
            "minimum_abs_energy": 0.01,
            "boundary_weight": 0.5,
            "eligible": True,
            **dict.fromkeys(
                (
                    "operator_phs_residual",
                    "spectral_phs_residual",
                    "hermiticity_residual",
                    "eigensystem_residual",
                ),
                0,
            ),
        },
        "indices": [1, 1, 1],
        "spatial": points,
        "boundary_window": {
            "state_count": 2,
            "distance_to_boundary": [0] * 100,
            "window": {
                "site_probability": [0.01] * 100,
                "strip_weights": {"1": 1, "2": 1, "3": 1},
                "mean_boundary_distance": 0,
            },
        },
        "chern_marker": {"status": "unavailable"},
        "majorana": {"states": []},
    }
    row = extract_record("regular", result, coordinates)
    assert row["center_weight_16_sites"] == pytest.approx(0.16)
    altered = deepcopy(result)
    altered["spatial"][0]["index"] = 0
    changed = extract_record("regular", altered, coordinates)
    assert changed["interior_nonzero_consistent_fraction"] == pytest.approx(8 / 9)
    assert changed["quality"] == row["quality"]
    assert changed["success"] == row["success"]
    altered["boundary_window"]["window"]["site_probability"] = [0.02] * 100
    with pytest.raises(ValueError, match="unnormalized"):
        extract_record("regular", altered, coordinates)
