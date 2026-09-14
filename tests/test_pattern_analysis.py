import numpy as np
import pytest

from toposc_lab.patterns.analysis import (
    CONTROL_NAMES,
    DIAGNOSTICS,
    ablation_analysis,
    cohort_analysis,
    feature_matrix,
    matched_quality_comparison,
)
from toposc_lab.patterns.interventions import MOTIFS


def experiment(seed, effect, *, parent=None, motif=MOTIFS[0]):
    before = dict.fromkeys(DIAGNOSTICS, 0.1)
    target = {**before, "quality": 0.1 + effect}
    return {
        "motif": motif,
        "parent": str(seed) if parent is None else parent,
        "seed": seed,
        "target_id": f"target-{seed}",
        "control_id": f"control-{seed}",
        "before": before,
        "target": target,
        "control": before,
        "before_indices": [1, 1, 1],
        "target_indices": [1, 1, 1],
        "control_indices": [1, 1, 1],
        "target_counts": dict.fromkeys((*MOTIFS, "diagonal_count"), 2),
        "control_counts": dict.fromkeys((*MOTIFS, "diagonal_count"), 3),
    }


def rows_with_qualities(qualities):
    return [
        {
            "id": str(i),
            "cohort": "fresh",
            "seed": i % 2,
            "generator": "random",
            "origins": ["random"],
            "metrics": {**dict.fromkeys(DIAGNOSTICS, 0.1), "quality": quality},
            "features": {
                **dict.fromkeys(CONTROL_NAMES, 1.0),
                **{motif: float(i % 3) for motif in MOTIFS},
            },
        }
        for i, quality in enumerate(qualities)
    ]


def test_ablation_averages_replicates_within_parent_and_parents_within_seed():
    experiments = [
        experiment(0, -0.02),
        experiment(0, -0.06),
        experiment(0, 0, parent="other"),
        experiment(1, -0.06),
    ]
    report = ablation_analysis(experiments)[0]
    assert report["pairs"] == 4
    assert report["parents"] == 3
    assert report["estimates"]["target_control_quality"]["group_means"] == pytest.approx(
        {"0": -0.02, "1": -0.06}
    )
    assert report["estimates"]["target_control_quality"]["mean"] == pytest.approx(-0.04)
    assert report["survived"] is False


def test_ablation_survival_requires_eight_seed_evidence_and_multiplicity():
    report = ablation_analysis([experiment(i, -0.02) for i in range(8)])[0]
    assert report["sign_flip_p"] == 2 / 256
    assert report["holm_p"] == 8 / 256
    assert report["survived"] is True
    small = ablation_analysis([experiment(i, -0.02) for i in range(6)])[0]
    assert small["holm_p"] == 0.125
    assert small["survived"] is False


def test_opposite_ablation_effect_fails_predeclared_benefit_hypothesis():
    report = ablation_analysis([experiment(i, 0.02) for i in range(8)])[0]
    assert report["survived"] is False
    assert report["opposite_direction_evidence"] is True
    assert report["status"] == "failed_predeclared_benefit_direction_opposite_effect"
    assert all(not r["survived"] for r in ablation_analysis([]))


def test_matching_exposes_imperfect_seed_and_structural_balance():
    rows = rows_with_qualities([0.01, 0.1, 0.02, 0.15])
    rows[3]["features"]["boundary_degree_mean"] = 4.0
    report = matched_quality_comparison(rows, 0.08)
    assert report["high_count"] == report["low_count"] == 2
    assert report["same_seed_pair_count"] == 0
    assert report["maximum_control_rms_distance"] > 0
    assert any(
        p["structural_control_differences"]["boundary_degree_mean"] == 3 for p in report["pairs"]
    )
    assert report["causal_interpretation"] is False


def test_no_success_is_unavailable_and_successes_get_actual_comparison():
    distances = np.ones((4, 4)) - np.eye(4)
    failed = cohort_analysis(
        rows_with_qualities([0.01, 0.1, 0.02, 0.15]), distances, fresh=True, cutoff=0.08
    )
    assert failed["success_count"] == 0
    assert failed["successful_vs_unsuccessful"] is None
    succeeded = cohort_analysis(
        rows_with_qualities([0.01, 0.2, 0.02, 0.25]), distances, fresh=True, cutoff=0.08
    )
    assert succeeded["success_count"] == 2
    assert succeeded["successful_vs_unsuccessful"]["cutoff"] == 0.2
    assert (
        succeeded["successful_vs_unsuccessful"]["status"]
        == "descriptive_frozen_success_vs_unsuccessful"
    )


def test_feature_matrix_excludes_constants_and_explicit_unavailability():
    rows = rows_with_qualities([0.1, 0.2])
    for row in rows:
        row["features"]["dimension"] = None
    x, names, excluded = feature_matrix(rows)
    assert x.shape == (2, len(MOTIFS))
    assert set(names) == set(MOTIFS)
    assert "dimension" in excluded
    rows[0]["features"]["dimension"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        feature_matrix(rows)
