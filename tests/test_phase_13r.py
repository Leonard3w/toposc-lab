from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from toposc_lab.active_learning.acquisition import AcquisitionConfig, acquire
from toposc_lab.active_learning.prediction import Prediction
from toposc_lab.ml import FeatureOODDetector


def test_extrapolation_flag_does_not_change_scores_or_make_physics_claims():
    x = np.array([[0.0], [10.0], [20.0]])
    ids = ("outside", "inside")
    old = FeatureOODDetector().fit(x).assess([[-1.0], [11.0]], record_ids=ids)
    new = (
        FeatureOODDetector(flag_extrapolation=True).fit(x).assess([[-1.0], [11.0]], record_ids=ids)
    )
    assert not old.is_ood[0] and new.is_ood[0]
    assert not new.is_ood[1]
    np.testing.assert_array_equal(old.scores, new.scores)
    assert old.caution == new.caution


def predictions():
    return tuple(
        Prediction(str(i), v, u, n, False, 0.0, "train", ())
        for i, (v, u, n) in enumerate(
            ((1.0, 0.01, 0.0), (0.99, 0.2, 0.1), (0.7, 0.3, 1.0), (0.6, 0.4, 2.0), (0.4, 0.5, 3.0))
        )
    )


def test_scaled_acquisition_invariant_to_target_units_and_offset():
    rows = predictions()
    config = AcquisitionConfig(scaling="target_scale")
    expected = acquire(rows, config)
    transformed = tuple(
        replace(p, value=1000 * p.value + 17, uncertainty=1000 * p.uncertainty) for p in rows
    )
    assert acquire(transformed, config) == expected


def test_diverse_batch_avoids_neighbors_and_records_exhaustion():
    rows = (replace(predictions()[0], value=2.0), *predictions()[1:])
    config = AcquisitionConfig(batch_size=2, scaling="target_scale", diversity_radius_fraction=0.5)
    coords = {str(i): np.array([v]) for i, v in enumerate((0.0, 0.01, 1.0, 2.0, 3.0))}
    selected = acquire(rows, config, feature_coordinates=coords, reference_spacing=2.0)
    assert selected[0].candidate_id == "0"
    assert selected[1].candidate_id != "1"
    collapsed = {p.candidate_id: np.array([0.0]) for p in rows}
    relaxed = acquire(rows, config, feature_coordinates=collapsed, reference_spacing=2.0)
    assert len(relaxed) == 2
    assert relaxed[-1].strategy.endswith("diversity_relaxed")


def test_recovery_statistics_exact_null_and_robust_effect():
    from toposc_lab.active_learning.recovery import paired_statistics

    tied = paired_statistics([0.0] * 20)
    assert tied["sign_flip_p_one_sided"] == 1
    assert not tied["passes"]
    improved = paired_statistics([1.0] * 20)
    assert improved["sign_flip_p_one_sided"] == 1 / 2**20
    assert improved["ci95"] == [1.0, 1.0]
    assert improved["passes"]
    fragile = paired_statistics([1.0] * 4 + [0.0] * 16)
    assert not fragile["passes"]
    with pytest.raises(ValueError):
        paired_statistics([float("nan")])


def test_recovery_cycle_checkpoint_and_holdout_reproducibility(tmp_path):
    from test_active_learning import campaign_config, cycle_fixture

    from toposc_lab.active_learning import run_campaign
    from toposc_lab.data import dataset_to_bytes

    space, dataset, evaluator = cycle_fixture()
    base = campaign_config()
    config = replace(
        base,
        cycle=replace(
            base.cycle,
            training=replace(base.cycle.training, flag_extrapolation=True),
            acquisition=replace(
                base.cycle.acquisition, scaling="target_scale", diversity_radius_fraction=0.5
            ),
        ),
    )
    run_campaign(
        space,
        dataset,
        config=config,
        evaluator=evaluator,
        checkpoint_path=tmp_path / "resume.json",
        stop_after=1,
    )
    resumed = run_campaign(
        space, dataset, config=config, evaluator=evaluator, checkpoint_path=tmp_path / "resume.json"
    )
    full = run_campaign(
        space, dataset, config=config, evaluator=evaluator, checkpoint_path=tmp_path / "full.json"
    )
    assert resumed.exact_attempts == 4
    assert resumed.history == full.history
    assert dataset_to_bytes(resumed.dataset) == dataset_to_bytes(full.dataset)
    for history in full.history:
        predictions = {p["candidate_id"]: p for p in history["predictions"]}
        for selected in history["selected"]:
            if selected["strategy"].startswith("exploitation"):
                assert not predictions[selected["candidate_id"]]["is_ood"]
