from __future__ import annotations

import json

from dataset_fixtures import representative_dataset_record

from toposc_lab.data import DatasetSplitConfig, ExactPhysicsDataset, split_dataset
from toposc_lab.geometry import chain
from toposc_lab.ml import (
    RidgeRegressor,
    bootstrap_rmse_interval,
    evaluate_regression_surrogates,
    save_evaluation_report,
)


def _dataset() -> tuple[ExactPhysicsDataset, dict[str, float]]:
    records = tuple(
        representative_dataset_record(
            geometry=chain(size), seed=size, family_label=f"chain_{size}"
        )
        for size in range(3, 23)
    )
    dataset = ExactPhysicsDataset(records)
    targets = {record.record_id: float(record.geometry.to_geometry().n_sites) for record in records}
    return dataset, targets


def test_evaluation_selects_only_on_validation_and_beats_trivial(tmp_path) -> None:
    dataset, targets = _dataset()
    split = split_dataset(
        dataset,
        config=DatasetSplitConfig(
            train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2, seed=1204
        ),
    )
    report, _model, schema = evaluate_regression_surrogates(
        dataset,
        targets,
        split=split,
        model_factories={"ridge": lambda: RidgeRegressor(alpha=1e-9)},
        bootstrap_samples=100,
    )

    assert report.selected_model == "ridge"
    assert not report.test_used_for_selection
    assert report.selected_test_metrics.rmse < report.trivial_test_metrics.rmse
    assert report.train_ids == split.train_ids
    assert report.dataset_fingerprint.startswith("exact-dataset-v1-sha256:")
    assert "graph::site_count" in schema.feature_names

    path = save_evaluation_report(tmp_path / "evaluation.json", report)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["schema_version"] == 1
    assert stored["test_used_for_selection"] is False


def test_bootstrap_interval_is_reproducible() -> None:
    first = bootstrap_rmse_interval([1, 2, 3], [1.1, 1.8, 3.2], seed=7, samples=50)
    second = bootstrap_rmse_interval([1, 2, 3], [1.1, 1.8, 3.2], seed=7, samples=50)

    assert first == second
    assert first.lower <= first.estimate <= first.upper
