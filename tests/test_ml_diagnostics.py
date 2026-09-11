from __future__ import annotations

import json

from dataset_fixtures import representative_dataset_record

from toposc_lab.geometry import chain
from toposc_lab.ml import analyze_regression_errors, save_error_analysis


def test_error_analysis_identifies_large_error_families_and_persists(tmp_path) -> None:
    records = tuple(
        representative_dataset_record(
            geometry=chain(size), seed=size, family_label="small" if size < 7 else "large"
        )
        for size in range(3, 11)
    )
    truth = [float(record.geometry.to_geometry().n_sites) for record in records]
    predictions = [value + (2.0 if value >= 7 else 0.1) for value in truth]

    report = analyze_regression_errors(
        records,
        truth,
        predictions,
        dataset_fingerprint="exact-dataset-v1-sha256:test",
        split_name="test",
        worst_count=2,
    )

    assert report.geometry_families[0].group == "large"
    assert report.geometry_families[0].rmse > report.geometry_families[1].rmse
    assert len(report.worst_record_ids) == 2
    stored = json.loads(save_error_analysis(tmp_path / "errors.json", report).read_text())
    assert stored["schema_version"] == 1
    assert stored["split_name"] == "test"
