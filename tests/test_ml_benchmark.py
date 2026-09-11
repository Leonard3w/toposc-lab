from __future__ import annotations

from dataset_fixtures import representative_dataset_record

from toposc_lab.data import DatasetSplitConfig, ExactPhysicsDataset, split_dataset
from toposc_lab.geometry import chain, ring
from toposc_lab.ml import benchmark_simple_models_vs_gnn


def test_strict_benchmark_retains_simple_model_without_verified_gnn_gain() -> None:
    records = tuple(
        representative_dataset_record(
            geometry=chain(size) if size % 2 else ring(size),
            seed=size,
            family_label=f"family_{size}",
        )
        for size in range(3, 33)
    )
    dataset = ExactPhysicsDataset(records)
    split = split_dataset(
        dataset,
        config=DatasetSplitConfig(
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
            seed=1211,
        ),
    )
    targets = {
        record.record_id: float(record.geometry.to_geometry().n_sites)
        for record in records
    }

    result = benchmark_simple_models_vs_gnn(
        dataset,
        split,
        targets,
        bootstrap_samples=100,
    )

    assert result.same_split_and_data_budget
    assert result.selected_default in {"ridge", "stump_boosting"}
    assert not result.gnn_meaningful_improvement
    assert result.test_metrics[result.selected_default].rmse < result.test_metrics[
        "trivial_mean"
    ].rmse
    assert result.train_record_count == len(split.train_ids)
    assert result.dataset_fingerprint.startswith("exact-dataset-v1-sha256:")
    assert "simple model retained" in result.selection_reason
