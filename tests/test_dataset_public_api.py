from __future__ import annotations

from toposc_lab import data


def test_dataset_foundation_is_available_from_existing_data_namespace() -> None:
    expected = {
        "DatasetRecord",
        "ExactPhysicsDataset",
        "GeometryRecord",
        "append_dataset_record",
        "assess_geometry_duplicate",
        "create_dataset_record",
        "load_dataset",
        "migrate_dataset_bytes",
        "save_dataset",
        "split_dataset",
        "validate_dataset",
    }

    assert expected.issubset(set(data.__all__))
    assert all(hasattr(data, name) for name in expected)
