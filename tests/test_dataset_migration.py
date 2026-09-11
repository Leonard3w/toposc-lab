from __future__ import annotations

import json

import pytest
from dataset_fixtures import representative_dataset_record

from toposc_lab.data.dataset_migration import migrate_dataset_bytes
from toposc_lab.data.dataset_storage import (
    DatasetStorageError,
    ExactPhysicsDataset,
    dataset_from_bytes,
    dataset_to_bytes,
)


def _legacy_container(version: int) -> bytes:
    current = json.loads(
        dataset_to_bytes(ExactPhysicsDataset((representative_dataset_record(),)))
    )
    current["schema_version"] = version
    return json.dumps(current).encode()


def test_explicit_compatible_migration_preserves_exact_records() -> None:
    result = migrate_dataset_bytes(_legacy_container(0))

    assert result.source_version == 0
    assert result.target_version == 1
    assert result.dataset.records[0].record_id == representative_dataset_record().record_id
    assert result.steps


def test_normal_loader_never_silently_migrates_legacy_data() -> None:
    with pytest.raises(DatasetStorageError, match="migrate explicitly"):
        dataset_from_bytes(_legacy_container(0))


def test_migration_rejects_unknown_version_and_missing_schema() -> None:
    with pytest.raises(DatasetStorageError, match="no compatible migration path"):
        migrate_dataset_bytes(_legacy_container(99))

    payload = json.loads(_legacy_container(0))
    del payload["schema_version"]
    with pytest.raises(DatasetStorageError, match="fields"):
        migrate_dataset_bytes(json.dumps(payload).encode())


def test_current_dataset_is_not_needlessly_rewritten_by_migration() -> None:
    current = dataset_to_bytes(ExactPhysicsDataset((representative_dataset_record(),)))
    with pytest.raises(DatasetStorageError, match="already current"):
        migrate_dataset_bytes(current)
