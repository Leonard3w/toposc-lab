from __future__ import annotations

import json
import os

import pytest
from dataset_fixtures import representative_dataset_record

from toposc_lab.data.dataset_storage import (
    DatasetStorageError,
    DuplicateRecordPolicy,
    ExactPhysicsDataset,
    append_dataset_record,
    dataset_from_bytes,
    dataset_to_bytes,
    load_dataset,
    save_dataset,
)


def test_dataset_write_load_round_trip_and_deterministic_bytes(tmp_path) -> None:
    record = representative_dataset_record()
    path = tmp_path / "nested" / "dataset.json"

    assert save_dataset(path, (record,)) == path
    restored = load_dataset(path)

    assert dataset_to_bytes(restored) == dataset_to_bytes(ExactPhysicsDataset((record,)))
    assert restored.records[0].geometry.archive == record.geometry.archive


def test_save_is_atomic_and_preserves_existing_file_on_replace_failure(tmp_path, monkeypatch) -> None:
    path = tmp_path / "dataset.json"
    first = representative_dataset_record(seed=1)
    second = representative_dataset_record(seed=2)
    save_dataset(path, (first,))
    original = path.read_bytes()

    def fail_replace(source, destination) -> None:
        raise PermissionError("simulated publication failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="publication"):
        save_dataset(path, (second,), overwrite=True)

    assert path.read_bytes() == original
    assert not tuple(tmp_path.glob(".dataset.json.*.tmp"))


def test_existing_dataset_is_not_silently_overwritten(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    save_dataset(path, (representative_dataset_record(),))

    with pytest.raises(FileExistsError):
        save_dataset(path, (), overwrite=False)


def test_append_duplicate_policy_is_explicit(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    record = representative_dataset_record()
    assert append_dataset_record(path, record)

    with pytest.raises(ValueError, match="duplicate"):
        append_dataset_record(path, record)
    assert not append_dataset_record(path, record, duplicate_policy=DuplicateRecordPolicy.SKIP)
    assert append_dataset_record(path, record, duplicate_policy=DuplicateRecordPolicy.REPLACE)
    assert len(load_dataset(path).records) == 1


def test_loader_rejects_incompatible_schema_without_implicit_migration() -> None:
    payload = dataset_to_bytes(ExactPhysicsDataset((representative_dataset_record(),)))
    decoded = json.loads(payload)
    decoded["schema_version"] = 0
    old_payload = json.dumps(decoded).encode()

    with pytest.raises(DatasetStorageError, match="migrate explicitly"):
        dataset_from_bytes(old_payload)


def test_loader_rejects_duplicate_json_fields_and_size_limit(tmp_path) -> None:
    with pytest.raises(DatasetStorageError, match="duplicate JSON field"):
        dataset_from_bytes(b'{"format":"x","format":"y","schema_version":1,"records":[]}')

    path = tmp_path / "large.json"
    path.write_bytes(b"{}")
    with pytest.raises(DatasetStorageError, match="exceeds"):
        load_dataset(path, max_bytes=1)
