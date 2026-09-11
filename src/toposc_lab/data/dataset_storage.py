"""Atomic persistence for versioned exact-physics datasets."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from pathlib import Path
from typing import Any, cast

from toposc_lab.data.dataset_codec import DatasetCodecError, record_from_dict, record_to_dict
from toposc_lab.data.dataset_schema import DATASET_FORMAT, DATASET_SCHEMA_VERSION, DatasetRecord

DEFAULT_DATASET_MAX_BYTES = 512 * 1024 * 1024


class DatasetStorageError(ValueError):
    """A dataset file is unsafe, incompatible, corrupt, or cannot be published."""


class DuplicateRecordPolicy(str, Enum):
    """Explicit behavior when an identical record identifier already exists."""

    REJECT = "reject"
    SKIP = "skip"
    REPLACE = "replace"


@dataclass(frozen=True, slots=True)
class ExactPhysicsDataset:
    """Immutable collection of unique exact-physics records."""

    records: tuple[DatasetRecord, ...]
    schema_version: int = field(default=DATASET_SCHEMA_VERSION, init=False)
    format: str = field(default=DATASET_FORMAT, init=False)

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if any(not isinstance(record, DatasetRecord) for record in records):
            raise TypeError("records must contain only DatasetRecord values")
        identifiers = tuple(record.record_id for record in records)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("dataset record identifiers must be unique")
        object.__setattr__(self, "records", records)

    def by_id(self) -> dict[str, DatasetRecord]:
        """Return a new identifier lookup without exposing mutable state."""
        return {record.record_id: record for record in self.records}


def save_dataset(
    path: str | Path,
    records: tuple[DatasetRecord, ...] | ExactPhysicsDataset,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically publish a complete validated dataset JSON file."""
    destination = _dataset_path(path)
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"dataset already exists: {destination}")
    dataset = records if isinstance(records, ExactPhysicsDataset) else ExactPhysicsDataset(records)
    payload = dataset_to_bytes(dataset)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def append_dataset_record(
    path: str | Path,
    record: DatasetRecord,
    *,
    duplicate_policy: DuplicateRecordPolicy = DuplicateRecordPolicy.REJECT,
) -> bool:
    """Atomically append one record and report whether storage changed."""
    if not isinstance(record, DatasetRecord):
        raise TypeError("record must be DatasetRecord")
    if not isinstance(duplicate_policy, DuplicateRecordPolicy):
        raise TypeError("duplicate_policy must be DuplicateRecordPolicy")
    destination = _dataset_path(path)
    dataset = load_dataset(destination) if destination.exists() else ExactPhysicsDataset(())
    existing = dataset.by_id()
    if record.record_id in existing:
        if duplicate_policy is DuplicateRecordPolicy.REJECT:
            raise ValueError(f"duplicate dataset record: {record.record_id}")
        if duplicate_policy is DuplicateRecordPolicy.SKIP:
            return False
        updated = tuple(
            record if item.record_id == record.record_id else item
            for item in dataset.records
        )
    else:
        updated = (*dataset.records, record)
    save_dataset(destination, updated, overwrite=destination.exists())
    return True


def load_dataset(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_DATASET_MAX_BYTES,
) -> ExactPhysicsDataset:
    """Load only the current schema; migrations must be requested explicitly."""
    source = _dataset_path(path)
    limit = _positive_integer(max_bytes, "max_bytes")
    try:
        size = source.stat().st_size
    except OSError as error:
        raise DatasetStorageError(f"could not inspect dataset: {source}") from error
    if size > limit:
        raise DatasetStorageError(f"dataset exceeds max_bytes={limit}")
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise DatasetStorageError(f"could not read dataset: {source}") from error
    return dataset_from_bytes(payload)


def dataset_to_bytes(dataset: ExactPhysicsDataset) -> bytes:
    """Serialize an immutable dataset deterministically."""
    if not isinstance(dataset, ExactPhysicsDataset):
        raise TypeError("dataset must be ExactPhysicsDataset")
    value = {
        "format": dataset.format,
        "schema_version": dataset.schema_version,
        "records": [record_to_dict(record) for record in dataset.records],
    }
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def dataset_from_bytes(payload: bytes) -> ExactPhysicsDataset:
    """Decode a current-schema dataset from UTF-8 JSON bytes."""
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
        data = _mapping(value, "dataset")
        if set(data) != {"format", "schema_version", "records"}:
            raise DatasetStorageError("dataset fields are missing or unknown")
        if data["format"] != DATASET_FORMAT:
            raise DatasetStorageError("dataset format is unsupported")
        version = _integer(data["schema_version"], "dataset.schema_version")
        if version != DATASET_SCHEMA_VERSION:
            raise DatasetStorageError(
                f"dataset schema version {version} is incompatible with current version "
                f"{DATASET_SCHEMA_VERSION}; migrate explicitly"
            )
        records_value = data["records"]
        if not isinstance(records_value, list):
            raise DatasetStorageError("dataset.records must be an array")
        records = tuple(record_from_dict(item) for item in records_value)
        return ExactPhysicsDataset(records)
    except DatasetStorageError:
        raise
    except (DatasetCodecError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise DatasetStorageError(f"invalid dataset payload: {error}") from error


def _dataset_path(path: str | Path) -> Path:
    result = Path(path)
    if result.suffix.lower() != ".json":
        raise ValueError("dataset path must use the .json suffix")
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetStorageError(f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetStorageError(f"{path} must be a JSON object")
    return cast(dict[str, Any], value)


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise DatasetStorageError(f"{path} must be an integer")
    return int(value)


def _positive_integer(value: object, name: str) -> int:
    result = _integer(value, name)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result
