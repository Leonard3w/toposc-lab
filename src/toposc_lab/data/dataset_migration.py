"""Explicit, auditable migrations between compatible dataset containers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from toposc_lab.data.dataset_schema import DATASET_FORMAT, DATASET_SCHEMA_VERSION
from toposc_lab.data.dataset_storage import (
    DatasetStorageError,
    ExactPhysicsDataset,
    dataset_from_bytes,
)


@dataclass(frozen=True, slots=True)
class DatasetMigrationResult:
    """Migrated current dataset plus the exact declared migration steps."""

    dataset: ExactPhysicsDataset
    source_version: int
    target_version: int
    steps: tuple[str, ...]


def migrate_dataset_bytes(
    payload: bytes,
    *,
    target_version: int = DATASET_SCHEMA_VERSION,
) -> DatasetMigrationResult:
    """Explicitly migrate the compatible container-only pilot schema v0.

    Version 0 used current v1 records but marked only the outer container as a
    pilot. No scientific field is synthesized. Any other legacy layout is
    rejected rather than guessed.
    """
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if isinstance(target_version, bool) or not isinstance(target_version, int):
        raise TypeError("target_version must be an integer")
    if target_version != DATASET_SCHEMA_VERSION:
        raise DatasetStorageError(f"unsupported migration target version: {target_version}")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, DatasetStorageError) as error:
        raise DatasetStorageError(f"invalid legacy dataset payload: {error}") from error
    data = _mapping(value)
    if set(data) != {"format", "schema_version", "records"}:
        raise DatasetStorageError("legacy dataset fields are missing or unknown")
    if data["format"] != DATASET_FORMAT:
        raise DatasetStorageError("legacy dataset format is unsupported")
    if "schema_version" not in data:
        raise DatasetStorageError("legacy dataset schema_version is required")
    source_version = data["schema_version"]
    if isinstance(source_version, bool) or not isinstance(source_version, int):
        raise DatasetStorageError("legacy dataset schema_version must be an integer")
    if source_version == DATASET_SCHEMA_VERSION:
        raise DatasetStorageError("dataset is already current; no migration is required")
    if source_version != 0:
        raise DatasetStorageError(f"no compatible migration path from schema version {source_version}")
    if not isinstance(data["records"], list):
        raise DatasetStorageError("legacy dataset records must be an array")

    migrated = dict(data)
    migrated["schema_version"] = DATASET_SCHEMA_VERSION
    migrated_bytes = json.dumps(
        migrated,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    dataset = dataset_from_bytes(migrated_bytes)
    return DatasetMigrationResult(
        dataset=dataset,
        source_version=source_version,
        target_version=target_version,
        steps=("promote compatible container-only pilot v0 marker to v1",),
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetStorageError(f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetStorageError("legacy dataset must be a JSON object")
    return cast(dict[str, Any], value)
