"""Versioned regression targets extracted only from exact dataset labels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Real

import numpy as np
from numpy.typing import NDArray

from toposc_lab.data import DATASET_SCHEMA_VERSION, DatasetRecord, ExactPhysicsDataset
from toposc_lab.data.dataset_schema import TopologyValidity
from toposc_lab.ml.evaluation import dataset_fingerprint

TARGET_SCHEMA_VERSION = 1


class RegressionTargetSource(str, Enum):
    OBSERVABLE = "observable"
    TOPOLOGY = "topology"
    ROBUSTNESS = "robustness"


@dataclass(frozen=True, slots=True)
class RegressionTargetDefinition:
    """Exact result selector with explicit version and numeric value key."""

    name: str
    source: RegressionTargetSource
    result_kind: str
    result_version: str
    value_key: str
    schema_version: int = TARGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("name", "result_kind", "result_version", "value_key"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.source, RegressionTargetSource):
            raise TypeError("source must be RegressionTargetSource")
        if self.schema_version != TARGET_SCHEMA_VERSION:
            raise ValueError("unsupported target schema version")


@dataclass(frozen=True, slots=True)
class RegressionTargets:
    """Aligned values/masks whose provenance points back to exact labels."""

    definition: RegressionTargetDefinition
    record_ids: tuple[str, ...]
    values: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]
    invalid_reasons: tuple[str | None, ...]
    dataset_schema_version: int
    dataset_fingerprint: str

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=float, copy=True)
        mask = np.array(self.valid_mask, dtype=bool, copy=True)
        expected = (len(self.record_ids),)
        if values.shape != expected or mask.shape != expected:
            raise ValueError("target arrays must align with record_ids")
        if len(self.invalid_reasons) != len(self.record_ids):
            raise ValueError("invalid reasons must align with record_ids")
        if np.any(mask & ~np.isfinite(values)):
            raise ValueError("valid target values must be finite")
        if any(reason is not None for reason, valid in zip(self.invalid_reasons, mask, strict=True) if valid):
            raise ValueError("valid targets cannot carry invalid reasons")
        if any(reason is None for reason, valid in zip(self.invalid_reasons, mask, strict=True) if not valid):
            raise ValueError("invalid targets require reasons")
        values.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "record_ids", tuple(self.record_ids))
        object.__setattr__(self, "invalid_reasons", tuple(self.invalid_reasons))

    def valid_values_by_id(self) -> dict[str, float]:
        return {
            record_id: float(value)
            for record_id, value, valid in zip(
                self.record_ids, self.values, self.valid_mask, strict=True
            )
            if valid
        }


def extract_regression_targets(
    dataset: ExactPhysicsDataset,
    definition: RegressionTargetDefinition,
    *,
    records: Sequence[DatasetRecord] | None = None,
) -> RegressionTargets:
    """Extract an exact numeric target and never coerce invalid physics claims."""
    if not isinstance(dataset, ExactPhysicsDataset):
        raise TypeError("dataset must be ExactPhysicsDataset")
    if not isinstance(definition, RegressionTargetDefinition):
        raise TypeError("definition must be RegressionTargetDefinition")
    if dataset.schema_version != DATASET_SCHEMA_VERSION:
        raise ValueError("dataset schema is incompatible with target extraction")
    selected = dataset.records if records is None else tuple(records)
    dataset_ids = dataset.by_id()
    if any(record.record_id not in dataset_ids for record in selected):
        raise ValueError("target records must belong to the exact dataset")

    values: list[float] = []
    masks: list[bool] = []
    reasons: list[str | None] = []
    for record in selected:
        value, reason = _extract_one(record, definition)
        values.append(np.nan if value is None else value)
        masks.append(value is not None)
        reasons.append(reason)
    return RegressionTargets(
        definition=definition,
        record_ids=tuple(record.record_id for record in selected),
        values=np.asarray(values, dtype=float),
        valid_mask=np.asarray(masks, dtype=bool),
        invalid_reasons=tuple(reasons),
        dataset_schema_version=dataset.schema_version,
        dataset_fingerprint=dataset_fingerprint(dataset),
    )


def _extract_one(
    record: DatasetRecord, definition: RegressionTargetDefinition
) -> tuple[float | None, str | None]:
    if definition.source is RegressionTargetSource.OBSERVABLE:
        matches = tuple(
            item
            for item in record.observables
            if item.kind == definition.result_kind and item.version == definition.result_version
        )
        if not matches:
            return None, "matching observable result is absent"
        raw = matches[0].values.get(definition.value_key)
    elif definition.source is RegressionTargetSource.TOPOLOGY:
        matches = tuple(
            item
            for item in record.topology
            if item.method == definition.result_kind and item.version == definition.result_version
        )
        if not matches:
            return None, "matching topology result is absent"
        topology = matches[0]
        if topology.validity is not TopologyValidity.VALID:
            return None, f"topology result is {topology.validity.value}: {topology.reason}"
        if definition.value_key != "invariant_value":
            return None, "unsupported topology numeric value key"
        raw = topology.invariant_value
    else:
        matches = tuple(
            item
            for item in record.robustness
            if item.protocol == definition.result_kind and item.version == definition.result_version
        )
        if not matches:
            return None, "matching robustness result is absent"
        raw = matches[0].statistics.get(definition.value_key)
    if isinstance(raw, bool) or not isinstance(raw, Real):
        return None, "selected exact result value is absent or non-numeric"
    value = float(raw)
    if not np.isfinite(value):
        return None, "selected exact result value is non-finite"
    return value, None
