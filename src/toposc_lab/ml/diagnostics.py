"""Compact held-out error diagnostics for geometry and parameter families."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from toposc_lab.data import DatasetRecord

ERROR_ANALYSIS_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ErrorGroupSummary:
    group: str
    sample_count: int
    mae: float
    rmse: float
    mean_signed_error: float
    maximum_absolute_error: float


@dataclass(frozen=True, slots=True)
class ErrorAnalysisReport:
    dataset_fingerprint: str
    split_name: str
    overall: ErrorGroupSummary
    geometry_families: tuple[ErrorGroupSummary, ...]
    site_count_bins: tuple[ErrorGroupSummary, ...]
    worst_record_ids: tuple[str, ...]
    schema_version: int = ERROR_ANALYSIS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_name": self.split_name,
            "overall": asdict(self.overall),
            "geometry_families": [asdict(item) for item in self.geometry_families],
            "site_count_bins": [asdict(item) for item in self.site_count_bins],
            "worst_record_ids": list(self.worst_record_ids),
        }


def analyze_regression_errors(
    records: Sequence[DatasetRecord],
    targets: Sequence[float],
    predictions: Sequence[float],
    *,
    dataset_fingerprint: str,
    split_name: str,
    worst_count: int = 10,
) -> ErrorAnalysisReport:
    """Aggregate errors without assigning scientific meaning to correlations."""
    selected = tuple(records)
    truth = np.asarray(targets, dtype=float)
    estimate = np.asarray(predictions, dtype=float)
    if not selected or truth.shape != (len(selected),) or estimate.shape != truth.shape:
        raise ValueError("records, targets, and predictions must be non-empty and aligned")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(estimate)):
        raise ValueError("targets and predictions must be finite")
    if worst_count < 1:
        raise ValueError("worst_count must be positive")
    errors = estimate - truth
    family_groups: dict[str, list[int]] = {}
    size_groups: dict[str, list[int]] = {}
    for index, record in enumerate(selected):
        family = record.geometry.family_label or (
            f"fingerprint:{record.geometry.family_fingerprint}"
        )
        family_groups.setdefault(family, []).append(index)
        size = record.geometry.to_geometry().n_sites
        size_groups.setdefault(_size_bin(size), []).append(index)
    worst = sorted(
        range(len(selected)),
        key=lambda index: (-abs(errors[index]), selected[index].record_id),
    )[:worst_count]
    return ErrorAnalysisReport(
        dataset_fingerprint=_text(dataset_fingerprint, "dataset_fingerprint"),
        split_name=_text(split_name, "split_name"),
        overall=_summary("all", errors, np.arange(len(errors))),
        geometry_families=_summaries(family_groups, errors),
        site_count_bins=_summaries(size_groups, errors),
        worst_record_ids=tuple(selected[index].record_id for index in worst),
    )


def save_error_analysis(path: str | Path, report: ErrorAnalysisReport) -> Path:
    """Atomically persist a compact diagnostic JSON artifact."""
    if not isinstance(report, ErrorAnalysisReport):
        raise TypeError("report must be ErrorAnalysisReport")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _summaries(
    groups: Mapping[str, list[int]], errors: np.ndarray
) -> tuple[ErrorGroupSummary, ...]:
    summaries = tuple(
        _summary(name, errors, np.asarray(indices, dtype=int))
        for name, indices in sorted(groups.items())
    )
    return tuple(sorted(summaries, key=lambda item: (-item.rmse, item.group)))


def _summary(group: str, errors: np.ndarray, indices: np.ndarray) -> ErrorGroupSummary:
    selected = errors[indices]
    return ErrorGroupSummary(
        group=group,
        sample_count=len(selected),
        mae=float(np.mean(np.abs(selected))),
        rmse=float(np.sqrt(np.mean(np.square(selected)))),
        mean_signed_error=float(np.mean(selected)),
        maximum_absolute_error=float(np.max(np.abs(selected))),
    )


def _size_bin(size: int) -> str:
    if size <= 8:
        return "sites:1-8"
    if size <= 32:
        return "sites:9-32"
    if size <= 128:
        return "sites:33-128"
    return "sites:129+"


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value
