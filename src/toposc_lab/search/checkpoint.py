"""Versioned, atomic snapshots at completed generation boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from numbers import Integral
from pathlib import Path
from types import MappingProxyType

from toposc_lab.search.generation_loop import GenerationLoopResult
from toposc_lab.search.novelty_score import PopulationNoveltyReport

SEARCH_CHECKPOINT_FORMAT = "toposc_lab_search_checkpoint"
SEARCH_CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_CHECKPOINT_MAX_BYTES = 512 * 1024 * 1024
DEFAULT_CHECKPOINT_MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024


class SearchCheckpointError(ValueError):
    """A checkpoint is malformed, unsupported, or inconsistent."""


@dataclass(frozen=True, slots=True)
class SearchCheckpoint:
    """Complete search prefix plus explicit external-policy provenance.

    ``result.config.generation_count`` counts completed transitions, whereas
    ``requested_generation_count`` retains the originally planned target.
    Callback implementations and their private state are not serializable.
    Their identifiers must name frozen, deterministic experiment policies.
    """

    result: GenerationLoopResult
    requested_generation_count: int
    evaluator_identifier: str
    code_version: str
    runtime_versions: Mapping[str, str]
    novelty_reports: tuple[PopulationNoveltyReport, ...] = ()
    schema_version: int = field(default=SEARCH_CHECKPOINT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.result, GenerationLoopResult):
            raise TypeError("result must be GenerationLoopResult")
        count = self.requested_generation_count
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError("requested_generation_count must be an integer")
        if count < self.result.config.generation_count:
            raise ValueError("requested_generation_count must cover every completed transition")
        object.__setattr__(self, "requested_generation_count", int(count))
        for name in ("evaluator_identifier", "code_version"):
            object.__setattr__(self, name, _label(getattr(self, name), name))
        if not isinstance(self.runtime_versions, Mapping) or not self.runtime_versions:
            raise ValueError("runtime_versions must be a nonempty mapping")
        runtime = {
            _label(key, "runtime key"): _label(value, "runtime version")
            for key, value in self.runtime_versions.items()
        }
        object.__setattr__(self, "runtime_versions", MappingProxyType(runtime))
        reports = tuple(self.novelty_reports)
        previous_generation = -1
        for report in reports:
            if not isinstance(report, PopulationNoveltyReport):
                raise TypeError("novelty_reports must contain PopulationNoveltyReport values")
            index = report.population.generation_index
            if index <= previous_generation or index >= len(self.result.populations):
                raise ValueError("novelty reports must follow unique completed generation order")
            if report.population is not self.result.populations[index]:
                raise ValueError("novelty report must refer to the exact stored population")
            previous_generation = index
        object.__setattr__(self, "novelty_reports", reports)

    @property
    def completed_generation_index(self) -> int:
        """Last fully evaluated generation in the snapshot."""
        return self.result.config.generation_count


def create_search_checkpoint(
    result: GenerationLoopResult,
    *,
    evaluator_identifier: str,
    code_version: str,
    requested_generation_count: int | None = None,
    novelty_reports: tuple[PopulationNoveltyReport, ...] = (),
) -> SearchCheckpoint:
    """Bind a completed prefix to explicit code and evaluator provenance."""
    if not isinstance(result, GenerationLoopResult):
        raise TypeError("result must be GenerationLoopResult")
    return SearchCheckpoint(
        result=result,
        requested_generation_count=(
            result.config.generation_count
            if requested_generation_count is None
            else requested_generation_count
        ),
        evaluator_identifier=evaluator_identifier,
        code_version=code_version,
        runtime_versions=_current_runtime_versions(),
        novelty_reports=novelty_reports,
    )


def _current_runtime_versions() -> dict[str, str]:
    """Use the same runtime inventory when recording and resuming a checkpoint."""
    runtime = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "toposc-lab"):
        try:
            runtime[package] = version(package)
        except PackageNotFoundError:
            runtime[package] = "unavailable"
    return runtime


def save_search_checkpoint(
    path: str | Path, checkpoint: SearchCheckpoint, *, overwrite: bool = True
) -> Path:
    """Validate, fsync, and atomically publish one explicit checkpoint path.

    An existing file is replaced only after serialization and validation have
    succeeded. Filesystem errors propagate; the previous file is retained if
    writing or replacement fails. The containing directory must already exist.
    With ``overwrite=False``, publish exclusively through a same-directory hard
    link instead of replacing a destination. An existing path raises
    FileExistsError, including if another writer publishes it concurrently.
    No delete-then-write fallback is used. The default retains the old contract.
    """
    from toposc_lab.search._checkpoint_codec import decode_checkpoint, encode_checkpoint

    if not isinstance(checkpoint, SearchCheckpoint):
        raise TypeError("checkpoint must be SearchCheckpoint")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be bool")
    destination = Path(path)
    payload = encode_checkpoint(checkpoint)
    # Reconstruct before any destination write, including all existing invariants.
    decode_checkpoint(payload)
    manifest = _json_bytes(
        {
            "format": SEARCH_CHECKPOINT_FORMAT,
            "schema_version": SEARCH_CHECKPOINT_SCHEMA_VERSION,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for member, data in (("manifest.json", manifest), ("payload.json", payload)):
                    info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, data)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_search_checkpoint(
    path: str | Path,
    *,
    max_archive_bytes: int = DEFAULT_CHECKPOINT_MAX_BYTES,
    max_uncompressed_bytes: int = DEFAULT_CHECKPOINT_MAX_UNCOMPRESSED_BYTES,
) -> SearchCheckpoint:
    """Load and validate data only; never execute evaluation or resume a search."""
    from toposc_lab.search._checkpoint_codec import decode_checkpoint

    for limit in (max_archive_bytes, max_uncompressed_bytes):
        if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1:
            raise ValueError("checkpoint size limits must be positive integers")
    with Path(path).open("rb") as handle:
        raw = handle.read(max_archive_bytes + 1)
    if len(raw) > max_archive_bytes:
        raise SearchCheckpointError("checkpoint exceeds max_archive_bytes")
    try:
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            members = archive.infolist()
            if len(members) != 2 or {item.filename for item in members} != {
                "manifest.json",
                "payload.json",
            }:
                raise SearchCheckpointError("checkpoint must contain exactly manifest and payload")
            if any(item.flag_bits & 1 for item in members):
                raise SearchCheckpointError("encrypted checkpoints are unsupported")
            if any(
                item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                for item in members
            ):
                raise SearchCheckpointError("unsupported checkpoint compression")
            if sum(item.file_size for item in members) > max_uncompressed_bytes:
                raise SearchCheckpointError("checkpoint exceeds max_uncompressed_bytes")
            manifest = _parse_json(archive.read("manifest.json"))
            if not isinstance(manifest, dict) or set(manifest) != {
                "format",
                "schema_version",
                "payload_sha256",
            }:
                raise SearchCheckpointError("invalid checkpoint manifest")
            if (
                manifest["format"] != SEARCH_CHECKPOINT_FORMAT
                or type(manifest["schema_version"]) is not int
                or manifest["schema_version"] != SEARCH_CHECKPOINT_SCHEMA_VERSION
            ):
                raise SearchCheckpointError("unsupported checkpoint format or schema version")
            payload = archive.read("payload.json")
        if hashlib.sha256(payload).hexdigest() != manifest["payload_sha256"]:
            raise SearchCheckpointError("checkpoint payload checksum mismatch")
        return decode_checkpoint(payload, max_geometry_bytes=max_uncompressed_bytes)
    except SearchCheckpointError:
        raise
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
        raise SearchCheckpointError(f"invalid checkpoint: {error}") from error
    except zipfile.BadZipFile as error:
        raise SearchCheckpointError("invalid checkpoint ZIP container") from error


def _label(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()


def _parse_json(data: bytes) -> object:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise SearchCheckpointError("duplicate JSON key")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise SearchCheckpointError(f"nonstandard JSON constant: {value}")

    return json.loads(data, object_pairs_hook=unique, parse_constant=reject)
