"""Versioned, checked artifacts for the one frozen Phase-10 campaign.

Uses the existing closed search codec, not pickle or a general dataset schema.
Completed records are immutable; partial executions remain as separate attempts.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from toposc_lab.search._checkpoint_codec import _Decoder, _Encoder
from toposc_lab.search.checkpoint import _parse_json

FORMAT = "toposc_phase10_research_record"
VERSION = 1
MAX_RECORD_BYTES = 512 * 1024 * 1024


class ResearchAbort(BaseException):
    """Campaign control-flow abort; must not become a failed physics evaluation.

    Like cancellation, this crosses the fitness callback's ordinary Exception
    boundary. The campaign/CLI converts it to a user-facing execution error.
    """


def encode_record(value: Any) -> bytes:
    """Encode only the existing whitelisted records and data types."""
    encoder = _Encoder()
    root = encoder.value(value)
    return json_bytes({"nodes": encoder.nodes, "root": root})


def decode_record(data: bytes) -> Any:
    """Validate and reconstruct a closed record graph with bounded expansion."""
    document = _parse_json(data)
    if not isinstance(document, dict) or set(document) != {"nodes", "root"}:
        raise ValueError("invalid research record graph")
    nodes = document["nodes"]
    if not isinstance(nodes, list) or len(nodes) > 1_000_000:
        raise ValueError("invalid research record count")
    decoder = _Decoder(MAX_RECORD_BYTES)
    for node in nodes:
        decoder.nodes.append(decoder.node(node))
    result = decoder.value(document["root"])
    if len(decoder.used) != len(nodes):
        raise ValueError("research record contains unused object nodes")
    return result


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


def write_exclusive(path: Path, data: bytes) -> None:
    """Publish fsynced bytes atomically without replacing any existing artifact."""
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except Exception as error:
        raise ResearchAbort(f"cannot seal {path}: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_record(path: Path, value: Any) -> str:
    """Seal a record and return the hash of its exact serialized contents."""
    try:
        payload = encode_record(value)
        digest = hashlib.sha256(payload).hexdigest()
        envelope = json_bytes(
            {
                "format": FORMAT,
                "version": VERSION,
                "sha256": digest,
                "payload": json.loads(payload),
            }
        )
        if len(envelope) > MAX_RECORD_BYTES:
            raise ValueError("research record exceeds size limit")
        write_exclusive(path, envelope)
        return digest
    except Exception as error:
        raise ResearchAbort(f"cannot store research record {path}: {error}") from error


def load_record(path: Path) -> Any:
    """Reject truncation, altered bytes, unknown schemas and invalid constructors."""
    try:
        if path.is_symlink() or path.stat().st_size > MAX_RECORD_BYTES:
            raise ValueError("unsupported research record path or size")
        envelope = _parse_json(path.read_bytes())
        if not isinstance(envelope, dict) or set(envelope) != {
            "format",
            "version",
            "sha256",
            "payload",
        }:
            raise ValueError("invalid research record envelope")
        if (
            envelope["format"] != FORMAT
            or type(envelope["version"]) is not int
            or envelope["version"] != VERSION
        ):
            raise ValueError("unsupported research record schema")
        payload = json_bytes(envelope["payload"])
        if hashlib.sha256(payload).hexdigest() != envelope["sha256"]:
            raise ValueError("research record checksum mismatch")
        return decode_record(payload)
    except Exception as error:
        raise ResearchAbort(f"cannot load research record {path}: {error}") from error


def publish_derived(path: Path, data: bytes) -> None:
    """Finish interrupted reporting only when already published bytes agree exactly."""
    if path.exists():
        if path.read_bytes() != data:
            raise ResearchAbort(f"derived report differs from previous bytes: {path}")
    else:
        write_exclusive(path, data)


class AttemptLedger:
    """Immediate input/outcome storage and deterministic replay checks for one unit."""

    def __init__(
        self, directory: Path, *, on_storage: Callable[[float], None] | None = None
    ) -> None:
        self.directory = directory
        self.on_storage = on_storage
        self.storage_seconds = 0.0
        directory.mkdir(exist_ok=True)
        self.previous: dict[str, bytes] = {}
        executions = sorted(directory.glob("execution_*"))
        for execution in executions:
            if not execution.is_dir() or execution.is_symlink():
                raise ResearchAbort("invalid execution directory")
            checkpoint = execution / "checkpoint.zip"
            if checkpoint.exists():
                from toposc_lab.search.checkpoint import load_search_checkpoint

                load_search_checkpoint(checkpoint)
            for path in sorted(execution.glob("*.json")):
                value = load_record(path)
                data = encode_record(value)
                if path.name in self.previous and self.previous[path.name] != data:
                    raise ResearchAbort("conflicting stored executions; no selective replay")
                self.previous[path.name] = data
        self.execution = directory / f"execution_{len(executions):04d}"
        self.execution.mkdir(exist_ok=False)
        self.recorded: set[str] = set()

    def record(self, name: str, value: Any) -> None:
        started = time.perf_counter()
        if Path(name).name != name or not name.endswith(".json"):
            raise ResearchAbort("invalid ledger record name")
        try:
            data = encode_record(value)
            if name in self.previous and self.previous[name] != data:
                raise ResearchAbort(
                    f"deterministic replay differs at {name}; previous outcome retained"
                )
            save_record(self.execution / name, value)
            self.recorded.add(name)
            self._measured_storage(started)
        except Exception as error:
            raise ResearchAbort(f"ledger failure: {error}") from error

    def seal(self, value: Any) -> None:
        started = time.perf_counter()
        if not set(self.previous).issubset(self.recorded):
            raise ResearchAbort("replay omitted previously recorded work")
        files = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(self.execution.iterdir())
            if path.suffix in (".json", ".zip")
        }
        save_record(
            self.directory / "sealed.json",
            {
                "execution": self.execution.name,
                "files": files,
                "result": value,
            },
        )
        self._measured_storage(started)

    def _measured_storage(self, started: float) -> None:
        elapsed = time.perf_counter() - started
        self.storage_seconds += elapsed
        if self.on_storage is not None:
            self.on_storage(elapsed)


def load_sealed(directory: Path) -> Any:
    """Verify every linked input/outcome, including superseded replay prefixes."""
    sealed = load_record(directory / "sealed.json")
    if not isinstance(sealed, dict) or set(sealed) != {"execution", "files", "result"}:
        raise ResearchAbort("invalid sealed unit")
    name = sealed["execution"]
    if not isinstance(name, str) or Path(name).name != name or not name.startswith("execution_"):
        raise ResearchAbort("invalid sealed execution path")
    execution = directory / name
    files = sealed["files"]
    if not isinstance(files, dict) or set(files) != {
        p.name for p in execution.iterdir() if p.suffix in (".json", ".zip")
    }:
        raise ResearchAbort("sealed execution inventory differs")
    current: dict[str, bytes] = {}
    for filename, digest in files.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ResearchAbort("unsafe sealed artifact path")
        path = execution / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ResearchAbort("sealed artifact hash differs")
        if path.suffix == ".json":
            current[filename] = encode_record(load_record(path))
        else:
            from toposc_lab.search.checkpoint import load_search_checkpoint

            load_search_checkpoint(path)
    for previous in directory.glob("execution_*"):
        if previous.is_symlink():
            raise ResearchAbort("symlinked execution is unsupported")
        if previous != execution:
            checkpoint = previous / "checkpoint.zip"
            if checkpoint.exists():
                from toposc_lab.search.checkpoint import load_search_checkpoint

                load_search_checkpoint(checkpoint)
            for path in previous.glob("*.json"):
                if current.get(path.name) != encode_record(load_record(path)):
                    raise ResearchAbort("superseded execution contradicts sealed outcomes")
    return sealed["result"]
