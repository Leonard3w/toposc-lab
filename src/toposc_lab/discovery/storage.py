"""Atomic, checksummed state and an OS-released single-writer lease."""

import json
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any

from toposc_lab.search._checkpoint_codec import _Decoder, _Encoder
from toposc_lab.search.lexicographic_fitness import DerivedEvaluationRun


def atomic_json(path: Path, payload: Any) -> None:
    data = json.dumps(payload, sort_keys=True, allow_nan=False, indent=2).encode()
    envelope = json.dumps(
        {"sha256": sha256(data).hexdigest(), "payload": payload},
        sort_keys=True,
        allow_nan=False,
        indent=2,
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(envelope)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    data = json.dumps(envelope["payload"], sort_keys=True, allow_nan=False, indent=2).encode()
    if sha256(data).hexdigest() != envelope["sha256"]:
        raise ValueError(f"corrupt discovery artifact: {path}")
    return envelope["payload"]


@contextmanager
def writer_lease(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "writer.lock").open("a+b") as stream:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def encode_run(run: DerivedEvaluationRun) -> dict[str, Any]:
    """Reuse the existing whitelisted data-only search codec, never pickle."""
    encoder = _Encoder()
    root = encoder.value(run)
    return {"nodes": encoder.nodes, "root": root}


def decode_run(payload: dict[str, Any]) -> DerivedEvaluationRun:
    decoder = _Decoder(32 * 1024 * 1024)
    for node in payload["nodes"]:
        decoder.nodes.append(decoder.node(node))
    result = decoder.value(payload["root"])
    if not isinstance(result, DerivedEvaluationRun) or not result.is_valid:
        raise ValueError("invalid exact run checkpoint")
    return result
