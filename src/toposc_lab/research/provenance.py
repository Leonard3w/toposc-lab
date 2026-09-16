"""Source snapshots and reproducibility guards shared by launch and resume."""

from __future__ import annotations

import os
import platform
import subprocess
import zipfile
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from toposc_lab.research.storage import utc_now

ROOT = Path(__file__).resolve().parents[3]


def source_files() -> list[Path]:
    return sorted((ROOT / "src").rglob("*.py"))


def source_digest() -> str:
    digest = sha256()
    for path in source_files():
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "pydantic", "toposc-lab"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "source-checkout"
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "packages": packages}


def capture(directory: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = "unavailable", True
    archive = directory / "source.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as stream:
        for path in source_files() + [ROOT / "pyproject.toml"]:
            stream.write(path, path.relative_to(ROOT).as_posix())
    return {"git_commit": commit, "git_dirty": dirty, "source_sha256": source_digest(),
            "source_zip_sha256": sha256(archive.read_bytes()).hexdigest(),
            "environment": environment(), "timestamp": utc_now(), "cpu_count": os.cpu_count(),
            "scientific_scope": "Finite fixed-site localizer evidence; no phase/Majorana certificate"}


def verify(directory: Path, manifest: dict[str, Any]) -> None:
    if manifest["source_sha256"] != source_digest():
        raise ValueError("Resume source mismatch; use archived source or clone a new experiment")
    if manifest["environment"] != environment():
        raise ValueError("Resume runtime mismatch")
    if sha256((directory / "source.zip").read_bytes()).hexdigest() != manifest["source_zip_sha256"]:
        raise ValueError("Corrupt source archive")
