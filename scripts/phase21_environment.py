"""Record the installed runtime without changing research manifests or schemas."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path


def environment_report() -> dict:
    import numpy as np
    import scipy
    from PySide6.QtCore import qVersion

    from toposc_lab.data import DATASET_SCHEMA_VERSION
    from toposc_lab.geometry.serialization import GEOMETRY_ARCHIVE_SCHEMA_VERSION
    from toposc_lab.research.config import ExperimentConfig
    from toposc_lab.research.embedded import EMBEDDED_ADAPTER_ID
    from toposc_lab.research.physics import ADAPTER_ID
    from toposc_lab.research.provenance import ROOT, environment, source_digest
    from toposc_lab.research.studio_config import preset
    from toposc_lab.research.studio_physics import STUDIO_ADAPTER_ID

    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    builds = {}
    for name, module in (("numpy", np), ("scipy", scipy)):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            module.show_config()
        builds[name] = stream.getvalue()
    packages = dict(sorted((d.metadata["Name"].lower(), d.version)
                           for d in metadata.distributions() if d.metadata["Name"]))
    lock = ROOT / "requirements/phase21-windows-py314.txt"
    pinned = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s]+)", lock.read_text(), re.MULTILINE))
    normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    installed = {normalize(name): value for name, value in packages.items()}
    mismatches = {name: {"expected": value, "installed": installed.get(normalize(name))}
                  for name, value in pinned.items() if installed.get(normalize(name)) != value}
    reference_platform = (sys.platform == "win32" and platform.machine().lower() in ("amd64", "x86_64")
                          and platform.python_version() == "3.14.7")
    return {
        **environment(), "report_schema": 1, "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"), "source_sha256": source_digest(),
        "python_executable": sys.executable, "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(), "architecture": platform.architecture(),
        "virtual_environment": sys.prefix != sys.base_prefix,
        "packages": packages,
        "reference_profile": {"status": "PASS" if reference_platform and not mismatches else "MISMATCH",
                              "platform_matches": reference_platform, "package_mismatches": mismatches},
        "qt_runtime": qVersion(), "blas_lapack_build_configuration": builds,
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")},
        "schema_versions": {
            "legacy_experiment": ExperimentConfig().schema_version,
            "studio_experiment": preset("quick_test")["schema_version"],
            "dataset": DATASET_SCHEMA_VERSION, "geometry_archive": GEOMETRY_ARCHIVE_SCHEMA_VERSION,
        },
        "physics_adapters": {"legacy": ADAPTER_ID, "embedded": EMBEDDED_ADAPTER_ID,
                             "studio": STUDIO_ADAPTER_ID},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/phase21_validation/environment.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(environment_report(), indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
