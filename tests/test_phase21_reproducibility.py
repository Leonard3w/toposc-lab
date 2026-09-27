"""Release checks exercise existing public CLI/UI paths with real tiny workers."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def verification():
    spec = importlib.util.spec_from_file_location("phase21_verify_tests", ROOT / "scripts/phase21_verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    output = tmp_path_factory.mktemp("phase21") / "verification"
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/phase21_verify.py"),
                             "--output", str(output)], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=150, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return output, json.loads((output / "reproducibility_report.json").read_text())


def test_environment_and_golden_reference(run):
    output, report = run
    environment = json.loads((output / "environment.json").read_text())
    assert environment["schema_versions"] == {
        "legacy_experiment": 1, "studio_experiment": 2, "dataset": 1, "geometry_archive": 1}
    assert environment["physics_adapters"]["studio"] == "phase20.configurable-chiral-p-wave.v1"
    assert all(environment["packages"][name] for name in ("numpy", "scipy", "pyside6", "streamlit"))
    assert "blas" in environment["blas_lapack_build_configuration"]["numpy"].lower()
    assert report["status"] == report["checks"]["golden_reference"]["status"] == "PASS"


def test_real_pause_resume_gui_and_exports(run):
    _, report = run
    for key in ("reference_integrity", "pause_resume", "gui_start_worker", "follow_up", "cli_gui_config_identity"):
        assert report["checks"][key]["status"] == "PASS"
    assert report["checks"]["pause_resume"]["exact_stages"] == 3
    assert report["checks"]["pause_resume"]["attempts"] == 3
    assert 1 <= report["checks"]["pause_resume"]["stages_before_pause"] < 3
    assert report["checks"]["follow_up"]["overlap_rejected"]
    assert report["checks"]["historical_data"]["status"] == "NOT_RUN"


def test_corrupt_export_fails_loudly(run, verification):
    output, _ = run
    path = output / "reference/reports/realizations.jsonl"
    before = path.read_bytes()
    path.write_text("{}\n", encoding="utf-8")
    try:
        with pytest.raises(AssertionError, match="raw export count"):
            verification.audit_run(output / "reference")
    finally:
        path.write_bytes(before)


def test_golden_config_and_legacy_hashes_stay_frozen():
    from toposc_lab.research.config import ExperimentConfig
    config = ExperimentConfig.from_file(ROOT / "tests/data/phase21_reproducibility_config.json")
    golden = json.loads((ROOT / "tests/data/phase21_golden.json").read_text())
    assert config.fingerprint == golden["canonical_config_sha256"]
    assert ExperimentConfig().fingerprint == "513fe0b990081133e181d4fb558a58d9154de407b6b5193a37c6f123b25204c4"
    altered = copy.deepcopy(config.to_dict())
    altered["output_directory"] += "-elsewhere"
    assert ExperimentConfig(**altered).fingerprint != config.fingerprint


@pytest.mark.parametrize("actual", [{"seed": 2, "Q": .2}, {"seed": 1, "Q": .20001},
                                    {"seed": 1, "Q": float("nan")}, {"seed": True, "Q": .2}])
def test_comparison_rejects_structure_seed_numeric_drift(verification, actual):
    with pytest.raises(AssertionError):
        verification.compare({"seed": 1, "Q": .2}, actual, 1e-10)
    verification.compare({"seed": 1, "Q": .2}, {"seed": 1, "Q": .2 + 1e-12}, 1e-10)


def test_real_native_startup_is_bounded(tmp_path):
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/phase21_startup_smoke.py"),
                             "--output", str(tmp_path / "startup")], cwd=ROOT,
                            text=True, capture_output=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "startup/startup.json").read_text())
    assert report["status"] == "PASS" and not report["experiment_started"]


def test_failed_verification_writes_failure_report(tmp_path):
    # An occupied reference directory is a supported invalid-state failure.
    output = tmp_path / "failed"
    (output / "reference").mkdir(parents=True)
    (output / "reference/occupied.txt").write_text("preserve", encoding="utf-8")
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/phase21_verify.py"),
                             "--output", str(output)], cwd=ROOT, capture_output=True,
                            text=True, timeout=30, check=False)
    assert result.returncode != 0
    report = json.loads((output / "reproducibility_report.json").read_text())
    assert report["status"] == "FAIL" and report["error"]
    assert (output / "reference/occupied.txt").read_text() == "preserve"


def test_fixed_cohort_reader_is_immutable(run, tmp_path, monkeypatch):
    """A local cohort fixture; actual Phase-18/19 archives are audited separately."""
    import sqlite3
    from hashlib import sha256

    from toposc_lab.research.service import ResearchService
    from toposc_lab.research.storage import ResearchStore
    from toposc_lab.research.studio_physics import StudioEvaluator

    source, _ = run
    target = tmp_path / "cohort"
    target.mkdir()
    with sqlite3.connect(source / "reference/research.sqlite3") as old, sqlite3.connect(target / "research.sqlite3") as new:
        old.backup(new)
    store = ResearchStore(target)
    config = store.get("config")
    store.save("validation_settings", {"widths": config["physics"]["disorder_widths"],
               "seeds": config["physics"]["disorder_seeds"], "clean_seed": config["physics"]["clean_seed"]})
    store.save("validation_cohort", {"candidates": store.all("candidate"), "domain": config["physics"]["domain"]})
    before = sha256(store.path.read_bytes()).hexdigest()
    monkeypatch.setattr(StudioEvaluator, "evaluate", lambda *args: pytest.fail("Read recomputed physics"))
    snapshot = ResearchService.snapshot(target)
    assert snapshot["state"]["historical_read_only"]
    assert len(snapshot["candidates"]) == 1
    candidate = ResearchService.candidate(target, snapshot["candidates"][0]["id"])
    assert len(candidate["exact_results"]) == 3
    with pytest.raises(ValueError, match="read-only"):
        ResearchService.control(target, "resume")
    assert sha256(store.path.read_bytes()).hexdigest() == before
