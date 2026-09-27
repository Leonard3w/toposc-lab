"""Small release verification through the supported CLI, Qt page and real workers.

No alternate solver or engine. Golden data is read-only: this command has no
bless/update mode. Use a new output directory on every invocation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def compare(expected, actual, tolerance: float, path: str = "root") -> None:
    """Exact structure/types/integers; absolute tolerance for finite real numbers."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and expected.keys() == actual.keys(), f"{path}: keys")
        for key in expected:
            compare(expected[key], actual[key], tolerance, f"{path}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(expected) == len(actual), f"{path}: length")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            compare(left, right, tolerance, f"{path}[{index}]")
    elif type(expected) is float:
        require(type(actual) in (float, int) and math.isfinite(actual)
                and abs(actual - expected) <= tolerance, f"{path}: numerical difference")
    else:
        require(type(expected) is type(actual) and expected == actual, f"{path}: exact difference")


def scientific_data(directory: Path) -> dict:
    from toposc_lab.research.embedded_cohort import geometry_id
    from toposc_lab.research.space import geometry_from_payload
    from toposc_lab.research.storage import ResearchStore
    from toposc_lab.research.studio_results import scalar_observables

    store = ResearchStore(directory)
    candidates = store.all("candidate")
    require(len(candidates) == 1, "one fixed verification geometry required")
    candidate = candidates[0]
    stages = {}
    for result in store.all("exact_result"):
        key = result["requested_stage"]["key"]
        stages[key] = {
            "stage": result["stage"], "requested_stage": result["requested_stage"],
            "status": result["status"], "primary_valid": result["primary_valid"],
            "adapter_id": result["adapter_id"], "spectrum": result["spectrum"],
            "onsite_offsets": result["onsite_offsets"], "indices": result["indices"],
            "localizer_gaps": result["localizer_gaps"], "spatial": result["spatial"],
            "chern_marker": result["chern_marker"],
            "observables": scalar_observables(result, candidate["geometry"]),
        }
    return {"candidate_id": candidate["id"],
            "geometry_sha256": geometry_id(geometry_from_payload(candidate["geometry"])),
            "generated": store.get("state")["generated"], "stages": stages}


def audit_run(directory: Path) -> dict:
    from toposc_lab.discovery.storage import read_json
    from toposc_lab.research.config import ExperimentConfig
    from toposc_lab.research.embedded_cohort import geometry_id
    from toposc_lab.research.provenance import verify
    from toposc_lab.research.storage import ResearchStore
    from toposc_lab.research.studio_results import scalar_observables
    from toposc_lab.research.studio_space import checked_geometry

    store = ResearchStore(directory)
    store.integrity_check()
    config, manifest, state = store.get("config"), store.get("manifest"), store.get("state")
    require(state["status"] == "COMPLETED", "run not completed")
    require(read(directory / "config.json") == config, "config file/database mismatch")
    require(read_json(directory / "manifest.json") == manifest, "manifest file/database mismatch")
    require(read_json(directory / "state.json") == state, "state export mismatch")
    require(read(directory / "reports/resolved_config.json") == config, "resolved export mismatch")
    require(ExperimentConfig(**config).fingerprint == manifest["config_sha256"], "config hash")
    verify(directory, manifest)
    with zipfile.ZipFile(directory / "source.zip") as archive:
        require(archive.testzip() is None, "source ZIP CRC")
        require("pyproject.toml" in archive.namelist(), "source project metadata")
        for name in archive.namelist():
            require(archive.read(name) == (ROOT / name).read_bytes(), f"archived source: {name}")
    candidates = {c["id"]: c for c in store.all("candidate")}
    require(state["generated"] == len(candidates), "proposal count")
    for candidate in candidates.values():
        require(geometry_id(checked_geometry(candidate)) == candidate["id"], "candidate identity")
        require(store.get("geometry", candidate["id"]) == candidate["geometry"], "geometry archive")
        require(read_json(directory / "candidates" / (candidate["id"] + ".json")) == candidate,
                "candidate export")
    attempts = store.attempts()
    require(all(a["status"] == "complete" for a in attempts), "incomplete attempts")
    require(len({(a["candidate"], a["stage"]) for a in attempts}) == len(attempts), "duplicate stages")
    with store.connect(readonly=True) as db:
        rows = db.execute("SELECT * FROM objects WHERE kind='exact_result' ORDER BY rowid").fetchall()
        checkpoints = db.execute("SELECT * FROM objects WHERE kind='checkpoint'").fetchall()
    require(len(rows) == len(attempts) == state["exact_evaluations"], "stage accounting")
    exported = [json.loads(line) for line in (directory / "reports/realizations.jsonl").read_text().splitlines()]
    require(len(exported) == len(rows), "raw export count")
    with (directory / "reports/realization_diagnostics.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == len(rows), "CSV count")
    for row, raw, scalar in zip(rows, exported, csv_rows, strict=True):
        identity = row["id"].partition(":")[0]
        result = store.decode(row)
        require(raw == {"candidate": identity, "result": result}, "JSONL differs from SQLite")
        require(scalar["candidate"] == identity and scalar["key"] == result["stage"]["key"], "CSV identity")
        for name, value in scalar_observables(result, candidates[identity]["geometry"]).items():
            if value is None:
                require(scalar[name] == "", f"CSV null: {name}")
            elif isinstance(value, bool):
                require(scalar[name] == str(value), f"CSV boolean: {name}")
            else:
                require(float(scalar[name]) == value, f"CSV number: {name}")
    require(bool(checkpoints), "no checkpoints")
    for row in checkpoints:
        require(read_json(directory / "checkpoints" / (row["id"] + ".json")) == store.decode(row), "checkpoint export")
    require((directory / "final_report.md").read_text(encoding="utf-8") == store.get("report")["markdown"], "final report")
    plot = directory / "plots/quality.png"
    if config["studio"]["output"]["save_plots"]:
        from PIL import Image
        with Image.open(plot) as picture:
            picture.verify()
    return {"status": "PASS", "exact_stages": len(rows), "attempts": len(attempts),
            "checkpoints": len(checkpoints), "config_sha256": manifest["config_sha256"],
            "plot_verified": plot.exists(), "source_sha256": manifest["source_sha256"]}


def cli(*args: str) -> str:
    result = subprocess.run([sys.executable, "-B", "-m", "toposc_lab.research", *map(str, args)],
                            cwd=ROOT, text=True, capture_output=True, timeout=120, check=False)
    require(result.returncode == 0, f"CLI {args}: {result.stderr}\n{result.stdout}")
    return result.stdout


def wait_for(predicate, *, app=None, timeout: float = 60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app:
            app.processEvents()
        value = predicate()
        if value:
            return value
        time.sleep(.005)
    raise TimeoutError("verification wait expired")


def gui_start(page, app, config: dict) -> None:
    from PySide6.QtWidgets import QDialogButtonBox

    page.set_config(config)
    page.start_run()
    require(not page.reviewed_preview["errors"], "GUI preview failed")
    require(not Path(config["output_directory"]).exists(), "preview created a run")
    button = page.preview_dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok)
    require(button.isEnabled(), "GUI START disabled")
    button.click()
    directory = Path(config["output_directory"])
    wait_for(lambda: (directory / "research.sqlite3").exists(), app=app)
    from toposc_lab.research.storage import ResearchStore
    wait_for(lambda: (ResearchStore(directory).get("state") or {}).get("status") == "COMPLETED", app=app)
    wait_for(lambda: page.operation is None or not page.operation.isRunning(), app=app)


def verify_all(output: Path, historical_root: Path | None = None) -> dict:
    from phase21_environment import environment_report
    from PySide6.QtWidgets import QApplication

    from toposc_lab.research.config import ExperimentConfig
    from toposc_lab.research.service import ResearchService
    from toposc_lab.research.storage import ResearchStore
    from toposc_live.research_page import ResearchPage

    report = {"report_schema": 1, "status": "FAIL", "checks": {key: {"status": "NOT_RUN"}
              for key in ("environment", "cli_gui_config_identity", "golden_reference",
                          "reference_integrity", "pause_resume", "gui_start_worker", "follow_up")},
              "scope": "Technical same-machine verification; no new physics or independent scientific validation."}
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checks = report["checks"]
    app = QApplication.instance() or QApplication([])
    page = None
    worker = None
    try:
        write(output / "environment.json", environment_report())
        checks["environment"] = {"status": "PASS"}
        raw = read(DATA / "phase21_reproducibility_config.json")
        golden = read(DATA / "phase21_golden.json")
        canonical = ExperimentConfig(**raw)
        require(canonical.fingerprint == golden["canonical_config_sha256"], "canonical config changed")
        tolerance = canonical.physics["tolerance"]
        # Relocation changes the full schema-2 hash: never pretend paths are excluded.
        def relocated(name):
            value = copy.deepcopy(raw)
            value["output_directory"] = str(output / name)
            return value

        reference = relocated("reference")
        reference_path = Path(reference["output_directory"])
        config_path = output / "reference-config.json"
        write(config_path, reference)
        preview = json.loads(cli("preview", "--config", config_path))
        require(not preview["errors"], "CLI preview")
        page = ResearchPage(output, ResearchService())
        page.timer.stop()
        page.set_config(reference)
        page.start_run()
        require(page.reviewed_preview == preview, "GUI/CLI preview identity")
        require(not reference_path.exists(), "preview wrote experiment")
        page.preview_dialog.reject()
        checks["cli_gui_config_identity"] = {"status": "PASS", "canonical_sha256": canonical.fingerprint,
                                             "relocated_sha256": preview["config_sha256"]}
        cli("create", reference_path, "--config", config_path)
        cli("run", reference_path)
        require(json.loads(cli("inspect", reference_path))["status"] == "COMPLETED", "CLI inspect")
        reference_data = scientific_data(reference_path)
        compare(golden["science"], reference_data, tolerance)
        checks["golden_reference"] = {"status": "PASS", "absolute_tolerance": tolerance,
                                      "canonical_sha256": canonical.fingerprint}
        checks["reference_integrity"] = audit_run(reference_path)

        resumed = relocated("resumed")
        resumed_path = Path(resumed["output_directory"])
        write(output / "resume-config.json", resumed)
        cli("create", resumed_path, "--config", output / "resume-config.json")
        store = ResearchStore(resumed_path)
        with (output / "pause-worker.log").open("w", encoding="utf-8") as log:
            worker = subprocess.Popen([sys.executable, "-B", "-m", "toposc_lab.research", "run", str(resumed_path)],
                                      cwd=ROOT, stdout=log, stderr=log)
            wait_for(lambda: store.count("exact_result") >= 1)
            ResearchService.control(resumed_path, "pause")
            require(worker.wait(timeout=60) == 0, "pause worker exit")
        require(store.get("state")["status"] == "PAUSED", "worker completed before pause; no pause evidence")
        saved = store.all("exact_result")
        require(0 < len(saved) < len(reference_data["stages"]), "partial pause required")
        page.directory = resumed_path
        page.snapshot = ResearchService.snapshot(resumed_path)
        page.request_control("resume")
        wait_for(lambda: store.get("state")["status"] == "COMPLETED", app=app)
        wait_for(lambda: not page.operation.isRunning(), app=app)
        compare(reference_data, scientific_data(resumed_path), tolerance)
        require(store.all("exact_result")[:len(saved)] == saved, "completed stage overwritten on resume")
        checks["pause_resume"] = {**audit_run(resumed_path), "stages_before_pause": len(saved),
                                  "new_worker_process": True, "completed_stages_unchanged": True}

        gui_config = relocated("gui")
        gui_start(page, app, gui_config)
        compare(reference_data, scientific_data(Path(gui_config["output_directory"])), tolerance)
        checks["gui_start_worker"] = audit_run(Path(gui_config["output_directory"]))
        identity = reference_data["candidate_id"]
        follow = ResearchService.follow_up(reference_path, [identity], widths=[3], seeds=[210101, 210102],
                                           output_directory=str(output / "follow-up"))
        try:
            ResearchService.follow_up(reference_path, [identity], widths=[3],
                                      seeds=raw["physics"]["disorder_seeds"], output_directory=str(output / "rejected"))
        except ValueError:
            pass
        else:
            raise AssertionError("overlapping seeds accepted")
        page.directory = reference_path
        page.prepare_follow_up([identity], "[3]", "[210101, 210102]", str(output / "follow-up"))
        require(page.studio_editor.raw() == follow, "GUI follow-up differs from service")
        gui_start(page, app, follow)
        following = ResearchService.candidate(output / "follow-up", identity)
        require(following["source_candidate_id"] == identity, "source candidate")
        require(Path(following["source_run"]) == reference_path, "source run")
        original = ResearchService.candidate(reference_path, identity)
        require(following["geometry"] == original["geometry"], "lossless follow-up archive")
        require(following["geometry_sha256"] == reference_data["geometry_sha256"], "physical hash")
        disorder = [r for r in following["exact_results"].values() if r["stage"]["kind"] == "disorder"]
        require([r["stage"]["seed"] for r in disorder] == [210101, 210102], "realized seed")
        require([r["requested_stage"]["seed"] for r in disorder] == [210101, 210102], "requested seed")
        checks["follow_up"] = {**audit_run(output / "follow-up"), "source_candidate_id": identity,
                                "geometry_sha256": following["geometry_sha256"], "overlap_rejected": True}
        if historical_root:
            from phase20_audit import audit

            from toposc_lab.research.studio_physics import StudioEvaluator
            for name in ("phase18-confirmation", "phase19-exploration-v2"):
                source = historical_root / name
                before = sha256((source / "research.sqlite3").read_bytes()).hexdigest()
                check = audit(source)
                page.directory = source
                page.update_snapshot(ResearchService.snapshot(source))
                require(all(not b.isEnabled() for b in page.controls.values()), "historical controls")
                candidate = ResearchService.candidate(source, page.snapshot["candidates"][0]["id"])
                # Guard against accidental evaluation while opening stored plots.
                evaluate = StudioEvaluator.evaluate
                def forbidden(*args, **kwargs):
                    raise AssertionError("historical inspection evaluated physics")
                StudioEvaluator.evaluate = forbidden
                try:
                    page._display_candidate_evidence(candidate)
                    app.processEvents()
                    require("plot unavailable" not in page.evidence_dialog.evidence_status.text(), "historical plot")
                    page.evidence_dialog.close()
                finally:
                    StudioEvaluator.evaluate = evaluate
                require(before == sha256((source / "research.sqlite3").read_bytes()).hexdigest(), "historical mutation")
                checks[name] = {"status": "PASS", **check, "explorer_stored_plot": True}
        else:
            checks["historical_data"] = {"status": "NOT_RUN", "reason": "External historical results not provided; use --historical-root."}
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=15)
        if page is not None:
            page.close()
            app.processEvents()
        write(output / "reproducibility_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/phase21_validation/run"))
    parser.add_argument("--historical-root", type=Path)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    report = verify_all(args.output, args.historical_root)
    print(json.dumps({"status": report["status"], "report": str(args.output / "reproducibility_report.json")}))


if __name__ == "__main__":
    main()
