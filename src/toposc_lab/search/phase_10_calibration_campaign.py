"""User-started campaign for the frozen square-neighborhood calibration."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout
from importlib.metadata import version
from io import StringIO
from pathlib import Path
from typing import Any, Literal

import numpy as np

from toposc_lab.search._research_report import wilson_interval
from toposc_lab.search._research_runtime import ResearchMonitor
from toposc_lab.search._research_storage import (
    AttemptLedger,
    ResearchAbort,
    json_bytes,
    load_record,
    load_sealed,
    publish_derived,
    save_record,
)
from toposc_lab.search.phase_9_8_evaluation import build_phase_9_8_primary_topology_inputs
from toposc_lab.search.phase_10_calibration import (
    CALIBRATION_PROTOCOL_COMMIT,
    CALIBRATION_PROTOCOL_ID,
    CALIBRATION_PROTOCOL_PATH,
    FULL_ROOTS,
    PREFLIGHT_ROOTS,
    build_calibration_trajectory,
    derive_calibration_schedule,
    ordered_calibration_cells,
)
from toposc_lab.search.phase_10_campaign import (
    _artifact_inventory,
    _campaign_lock,
    _verify_completion,
)
from toposc_lab.search.phase_10_research import (
    CONFIRMATION_SEEDS,
    PREFLIGHT_SEEDS,
    REFERENCE_SEEDS,
    THREAD_VARIABLES,
    TRIAL_SEEDS,
    VALIDATION_SEEDS,
    evaluate_research_geometry,
)


def calibration_environment() -> dict[str, Any]:
    """Require the committed frozen protocol and numerical runtime."""
    root = Path(__file__).resolve().parents[3]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True,
            text=True, encoding="utf-8"
        ).stdout.strip()

    if sys.version_info[:2] != (3, 14) or not sys.dont_write_bytecode:
        raise ValueError("Python 3.14 und PYTHONDONTWRITEBYTECODE=1 sind erforderlich")
    if any(os.environ.get(name) != "1" for name in THREAD_VARIABLES):
        raise ValueError(
            "Vor Pythonstart OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS "
            "und BLIS_NUM_THREADS auf 1 setzen"
        )
    code = git("rev-parse", "HEAD")
    if code == CALIBRATION_PROTOCOL_COMMIT:
        raise ValueError("Die Implementierung muss nach dem Protokoll separat committed sein")
    git("merge-base", "--is-ancestor", CALIBRATION_PROTOCOL_COMMIT, code)
    allowed = {
        "src/toposc_lab/observables/__pycache__/__init__.cpython-314.pyc",
        "src/toposc_lab/observables/__pycache__/spectrum.cpython-314.pyc",
        "geometry_demo.npz",
    }
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout
    unexpected = [
        line for line in status.splitlines() if line[3:].replace("\\", "/") not in allowed
    ]
    if unexpected:
        raise ValueError("Worktree vor Kalibrationslauf committen: " + "; ".join(unexpected))
    frozen = subprocess.run(
        ["git", "-C", str(root), "show", f"{CALIBRATION_PROTOCOL_COMMIT}:{CALIBRATION_PROTOCOL_PATH}"],
        check=True, capture_output=True
    ).stdout.replace(b"\r\n", b"\n")
    current = (root / CALIBRATION_PROTOCOL_PATH).read_bytes().replace(b"\r\n", b"\n")
    if current != frozen:
        raise ValueError("Kalibrationsprotokoll weicht vom eingefrorenen Commit ab")
    config = StringIO()
    with redirect_stdout(config):
        np.show_config()
    return {
        "code_commit": code,
        "protocol_commit": CALIBRATION_PROTOCOL_COMMIT,
        "protocol_sha256": hashlib.sha256(frozen).hexdigest(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "matplotlib", "pydantic")},
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "numpy_backend": config.getvalue(),
        "threads": {name: os.environ[name] for name in THREAD_VARIABLES},
    }


def run_calibration_campaign(
    output: Path, *, mode: Literal["preflight", "full", "resume"]
) -> Path:
    """Run only the explicitly requested stage of TOPOSC-P10-CAL-001."""
    environment = calibration_environment()
    root = _output_path(output)
    if mode == "preflight":
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        raise ValueError("Zuerst den Kalibrationsvorlauf in einem neuen Verzeichnis starten")
    try:
        with _campaign_lock(root):
            if mode == "resume":
                mode = "full" if (root / "full").is_dir() else "preflight"
                resume = True
            else:
                resume = False
            if mode == "full":
                preflight = root / "preflight"
                complete = load_record(preflight / "complete.json")
                _verify_completion(preflight)
                if complete["preflight_passed"] is not True:
                    raise ResearchAbort("Passender bestandener Kalibrationsvorlauf erforderlich")
                manifest = load_record(preflight / "manifest.json")
                if manifest["environment"] != environment:
                    raise ResearchAbort("Vorlauf und Hauptlauf benötigen dieselbe Umgebung")
            stage = root / mode
            if not resume:
                stage.mkdir(exist_ok=False)
            with ResearchMonitor(root / "events.jsonl") as monitor:
                try:
                    return _execute_stage(stage, mode, environment, monitor, resume=resume)
                except BaseException as error:
                    monitor.emit("interrupted", error_type=type(error).__name__, error=str(error))
                    raise
    except ResearchAbort as error:
        raise RuntimeError(str(error)) from error


def _execute_stage(
    stage: Path,
    mode: str,
    environment: dict[str, Any],
    monitor: ResearchMonitor,
    *,
    resume: bool,
) -> Path:
    roots = PREFLIGHT_ROOTS if mode == "preflight" else FULL_ROOTS
    total = len(roots) * 24 + 2
    protocol_file = Path(__file__).resolve().parents[3] / CALIBRATION_PROTOCOL_PATH
    metadata = {
        "protocol_id": CALIBRATION_PROTOCOL_ID,
        "protocol_commit": CALIBRATION_PROTOCOL_COMMIT,
        "environment": environment,
        "mode": mode,
        "total_evaluation_attempts": total,
        "protocol_text": protocol_file.read_text(encoding="utf-8"),
        "roots": roots,
    }
    manifest_path = stage / "manifest.json"
    if manifest_path.exists():
        if load_record(manifest_path) != metadata:
            raise ResearchAbort("Protokoll, Code oder Umgebung stimmen beim Resume nicht überein")
    else:
        save_record(manifest_path, metadata)
    if (stage / "complete.json").exists():
        _verify_completion(stage)
        monitor.emit("complete_already_sealed", stage=mode, completed=total, total=total)
        return stage / "report.md"

    monitor.emit("constructing", stage=mode, block="construction", completed=0, total=total)
    plan_path = stage / "construction_plan.json"
    if plan_path.exists():
        plans = tuple(load_record(plan_path))
    else:
        plans = tuple(build_calibration_trajectory(root) for root in roots)
        save_record(plan_path, plans)
    _audit_seed_roles(plans)
    cells = ordered_calibration_cells(plans)
    if len(cells) != total:
        raise ResearchAbort("Kalibrationsplan hat nicht das eingefrorene Slotbudget")

    outcomes = []
    started = time.perf_counter()
    for index, cell in enumerate(cells):
        directory = stage / "evaluations" / f"cell_{index:04d}"
        if (directory / "sealed.json").exists():
            record = load_sealed(directory)
        else:
            record = _evaluate_cell(
                directory, cell, environment["code_commit"], monitor, mode, index, total
            )
        outcomes.append(record)
        if index == 0:
            scientific = record.get("scientific")
            if scientific is None or scientific["clean_eligible"] is not True:
                raise ResearchAbort("Die Anfangs-Nullkontrolle besteht die sauberen Gates nicht")

    summary = calibration_summary(tuple(outcomes), preflight=mode == "preflight")
    summary["environment"] = environment
    summary["protocol_commit"] = CALIBRATION_PROTOCOL_COMMIT
    summary["total_evaluation_attempts"] = total
    analysis_path = stage / "analysis.json"
    if analysis_path.exists():
        if load_record(analysis_path) != summary:
            raise ResearchAbort("Wiederholte Kalibrationsanalyse stimmt nicht überein")
    else:
        save_record(analysis_path, summary)
    publish_derived(stage / "summary.json", json_bytes(summary))
    publish_derived(stage / "report.md", _render_report(summary).encode("utf-8"))
    preflight_passed = bool(
        mode == "preflight" and summary["control_valid"] and summary["all_evaluations_available"]
    )
    save_record(
        stage / "complete.json",
        {
            "preflight_passed": preflight_passed,
            "artifacts": _artifact_inventory(stage),
            "seconds_per_evaluation": (time.perf_counter() - started) / total,
        },
    )
    monitor.emit("complete", stage=mode, completed=total, total=total,
                 last_sealed=f"{mode}/complete")
    return stage / "report.md"


def _evaluate_cell(
    directory: Path,
    cell: dict[str, Any],
    code: str,
    monitor: ResearchMonitor,
    mode: str,
    index: int,
    total: int,
) -> dict[str, Any]:
    directory.parent.mkdir(exist_ok=True)
    ledger = AttemptLedger(
        directory,
        on_storage=lambda seconds: monitor.emit(
            "artifact_saved", category="storage", operation_seconds=seconds
        ),
    )
    identity = {key: cell[key] for key in ("block", "root", "amplitude", "edge_step", "edge_distance")}
    ledger.record("input.json", {**identity, "genome": cell["genome"]})
    monitor.emit(
        "evaluating", stage=mode, block=cell["block"], trial=cell["root"],
        amplitude=cell["amplitude"], edge_step=cell["edge_step"],
        slot=index + 1, completed=index, total=total
    )
    started = time.perf_counter()
    try:
        geometry = cell["genome"].to_geometry()
        topology_inputs = build_phase_9_8_primary_topology_inputs(geometry)
        coordinates = geometry.coordinates
        assert coordinates is not None
        unique_order = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
        input_evidence = {
            "unique_coordinate_order": unique_order,
            "position_areas_in_coordinate_order": topology_inputs.position_areas,
            "bulk_masks_in_coordinate_order": topology_inputs.bulk_masks,
        }
        evaluated = evaluate_research_geometry(geometry, code)
        record = {**identity, "run": evaluated.run, "scientific": evaluated.scientific,
                  "topology_input_evidence": input_evidence, "error": None}
    except Exception as error:  # noqa: BLE001 - every operational failure remains in-budget
        record = {**identity, "run": None, "scientific": None,
                  "topology_input_evidence": None,
                  "error": {"type": type(error).__name__, "message": str(error)}}
    elapsed = time.perf_counter() - started
    ledger.record("outcome.json", record)
    ledger.seal(record)
    scientific = record["scientific"]
    monitor.emit(
        "evaluated" if scientific is not None else "evaluation_failed",
        stage=mode, block=cell["block"], trial=cell["root"],
        amplitude=cell["amplitude"], edge_step=cell["edge_step"],
        clean_eligible=None if scientific is None else scientific["clean_eligible"],
        slot=index + 1, completed=index + 1, total=total,
        last_sealed=f"{mode}/evaluations/cell_{index:04d}",
        category="physics", operation_seconds=elapsed
    )
    return record


def calibration_summary(outcomes: tuple[dict[str, Any], ...], *, preflight: bool) -> dict[str, Any]:
    """Summarize fixed cells while retaining trajectory-level denominators."""
    if len(outcomes) < 2:
        raise ValueError("calibration summary requires both controls")
    start, end = outcomes[0], outcomes[-1]
    control_valid = (
        start["scientific"] is not None
        and end["scientific"] is not None
        and start["scientific"]["clean_eligible"] is True
        and end["scientific"]["clean_eligible"] is True
        and _controls_agree(start["scientific"], end["scientific"])
    )
    grouped: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for record in outcomes[1:-1]:
        grouped[(record["amplitude"], record["edge_step"])].append(record)
    cells = []
    for (amplitude, edge_step), records in sorted(grouped.items()):
        available = [record for record in records if record["scientific"] is not None]
        hits = sum(record["scientific"]["clean_eligible"] for record in available)
        count = len(records)
        trajectory_results = [_trajectory_result(record) for record in records]
        protection_values = [
            record["scientific"]["localizer_protection_proxy"]
            for record in available
            if record["scientific"]["localizer_protection_proxy"] is not None
        ]
        boundary_values = [
            record["scientific"]["minimum_boundary_weight_first_four"]
            for record in available
            if record["scientific"]["minimum_boundary_weight_first_four"] is not None
        ]
        cells.append(
            {
                "amplitude": amplitude,
                "edge_step": edge_step,
                "trajectory_count": count,
                "available_count": len(available),
                "operational_failure_count": count - len(available),
                "clean_eligible_count": hits,
                "clean_eligible_fraction_all": hits / count,
                "clean_eligible_fraction_available": None if not available else hits / len(available),
                "wilson_95_all": list(wilson_interval(hits, count)),
                "gate_reason_counts": _reason_counts(available),
                "protection_proxy_summary": _numeric_summary(protection_values),
                "minimum_boundary_weight_first_four_summary": _numeric_summary(boundary_values),
                "trajectory_results": trajectory_results,
            }
        )
    return {
        "protocol_id": CALIBRATION_PROTOCOL_ID,
        "preflight_only": preflight,
        "control_valid": control_valid,
        "physical_interpretation_allowed": control_valid,
        "all_evaluations_available": all(record["scientific"] is not None for record in outcomes),
        "cells": cells,
        "claim_boundary": "finite 8x8 square-neighborhood calibration; descriptive only",
    }


def _controls_agree(first: dict[str, Any], second: dict[str, Any]) -> bool:
    a, b = _control_signature(first), _control_signature(second)
    return a[0] == b[0] and np.allclose(a[1], b[1], rtol=0.0, atol=1e-10)


def _control_signature(scientific: dict[str, Any]) -> tuple[Any, list[float]]:
    grid = scientific["topology_grid"]
    boundary = scientific["boundary_signature"]
    discrete = (
        scientific["clean_eligible"], tuple(scientific["gate_reasons"]),
        tuple(item["index"] for item in grid["bott"]),
        tuple(item["chern_number"] for item in grid["local_chern"]),
        tuple(item["local_chern_number"] for item in grid["localizer"]),
    )
    floats = (
        [item["estimate"] for item in grid["bott"]]
        + [item["bulk_chern_estimate"] for item in grid["local_chern"]]
        + [item["localizer_gap"] for item in grid["localizer"]]
        + [item["energy"] for item in boundary["states"]]
    )
    return discrete, floats


def _reason_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        for reason in record["scientific"]["gate_reasons"]:
            result[reason] = result.get(reason, 0) + 1
    return result


def _trajectory_result(record: dict[str, Any]) -> dict[str, Any]:
    scientific = record["scientific"]
    if scientific is None:
        return {
            "root": record["root"],
            "edge_distance": record["edge_distance"],
            "available": False,
            "error": record["error"],
        }
    grid = scientific["topology_grid"]
    boundary = scientific["boundary_signature"]
    return {
        "root": record["root"],
        "edge_distance": record["edge_distance"],
        "available": True,
        "clean_eligible": scientific["clean_eligible"],
        "gate_reasons": scientific["gate_reasons"],
        "localizer_protection_proxy": scientific["localizer_protection_proxy"],
        "minimum_boundary_weight_first_four": scientific[
            "minimum_boundary_weight_first_four"
        ],
        "bott": [
            {key: item[key] for key in ("estimate", "index", "is_quantized")}
            for item in grid["bott"]
        ],
        "local_chern": [
            {
                key: item[key]
                for key in (
                    "bulk_site_count", "bulk_chern_estimate", "chern_number", "is_quantized"
                )
            }
            for item in grid["local_chern"]
        ],
        "localizer": [
            {key: item[key] for key in ("kappa", "local_chern_number", "localizer_gap")}
            for item in grid["localizer"]
        ],
        "boundary_localized_count": None
        if boundary is None
        else boundary["boundary_localized_count"],
        "maximum_pair_residual": None if boundary is None else boundary["maximum_pair_residual"],
    }


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "minimum": None if not values else min(values),
        "median": None if not values else float(np.median(values)),
        "maximum": None if not values else max(values),
        "values": values,
    }


def _audit_seed_roles(plans: tuple[dict[str, Any], ...]) -> None:
    old = set(PREFLIGHT_SEEDS + TRIAL_SEEDS + REFERENCE_SEEDS + VALIDATION_SEEDS + CONFIRMATION_SEEDS)
    seen = set()
    plan_roots = tuple(plan["schedule"]["root"] for plan in plans)
    if plan_roots == PREFLIGHT_ROOTS:
        expected_schedules = tuple(plan["schedule"] for plan in plans)
    elif plan_roots == FULL_ROOTS:
        expected_schedules = tuple(
            derive_calibration_schedule(root) for root in PREFLIGHT_ROOTS
        ) + tuple(plan["schedule"] for plan in plans)
    else:
        raise ResearchAbort("Kalibrationsplan enthält nicht die eingefrorenen Wurzelseeds")
    schedules = {
        schedule["root"]: schedule
        for schedule in expected_schedules
    }
    for schedule in schedules.values():
        values = (schedule["root"], schedule["position_seed"], *schedule["step_seeds"])
        for value in values:
            if value in old or value in seen:
                raise ResearchAbort(f"Kalibrationsseed-Kollision: {value}")
            seen.add(value)
    for plan in plans:
        schedule = plan["schedule"]
        if derive_calibration_schedule(schedule["root"]) != schedule:
            raise ResearchAbort("Versiegelter Kalibrationsseedplan weicht vom Protokoll ab")


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase-10-Quadratkalibration", "", f"Protokoll: `{CALIBRATION_PROTOCOL_ID}`", "",
        f"Nullkontrollen gültig: **{summary['control_valid']}**", "",
        "| Positionsamplitude | Kantenschritte | Sauber geeignet | Verfügbar | L Median [Min, Max] |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for cell in summary["cells"]:
        proxy = cell["protection_proxy_summary"]
        proxy_text = (
            "nicht verfügbar"
            if proxy["count"] == 0
            else f"{proxy['median']:.6g} [{proxy['minimum']:.6g}, {proxy['maximum']:.6g}]"
        )
        lines.append(
            f"| {cell['amplitude']:.3g} | {cell['edge_step']} | "
            f"{cell['clean_eligible_count']}/{cell['trajectory_count']} | "
            f"{cell['available_count']}/{cell['trajectory_count']} | {proxy_text} |"
        )
    lines += ["", "Alle Rohwerte, Gategründe und Fehler stehen in `summary.json`.", "",
              "Diese finite, deskriptive Kalibration beweist keine Robustheit und keinen neuen Mechanismus.", ""]
    return "\n".join(lines)


def _output_path(output: Path) -> Path:
    root = output.resolve()
    results = Path(__file__).resolve().parents[3] / "results"
    if not root.is_relative_to(results.resolve()) or not root.name.startswith("phase_10_calibration"):
        raise ValueError("Ein neues results/phase_10_calibration...-Verzeichnis verwenden")
    return root
