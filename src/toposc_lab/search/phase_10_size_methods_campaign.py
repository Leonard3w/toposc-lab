"""User-started campaign for the frozen Phase-10 size/methods protocol."""

from __future__ import annotations

import hashlib
import json
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

from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.search._research_report import wilson_interval
from toposc_lab.search._research_runtime import ResearchMonitor
from toposc_lab.search._research_storage import (
    AttemptLedger,
    ResearchAbort,
    encode_record,
    json_bytes,
    load_record,
    load_sealed,
    publish_derived,
    save_record,
)
from toposc_lab.search.phase_9_8_evaluation import (
    build_phase_9_8_boundary_signature,
    evaluate_phase_9_8_descriptive_geometry,
)
from toposc_lab.search.phase_10_calibration import (
    FULL_ROOTS as OLD_CALIBRATION_FULL_ROOTS,
)
from toposc_lab.search.phase_10_calibration import (
    PREFLIGHT_ROOTS as OLD_CALIBRATION_PREFLIGHT_ROOTS,
)
from toposc_lab.search.phase_10_calibration import (
    derive_calibration_schedule,
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
)
from toposc_lab.search.phase_10_size_methods import (
    FULL_ROOTS,
    MASK_NAMES,
    PREFLIGHT_ROOTS,
    SIZE_METHODS_PROTOCOL_COMMIT,
    SIZE_METHODS_PROTOCOL_ID,
    SIZE_METHODS_PROTOCOL_PATH,
    SIZES,
    build_size_topology_inputs,
    build_size_trajectory,
    derive_size_schedule,
    ordered_size_cells,
    validate_size_geometry,
)
from toposc_lab.search.random_search_experiment import (
    _encode_pipeline_run,
    _encode_topology_grid,
)


def size_methods_environment() -> dict[str, Any]:
    """Require the frozen protocol commit and controlled numerical runtime."""
    root = Path(__file__).resolve().parents[3]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()

    if sys.version_info[:2] != (3, 14) or not sys.dont_write_bytecode:
        raise ValueError("Python 3.14 und PYTHONDONTWRITEBYTECODE=1 sind erforderlich")
    if any(os.environ.get(name) != "1" for name in THREAD_VARIABLES):
        raise ValueError(
            "Vor Pythonstart OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS "
            "und BLIS_NUM_THREADS auf 1 setzen"
        )
    code = git("rev-parse", "HEAD")
    if code == SIZE_METHODS_PROTOCOL_COMMIT:
        raise ValueError("Die Implementierung muss nach dem Protokoll separat committed sein")
    git("merge-base", "--is-ancestor", SIZE_METHODS_PROTOCOL_COMMIT, code)
    allowed = {
        "src/toposc_lab/observables/__pycache__/__init__.cpython-314.pyc",
        "src/toposc_lab/observables/__pycache__/spectrum.cpython-314.pyc",
        "geometry_demo.npz",
    }
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    unexpected = [
        line for line in status.splitlines() if line[3:].replace("\\", "/") not in allowed
    ]
    if unexpected:
        raise ValueError("Worktree vor Größen-/Methodenlauf committen: " + "; ".join(unexpected))
    frozen = subprocess.run(
        ["git", "-C", str(root), "show", f"{SIZE_METHODS_PROTOCOL_COMMIT}:{SIZE_METHODS_PROTOCOL_PATH}"],
        check=True,
        capture_output=True,
    ).stdout.replace(b"\r\n", b"\n")
    current = (root / SIZE_METHODS_PROTOCOL_PATH).read_bytes().replace(b"\r\n", b"\n")
    if current != frozen:
        raise ValueError("Größen-/Methodenprotokoll weicht vom eingefrorenen Commit ab")
    config = StringIO()
    with redirect_stdout(config):
        np.show_config()
    return {
        "code_commit": code,
        "protocol_commit": SIZE_METHODS_PROTOCOL_COMMIT,
        "protocol_sha256": hashlib.sha256(frozen).hexdigest(),
        "python": platform.python_version(),
        "packages": {
            name: version(name) for name in ("numpy", "scipy", "matplotlib", "pydantic")
        },
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "numpy_backend": config.getvalue(),
        "threads": {name: os.environ[name] for name in THREAD_VARIABLES},
    }


def run_size_methods_campaign(
    output: Path, *, mode: Literal["preflight", "full", "resume"]
) -> Path:
    """Run only the explicitly requested stage of TOPOSC-P10-SIZE-001."""
    environment = size_methods_environment()
    root = _output_path(output)
    if mode == "preflight":
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        raise ValueError("Zuerst den Größen-/Methodenvorlauf in einem neuen Verzeichnis starten")
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
                    raise ResearchAbort("Passender bestandener Größen-/Methodenvorlauf erforderlich")
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
    total = len(SIZES) * (len(roots) * 3 + 2)
    protocol_file = Path(__file__).resolve().parents[3] / SIZE_METHODS_PROTOCOL_PATH
    metadata = {
        "protocol_id": SIZE_METHODS_PROTOCOL_ID,
        "protocol_commit": SIZE_METHODS_PROTOCOL_COMMIT,
        "environment": environment,
        "mode": mode,
        "total_evaluation_attempts": total,
        "protocol_text": protocol_file.read_text(encoding="utf-8"),
        "roots": roots,
        "sizes": SIZES,
        "mask_names": MASK_NAMES,
    }
    manifest_path = stage / "manifest.json"
    if manifest_path.exists():
        if encode_record(load_record(manifest_path)) != encode_record(metadata):
            raise ResearchAbort("Protokoll, Code oder Umgebung stimmen beim Resume nicht überein")
    else:
        save_record(manifest_path, metadata)
    if (stage / "complete.json").exists():
        _verify_completion(stage)
        monitor.emit("complete_already_sealed", stage=mode, completed=total, total=total)
        return stage / "report.md"

    stage_started = time.perf_counter()
    construction_started = stage_started
    monitor.emit("constructing", stage=mode, block="construction", completed=0, total=total)
    plan_path = stage / "construction_plan.json"
    if plan_path.exists():
        plans = tuple(load_record(plan_path))
    else:
        plans = tuple(build_size_trajectory(root) for root in roots)
        save_record(plan_path, plans)
    _audit_seed_roles(plans)
    cells = ordered_size_cells(plans)
    if len(cells) != total:
        raise ResearchAbort("Größen-/Methodenplan hat nicht das eingefrorene Slotbudget")
    prepared = tuple(_prepare_cell(cell) for cell in cells)
    input_plan_path = stage / "input_plan.json"
    if input_plan_path.exists():
        if encode_record(load_record(input_plan_path)) != encode_record(prepared):
            raise ResearchAbort("Gespeicherter Zellinputplan weicht von der Konstruktion ab")
    else:
        save_record(input_plan_path, prepared)
    monitor.emit(
        "constructed",
        stage=mode,
        block="construction",
        completed=0,
        total=total,
        category="construction",
        operation_seconds=time.perf_counter() - construction_started,
    )

    size_estimates: dict[int, float] = {}
    if mode == "full":
        preflight_complete = load_record(stage.parent / "preflight" / "complete.json")
        size_estimates = {
            int(n): float(seconds)
            for n, seconds in preflight_complete["seconds_per_size"].items()
        }

    outcomes = []
    started = time.perf_counter()
    for index, cell in enumerate(prepared):
        directory = stage / "evaluations" / f"cell_{index:04d}"
        if (directory / "sealed.json").exists():
            record = _load_bound_sealed(directory, cell)
        else:
            record = _evaluate_cell(
                directory,
                cell,
                environment["code_commit"],
                monitor,
                mode,
                index,
                total,
                estimated_seconds=size_estimates.get(cell["n"]),
            )
        outcomes.append(record)
        if cell["block"] == "control_start" and not _control_is_valid(record):
            raise ResearchAbort(f"Anfangskontrolle für n={cell['n']} ist ungültig")

    summary = size_methods_summary(tuple(outcomes), preflight=mode == "preflight")
    summary["environment"] = environment
    summary["protocol_commit"] = SIZE_METHODS_PROTOCOL_COMMIT
    summary["total_evaluation_attempts"] = total
    analysis_path = stage / "analysis.json"
    if analysis_path.exists():
        if encode_record(load_record(analysis_path)) != encode_record(summary):
            raise ResearchAbort("Wiederholte Größen-/Methodenanalyse stimmt nicht überein")
    else:
        save_record(analysis_path, summary)
    publish_derived(stage / "summary.json", json_bytes(summary))
    publish_derived(stage / "report.md", _render_report(summary).encode("utf-8"))
    preflight_passed = bool(
        mode == "preflight"
        and summary["control_valid"]
        and summary["all_evaluations_available"]
    )
    save_record(
        stage / "complete.json",
        {
            "preflight_passed": preflight_passed,
            "artifacts": _artifact_inventory(stage),
            "seconds_per_evaluation": (time.perf_counter() - started) / total,
            "seconds_per_size": _observed_seconds_by_size(stage.parent / "events.jsonl", mode),
            "total_stage_seconds": time.perf_counter() - stage_started,
        },
    )
    monitor.emit(
        "complete", stage=mode, completed=total, total=total, last_sealed=f"{mode}/complete"
    )
    return stage / "report.md"


def _prepare_cell(cell: dict[str, Any]) -> dict[str, Any]:
    geometry = cell["genome"].to_geometry()
    report = validate_size_geometry(geometry, cell["n"])
    if not report["is_valid"]:
        raise ResearchAbort(f"Ungültige Geometrie im Zellinputplan: {report['issues']}")
    _, evidence = build_size_topology_inputs(geometry, cell["n"])
    return {
        **cell,
        "geometry_id": exact_geometry_id(geometry),
        "validation": report,
        "topology_input_evidence": evidence,
        "evaluation_seed": None,
    }


def _evaluate_cell(
    directory: Path,
    cell: dict[str, Any],
    code: str,
    monitor: ResearchMonitor,
    mode: str,
    index: int,
    total: int,
    *,
    estimated_seconds: float | None,
) -> dict[str, Any]:
    directory.parent.mkdir(exist_ok=True)
    ledger = AttemptLedger(
        directory,
        on_storage=lambda seconds: monitor.emit(
            "artifact_saved", category="storage", operation_seconds=seconds
        ),
    )
    ledger.record("input.json", cell)
    identity = _identity(cell)
    construction_metrics = {
        key: cell[key]
        for key in (
            "measurements",
            "degree_sequence",
            "boundary_fraction",
            "edge_endpoint_class_counts",
        )
    }
    progress: dict[str, Any] = {
        "stage": mode,
        "block": cell["block"],
        "trial": cell["root"],
        "size": cell["n"],
        "slot": index + 1,
        "completed": index,
        "total": total,
    }
    if estimated_seconds is not None:
        progress["estimated_seconds_per_evaluation"] = estimated_seconds
    monitor.emit("evaluating", **progress)
    started = time.perf_counter()
    timings: dict[str, float] = {}

    def record_timing(category: str, seconds: float) -> None:
        timings[category] = timings.get(category, 0.0) + seconds

    try:
        geometry = cell["genome"].to_geometry()
        if exact_geometry_id(geometry) != cell["geometry_id"]:
            raise ValueError("geometry ID differs from sealed cell input")
        inputs, evidence = build_size_topology_inputs(geometry, cell["n"])
        if encode_record(evidence) != encode_record(cell["topology_input_evidence"]):
            raise ValueError("topology inputs differ from sealed cell input")
        run, grid = evaluate_phase_9_8_descriptive_geometry(
            geometry,
            inputs=inputs,
            code_version=code,
            timing_callback=record_timing,
        )
        boundary = None
        if run.is_valid and run.evaluation is not None:
            boundary = build_phase_9_8_boundary_signature(run.evaluation)
        scientific = _encode_size_scientific(run, grid, boundary)
        record = {
            **identity,
            "construction_metrics": construction_metrics,
            "run": run,
            "scientific": scientific,
            "error": None,
        }
    except Exception as error:  # noqa: BLE001 - operational failures remain in-budget
        record = {
            **identity,
            "construction_metrics": construction_metrics,
            "run": None,
            "scientific": None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    elapsed = time.perf_counter() - started
    timings["total"] = elapsed
    ledger.record("outcome.json", record)
    ledger.seal(record)
    monitor.emit(
        "evaluated" if _scientific_available(record.get("scientific")) else "evaluation_failed",
        stage=mode,
        block=cell["block"],
        trial=cell["root"],
        size=cell["n"],
        screening_pass=None
        if record["scientific"] is None
        else record["scientific"]["screening_pass"],
        slot=index + 1,
        completed=index + 1,
        total=total,
        last_sealed=f"{mode}/evaluations/cell_{index:04d}",
        category="physics",
        operation_seconds=elapsed,
        stage_timings=timings,
    )
    return record


def _encode_size_scientific(run: Any, grid: Any, boundary: Any) -> dict[str, Any]:
    available = bool(
        run.is_valid
        and run.simulation_result is not None
        and run.evaluation is not None
        and grid is not None
        and len(grid.bott) == 3
        and len(grid.local_chern) == 5
        and len(grid.localizer) == 3
        and boundary is not None
    )
    if not available:
        return {
            "available": False,
            "screening_pass": False,
            "control_pass": False,
            "gate_reasons": ["required_pipeline_or_method_result_unavailable"],
            "pipeline": _encode_pipeline_run(run),
            "topology_grid": None if grid is None else _encode_topology_grid(grid),
            "marker_maps": [],
            "boundary_signature": None,
            "localizer_protection_proxy": None,
        }
    encoded_grid = _encode_topology_grid(grid)
    marker_maps = [
        {
            "name": name,
            "positions": result.positions,
            "local_marker": result.local_marker,
            "position_areas": result.position_areas,
            "bulk_mask": result.bulk_mask,
        }
        for name, result in zip(MASK_NAMES, grid.local_chern, strict=True)
    ]
    reasons = _screening_reasons(grid, boundary)
    all_positive = (
        all(item.bott_index == 1 for item in grid.bott)
        and all(item.chern_number == 1 for item in grid.local_chern)
        and all(item.local_chern_number == 1 for item in grid.localizer)
        and grid.localizer_protection_proxy >= 0.20
        and boundary.passes
    )
    return {
        "available": True,
        "screening_pass": not reasons,
        "control_pass": all_positive,
        "gate_reasons": reasons,
        "pipeline": _encode_pipeline_run(run),
        "topology_grid": encoded_grid,
        "marker_maps": marker_maps,
        "boundary_signature": _encode_boundary(boundary),
        "localizer_protection_proxy": grid.localizer_protection_proxy,
    }


def _screening_reasons(grid: Any, boundary: Any) -> list[str]:
    reasons = []
    values = (
        tuple(item.bott_index for item in grid.bott)
        + tuple(item.chern_number for item in grid.local_chern[:2])
        + tuple(item.local_chern_number for item in grid.localizer)
    )
    if any(value is None for value in values):
        reasons.append("topology_unresolved")
    elif any(abs(value) != 1 for value in values):
        reasons.append("topology_magnitude_not_one")
    elif len(set(values)) != 1:
        reasons.append("topology_signed_disagreement")
    if grid.localizer_protection_proxy < 0.20:
        reasons.append("localizer_protection_proxy_below_0_20")
    reasons.extend(f"boundary:{reason}" for reason in boundary.reasons)
    return reasons


def _encode_boundary(boundary: Any) -> dict[str, Any]:
    return {
        "particle_hole_pairs_by_state_index": [list(pair) for pair in boundary.particle_hole_pairs],
        "pairing_cost": boundary.pairing_cost,
        "maximum_pair_residual": boundary.maximum_pair_residual,
        "boundary_localized_count": boundary.boundary_localized_count,
        "minimum_boundary_weight_first_four": boundary.minimum_boundary_weight_first_four,
        "reasons": list(boundary.reasons),
        "states": [
            {
                "state_index": state.state_index,
                "energy": state.energy,
                "ipr": state.ipr,
                "boundary_weight": state.boundary_weight,
                "site_probability": state.localization.probability,
                "component_probabilities": state.localization.component_probabilities,
                "majorana_site_probability": state.majorana.site_probability,
                "majorana_particle_probability": state.majorana.particle_probability,
                "majorana_hole_probability": state.majorana.hole_probability,
                "majorana_polarization": state.majorana.polarization,
                "majorana_polarization_magnitude": state.majorana.polarization_magnitude,
                "majorana_total_polarization": state.majorana.total_polarization,
                "majorana_self_conjugacy": state.majorana.self_conjugacy,
                "majorana_polarization_norm": state.majorana.polarization_norm,
                "majorana_particle_weight": state.majorana.particle_weight,
                "majorana_hole_weight": state.majorana.hole_weight,
            }
            for state in boundary.states
        ],
    }


def size_methods_summary(
    outcomes: tuple[dict[str, Any], ...], *, preflight: bool
) -> dict[str, Any]:
    """Summarize each fixed condition without dropping failed attempts."""
    controls: dict[int, list[dict[str, Any]]] = defaultdict(list)
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in outcomes:
        if record["block"].startswith("control_"):
            controls[record["n"]].append(record)
        else:
            grouped[(record["n"], record["block"])].append(record)
    control_results = {}
    for n in SIZES:
        records = controls[n]
        valid = (
            len(records) == 2
            and all(_control_is_valid(record) for record in records)
            and _controls_agree(records[0], records[1])
        )
        control_results[str(n)] = {
            "valid": valid,
            "start_available": bool(records and _scientific_available(records[0].get("scientific"))),
            "end_available": bool(len(records) == 2 and _scientific_available(records[1].get("scientific"))),
        }
    cells = []
    for (n, block), records in sorted(grouped.items()):
        available = [record for record in records if _scientific_available(record.get("scientific"))]
        screening_hits = sum(record["scientific"]["screening_pass"] for record in available)
        central_values = [
            record["scientific"]["topology_grid"]["local_chern"][4]["bulk_chern_estimate"]
            for record in available
        ]
        mask_counts = []
        for index, name in enumerate(MASK_NAMES):
            hits = sum(
                record["scientific"]["topology_grid"]["local_chern"][index]["is_quantized"]
                and record["scientific"]["topology_grid"]["local_chern"][index]["chern_number"] == 1
                for record in available
            )
            mask_counts.append(
                {
                    "name": name,
                    "positive_quantized_count": hits,
                    "positive_quantized_fraction_all": hits / len(records),
                    "positive_quantized_fraction_available": None
                    if not available
                    else hits / len(available),
                    "wilson_95_all": list(wilson_interval(hits, len(records))),
                }
            )
        control_central = _control_central(controls[n][0]) if controls[n] else None
        cells.append(
            {
                "n": n,
                "block": block,
                "attempt_count": len(records),
                "available_count": len(available),
                "operational_failure_count": len(records) - len(available),
                "screening_pass_count": screening_hits,
                "screening_fraction_all": screening_hits / len(records),
                "screening_fraction_available": None
                if not available
                else screening_hits / len(available),
                "screening_wilson_95_all": list(wilson_interval(screening_hits, len(records))),
                "central_absolute_error": _numeric_summary(
                    [abs(value - 1.0) for value in central_values]
                ),
                "central_difference_from_square": _numeric_summary(
                    []
                    if control_central is None
                    else [value - control_central for value in central_values]
                ),
                "mask_positive_quantized": mask_counts,
                "gate_reason_counts": _reason_counts(available),
                "trajectory_results": [_trajectory_result(record) for record in records],
            }
        )
    return {
        "protocol_id": SIZE_METHODS_PROTOCOL_ID,
        "preflight_only": preflight,
        "control_valid": all(item["valid"] for item in control_results.values()),
        "control_results": control_results,
        "all_evaluations_available": all(
            _scientific_available(record.get("scientific")) for record in outcomes
        ),
        "physical_interpretation_allowed": all(
            item["valid"] for item in control_results.values()
        ),
        "physical_interpretation_allowed_by_size": {
            n: item["valid"] for n, item in control_results.items()
        },
        "cells": cells,
        "paired_central_differences": _paired_differences(grouped),
        "claim_boundary": "finite 8x8/10x10/12x12 methods comparison; descriptive only",
    }


def _paired_differences(
    grouped: dict[tuple[int, str], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    results = []
    for n in SIZES:
        by_block = {
            block: {
                record["root"]: record
                for record in grouped[(n, block)]
                if _scientific_available(record.get("scientific"))
            }
            for block in ("positions", "connectivity", "combined")
        }
        roots = sorted(set.intersection(*(set(records) for records in by_block.values())))
        combined_minus_positions = []
        combined_minus_connectivity = []
        for root in roots:
            estimates = {
                block: by_block[block][root]["scientific"]["topology_grid"]["local_chern"][4][
                    "bulk_chern_estimate"
                ]
                for block in by_block
            }
            combined_minus_positions.append(estimates["combined"] - estimates["positions"])
            combined_minus_connectivity.append(estimates["combined"] - estimates["connectivity"])
        results.append(
            {
                "n": n,
                "paired_available_count": len(roots),
                "combined_minus_positions": _numeric_summary(combined_minus_positions),
                "combined_minus_connectivity": _numeric_summary(combined_minus_connectivity),
            }
        )
    return results


def _controls_agree(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_run, second_run = first.get("run"), second.get("run")
    if (
        first_run is None
        or second_run is None
        or first_run.simulation_result is None
        or second_run.simulation_result is None
    ):
        return False
    first_scientific = first["scientific"]
    second_scientific = second["scientific"]
    first_discrete, first_floats = _control_signature(first_scientific)
    second_discrete, second_floats = _control_signature(second_scientific)
    return (
        first_discrete == second_discrete
        and np.allclose(first_floats, second_floats, rtol=0.0, atol=1.0e-10)
        and np.allclose(
            first_run.simulation_result.eigenvalues,
            second_run.simulation_result.eigenvalues,
            rtol=0.0,
            atol=1.0e-10,
        )
    )


def _control_signature(scientific: dict[str, Any]) -> tuple[Any, list[float]]:
    grid = scientific["topology_grid"]
    boundary = scientific["boundary_signature"]
    discrete = (
        scientific["control_pass"],
        scientific["screening_pass"],
        tuple(scientific["gate_reasons"]),
        tuple((item["index"], item.get("is_quantized")) for item in grid["bott"]),
        tuple(
            (item["chern_number"], item.get("is_quantized"), item.get("bulk_site_count"))
            for item in grid["local_chern"]
        ),
        tuple(
            (item["local_chern_number"], item.get("is_invertible"), item.get("signature"))
            for item in grid["localizer"]
        ),
        boundary["boundary_localized_count"],
        tuple(tuple(pair) for pair in boundary.get("particle_hole_pairs_by_state_index", ())),
        tuple(boundary.get("reasons", ())),
    )
    floats = (
        [item["estimate"] for item in grid["bott"]]
        + [item["bulk_chern_estimate"] for item in grid["local_chern"]]
        + [item["localizer_gap"] for item in grid["localizer"]]
        + [item["energy"] for item in boundary["states"]]
    )
    return discrete, floats


def _control_is_valid(record: dict[str, Any]) -> bool:
    scientific = record.get("scientific")
    if not _scientific_available(scientific) or not isinstance(scientific, dict):
        return False
    return scientific["control_pass"] is True


def _scientific_available(scientific: Any) -> bool:
    if not isinstance(scientific, dict) or scientific.get("available") is not True:
        return False
    pipeline = scientific.get("pipeline")
    grid = scientific.get("topology_grid")
    return bool(
        isinstance(pipeline, dict)
        and pipeline.get("is_valid") is True
        and isinstance(grid, dict)
        and len(grid.get("bott", ())) == 3
        and len(grid.get("local_chern", ())) == 5
        and len(grid.get("localizer", ())) == 3
        and scientific.get("boundary_signature") is not None
        and len(scientific.get("marker_maps", ())) == 5
    )


def _load_bound_sealed(directory: Path, expected: dict[str, Any]) -> dict[str, Any]:
    sealed = load_record(directory / "sealed.json")
    execution = directory / sealed["execution"]
    stored_input = load_record(execution / "input.json")
    if encode_record(stored_input) != encode_record(expected):
        raise ResearchAbort("Versiegelter Zellinput stimmt nicht mit dem Zellplan überein")
    record = load_sealed(directory)
    if not isinstance(record, dict):
        raise ResearchAbort("Versiegeltes Zellergebnis ist keine Zuordnung")
    if any(record.get(key) != value for key, value in _identity(expected).items()):
        raise ResearchAbort("Versiegeltes Zellergebnis hat eine falsche Identität")
    return record


def _identity(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        key: cell[key]
        for key in (
            "n",
            "block",
            "root",
            "amplitude",
            "edge_step",
            "edge_distance",
            "geometry_id",
        )
    }


def _control_central(record: dict[str, Any]) -> float | None:
    if not _scientific_available(record.get("scientific")):
        return None
    return float(
        record["scientific"]["topology_grid"]["local_chern"][4]["bulk_chern_estimate"]
    )


def _trajectory_result(record: dict[str, Any]) -> dict[str, Any]:
    scientific = record.get("scientific")
    if not _scientific_available(scientific):
        return {
            "root": record["root"],
            "available": False,
            "edge_distance": record["edge_distance"],
            "error": record["error"],
        }
    assert scientific is not None
    return {
        "root": record["root"],
        "available": True,
        "edge_distance": record["edge_distance"],
        "edge_distance_fraction": record["edge_distance"] / (2 * record["n"] * (record["n"] - 1)),
        "attempted_step_fraction": record["edge_step"] / (2 * record["n"] * (record["n"] - 1)),
        "screening_pass": scientific["screening_pass"],
        "gate_reasons": scientific["gate_reasons"],
        "localizer_protection_proxy": scientific["localizer_protection_proxy"],
        "bott": scientific["topology_grid"]["bott"],
        "local_chern": scientific["topology_grid"]["local_chern"],
        "localizer": scientific["topology_grid"]["localizer"],
        "boundary_localized_count": scientific["boundary_signature"][
            "boundary_localized_count"
        ],
        "maximum_pair_residual": scientific["boundary_signature"]["maximum_pair_residual"],
        "construction_metrics": record.get("construction_metrics"),
    }


def _reason_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        for reason in record["scientific"]["gate_reasons"]:
            result[reason] = result.get(reason, 0) + 1
    return result


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "minimum": None if not values else min(values),
        "median": None if not values else float(np.median(values)),
        "maximum": None if not values else max(values),
        "values": values,
    }


def _observed_seconds_by_size(path: Path, stage: str) -> dict[str, float]:
    """Read the latest sealed-evaluation duration for every stage slot."""
    observations: dict[tuple[int, int], float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            event.get("stage") == stage
            and event.get("event") in ("evaluated", "evaluation_failed")
            and type(event.get("size")) is int
            and type(event.get("slot")) is int
            and isinstance(event.get("operation_seconds"), (int, float))
        ):
            observations[(event["size"], event["slot"])] = float(event["operation_seconds"])
    result = {}
    for n in SIZES:
        values = [seconds for (size, _), seconds in observations.items() if size == n]
        if values:
            result[str(n)] = float(np.median(values))
    return result


def _audit_seed_roles(plans: tuple[dict[str, Any], ...]) -> None:
    old = set(
        PREFLIGHT_SEEDS
        + TRIAL_SEEDS
        + REFERENCE_SEEDS
        + VALIDATION_SEEDS
        + CONFIRMATION_SEEDS
    )
    for root in OLD_CALIBRATION_PREFLIGHT_ROOTS + OLD_CALIBRATION_FULL_ROOTS:
        schedule = derive_calibration_schedule(root)
        old.update((root, schedule["position_seed"], *schedule["step_seeds"]))
    plan_roots = tuple(plan["root"] for plan in plans)
    if plan_roots == PREFLIGHT_ROOTS:
        schedules = tuple(plan["schedule"] for plan in plans)
        reserved_other_roots = set(FULL_ROOTS)
    elif plan_roots == FULL_ROOTS:
        schedules = tuple(derive_size_schedule(root) for root in PREFLIGHT_ROOTS) + tuple(
            plan["schedule"] for plan in plans
        )
        reserved_other_roots = set()
    else:
        raise ResearchAbort("Größen-/Methodenplan enthält nicht die eingefrorenen Wurzelseeds")
    seen: set[int] = set()
    for schedule in schedules:
        values = [schedule["root"]]
        for item in schedule["sizes"]:
            values.extend((item["size_seed"], item["position_seed"], *item["step_seeds"]))
        for value in values:
            if value in old or value in seen or value in reserved_other_roots:
                raise ResearchAbort(f"Größen-/Methodenseed-Kollision: {value}")
            seen.add(value)
    for plan in plans:
        if derive_size_schedule(plan["root"]) != plan["schedule"]:
            raise ResearchAbort("Versiegelter Größen-/Methodenseedplan weicht vom Protokoll ab")


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase-10-Größen-/Methodenvergleich",
        "",
        f"Protokoll: `{SIZE_METHODS_PROTOCOL_ID}`",
        "",
        f"Nullkontrollen gültig: **{summary['control_valid']}**",
        "",
        "| Größe | Variante | Screening bestanden | Verfügbar | zentraler |C-1| Median [Min, Max] |",
        "| ---: | --- | ---: | ---: | --- |",
    ]
    for cell in summary["cells"]:
        central = cell["central_absolute_error"]
        central_text = (
            "nicht verfügbar"
            if central["count"] == 0
            else f"{central['median']:.6g} [{central['minimum']:.6g}, {central['maximum']:.6g}]"
        )
        lines.append(
            f"| {cell['n']}×{cell['n']} | {cell['block']} | "
            f"{cell['screening_pass_count']}/{cell['attempt_count']} | "
            f"{cell['available_count']}/{cell['attempt_count']} | {central_text} |"
        )
    lines += [
        "",
        "Alle fünf Masken, Markerfelder, Fehlernenner und gepaarten Differenzen stehen in `summary.json`.",
        "",
        "Dieser finite Methodenvergleich ist deskriptiv und kein thermodynamischer Nachweis.",
        "",
    ]
    return "\n".join(lines)


def _output_path(output: Path) -> Path:
    root = output.resolve()
    results = Path(__file__).resolve().parents[3] / "results"
    if not root.is_relative_to(results.resolve()) or not root.name.startswith(
        "phase_10_size_methods"
    ):
        raise ValueError("Ein neues results/phase_10_size_methods...-Verzeichnis verwenden")
    return root
