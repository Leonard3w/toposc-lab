"""User-started campaign for the frozen Phase-10 edge-location protocol."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
import time
from contextlib import redirect_stdout
from importlib.metadata import version
from io import StringIO
from pathlib import Path
from typing import Any, Literal

import numpy as np

from toposc_lab.evaluation.reproducibility import exact_geometry_id
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
from toposc_lab.search.phase_10_campaign import (
    _artifact_inventory,
    _campaign_lock,
    _verify_completion,
)
from toposc_lab.search.phase_10_edge_location import (
    EDGE_LOCATION_PROTOCOL_COMMIT,
    EDGE_LOCATION_PROTOCOL_ID,
    EDGE_LOCATION_PROTOCOL_PATH,
    FULL_PAIR_IDS,
    PREFLIGHT_PAIR_IDS,
    N,
    build_edge_location_plan,
)
from toposc_lab.search.phase_10_research import THREAD_VARIABLES
from toposc_lab.search.phase_10_size_methods import MASK_NAMES, build_size_topology_inputs
from toposc_lab.search.phase_10_size_methods_campaign import (
    _control_is_valid,
    _controls_agree,
    _encode_size_scientific,
    _numeric_summary,
    _scientific_available,
)


def edge_location_environment() -> dict[str, Any]:
    """Require the frozen protocol, committed code and controlled runtime."""
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
    if code == EDGE_LOCATION_PROTOCOL_COMMIT:
        raise ValueError("Die Implementierung muss nach dem Protokoll separat committed sein")
    git("merge-base", "--is-ancestor", EDGE_LOCATION_PROTOCOL_COMMIT, code)
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
        raise ValueError("Worktree vor Rand-/Innenlauf committen: " + "; ".join(unexpected))
    frozen = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "show",
            f"{EDGE_LOCATION_PROTOCOL_COMMIT}:{EDGE_LOCATION_PROTOCOL_PATH}",
        ],
        check=True,
        capture_output=True,
    ).stdout.replace(b"\r\n", b"\n")
    current = (root / EDGE_LOCATION_PROTOCOL_PATH).read_bytes().replace(b"\r\n", b"\n")
    if current != frozen:
        raise ValueError("Rand-/Innenprotokoll weicht vom eingefrorenen Commit ab")
    config = StringIO()
    with redirect_stdout(config):
        np.show_config()
    return {
        "code_commit": code,
        "protocol_commit": EDGE_LOCATION_PROTOCOL_COMMIT,
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


def run_edge_location_campaign(
    output: Path, *, mode: Literal["preflight", "full", "resume"]
) -> Path:
    """Run only the explicitly requested stage of the frozen location experiment."""
    environment = edge_location_environment()
    root = _output_path(output)
    if mode == "preflight":
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        raise ValueError("Zuerst den Rand-/Innenvorlauf in einem neuen Verzeichnis starten")
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
                    raise ResearchAbort("Passender bestandener Rand-/Innenvorlauf erforderlich")
                if load_record(preflight / "manifest.json")["environment"] != environment:
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
    del resume
    preflight = mode == "preflight"
    pair_ids = PREFLIGHT_PAIR_IDS if preflight else FULL_PAIR_IDS
    total = 6 if preflight else 82
    protocol_file = Path(__file__).resolve().parents[3] / EDGE_LOCATION_PROTOCOL_PATH
    metadata = {
        "protocol_id": EDGE_LOCATION_PROTOCOL_ID,
        "protocol_commit": EDGE_LOCATION_PROTOCOL_COMMIT,
        "environment": environment,
        "mode": mode,
        "total_evaluation_attempts": total,
        "protocol_text": protocol_file.read_text(encoding="utf-8"),
        "pair_ids": pair_ids,
        "mask_names": MASK_NAMES,
        "evaluation_seeds": (),
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
    monitor.emit("constructing", stage=mode, block="construction", completed=0, total=total)
    construction_started = time.perf_counter()
    constructed = build_edge_location_plan(preflight=preflight)
    prepared = tuple(_prepare_cell(cell) for cell in constructed)
    _audit_plan(prepared, preflight=preflight)
    plan_path = stage / "input_plan.json"
    if plan_path.exists():
        if encode_record(load_record(plan_path)) != encode_record(prepared):
            raise ResearchAbort("Gespeicherter Zellinputplan weicht von der Konstruktion ab")
    else:
        save_record(plan_path, prepared)
    monitor.emit(
        "constructed",
        stage=mode,
        block="construction",
        completed=0,
        total=total,
        category="construction",
        operation_seconds=time.perf_counter() - construction_started,
    )

    estimate = None
    if mode == "full":
        estimate = float(load_record(stage.parent / "preflight" / "complete.json")
                         ["seconds_per_evaluation"])
    outcomes = []
    evaluations_started = time.perf_counter()
    for index, cell in enumerate(prepared):
        directory = stage / "evaluations" / f"cell_{index:04d}"
        if (directory / "sealed.json").exists():
            record = _load_bound_sealed(directory, cell)
        else:
            record = _evaluate_cell(
                directory, cell, environment["code_commit"], monitor, mode, index, total,
                estimated_seconds=estimate,
            )
        outcomes.append(record)
        if cell["block"] == "control_start" and not _control_is_valid(record):
            raise ResearchAbort("Anfangskontrolle für den Rand-/Innenlauf ist ungültig")

    summary = edge_location_summary(tuple(outcomes), preflight=preflight)
    summary["environment"] = environment
    summary["protocol_commit"] = EDGE_LOCATION_PROTOCOL_COMMIT
    summary["total_evaluation_attempts"] = total
    analysis_path = stage / "analysis.json"
    if analysis_path.exists():
        if encode_record(load_record(analysis_path)) != encode_record(summary):
            raise ResearchAbort("Wiederholte Rand-/Innenanalyse stimmt nicht überein")
    else:
        save_record(analysis_path, summary)
    publish_derived(stage / "summary.json", json_bytes(summary))
    publish_derived(stage / "report.md", _render_report(summary).encode("utf-8"))
    preflight_passed = bool(
        preflight and summary["control_valid"] and summary["all_evaluations_available"]
    )
    save_record(
        stage / "complete.json",
        {
            "preflight_passed": preflight_passed,
            "artifacts": _artifact_inventory(stage),
            "seconds_per_evaluation": (time.perf_counter() - evaluations_started) / total,
            "total_stage_seconds": time.perf_counter() - stage_started,
        },
    )
    monitor.emit(
        "complete", stage=mode, completed=total, total=total, last_sealed=f"{mode}/complete"
    )
    return stage / "report.md"


def _prepare_cell(cell: dict[str, Any]) -> dict[str, Any]:
    geometry = cell["genome"].to_geometry()
    if cell["geometry_id"] != exact_geometry_id(geometry):
        raise ResearchAbort("Konstruiertes Geometry-ID-Audit ist fehlgeschlagen")
    report = cell["validation"]
    if not report["is_valid"]:
        raise ResearchAbort(f"Ungültige Geometrie im Zellinputplan: {report['issues']}")
    _, evidence = build_size_topology_inputs(geometry, N)
    expected_order = np.arange(N * N, dtype=np.int64)
    if not np.array_equal(evidence["unique_coordinate_order"], expected_order):
        raise ResearchAbort("Marker-Ortsordnung weicht von der physischen Site-Reihenfolge ab")
    if not np.allclose(
        evidence["position_areas_in_coordinate_order"], 1.0, rtol=0.0, atol=1.0e-10
    ):
        raise ResearchAbort("Rand-/Innenlauf verlangt einheitliche Voronoi-Flächen")
    return {**cell, "topology_input_evidence": evidence}


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
    trial = None
    if cell["pair_id"] is not None:
        trial = (
            f"p={cell['pair_id']}/m={cell['mirror']}/r={cell['rotation']}/"
            f"y={cell['anchor_y']}"
        )
    progress: dict[str, Any] = {
        "stage": mode,
        "block": cell["block"],
        "trial": trial,
        "arm": cell["arm"],
        "size": N,
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
        inputs, evidence = build_size_topology_inputs(geometry, N)
        if encode_record(evidence) != encode_record(cell["topology_input_evidence"]):
            raise ValueError("topology inputs differ from sealed cell input")
        run, grid = evaluate_phase_9_8_descriptive_geometry(
            geometry, inputs=inputs, code_version=code, timing_callback=record_timing
        )
        boundary = None
        if run.is_valid and run.evaluation is not None:
            boundary = build_phase_9_8_boundary_signature(run.evaluation)
        scientific = _encode_size_scientific(run, grid, boundary)
        record = {
            **identity,
            "construction_metrics": _construction_metrics(cell),
            "run": run,
            "scientific": scientific,
            "error": None,
        }
    except Exception as error:  # noqa: BLE001 - failures stay in the fixed budget
        record = {
            **identity,
            "construction_metrics": _construction_metrics(cell),
            "run": None,
            "scientific": None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    elapsed = time.perf_counter() - started
    timings["total"] = elapsed
    ledger.record("outcome.json", record)
    ledger.seal(record)
    final_progress = {
        **progress,
        "completed": index + 1,
        "screening_pass": None if record["scientific"] is None
        else record["scientific"]["screening_pass"],
        "last_sealed": f"{mode}/evaluations/cell_{index:04d}",
        "category": "physics",
        "operation_seconds": elapsed,
        "stage_timings": timings,
    }
    monitor.emit(
        "evaluated" if _scientific_available(record.get("scientific")) else "evaluation_failed",
        **final_progress,
    )
    return record


def edge_location_summary(
    outcomes: tuple[dict[str, Any], ...], *, preflight: bool
) -> dict[str, Any]:
    """Summarize fixed arms and pairs without treating placements as iid draws."""
    controls = [record for record in outcomes if record["block"].startswith("control_")]
    variants = [record for record in outcomes if record["block"] == "edge_location"]
    control_valid = bool(
        len(controls) == 2
        and all(_control_is_valid(record) for record in controls)
        and _controls_agree(controls[0], controls[1])
    )
    reference_marker = _marker(controls[0]) if controls else None
    reference_chern = _chern_values(controls[0]) if controls else None
    rows = [
        _variant_summary(record, reference_marker, reference_chern) for record in variants
    ]
    by_key = {(row["pair_id"], row["arm"]): row for row in rows}
    pair_rows = []
    pair_ids = PREFLIGHT_PAIR_IDS if preflight else FULL_PAIR_IDS
    for pair_id in pair_ids:
        boundary = by_key.get((pair_id, "boundary"))
        interior = by_key.get((pair_id, "interior"))
        pair_available = bool(
            boundary and interior and boundary["available"] and interior["available"]
        )
        central_difference = None
        patch_absolute_difference = None
        patch_signed_difference = None
        if pair_available and boundary is not None and interior is not None:
            central_difference = (
                float(interior["central_absolute_error"])
                - float(boundary["central_absolute_error"])
            )
            patch_absolute_difference = (
                float(interior["patch_marker_change"]["mean_absolute"])
                - float(boundary["patch_marker_change"]["mean_absolute"])
            )
            patch_signed_difference = (
                float(interior["patch_marker_change"]["signed_mean"])
                - float(boundary["patch_marker_change"]["signed_mean"])
            )
        pair_rows.append(
            {
                "pair_id": pair_id,
                "mirror": None if boundary is None else boundary["mirror"],
                "rotation": None if boundary is None else boundary["rotation"],
                "anchor_y": None if boundary is None else boundary["anchor_y"],
                "available": pair_available,
                "central_abs_error_interior_minus_boundary": central_difference,
                "patch_mean_abs_delta_interior_minus_boundary": patch_absolute_difference,
                "patch_signed_mean_delta_interior_minus_boundary": patch_signed_difference,
            }
        )
    available_pairs = [row for row in pair_rows if row["available"]]
    primary_values = []
    patch_values = []
    for row in available_pairs:
        primary_value = row["central_abs_error_interior_minus_boundary"]
        patch_value = row["patch_mean_abs_delta_interior_minus_boundary"]
        if not isinstance(primary_value, (int, float)) or not isinstance(
            patch_value, (int, float)
        ):
            raise ResearchAbort("Verfügbares Rand-/Innenpaar hat keinen numerischen Kontrast")
        primary_values.append(float(primary_value))
        patch_values.append(float(patch_value))
    arms = {}
    for arm in ("boundary", "interior"):
        selected = [row for row in rows if row["arm"] == arm]
        arm_available = [row for row in selected if row["available"]]
        arms[arm] = {
            "attempt_count": len(selected),
            "available_count": len(arm_available),
            "operational_failure_count": len(selected) - len(arm_available),
            "screening_pass_count": sum(row["screening_pass"] for row in arm_available),
            "central_absolute_error": _numeric_summary(
                [float(row["central_absolute_error"]) for row in arm_available]
            ),
            "patch_mean_absolute_marker_change": _numeric_summary(
                [
                    float(row["patch_marker_change"]["mean_absolute"])
                    for row in arm_available
                ]
            ),
        }
    return {
        "protocol_id": EDGE_LOCATION_PROTOCOL_ID,
        "preflight_only": preflight,
        "control_valid": control_valid,
        "control_start_available": bool(
            controls and _scientific_available(controls[0].get("scientific"))
        ),
        "control_end_available": bool(
            len(controls) == 2 and _scientific_available(controls[1].get("scientific"))
        ),
        "all_evaluations_available": all(
            _scientific_available(record.get("scientific")) for record in outcomes
        ),
        "physical_interpretation_allowed": control_valid,
        "arms": arms,
        "primary_paired_difference": _numeric_summary(primary_values),
        "paired_patch_absolute_difference": _numeric_summary(patch_values),
        "pair_results": pair_rows,
        "variant_results": rows,
        "independence_warning": (
            "deterministic and symmetry-related placements; not independent random samples"
        ),
        "claim_boundary": "finite 12x12 deterministic edge-location methods comparison",
    }


def _variant_summary(
    record: dict[str, Any],
    reference_marker: np.ndarray | None,
    reference_chern: tuple[float, ...] | None,
) -> dict[str, Any]:
    identity = {key: record[key] for key in ("pair_id", "arm", "mirror", "rotation", "anchor_y")}
    scientific = record.get("scientific")
    if not _scientific_available(scientific) or reference_marker is None or reference_chern is None:
        return {**identity, "available": False, "error": record.get("error")}
    assert isinstance(scientific, dict)
    marker = _marker(record)
    assert marker is not None
    delta = marker - reference_marker
    sites = np.asarray(record["construction_metrics"]["measurement_sites"], dtype=np.intp)
    patch = delta[sites]
    rings = []
    for depth in range(6):
        ring_sites = np.asarray(
            [site for site in range(N * N) if _site_depth(site) == depth], dtype=np.intp
        )
        values = delta[ring_sites]
        rings.append(
            {
                "depth": depth,
                "site_count": len(ring_sites),
                "mean_absolute": float(np.mean(np.abs(values))),
                "signed_mean": float(np.mean(values)),
            }
        )
    chern = _chern_values(record)
    assert chern is not None
    decomposition = []
    for index in (0, 1):
        mask = np.asarray(scientific["marker_maps"][index]["bulk_mask"], dtype=bool)
        reference_on_variant_mask = float(np.mean(reference_marker[mask]))
        decomposition.append(
            {
                "name": MASK_NAMES[index],
                "field_and_area_term": chern[index] - reference_on_variant_mask,
                "reference_mask_selection_term": (
                    reference_on_variant_mask - reference_chern[index]
                ),
            }
        )
    central = chern[4]
    return {
        **identity,
        "available": True,
        "central_chern_estimate": central,
        "central_absolute_error": abs(central - 1.0),
        "screening_pass": scientific["screening_pass"],
        "gate_reasons": scientific["gate_reasons"],
        "localizer_protection_proxy": scientific["localizer_protection_proxy"],
        "mask_positive_quantized": [
            bool(item["is_quantized"] and item["chern_number"] == 1)
            for item in scientific["topology_grid"]["local_chern"]
        ],
        "patch_marker_change": {
            "site_count": len(sites),
            "mean_absolute": float(np.mean(np.abs(patch))),
            "signed_mean": float(np.mean(patch)),
        },
        "marker_change_by_original_depth": rings,
        "graph_mask_decomposition": decomposition,
        "topology_grid": scientific["topology_grid"],
        "boundary_signature": {
            key: scientific["boundary_signature"][key]
            for key in (
                "boundary_localized_count",
                "minimum_boundary_weight_first_four",
                "maximum_pair_residual",
                "pairing_cost",
                "reasons",
            )
        },
        "construction_metrics": record["construction_metrics"],
    }


def _marker(record: dict[str, Any]) -> np.ndarray | None:
    scientific = record.get("scientific")
    if not _scientific_available(scientific):
        return None
    assert isinstance(scientific, dict)
    marker_maps = scientific["marker_maps"]
    first = np.asarray(marker_maps[0]["local_marker"], dtype=float)
    if first.shape != (N * N,) or any(
        not np.array_equal(first, np.asarray(item["local_marker"], dtype=float))
        for item in marker_maps[1:]
    ):
        raise ResearchAbort("Gespeicherte Markerfelder sind nicht maskenübergreifend identisch")
    return first


def _chern_values(record: dict[str, Any]) -> tuple[float, ...] | None:
    scientific = record.get("scientific")
    if not _scientific_available(scientific):
        return None
    assert isinstance(scientific, dict)
    return tuple(
        float(item["bulk_chern_estimate"])
        for item in scientific["topology_grid"]["local_chern"]
    )


def _site_depth(site: int) -> int:
    x_value, y_value = divmod(site, N)
    return min(x_value, y_value, N - 1 - x_value, N - 1 - y_value)


def _construction_metrics(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        key: cell[key]
        for key in (
            "affected_sites", "removed_edges", "added_edges", "measurement_sites",
            "intervention_depth", "degree_sequence", "edge_distance",
            "removed_edge_lengths", "added_edge_lengths", "validation",
            "removed_edge_displacements", "added_edge_displacements",
        )
    }


def _audit_plan(cells: tuple[dict[str, Any], ...], *, preflight: bool) -> None:
    expected_ids = PREFLIGHT_PAIR_IDS if preflight else FULL_PAIR_IDS
    variants = [cell for cell in cells if cell["block"] == "edge_location"]
    if tuple(sorted({cell["pair_id"] for cell in variants})) != expected_ids:
        raise ResearchAbort("Rand-/Innenplan enthält nicht die eingefrorenen Paar-IDs")
    if len({cell["geometry_id"] for cell in variants}) != len(variants):
        raise ResearchAbort("Rand-/Innenplan enthält doppelte Varianten-Geometrien")
    for pair_id in expected_ids:
        paired = [cell for cell in variants if cell["pair_id"] == pair_id]
        if {cell["arm"] for cell in paired} != {"boundary", "interior"}:
            raise ResearchAbort("Rand-/Innenplan enthält ein unvollständiges Paar")
        first, second = paired
        for key in ("mirror", "rotation", "anchor_y", "removed_edge_lengths",
                    "added_edge_lengths", "removed_edge_displacements",
                    "added_edge_displacements", "degree_sequence", "edge_distance"):
            if first[key] != second[key]:
                raise ResearchAbort(f"Paarressource stimmt nicht überein: {key}")


def _load_bound_sealed(directory: Path, expected: dict[str, Any]) -> dict[str, Any]:
    sealed = load_record(directory / "sealed.json")
    stored = load_record(directory / sealed["execution"] / "input.json")
    if encode_record(stored) != encode_record(expected):
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
            "n", "block", "pair_id", "arm", "mirror", "rotation", "anchor_y",
            "geometry_id", "evaluation_seed",
        )
    }


def _render_report(summary: dict[str, Any]) -> str:
    primary = summary["primary_paired_difference"]
    primary_text = "nicht verfügbar" if not primary["count"] else (
        f"{primary['median']:.6g} [{primary['minimum']:.6g}, {primary['maximum']:.6g}]"
    )
    lines = [
        "# Phase-10-Rand-/Innenvergleich",
        "",
        f"Protokoll: `{EDGE_LOCATION_PROTOCOL_ID}`",
        "",
        f"Nullkontrollen gültig: **{summary['control_valid']}**",
        "",
        "| Arm | Screening bestanden | Verfügbar | zentraler |C-1| Median [Min, Max] |",
        "| --- | ---: | ---: | --- |",
    ]
    for arm in ("boundary", "interior"):
        item = summary["arms"][arm]
        central = item["central_absolute_error"]
        central_text = "nicht verfügbar" if not central["count"] else (
            f"{central['median']:.6g} [{central['minimum']:.6g}, {central['maximum']:.6g}]"
        )
        lines.append(
            f"| {arm} | {item['screening_pass_count']}/{item['attempt_count']} | "
            f"{item['available_count']}/{item['attempt_count']} | {central_text} |"
        )
    lines += [
        "",
        (
            "Primärkontrast `|C_interior-1| - |C_boundary-1|` als Median [Min, Max]: "
            f"**{primary_text}**."
        ),
        "",
        (
            "Die Platzierungen sind deterministisch und symmetrieverwandt, nicht unabhängige "
            "Zufallsstichproben. Alle Paar-, Marker-, Masken- und Fehlerdaten stehen in "
            "`summary.json`."
        ),
        "",
        "Dieser finite 12×12-Methodenvergleich ist kein Robustheits- oder Phasennachweis.",
        "",
    ]
    return "\n".join(lines)


def _output_path(output: Path) -> Path:
    root = output.resolve()
    results = Path(__file__).resolve().parents[3] / "results"
    if not root.is_relative_to(results.resolve()) or not root.name.startswith(
        "phase_10_edge_location"
    ):
        raise ValueError("Ein neues results/phase_10_edge_location...-Verzeichnis verwenden")
    return root
