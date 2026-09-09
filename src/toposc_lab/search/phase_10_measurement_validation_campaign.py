"""User-started runner for TOPOSC-P10-MEAS-VAL-001."""

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

from toposc_lab.core.model import BaseModel
from toposc_lab.evaluation import (
    GeometryEvaluationConfig,
    GeometryModelAdapter,
    ModelGeometryRequirements,
    evaluate_geometry,
)
from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.hamiltonians import NambuBasis
from toposc_lab.models.chiral_p_wave import ChiralPWaveModel, ChiralPWaveParameters
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
from toposc_lab.search.phase_9_8_evaluation import build_phase_9_8_boundary_signature
from toposc_lab.search.phase_10_campaign import (
    _artifact_inventory,
    _campaign_lock,
    _verify_completion,
)
from toposc_lab.search.phase_10_measurement_validation import (
    FULL_SIZES,
    MEASUREMENT_VALIDATION_PROTOCOL_COMMIT,
    MEASUREMENT_VALIDATION_PROTOCOL_ID,
    MEASUREMENT_VALIDATION_PROTOCOL_PATH,
    MODEL_ROLES,
    PREFLIGHT_SIZE,
    build_measurement_plan,
    build_measurement_topology_inputs,
)
from toposc_lab.search.phase_10_research import THREAD_VARIABLES
from toposc_lab.search.phase_10_size_methods import MASK_NAMES
from toposc_lab.search.random_search_experiment import _encode_pipeline_run
from toposc_lab.topology import (
    SymmetryClassification,
    bott_index,
    local_chern_marker,
    spectral_localizer,
)

Mode = Literal["preflight", "full", "resume"]
NUMERICAL_TOLERANCE = 1.0e-10
SIZE_STABILITY_TOLERANCE = 5.0e-3
LOCALIZER_PROTECTION_THRESHOLD = 0.20
PH_TOLERANCE = 1.0e-8

_EVALUATION_CONFIG = GeometryEvaluationConfig(
    reference_energy=0.0,
    zero_mode_tolerance=NUMERICAL_TOLERANCE,
    low_energy_count=16,
    boundary_localization_threshold=0.8,
    numerical_tolerance=NUMERICAL_TOLERANCE,
)


def measurement_validation_environment() -> dict[str, Any]:
    """Require the frozen protocol, a later clean commit and controlled runtime."""
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
    if code == MEASUREMENT_VALIDATION_PROTOCOL_COMMIT:
        raise ValueError("Die Implementierung muss nach dem Protokoll separat committed sein")
    git("merge-base", "--is-ancestor", MEASUREMENT_VALIDATION_PROTOCOL_COMMIT, code)
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
        raise ValueError("Worktree vor Messvalidierung committen: " + "; ".join(unexpected))
    frozen = subprocess.run(
        [
            "git", "-C", str(root), "show",
            f"{MEASUREMENT_VALIDATION_PROTOCOL_COMMIT}:{MEASUREMENT_VALIDATION_PROTOCOL_PATH}",
        ],
        check=True,
        capture_output=True,
    ).stdout.replace(b"\r\n", b"\n")
    current = (root / MEASUREMENT_VALIDATION_PROTOCOL_PATH).read_bytes().replace(b"\r\n", b"\n")
    if current != frozen:
        raise ValueError("Messvalidierungsprotokoll weicht vom eingefrorenen Commit ab")
    config = StringIO()
    with redirect_stdout(config):
        np.show_config()
    return {
        "code_commit": code,
        "protocol_commit": MEASUREMENT_VALIDATION_PROTOCOL_COMMIT,
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


def run_measurement_validation_campaign(output: Path, *, mode: Mode) -> Path:
    """Run only the explicitly requested frozen validation stage."""
    environment = measurement_validation_environment()
    root = _output_path(output)
    if mode == "preflight":
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        raise ValueError("Zuerst den Messvalidierungsvorlauf in einem neuen Verzeichnis starten")
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
                    raise ResearchAbort("Passender bestandener Messvalidierungsvorlauf erforderlich")
                if load_record(preflight / "manifest.json")["environment"] != environment:
                    raise ResearchAbort("Vorlauf und Hauptlauf benötigen dieselbe Umgebung")
            stage = root / mode
            if not resume:
                stage.mkdir(exist_ok=False)
            with ResearchMonitor(root / "events.jsonl") as monitor:
                try:
                    return _execute_stage(stage, mode, environment, monitor)
                except BaseException as error:
                    monitor.emit("interrupted", error_type=type(error).__name__, error=str(error))
                    raise
    except ResearchAbort as error:
        raise RuntimeError(str(error)) from error


def _execute_stage(
    stage: Path,
    mode: Literal["preflight", "full"],
    environment: dict[str, Any],
    monitor: ResearchMonitor,
) -> Path:
    preflight = mode == "preflight"
    total = 6 if preflight else 36
    protocol_file = Path(__file__).resolve().parents[3] / MEASUREMENT_VALIDATION_PROTOCOL_PATH
    metadata = {
        "protocol_id": MEASUREMENT_VALIDATION_PROTOCOL_ID,
        "protocol_commit": MEASUREMENT_VALIDATION_PROTOCOL_COMMIT,
        "environment": environment,
        "mode": mode,
        "total_evaluation_attempts": total,
        "protocol_text": protocol_file.read_text(encoding="utf-8"),
        "sizes": (PREFLIGHT_SIZE,) if preflight else FULL_SIZES,
        "model_roles": MODEL_ROLES,
        "model_role_parameters": {
            "topological": {
                "hopping": 1.0, "chemical_potential": 2.0, "pairing": 1.0,
                "chirality": 1, "plane_axes": [0, 1],
            },
            "trivial": {
                "hopping": 1.0, "chemical_potential": 12.0, "pairing": 1.0,
                "chirality": 1, "plane_axes": [0, 1],
            },
            "undefined": {
                "hopping": 0.0, "chemical_potential": 0.0, "pairing": 0.0,
                "chirality": 1, "plane_axes": [0, 1],
            },
        },
        "mask_names": MASK_NAMES,
        "numerical_contract": {
            "solver": "numpy.linalg.eigh",
            "spectrum": "full",
            "reference_energy": 0.0,
            "numerical_tolerance": NUMERICAL_TOLERANCE,
            "bott_quantization_tolerance": 1.0e-6,
            "local_chern_quantization_tolerance": 5.0e-3,
            "localizer_protection_threshold": LOCALIZER_PROTECTION_THRESHOLD,
            "particle_hole_tolerance": PH_TOLERANCE,
            "boundary_weight_threshold": 0.8,
            "size_stability_tolerance": SIZE_STABILITY_TOLERANCE,
        },
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
    prepared = tuple(_prepare_cell(cell) for cell in build_measurement_plan(preflight=preflight))
    _audit_plan(prepared, preflight=preflight)
    plan_path = stage / "input_plan.json"
    if plan_path.exists():
        if encode_record(load_record(plan_path)) != encode_record(prepared):
            raise ResearchAbort("Gespeicherter Messvalidierungsplan weicht von der Konstruktion ab")
    else:
        save_record(plan_path, prepared)
    monitor.emit(
        "constructed", stage=mode, block="construction", completed=0, total=total,
        category="construction", operation_seconds=time.perf_counter() - construction_started,
    )

    estimate = None
    if mode == "full":
        estimate = float(
            load_record(stage.parent / "preflight" / "complete.json")["seconds_per_evaluation"]
        )
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
            raise ResearchAbort(
                f"Ungültige Anfangskontrolle: n={cell['n']} role={cell['model_role']}"
            )

    summary = measurement_validation_summary(tuple(outcomes), preflight=preflight)
    summary["environment"] = environment
    summary["protocol_commit"] = MEASUREMENT_VALIDATION_PROTOCOL_COMMIT
    summary["total_evaluation_attempts"] = total
    analysis_path = stage / "analysis.json"
    if analysis_path.exists():
        if encode_record(load_record(analysis_path)) != encode_record(summary):
            raise ResearchAbort("Wiederholte Messvalidierungsanalyse stimmt nicht überein")
    else:
        save_record(analysis_path, summary)
    publish_derived(stage / "summary.json", json_bytes(summary))
    publish_derived(stage / "report.md", _render_report(summary).encode("utf-8"))
    preflight_passed = bool(preflight and summary["controls_passed"])
    save_record(
        stage / "complete.json",
        {
            "preflight_passed": preflight_passed,
            "artifacts": _artifact_inventory(stage),
            "seconds_per_evaluation": (time.perf_counter() - evaluations_started) / total,
            "total_stage_seconds": time.perf_counter() - stage_started,
        },
    )
    monitor.emit("complete", stage=mode, completed=total, total=total, last_sealed=f"{mode}/complete")
    return stage / "report.md"


def _prepare_cell(cell: dict[str, Any]) -> dict[str, Any]:
    geometry = cell["genome"].to_geometry()
    if exact_geometry_id(geometry) != cell["geometry_id"]:
        raise ResearchAbort("Konstruiertes Geometry-ID-Audit ist fehlgeschlagen")
    if not cell["validation"]["is_valid"]:
        raise ResearchAbort("Ungültige Geometrie im Messvalidierungsplan")
    _, evidence = build_measurement_topology_inputs(geometry, cell["n"])
    expected_order = np.arange(cell["n"] * cell["n"], dtype=np.int64)
    if not np.array_equal(evidence["unique_coordinate_order"], expected_order):
        raise ResearchAbort("Marker-Ortsordnung weicht von der physischen Site-Reihenfolge ab")
    if not np.allclose(
        evidence["position_areas_in_coordinate_order"], 1.0, rtol=0.0, atol=1.0e-10
    ):
        raise ResearchAbort("Messvalidierung verlangt einheitliche Voronoi-Flächen")
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
    progress: dict[str, Any] = {
        "stage": mode,
        "block": cell["block"],
        "trial": cell["offset"],
        "arm": cell["arm"],
        "model_role": cell["model_role"],
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
    try:
        record = _calculate_cell(cell, code, monitor, progress, timings)
    except Exception as error:  # noqa: BLE001 - operational failures remain in budget
        record = {
            **_identity(cell),
            "run": None,
            "hamiltonian": None,
            "methods": None,
            "classification": "operational_failure",
            "positive_screening": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    elapsed = time.perf_counter() - started
    timings["total"] = elapsed
    record["timings"] = timings
    ledger.record("outcome.json", record)
    ledger.seal(record)
    monitor.emit(
        "evaluated" if record["classification"] != "operational_failure" else "evaluation_failed",
        **{
            **progress,
            "completed": index + 1,
            "classification": record["classification"],
            "screening_pass": record["positive_screening"],
            "last_sealed": f"{mode}/evaluations/cell_{index:04d}",
            "category": "physics",
            "operation_seconds": elapsed,
            "stage_timings": timings,
        },
    )
    return record


def _calculate_cell(
    cell: dict[str, Any],
    code: str,
    monitor: ResearchMonitor,
    progress: dict[str, Any],
    timings: dict[str, float],
) -> dict[str, Any]:
    geometry = cell["genome"].to_geometry()
    if exact_geometry_id(geometry) != cell["geometry_id"]:
        raise ValueError("geometry ID differs from sealed input")
    inputs, evidence = build_measurement_topology_inputs(geometry, cell["n"])
    if encode_record(evidence) != encode_record(cell["topology_input_evidence"]):
        raise ValueError("topology inputs differ from sealed input")
    parameters = ChiralPWaveParameters.model_validate(cell["model_parameters"])
    model = ChiralPWaveModel(geometry, parameters)
    stage_started = time.perf_counter()
    hamiltonian = np.asarray(model.hamiltonian(), dtype=complex)
    timings["hamiltonian"] = time.perf_counter() - stage_started
    if cell["model_role"] == "undefined" and np.count_nonzero(hamiltonian):
        raise ValueError("undefined control Hamiltonian must be exactly zero")
    run = evaluate_geometry(
        geometry,
        adapter=_model_adapter(parameters),
        config=_EVALUATION_CONFIG,
        seed=None,
        code_version=code,
        timing_callback=lambda category, seconds: timings.__setitem__(
            category, timings.get(category, 0.0) + seconds
        ),
    )
    if not run.is_valid or run.simulation_result is None or run.evaluation is None:
        return {
            **_identity(cell),
            "run": run,
            "hamiltonian": hamiltonian,
            "methods": None,
            "classification": "operational_failure",
            "positive_screening": False,
            "error": None if run.failure is None else {
                "type": run.failure.error_type,
                "message": run.failure.message,
            },
        }
    actual_parameters = (
        None
        if run.reproducibility is None
        else ChiralPWaveParameters.model_validate(
            dict(run.reproducibility.model_parameters)
        )
    )
    if actual_parameters != parameters:
        raise ValueError("actual model provenance differs from the sealed role parameters")
    residual = float(
        np.max(
            np.abs(
                hamiltonian @ run.simulation_result.eigenvectors
                - run.simulation_result.eigenvectors * run.simulation_result.eigenvalues
            )
        )
    )
    if residual > 1.0e-8:
        raise ValueError("archived eigensystem does not solve the archived Hamiltonian")
    minimum_fermi_distance = float(np.min(np.abs(run.simulation_result.eigenvalues)))
    methods = _evaluate_methods(hamiltonian, inputs, monitor, progress, timings)
    classification, regional_sensitivity = _classify(methods, minimum_fermi_distance)
    classification_reason = _classification_reason(
        methods, minimum_fermi_distance, classification
    )
    boundary = None
    boundary_error = None
    if minimum_fermi_distance > NUMERICAL_TOLERANCE:
        try:
            boundary = build_phase_9_8_boundary_signature(run.evaluation)
        except Exception as error:  # noqa: BLE001 - separate diagnostic availability
            boundary_error = {"type": type(error).__name__, "message": str(error)}
    localizer_protection = _minimum_localizer_gap(methods)
    positive = bool(
        classification == "primary_topological"
        and localizer_protection is not None
        and localizer_protection >= LOCALIZER_PROTECTION_THRESHOLD
        and boundary is not None
        and boundary.maximum_pair_residual <= PH_TOLERANCE
        and boundary.boundary_localized_count >= 4
    )
    return {
        **_identity(cell),
        "run": run,
        "pipeline": _encode_pipeline_run(run),
        "hamiltonian": hamiltonian,
        "eigensystem_residual": residual,
        "minimum_fermi_distance": minimum_fermi_distance,
        "methods": methods,
        "classification": classification,
        "classification_reason": classification_reason,
        "regional_sensitivity": regional_sensitivity,
        "positive_screening": positive,
        "old_size_screening": _old_size_screening(methods, boundary),
        "boundary_signature": boundary,
        "boundary_error": boundary_error,
        "error": None,
    }


def _evaluate_methods(
    hamiltonian: np.ndarray,
    inputs: Any,
    monitor: ResearchMonitor,
    progress: dict[str, Any],
    timings: dict[str, float],
) -> dict[str, list[dict[str, Any]]]:
    classification = SymmetryClassification.from_signature(
        time_reversal_square=None, particle_hole_square=1, chiral_symmetry=False
    )
    result: dict[str, list[dict[str, Any]]] = {"bott": [], "local_chern": [], "localizer": []}
    for index, periods in enumerate(inputs.bott_periods):
        monitor.emit("method", **progress, method=f"bott_{index}")
        started = time.perf_counter()
        result["bott"].append(
            _method_call(
                lambda periods=periods: bott_index(
                    hamiltonian, inputs.basis_coordinates, periods, classification,
                    fermi_energy=0.0, tolerance=NUMERICAL_TOLERANCE,
                    quantization_tolerance=1.0e-6,
                )
            )
        )
        timings["bott"] = timings.get("bott", 0.0) + time.perf_counter() - started
    assert inputs.position_areas is not None
    for index, mask in enumerate(inputs.bulk_masks):
        monitor.emit("method", **progress, method=f"local_chern_{MASK_NAMES[index]}")
        started = time.perf_counter()
        result["local_chern"].append(
            _method_call(
                lambda mask=mask: local_chern_marker(
                    hamiltonian, inputs.basis_coordinates, inputs.position_areas, mask,
                    classification, fermi_energy=0.0, tolerance=NUMERICAL_TOLERANCE,
                    quantization_tolerance=5.0e-3,
                )
            )
        )
        timings["local_chern"] = (
            timings.get("local_chern", 0.0) + time.perf_counter() - started
        )
    for index, kappa in enumerate(inputs.localizer_kappas):
        monitor.emit("method", **progress, method=f"localizer_{index}")
        started = time.perf_counter()
        result["localizer"].append(
            _method_call(
                lambda kappa=kappa: spectral_localizer(
                    hamiltonian, inputs.basis_coordinates, inputs.localizer_probe,
                    classification, energy=0.0, kappa=kappa,
                    tolerance=NUMERICAL_TOLERANCE,
                )
            )
        )
        timings["localizer"] = (
            timings.get("localizer", 0.0) + time.perf_counter() - started
        )
    return result


def _method_call(function: Any) -> dict[str, Any]:
    try:
        return {"status": "available", "result": function(), "error": None}
    except Exception as error:  # noqa: BLE001 - method failures are first-class evidence
        return {
            "status": "rejected",
            "result": None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }


def _classify(
    methods: dict[str, list[dict[str, Any]]], minimum_fermi_distance: float
) -> tuple[str, bool]:
    if minimum_fermi_distance <= NUMERICAL_TOLERANCE:
        return "undefined_fermi_projector", False
    bott = [_method_value(item, "bott_index") for item in methods["bott"]]
    local = [_method_value(item, "chern_number") for item in methods["local_chern"]]
    localizer = [_method_value(item, "local_chern_number") for item in methods["localizer"]]
    primary = (*bott, *localizer, local[4])
    if all(value == 1 for value in primary):
        status = "primary_topological"
    elif all(value == 0 for value in primary):
        status = "primary_trivial"
    else:
        status = "mixed_or_unresolved"
    regional_sensitivity = bool(
        status in ("primary_topological", "primary_trivial")
        and any(value != primary[-1] for value in local[:4])
    )
    return status, regional_sensitivity


def _classification_reason(
    methods: dict[str, list[dict[str, Any]]],
    minimum_fermi_distance: float,
    status: str,
) -> str:
    if minimum_fermi_distance <= NUMERICAL_TOLERANCE:
        return "fermi_level_eigenstate"
    values = (
        [_method_value(item, "bott_index") for item in methods["bott"]]
        + [_method_value(item, "local_chern_number") for item in methods["localizer"]]
        + [_method_value(methods["local_chern"][4], "chern_number")]
    )
    if any(value is None for value in values):
        return "primary_method_unavailable_or_unresolved"
    if len(set(values)) == 1 and values[0] not in (0, 1):
        return f"unexpected_common_integer_{values[0]}"
    if status == "mixed_or_unresolved":
        return "primary_method_disagreement"
    return "unanimous_primary_panel"


def _has_unexpected_failure(record: dict[str, Any]) -> bool:
    if record["classification"] == "operational_failure":
        return True
    methods = record.get("methods")
    if not isinstance(methods, dict):
        return True
    if record["model_role"] == "undefined":
        return bool(
            any(item["status"] != "rejected" for item in methods["bott"])
            or any(item["status"] != "rejected" for item in methods["local_chern"])
            or any(item["status"] != "available" for item in methods["localizer"])
        )
    return any(item["status"] != "available" for group in methods.values() for item in group)


def _method_value(item: dict[str, Any], attribute: str) -> Any:
    value = item.get("result")
    return None if item.get("status") != "available" or value is None else getattr(value, attribute)


def _minimum_localizer_gap(
    methods: dict[str, list[dict[str, Any]]]
) -> float | None:
    values = [
        float(item["result"].localizer_gap)
        for item in methods["localizer"]
        if item["status"] == "available"
    ]
    return min(values) if len(values) == 3 else None


def _old_size_screening(methods: dict[str, list[dict[str, Any]]], boundary: Any) -> bool | None:
    if boundary is None or any(
        item["status"] != "available" for group in methods.values() for item in group
    ):
        return None
    values = (
        tuple(_method_value(item, "bott_index") for item in methods["bott"])
        + tuple(_method_value(item, "chern_number") for item in methods["local_chern"][:2])
        + tuple(_method_value(item, "local_chern_number") for item in methods["localizer"])
    )
    protection = _minimum_localizer_gap(methods)
    return bool(
        all(value is not None and abs(value) == 1 for value in values)
        and len(set(values)) == 1
        and protection is not None
        and protection >= LOCALIZER_PROTECTION_THRESHOLD
        and boundary.passes
    )


def measurement_validation_summary(
    outcomes: tuple[dict[str, Any], ...], *, preflight: bool
) -> dict[str, Any]:
    """Apply the frozen control, stability and verdict rules without extension."""
    controls: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    variants = []
    for record in outcomes:
        if record["block"].startswith("control_"):
            controls[(record["n"], record["model_role"])].append(record)
        else:
            variants.append(record)
    control_rows = []
    for key, records in sorted(controls.items()):
        valid = len(records) == 2 and all(_control_is_valid(item) for item in records)
        agrees = valid and _controls_agree(records[0], records[1])
        control_rows.append(
            {
                "n": key[0],
                "model_role": key[1],
                "valid": valid,
                "start_end_agree": agrees,
                "start": None if not records else _control_summary(records[0]),
                "end": None if len(records) != 2 else _control_summary(records[1]),
            }
        )
    controls_passed = bool(control_rows and all(row["valid"] and row["start_end_agree"] for row in control_rows))
    variant_rows = [_variant_row(record, controls) for record in variants]
    trivial_variants_passed = all(
        row["classification"] == "primary_trivial"
        for row in variant_rows if row["model_role"] == "trivial"
    )
    topological_variants_passed = all(
        row["classification"] == "primary_topological" and row["positive_screening"]
        for row in variant_rows if row["model_role"] == "topological"
    )
    stability_rows = []
    if not preflight:
        keyed = {
            (row["n"], row["model_role"], row["offset"], row["arm"]): row
            for row in variant_rows
        }
        for offset in (-2, 0, 2):
            for arm in ("boundary", "interior"):
                first = keyed[(16, "topological", offset, arm)]["central_chern"]
                second = keyed[(20, "topological", offset, arm)]["central_chern"]
                difference = None if first is None or second is None else abs(second - first)
                stability_rows.append(
                    {
                        "offset": offset,
                        "arm": arm,
                        "absolute_central_chern_difference": difference,
                        "passes": difference is not None and difference <= SIZE_STABILITY_TOLERANCE,
                    }
                )
    stability_passed = bool(stability_rows and all(row["passes"] for row in stability_rows))
    if preflight:
        verdict = "preflight_passed" if controls_passed else "control_validation_failed"
    elif not controls_passed or not trivial_variants_passed:
        verdict = "control_validation_failed"
    elif not topological_variants_passed or not stability_passed:
        verdict = "controls_passed_intervention_panel_not_stable"
    else:
        verdict = "bounded_measurement_contract_accepted"
    return {
        "protocol_id": MEASUREMENT_VALIDATION_PROTOCOL_ID,
        "preflight_only": preflight,
        "slots_processed": len(outcomes),
        "unexpected_failures": sum(_has_unexpected_failure(record) for record in outcomes),
        "expected_method_rejections": sum(
            item["status"] == "rejected"
            for record in outcomes if record["model_role"] == "undefined" and record["methods"]
            for method in ("bott", "local_chern") for item in record["methods"][method]
        ),
        "controls": control_rows,
        "controls_passed": controls_passed,
        "variants": variant_rows,
        "paired_differences": _paired_differences(variant_rows),
        "variant_groups": _variant_groups(variant_rows),
        "trivial_variants_passed": trivial_variants_passed,
        "topological_variants_passed": topological_variants_passed,
        "size_stability": stability_rows,
        "size_stability_passed": stability_passed,
        "verdict": verdict,
    }


def _control_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "classification": record["classification"],
        "classification_reason": record.get("classification_reason"),
        "positive_screening": record["positive_screening"],
        "minimum_fermi_distance": record.get("minimum_fermi_distance"),
        "methods": _compact_methods(record.get("methods")),
        "boundary_signature": _compact_boundary(record.get("boundary_signature")),
        "error": record.get("error"),
    }


def _variant_row(
    record: dict[str, Any], controls: dict[tuple[int, str], list[dict[str, Any]]]
) -> dict[str, Any]:
    central = _central_chern(record)
    reference_records = controls.get((record["n"], record["model_role"]), [])
    reference = _central_chern(reference_records[0]) if reference_records else None
    marker_change = _marker_change(record, reference_records[0] if reference_records else None)
    expected = 1.0 if record["model_role"] == "topological" else 0.0
    return {
        "n": record["n"],
        "model_role": record["model_role"],
        "offset": record["offset"],
        "arm": record["arm"],
        "classification": record["classification"],
        "classification_reason": record.get("classification_reason"),
        "positive_screening": record["positive_screening"],
        "regional_sensitivity": record.get("regional_sensitivity"),
        "central_chern": central,
        "central_absolute_error": None if central is None else abs(central - expected),
        "central_change_from_reference": (
            None if central is None or reference is None else central - reference
        ),
        "patch_marker_change": marker_change,
        "depth_marker_change": _depth_marker_change(
            record, reference_records[0] if reference_records else None
        ),
        "graph_mask_change": _graph_mask_change(
            record, reference_records[0] if reference_records else None
        ),
        "localizer_protection_proxy": (
            None if record.get("methods") is None
            else _minimum_localizer_gap(record["methods"])
        ),
        "boundary_signature": _compact_boundary(record.get("boundary_signature")),
        "methods": _compact_methods(record.get("methods")),
    }


def _paired_differences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed = {
        (row["n"], row["model_role"], row["offset"], row["arm"]): row
        for row in rows
    }
    result = []
    for n in FULL_SIZES:
        for role in ("topological", "trivial"):
            for offset in (-2, 0, 2):
                boundary = keyed.get((n, role, offset, "boundary"))
                interior = keyed.get((n, role, offset, "interior"))
                if boundary is None or interior is None:
                    continue
                result.append(
                    {
                        "n": n,
                        "model_role": role,
                        "offset": offset,
                        "central_absolute_error_interior_minus_boundary": _difference(
                            interior["central_absolute_error"],
                            boundary["central_absolute_error"],
                        ),
                        "patch_mean_absolute_interior_minus_boundary": _difference(
                            _nested_value(interior, "patch_marker_change", "mean_absolute"),
                            _nested_value(boundary, "patch_marker_change", "mean_absolute"),
                        ),
                        "patch_signed_mean_interior_minus_boundary": _difference(
                            _nested_value(interior, "patch_marker_change", "signed_mean"),
                            _nested_value(boundary, "patch_marker_change", "signed_mean"),
                        ),
                    }
                )
    return result


def _variant_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for n in FULL_SIZES:
        for role in ("topological", "trivial"):
            for arm in ("boundary", "interior"):
                selected = [
                    row for row in rows
                    if row["n"] == n and row["model_role"] == role and row["arm"] == arm
                ]
                if not selected:
                    continue
                result.append(
                    {
                        "n": n,
                        "model_role": role,
                        "arm": arm,
                        "count": len(selected),
                        "central_absolute_error": _numeric_summary(
                            row["central_absolute_error"] for row in selected
                        ),
                        "localizer_protection_proxy": _numeric_summary(
                            row["localizer_protection_proxy"] for row in selected
                        ),
                        "patch_mean_absolute_marker_change": _numeric_summary(
                            _nested_value(row, "patch_marker_change", "mean_absolute")
                            for row in selected
                        ),
                    }
                )
    return result


def _compact_methods(methods: Any) -> Any:
    if not isinstance(methods, dict):
        return None
    return {
        "bott": [
            {"status": item["status"], "index": _method_value(item, "bott_index"),
             "estimate": _method_value(item, "bott_estimate"), "error": item["error"]}
            for item in methods["bott"]
        ],
        "local_chern": [
            {"name": name, "status": item["status"],
             "chern_number": _method_value(item, "chern_number"),
             "estimate": _method_value(item, "bulk_chern_estimate"), "error": item["error"]}
            for name, item in zip(MASK_NAMES, methods["local_chern"], strict=True)
        ],
        "localizer": [
            {"status": item["status"], "index": _method_value(item, "local_chern_number"),
             "gap": _method_value(item, "localizer_gap"), "error": item["error"]}
            for item in methods["localizer"]
        ],
    }


def _central_chern(record: dict[str, Any]) -> float | None:
    methods = record.get("methods")
    if not isinstance(methods, dict) or len(methods.get("local_chern", ())) != 5:
        return None
    value = methods["local_chern"][4].get("result")
    return None if value is None else float(value.bulk_chern_estimate)


def _marker_change(record: dict[str, Any], reference: dict[str, Any] | None) -> dict[str, Any] | None:
    if reference is None or not record["measurement_sites"]:
        return None
    methods = record.get("methods")
    reference_methods = reference.get("methods")
    if not isinstance(methods, dict) or not isinstance(reference_methods, dict):
        return None
    current = methods["local_chern"][4].get("result")
    baseline = reference_methods["local_chern"][4].get("result")
    if current is None or baseline is None:
        return None
    sites = np.asarray(record["measurement_sites"], dtype=np.intp)
    delta = current.local_marker[sites] - baseline.local_marker[sites]
    return {
        "site_ids": record["measurement_sites"],
        "values": tuple(float(value) for value in delta),
        "signed_mean": float(np.mean(delta)),
        "mean_absolute": float(np.mean(np.abs(delta))),
    }


def _depth_marker_change(
    record: dict[str, Any], reference: dict[str, Any] | None
) -> list[dict[str, Any]] | None:
    delta = _marker_delta(record, reference)
    if delta is None:
        return None
    n = record["n"]
    result = []
    for depth in range(n // 2):
        sites = np.asarray(
            [
                site for site in range(n * n)
                if min(
                    site // n, site % n, n - 1 - site // n, n - 1 - site % n
                ) == depth
            ],
            dtype=np.intp,
        )
        values = delta[sites]
        result.append(
            {
                "depth": depth,
                "site_count": int(sites.size),
                "signed_mean": float(np.mean(values)),
                "mean_absolute": float(np.mean(np.abs(values))),
            }
        )
    return result


def _graph_mask_change(
    record: dict[str, Any], reference: dict[str, Any] | None
) -> list[dict[str, Any]] | None:
    if reference is None:
        return None
    methods = record.get("methods")
    reference_methods = reference.get("methods")
    if not isinstance(methods, dict) or not isinstance(reference_methods, dict):
        return None
    central_delta = _marker_delta(record, reference)
    if central_delta is None:
        return None
    result = []
    for graph_index, fixed_index in ((0, 2), (1, 3)):
        graph = methods["local_chern"][graph_index].get("result")
        fixed = methods["local_chern"][fixed_index].get("result")
        reference_graph = reference_methods["local_chern"][graph_index].get("result")
        if graph is None or fixed is None or reference_graph is None:
            return None
        graph_sites = set(np.flatnonzero(graph.bulk_mask).tolist())
        fixed_sites = set(np.flatnonzero(fixed.bulk_mask).tolist())
        changed_sites = tuple(sorted(graph_sites.symmetric_difference(fixed_sites)))
        result.append(
            {
                "graph_mask": MASK_NAMES[graph_index],
                "fixed_mask": MASK_NAMES[fixed_index],
                "added_site_ids": tuple(sorted(graph_sites - fixed_sites)),
                "removed_site_ids": tuple(sorted(fixed_sites - graph_sites)),
                "changed_site_contributions": [
                    {
                        "site_id": site,
                        "current_marker": float(graph.local_marker[site]),
                        "reference_marker": float(reference_graph.local_marker[site]),
                        "marker_delta": float(central_delta[site]),
                    }
                    for site in changed_sites
                ],
            }
        )
    return result


def _marker_delta(
    record: dict[str, Any], reference: dict[str, Any] | None
) -> np.ndarray | None:
    if reference is None:
        return None
    methods = record.get("methods")
    reference_methods = reference.get("methods")
    if not isinstance(methods, dict) or not isinstance(reference_methods, dict):
        return None
    current = methods["local_chern"][4].get("result")
    baseline = reference_methods["local_chern"][4].get("result")
    if current is None or baseline is None:
        return None
    return np.asarray(current.local_marker) - np.asarray(baseline.local_marker)


def _compact_boundary(boundary: Any) -> dict[str, Any] | None:
    if boundary is None:
        return None
    return {
        "particle_hole_pairs_by_state_index": tuple(boundary.particle_hole_pairs),
        "pairing_cost": boundary.pairing_cost,
        "maximum_pair_residual": boundary.maximum_pair_residual,
        "boundary_localized_count": boundary.boundary_localized_count,
        "minimum_boundary_weight_first_four": boundary.minimum_boundary_weight_first_four,
        "reasons": boundary.reasons,
        "states": [
            {
                "state_index": state.state_index,
                "energy": state.energy,
                "boundary_weight": state.boundary_weight,
            }
            for state in boundary.states
        ],
    }


def _difference(first: Any, second: Any) -> float | None:
    if not isinstance(first, (int, float)) or not isinstance(second, (int, float)):
        return None
    return float(first - second)


def _nested_value(record: dict[str, Any], outer: str, inner: str) -> Any:
    value = record.get(outer)
    return None if not isinstance(value, dict) else value.get(inner)


def _numeric_summary(values: Any) -> dict[str, Any]:
    available = [float(value) for value in values if isinstance(value, (int, float))]
    return {
        "count": len(available),
        "minimum": None if not available else min(available),
        "median": None if not available else float(np.median(available)),
        "maximum": None if not available else max(available),
        "values": available,
    }


def _control_is_valid(record: dict[str, Any]) -> bool:
    role = record["model_role"]
    methods = record.get("methods")
    if record["classification"] == "operational_failure" or not isinstance(methods, dict):
        return False
    if role == "topological":
        return bool(
            record["classification"] == "primary_topological"
            and record["positive_screening"]
            and all(_method_value(item, "chern_number") == 1 for item in methods["local_chern"])
        )
    if role == "trivial":
        return bool(
            record["classification"] == "primary_trivial"
            and not record["positive_screening"]
            and all(_method_value(item, "chern_number") == 0 for item in methods["local_chern"])
        )
    def expected_gap(kappa: float) -> float:
        return float(kappa / np.sqrt(2.0))
    return bool(
        record["classification"] == "undefined_fermi_projector"
        and all(item["status"] == "rejected" for item in methods["bott"])
        and all(item["status"] == "rejected" for item in methods["local_chern"])
        and all(
            item["status"] == "available"
            and _method_value(item, "local_chern_number") == 0
            and np.isclose(
                _method_value(item, "localizer_gap"), expected_gap(kappa),
                rtol=0.0, atol=NUMERICAL_TOLERANCE,
            )
            for item, kappa in zip(methods["localizer"], (0.1, 0.2, 0.3), strict=True)
        )
    )


def _controls_agree(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if first["classification"] != second["classification"]:
        return False
    first_run, second_run = first.get("run"), second.get("run")
    if (
        first_run is None or second_run is None
        or first_run.simulation_result is None or second_run.simulation_result is None
    ):
        return False
    if not np.allclose(
        first_run.simulation_result.eigenvalues,
        second_run.simulation_result.eigenvalues,
        rtol=0.0,
        atol=NUMERICAL_TOLERANCE,
    ):
        return False
    first_signature = _method_signature(first.get("methods"))
    second_signature = _method_signature(second.get("methods"))
    if first_signature is None or second_signature is None:
        return False
    return first_signature[0] == second_signature[0] and np.allclose(
        first_signature[1], second_signature[1], rtol=0.0, atol=NUMERICAL_TOLERANCE
    )


def _method_signature(methods: Any) -> Any:
    compact = _compact_methods(methods)
    if compact is None:
        return None
    discrete = (
        tuple((item["status"], item["index"], _error_signature(item["error"]))
              for item in compact["bott"]),
        tuple((item["status"], item["chern_number"], _error_signature(item["error"]))
              for item in compact["local_chern"]),
        tuple((item["status"], item["index"], _error_signature(item["error"]))
              for item in compact["localizer"]),
    )
    floats = np.asarray(
        [item["estimate"] for item in compact["bott"] if item["estimate"] is not None]
        + [item["estimate"] for item in compact["local_chern"] if item["estimate"] is not None]
        + [item["gap"] for item in compact["localizer"] if item["gap"] is not None]
    )
    return discrete, floats


def _error_signature(error: Any) -> Any:
    if not isinstance(error, dict):
        return None
    return error.get("type"), error.get("message")


def _audit_plan(cells: tuple[dict[str, Any], ...], *, preflight: bool) -> None:
    expected = 6 if preflight else 36
    if len(cells) != expected or len({_identity_key(cell) for cell in cells}) != expected:
        raise ResearchAbort("Messvalidierungsplan verletzt Budget oder Zellidentität")
    if any(cell["evaluation_seed"] is not None for cell in cells):
        raise ResearchAbort("Messvalidierungsplan darf keine Zufallsseeds enthalten")


def _load_bound_sealed(directory: Path, expected: dict[str, Any]) -> dict[str, Any]:
    sealed = load_record(directory / "sealed.json")
    execution = directory / sealed["execution"]
    if encode_record(load_record(execution / "input.json")) != encode_record(expected):
        raise ResearchAbort("Versiegelter Zellinput stimmt nicht mit dem Messplan überein")
    record = load_sealed(directory)
    if not isinstance(record, dict) or any(
        record.get(key) != value for key, value in _identity(expected).items()
    ):
        raise ResearchAbort("Versiegeltes Zellergebnis hat eine falsche Identität")
    return record


def _identity(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        key: cell[key]
        for key in ("n", "block", "model_role", "offset", "arm", "geometry_id", "evaluation_seed")
    }


def _identity_key(cell: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(_identity(cell).values())


def _model_adapter(parameters: ChiralPWaveParameters) -> GeometryModelAdapter:
    def model_factory(geometry: Any) -> BaseModel:
        return ChiralPWaveModel(geometry, parameters)

    def nambu_basis_resolver(model: BaseModel) -> NambuBasis:
        if not isinstance(model, ChiralPWaveModel):
            raise TypeError("measurement validation requires ChiralPWaveModel")
        return model.nambu_basis

    return GeometryModelAdapter(
        model_factory=model_factory,
        requirements=ModelGeometryRequirements(
            require_connected=True,
            require_edges=True,
            require_boundary_sites=True,
            required_spatial_axes=(0, 1),
        ),
        nambu_basis_resolver=nambu_basis_resolver,
    )


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase-10-Messvertragsvalidierung",
        "",
        f"Protokoll: `{MEASUREMENT_VALIDATION_PROTOCOL_ID}`",
        "",
        f"Bearbeitete Slots: **{summary['slots_processed']}**",
        "",
        f"Kontrollen bestanden: **{summary['controls_passed']}**",
        "",
        f"Urteil: **{summary['verdict']}**",
        "",
        "| Größe | Rolle | gültig | Anfang/Ende gleich |",
        "| ---: | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['n']}×{row['n']} | {row['model_role']} | {row['valid']} | "
        f"{row['start_end_agree']} |"
        for row in summary["controls"]
    )
    lines += [
        "",
        "Alle Methodenwerte, erwarteten Ablehnungen, Varianten und Größenvergleiche ",
        "stehen vollständig in `summary.json`; die versiegelten Zellergebnisse enthalten ",
        "Hamiltonian, Eigenpaare, Markerfelder und Provenienz.",
        "",
        "Das Urteil gilt nur für das vorab begrenzte Quadratpanel.",
        "",
    ]
    return "\n".join(lines)


def _output_path(output: Path) -> Path:
    root = output.resolve()
    results = Path(__file__).resolve().parents[3] / "results"
    if not root.is_relative_to(results.resolve()) or not root.name.startswith(
        "phase_10_measurement_validation"
    ):
        raise ValueError("Ein neues results/phase_10_measurement_validation...-Verzeichnis verwenden")
    return root
