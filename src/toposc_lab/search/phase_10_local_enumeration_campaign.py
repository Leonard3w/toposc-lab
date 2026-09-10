"""User-started runner for the frozen local geometry enumeration protocol."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
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
from toposc_lab.search.phase_10_campaign import (
    _artifact_inventory,
    _campaign_lock,
    _verify_completion,
)
from toposc_lab.search.phase_10_local_enumeration import (
    ALL_KAPPAS,
    CATALOG_COMMIT,
    CATALOG_IDS_SHA256,
    EXTRA_KAPPAS,
    FULL_BUDGET,
    LOCAL_ENUMERATION_PROTOCOL_COMMIT,
    LOCAL_ENUMERATION_PROTOCOL_ID,
    LOCAL_ENUMERATION_PROTOCOL_PATH,
    PREFLIGHT_BUDGET,
    PRIMARY_KAPPAS,
    build_local_enumeration_plan,
    paired_candidate_ids,
)
from toposc_lab.search.phase_10_measurement_validation import (
    FULL_SIZES,
    build_measurement_topology_inputs,
    validate_measurement_geometry,
)
from toposc_lab.search.phase_10_measurement_validation_campaign import (
    NUMERICAL_TOLERANCE,
    _compact_boundary,
    _compact_methods,
    _depth_marker_change,
    _graph_mask_change,
    _method_call,
    _method_signature,
    _method_value,
    _result_field,
)
from toposc_lab.search.phase_10_measurement_validation_campaign import (
    _calculate_cell as _calculate_primary_cell,
)
from toposc_lab.search.phase_10_research import THREAD_VARIABLES
from toposc_lab.topology import SymmetryClassification, spectral_localizer

Mode = Literal["preflight", "full", "resume"]
PROXY_GAIN = 0.01
DECISION_EPSILON = 1.0e-8
PAIR_CHERN_TOLERANCE = 5.0e-3


def local_enumeration_environment() -> dict[str, Any]:
    """Require the frozen protocol, committed implementation, and fixed runtime."""
    root = Path(__file__).resolve().parents[3]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True,
            text=True, encoding="utf-8",
        ).stdout.strip()

    if sys.version_info[:2] != (3, 14) or not sys.dont_write_bytecode:
        raise ValueError("Python 3.14 und PYTHONDONTWRITEBYTECODE=1 sind erforderlich")
    if any(os.environ.get(name) != "1" for name in THREAD_VARIABLES):
        raise ValueError(
            "Vor Pythonstart OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS "
            "und BLIS_NUM_THREADS auf 1 setzen"
        )
    code = git("rev-parse", "HEAD")
    if code == LOCAL_ENUMERATION_PROTOCOL_COMMIT:
        raise ValueError("Die Implementierung muss nach dem Protokoll separat committed sein")
    git("merge-base", "--is-ancestor", LOCAL_ENUMERATION_PROTOCOL_COMMIT, code)
    git("merge-base", "--is-ancestor", CATALOG_COMMIT, code)
    allowed = {
        "src/toposc_lab/observables/__pycache__/__init__.cpython-314.pyc",
        "src/toposc_lab/observables/__pycache__/spectrum.cpython-314.pyc",
        "geometry_demo.npz",
    }
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout
    unexpected = [
        line for line in status.splitlines() if line[3:].replace("\\", "/") not in allowed
    ]
    if unexpected:
        raise ValueError("Worktree vor Enumeration committen: " + "; ".join(unexpected))
    frozen = subprocess.run(
        ["git", "-C", str(root), "show",
         f"{LOCAL_ENUMERATION_PROTOCOL_COMMIT}:{LOCAL_ENUMERATION_PROTOCOL_PATH}"],
        check=True, capture_output=True,
    ).stdout.replace(b"\r\n", b"\n")
    current = (root / LOCAL_ENUMERATION_PROTOCOL_PATH).read_bytes().replace(b"\r\n", b"\n")
    if current != frozen:
        raise ValueError("Enumerationsprotokoll weicht vom eingefrorenen Commit ab")
    configuration = StringIO()
    with redirect_stdout(configuration):
        np.show_config()
    return {
        "code_commit": code,
        "protocol_commit": LOCAL_ENUMERATION_PROTOCOL_COMMIT,
        "protocol_sha256": hashlib.sha256(frozen).hexdigest(),
        "catalog_commit": CATALOG_COMMIT,
        "catalog_ids_sha256": CATALOG_IDS_SHA256,
        "python": platform.python_version(),
        "packages": {
            name: version(name) for name in ("numpy", "scipy", "matplotlib", "pydantic")
        },
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "numpy_backend": configuration.getvalue(),
        "threads": {name: os.environ[name] for name in THREAD_VARIABLES},
    }


def run_local_enumeration_campaign(output: Path, *, mode: Mode) -> Path:
    """Run exactly one user-requested frozen enumeration stage."""
    environment = local_enumeration_environment()
    root = _output_path(output)
    if mode == "preflight":
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        raise ValueError("Zuerst den Enumerationsvorlauf in einem neuen Verzeichnis starten")
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
                    raise ResearchAbort("Passender bestandener Enumerationsvorlauf erforderlich")
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
    total = PREFLIGHT_BUDGET if preflight else FULL_BUDGET
    protocol_file = Path(__file__).resolve().parents[3] / LOCAL_ENUMERATION_PROTOCOL_PATH
    metadata = {
        "protocol_id": LOCAL_ENUMERATION_PROTOCOL_ID,
        "protocol_commit": LOCAL_ENUMERATION_PROTOCOL_COMMIT,
        "protocol_text": protocol_file.read_text(encoding="utf-8"),
        "environment": environment,
        "mode": mode,
        "total_evaluation_attempts": total,
        "primary_kappas": PRIMARY_KAPPAS,
        "extra_kappas": EXTRA_KAPPAS,
        "proxy_gain": PROXY_GAIN,
        "decision_epsilon": DECISION_EPSILON,
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
    monitor.emit(
        "constructing", stage=mode, block="construction", completed=0, total=total,
        screening_passes=0, primary_hits=0, qualified_hits=0,
    )
    constructed = time.perf_counter()
    prepared = tuple(_prepare_cell(cell) for cell in build_local_enumeration_plan(preflight=preflight))
    _audit_plan(prepared, preflight=preflight)
    plan_path = stage / "input_plan.json"
    if plan_path.exists():
        if encode_record(load_record(plan_path)) != encode_record(prepared):
            raise ResearchAbort("Gespeicherter Enumerationsplan weicht von der Konstruktion ab")
    else:
        save_record(plan_path, prepared)
    monitor.emit(
        "constructed", stage=mode, block="construction", completed=0, total=total,
        category="construction", operation_seconds=time.perf_counter() - constructed,
    )

    estimate = None
    if mode == "full":
        estimate = float(load_record(stage.parent / "preflight" / "complete.json")[
            "seconds_per_evaluation"
        ])
    outcomes: list[dict[str, Any]] = []
    evaluations_started = time.perf_counter()
    for index, cell in enumerate(prepared):
        directory = stage / "evaluations" / f"cell_{index:04d}"
        record = (
            _load_bound_sealed(directory, cell)
            if (directory / "sealed.json").exists()
            else _evaluate_cell(
                directory, cell, environment["code_commit"], monitor, mode, index, total,
                estimated_seconds=estimate,
            )
        )
        outcomes.append(record)
        if cell["block"] == "control_start" and not _control_is_valid(record):
            raise ResearchAbort(
                f"Ungültige Anfangskontrolle: n={cell['n']} role={cell['model_role']}"
            )
        counters = _live_counts(outcomes)
        monitor.emit("progress", stage=mode, completed=index + 1, total=total, **counters)

    summary = local_enumeration_summary(tuple(outcomes), preflight=preflight)
    summary["environment"] = environment
    summary["protocol_commit"] = LOCAL_ENUMERATION_PROTOCOL_COMMIT
    summary["total_evaluation_attempts"] = total
    analysis = stage / "analysis.json"
    if analysis.exists():
        if encode_record(load_record(analysis)) != encode_record(summary):
            raise ResearchAbort("Wiederholte Enumerationsanalyse stimmt nicht überein")
    else:
        save_record(analysis, summary)
    publish_derived(stage / "summary.json", json_bytes(summary))
    publish_derived(stage / "report.md", _render_report(summary).encode("utf-8"))
    preflight_passed = bool(preflight and summary["verdict"] == "preflight_passed")
    save_record(
        stage / "complete.json",
        {
            "preflight_passed": preflight_passed,
            "artifacts": _artifact_inventory(stage),
            "seconds_per_evaluation": (time.perf_counter() - evaluations_started) / total,
            "total_stage_seconds": time.perf_counter() - stage_started,
        },
    )
    monitor.emit("complete", stage=mode, completed=total, total=total,
                 last_sealed=f"{mode}/complete", **_live_counts(outcomes))
    return stage / "report.md"


def _prepare_cell(cell: dict[str, Any]) -> dict[str, Any]:
    geometry = cell["genome"].to_geometry()
    if exact_geometry_id(geometry) != cell["geometry_id"]:
        raise ResearchAbort("Konstruiertes Geometry-ID-Audit ist fehlgeschlagen")
    if not cell["validation"]["is_valid"]:
        raise ResearchAbort("Ungültige Geometrie im Enumerationsplan")
    validation = validate_measurement_geometry(geometry, cell["n"])
    if not validation["is_valid"]:
        raise ResearchAbort("Geometrie verletzt den vollständigen Messvalidator")
    _, evidence = build_measurement_topology_inputs(geometry, cell["n"])
    expected = np.arange(cell["n"] * cell["n"], dtype=np.int64)
    if not np.array_equal(evidence["unique_coordinate_order"], expected):
        raise ResearchAbort("Marker-Ortsordnung weicht von der Site-Reihenfolge ab")
    if not np.allclose(evidence["position_areas_in_coordinate_order"], 1.0,
                       rtol=0.0, atol=NUMERICAL_TOLERANCE):
        raise ResearchAbort("Enumeration verlangt einheitliche Voronoi-Flächen")
    return {
        **cell,
        "measurement_validation": validation,
        "topology_input_evidence": evidence,
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
    ledger = AttemptLedger(directory, on_storage=lambda seconds: monitor.emit(
        "artifact_saved", category="storage", operation_seconds=seconds
    ))
    ledger.record("input.json", cell)
    progress: dict[str, Any] = {
        "stage": mode, "block": cell["block"], "model_role": cell["model_role"],
        "size": cell["n"], "candidate_id": cell["candidate_id"], "slot": index + 1,
        "completed": index, "total": total,
    }
    if estimated_seconds is not None:
        progress["estimated_seconds_per_evaluation"] = estimated_seconds
    monitor.emit("evaluating", **progress)
    started = time.perf_counter()
    timings: dict[str, float] = {}
    try:
        record = _calculate_cell(cell, code, monitor, progress, timings)
    except Exception as error:  # noqa: BLE001
        record = {
            **_identity(cell), "run": None, "hamiltonian": None, "methods": None,
            "classification": "operational_failure", "positive_screening": False,
            "numerical_valid": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    elapsed = time.perf_counter() - started
    timings["total"] = elapsed
    for key in (
        "candidate_id", "origins", "known_measurement_variant", "rotation_signature",
        "dihedral_signature", "removed_edges", "added_edges", "affected_sites",
        "intervention_depth",
    ):
        record[key] = cell.get(key)
    record["timings"] = timings
    ledger.record("outcome.json", record)
    ledger.seal(record)
    monitor.emit(
        "evaluated" if record["classification"] != "operational_failure" else "evaluation_failed",
        **progress, completed=index + 1, classification=record["classification"],
        screening_pass=record["positive_screening"],
        last_sealed=f"{mode}/evaluations/cell_{index:04d}", category="physics",
        operation_seconds=elapsed, stage_timings=timings,
    )
    return record


def _calculate_cell(
    cell: dict[str, Any],
    code: str,
    monitor: ResearchMonitor,
    progress: dict[str, Any],
    timings: dict[str, float],
) -> dict[str, Any]:
    with _count_linear_algebra_calls() as calls:
        record = _calculate_cell_core(cell, code, monitor, progress, timings)
    record["linear_algebra_calls"] = calls
    return record


def _calculate_cell_core(
    cell: dict[str, Any],
    code: str,
    monitor: ResearchMonitor,
    progress: dict[str, Any],
    timings: dict[str, float],
) -> dict[str, Any]:
    record = _calculate_primary_cell(cell, code, monitor, progress, timings)
    if record.get("methods") is None or record.get("run") is None:
        return {**record, "numerical_valid": False}
    geometry = cell["genome"].to_geometry()
    inputs, _ = build_measurement_topology_inputs(geometry, cell["n"])
    hamiltonian = np.asarray(record["hamiltonian"], dtype=complex)
    classification = SymmetryClassification.from_signature(
        time_reversal_square=None, particle_hole_square=1, chiral_symmetry=False
    )
    methods = record["methods"]
    for kappa in EXTRA_KAPPAS:
        monitor.emit("method", **progress, method=f"localizer_kappa_{kappa:g}")
        started = time.perf_counter()
        methods["localizer"].append(_method_call(lambda kappa=kappa: spectral_localizer(
            hamiltonian, inputs.basis_coordinates, inputs.localizer_probe, classification,
            energy=0.0, kappa=kappa, tolerance=NUMERICAL_TOLERANCE,
        )))
        timings["localizer_extra"] = (
            timings.get("localizer_extra", 0.0) + time.perf_counter() - started
        )
    simulation = record["run"].simulation_result
    if simulation is None:
        return {**record, "numerical_valid": False}
    values = np.asarray(simulation.eigenvalues)
    vectors = np.asarray(simulation.eigenvectors)
    dimension = hamiltonian.shape[0]
    residual = float(np.linalg.norm(hamiltonian @ vectors - vectors * values, ord="fro")
                     / max(1.0, float(np.linalg.norm(hamiltonian, ord="fro"))))
    orthonormality = float(np.linalg.norm(
        vectors.conj().T @ vectors - np.eye(dimension), ord="fro"
    ) / np.sqrt(dimension))
    hermiticity = float(np.max(np.abs(hamiltonian - hamiltonian.conj().T)))
    finite = _all_finite((hamiltonian, values, vectors, methods))
    numerical_valid = bool(
        finite and hermiticity <= NUMERICAL_TOLERANCE
        and residual <= NUMERICAL_TOLERANCE
        and orthonormality <= NUMERICAL_TOLERANCE
    )
    if not numerical_valid:
        record["classification"] = "numerical_failure"
        record["positive_screening"] = False
    return {
        **record,
        "normalized_eigensystem_residual": residual,
        "orthonormality_residual": orthonormality,
        "hamiltonian_hermiticity_residual": hermiticity,
        "all_numeric_values_finite": finite,
        "numerical_valid": numerical_valid,
        "method_call_count": 17,
    }


@contextmanager
def _count_linear_algebra_calls() -> Iterator[dict[str, int]]:
    """Count the actual dense decompositions used by this single-threaded slot."""
    names = ("eigh", "eigvalsh", "eigvals", "svd")
    originals: dict[str, Any] = {name: getattr(np.linalg, name) for name in names}
    counts = {name: 0 for name in names}
    try:
        for name, function in originals.items():
            def counted(*args: Any, _name: str = name, _function: Any = function,
                        **kwargs: Any) -> Any:
                counts[_name] += 1
                return _function(*args, **kwargs)

            setattr(np.linalg, name, counted)
        yield counts
    finally:
        for name, function in originals.items():
            setattr(np.linalg, name, function)


def local_enumeration_summary(
    outcomes: tuple[dict[str, Any], ...], *, preflight: bool
) -> dict[str, Any]:
    """Apply the frozen comparison and verdict rules without extending the budget."""
    controls: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    candidates = []
    for record in outcomes:
        if record["block"].startswith("control_"):
            controls[(record["n"], record["model_role"])].append(record)
        else:
            candidates.append(record)
    control_rows = []
    for (n, role), records in sorted(controls.items()):
        valid = len(records) == 2 and all(_control_is_valid(record) for record in records)
        agrees = valid and _controls_agree(records[0], records[1])
        control_rows.append({
            "n": n, "model_role": role, "valid": valid, "start_end_agree": agrees,
            "start": None if not records else _compact_record(records[0]),
            "end": None if len(records) != 2 else _compact_record(records[1]),
        })
    controls_passed = bool(control_rows and all(
        row["valid"] and row["start_end_agree"] for row in control_rows
    ))
    references = {
        n: records[0] for (n, role), records in controls.items()
        if role == "topological" and records
    }
    rows = [_candidate_row(record, references.get(record["n"])) for record in candidates]
    expected_total = PREFLIGHT_BUDGET if preflight else FULL_BUDGET
    expected_candidates = 2 if preflight else 256
    inventory_complete = len(outcomes) == expected_total and len(rows) == expected_candidates
    unresolved = not inventory_complete or any(not row["complete"] for row in rows)
    rankings = {
        str(n): _ranking([row for row in rows if row["n"] == n]) for n in FULL_SIZES
    }
    pairs = [] if preflight else _pair_rows(rows)
    primary_hits = sum(bool(row["primary_proxy_hit"]) for row in rows)
    qualified_hits = sum(bool(row["sensitivity_qualified_hit"]) for row in rows)
    paired_hits = sum(bool(row["paired_proxy_candidate"]) for row in pairs)
    borderline = any(row["threshold_status"] == "threshold_borderline" for row in rows)
    if preflight:
        interventions_passed = len(rows) == 2 and all(
            row["complete"] and row["positive_screening"]
            and row["extra_localizers_all_topological"] for row in rows
        )
        verdict = "preflight_passed" if controls_passed and interventions_passed else (
            "control_validation_failed" if not controls_passed else "preflight_intervention_failed"
        )
    elif not controls_passed:
        verdict = "control_validation_failed"
    elif unresolved:
        verdict = "incomplete_or_unresolved"
    elif paired_hits:
        verdict = "paired_proxy_candidates_found"
    elif primary_hits:
        verdict = "single_size_or_setting_dependent_hits"
    elif borderline:
        verdict = "threshold_borderline_only"
    else:
        verdict = "no_relevant_proxy_gain_in_catalog"
    return {
        "preflight": preflight,
        "sealed_slots": len(outcomes),
        "inventory_complete": inventory_complete,
        "controls": control_rows,
        "controls_passed": controls_passed,
        "candidates": rows,
        "rankings": rankings,
        "pairs": pairs,
        "screening_passes": sum(bool(row["positive_screening"]) for row in rows),
        "primary_hits": primary_hits,
        "qualified_hits": qualified_hits,
        "paired_hits": paired_hits,
        "verdict": verdict,
    }


def _candidate_row(record: dict[str, Any], reference: dict[str, Any] | None) -> dict[str, Any]:
    protection = _localizer_min(record, range(3))
    reference_protection = _localizer_min(reference, range(3))
    delta = _difference(protection, reference_protection)
    margin = None if delta is None or reference_protection is None else (
        delta - PROXY_GAIN * reference_protection
    )
    if not record.get("positive_screening") or margin is None or not record.get("numerical_valid"):
        threshold = "not_eligible"
    elif margin > DECISION_EPSILON:
        threshold = "primary_proxy_hit"
    elif abs(margin) <= DECISION_EPSILON:
        threshold = "threshold_borderline"
    else:
        threshold = "below_threshold"
    primary_hit = threshold == "primary_proxy_hit"
    extra_indices = _localizer_indices(record, range(3, 9))
    all_extra = len(extra_indices) == 6 and all(value == 1 for value in extra_indices)
    all_gaps_noninferior = _all_gap_differences_noninferior(record, reference)
    extra = _localizer_min(record, range(3, 9))
    reference_extra = _localizer_min(reference, range(3, 9))
    extra_margin = None if extra is None or reference_extra is None else (
        extra - reference_extra - PROXY_GAIN * reference_extra
    )
    qualified = bool(
        primary_hit and all_extra and all_gaps_noninferior
        and extra_margin is not None and extra_margin > DECISION_EPSILON
    )
    complete = _candidate_complete(record)
    origins = [_origin_marker_change(record, reference, origin) for origin in record.get("origins", ())]
    n = record["n"]
    affected = tuple(record.get("affected_sites", ()))
    central_overlap = tuple(
        site for site in affected
        if min(site // n, site % n, n - 1 - site // n, n - 1 - site % n) >= n / 4
    )
    return {
        **_identity(record),
        "removed_edges": record.get("removed_edges"),
        "added_edges": record.get("added_edges"),
        "affected_sites": affected,
        "intervention_depth": record.get("intervention_depth"),
        "affected_central_overlap_sites": central_overlap,
        "known_measurement_variant": record.get("known_measurement_variant"),
        "rotation_signature": record.get("rotation_signature"),
        "dihedral_signature": record.get("dihedral_signature"),
        "classification": record["classification"],
        "positive_screening": record.get("positive_screening", False),
        "numerical_valid": record.get("numerical_valid", False),
        "complete": complete,
        "localizer_protection": protection,
        "reference_localizer_protection": reference_protection,
        "delta_L": delta,
        "relative_gain": None if delta is None or reference_protection in (None, 0.0)
        else delta / reference_protection,
        "margin": margin,
        "threshold_status": threshold,
        "primary_proxy_hit": primary_hit,
        "extra_localizer_protection": extra,
        "reference_extra_localizer_protection": reference_extra,
        "extra_margin": extra_margin,
        "extra_localizers_all_topological": all_extra,
        "all_nine_gaps_noninferior": all_gaps_noninferior,
        "sensitivity_qualified_hit": qualified,
        "central_chern": _central_chern(record),
        "origin_marker_change": origins,
        "depth_marker_change": _depth_marker_change(record, reference),
        "graph_mask_change": _graph_mask_change(record, reference),
        "methods": _compact_methods(record.get("methods")),
        "boundary": _compact_boundary(record.get("boundary_signature")),
        "error": record.get("error"),
    }


def _pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {row["candidate_id"]: row for row in rows}
    result = []
    for first_id, second_id in paired_candidate_ids():
        first, second = by_id.get(first_id), by_id.get(second_id)
        difference = None if first is None or second is None else _difference(
            second["central_chern"], first["central_chern"]
        )
        paired = bool(
            first is not None and second is not None
            and first["sensitivity_qualified_hit"] and second["sensitivity_qualified_hit"]
            and difference is not None and abs(difference) <= PAIR_CHERN_TOLERANCE
        )
        result.append({
            "candidate_id_16": first_id, "candidate_id_20": second_id,
            "central_chern_difference_20_minus_16": difference,
            "threshold_status_16": None if first is None else first["threshold_status"],
            "threshold_status_20": None if second is None else second["threshold_status"],
            "qualified_16": False if first is None else first["sensitivity_qualified_hit"],
            "qualified_20": False if second is None else second["sensitivity_qualified_hit"],
            "paired_proxy_candidate": paired,
        })
    return result


def _ranking(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["complete"] and row["positive_screening"]
                and row["delta_L"] is not None]
    eligible.sort(key=lambda row: (-row["delta_L"], row["candidate_id"]))
    maximum = None if not eligible else eligible[0]["delta_L"]
    ties = [] if maximum is None else sorted(
        row["candidate_id"] for row in eligible
        if maximum - row["delta_L"] <= DECISION_EPSILON
    )
    return {
        "eligible_count": len(eligible), "maximum_delta_L": maximum,
        "top_tied_candidate_ids": ties,
        "ordered_candidate_ids": [row["candidate_id"] for row in eligible],
    }


def _control_is_valid(record: dict[str, Any]) -> bool:
    methods = record.get("methods")
    if not record.get("numerical_valid") or not isinstance(methods, dict):
        return False
    role = record["model_role"]
    expected = 1 if role == "topological" else 0
    localizers = methods.get("localizer", ())
    if len(localizers) != 9:
        return False
    if role == "undefined":
        return bool(
            record["classification"] == "undefined_fermi_projector"
            and all(_is_projector_rejection(item) for item in methods["bott"])
            and all(_is_projector_rejection(item) for item in methods["local_chern"])
            and all(
                item["status"] == "available"
                and _method_value(item, "local_chern_number") == 0
                and np.isclose(_method_value(item, "localizer_gap"), kappa / np.sqrt(2.0),
                               rtol=0.0, atol=NUMERICAL_TOLERANCE)
                for item, kappa in zip(localizers, ALL_KAPPAS, strict=True)
            )
        )
    return bool(
        record["classification"] == ("primary_topological" if expected else "primary_trivial")
        and bool(record["positive_screening"]) == bool(expected)
        and all(_method_value(item, "chern_number") == expected
                for item in methods["local_chern"])
        and all(_method_value(item, "local_chern_number") == expected for item in localizers)
    )


def _controls_agree(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if first["classification"] != second["classification"]:
        return False
    first_run, second_run = first.get("run"), second.get("run")
    if first_run is None or second_run is None:
        return False
    first_sim, second_sim = first_run.simulation_result, second_run.simulation_result
    if first_sim is None or second_sim is None or not np.allclose(
        first_sim.eigenvalues, second_sim.eigenvalues, rtol=0.0, atol=NUMERICAL_TOLERANCE
    ):
        return False
    signatures = (_method_signature(first.get("methods")), _method_signature(second.get("methods")))
    if any(signature is None for signature in signatures):
        return False
    if signatures[0][0] != signatures[1][0] or not np.allclose(
        signatures[0][1], signatures[1][1], rtol=0.0, atol=NUMERICAL_TOLERANCE
    ):
        return False
    first_boundary = _continuous_boundary(first)
    second_boundary = _continuous_boundary(second)
    if first_boundary is None or second_boundary is None:
        return first_boundary is second_boundary and first["model_role"] == "undefined"
    return first_boundary[1] == second_boundary[1] and np.allclose(
        first_boundary[0], second_boundary[0], rtol=0.0, atol=NUMERICAL_TOLERANCE
    )


def _continuous_boundary(record: dict[str, Any]) -> Any:
    boundary = _compact_boundary(record.get("boundary_signature"))
    if boundary is None:
        return None
    values = [boundary["maximum_pair_residual"]]
    values.extend(
        value
        for state in boundary["states"]
        for value in (state["energy"], state["boundary_weight"])
    )
    return np.asarray(values, dtype=float), boundary["boundary_localized_count"]


def _candidate_complete(record: dict[str, Any]) -> bool:
    methods = record.get("methods")
    if record.get("classification") in ("operational_failure", "numerical_failure"):
        return False
    if not record.get("numerical_valid") or not isinstance(methods, dict):
        return False
    if [len(methods.get(name, ())) for name in ("bott", "local_chern", "localizer")] != [3, 5, 9]:
        return False
    fermi_undefined = record.get("minimum_fermi_distance", 1.0) <= NUMERICAL_TOLERANCE
    for name in ("bott", "local_chern"):
        for item in methods[name]:
            if item["status"] != "available" and not (
                fermi_undefined and _is_projector_rejection(item)
            ):
                return False
    return all(item["status"] == "available" for item in methods["localizer"])


def _is_projector_rejection(item: dict[str, Any]) -> bool:
    error = item.get("error")
    return bool(
        item.get("status") == "rejected" and isinstance(error, dict)
        and error.get("type") == "ValueError"
        and "Fermi-level eigenstate" in str(error.get("message"))
    )


def _localizer_indices(record: dict[str, Any] | None, indices: range) -> list[Any]:
    if record is None or not isinstance(record.get("methods"), dict):
        return []
    values = record["methods"]["localizer"]
    return [_method_value(values[index], "local_chern_number") for index in indices
            if index < len(values) and values[index]["status"] == "available"]


def _localizer_min(record: dict[str, Any] | None, indices: range) -> float | None:
    if record is None or not isinstance(record.get("methods"), dict):
        return None
    values = record["methods"]["localizer"]
    gaps = [float(_method_value(values[index], "localizer_gap")) for index in indices
            if index < len(values) and values[index]["status"] == "available"]
    return min(gaps) if len(gaps) == len(indices) else None


def _all_gap_differences_noninferior(
    record: dict[str, Any], reference: dict[str, Any] | None
) -> bool:
    if reference is None:
        return False
    return all(
        candidate is not None and baseline is not None
        and candidate - baseline >= -DECISION_EPSILON
        for candidate, baseline in zip(
            (_localizer_gap(record, index) for index in range(9)),
            (_localizer_gap(reference, index) for index in range(9)), strict=True,
        )
    )


def _localizer_gap(record: dict[str, Any], index: int) -> float | None:
    methods = record.get("methods")
    if not isinstance(methods, dict) or len(methods["localizer"]) <= index:
        return None
    value = _method_value(methods["localizer"][index], "localizer_gap")
    return None if value is None else float(value)


def _central_chern(record: dict[str, Any]) -> float | None:
    methods = record.get("methods")
    if not isinstance(methods, dict) or len(methods["local_chern"]) != 5:
        return None
    value = _method_value(methods["local_chern"][4], "bulk_chern_estimate")
    return None if value is None else float(value)


def _origin_marker_change(
    record: dict[str, Any], reference: dict[str, Any] | None, origin: dict[str, Any]
) -> dict[str, Any]:
    n = record["n"]
    sites_tuple = tuple(origin["measurement_sites"])
    result = {
        "offset": origin["offset"], "arm": origin["arm"],
        "measurement_sites": sites_tuple,
        "minimum_patch_depth": min(
            min(site // n, site % n, n - 1 - site // n, n - 1 - site % n)
            for site in sites_tuple
        ),
        "central_overlap_sites": tuple(
            site for site in sites_tuple
            if min(site // n, site % n, n - 1 - site // n, n - 1 - site % n) >= n / 4
        ),
    }
    if reference is None:
        return {**result, "values": None, "signed_mean": None, "mean_absolute": None}
    methods, baseline_methods = record.get("methods"), reference.get("methods")
    if not isinstance(methods, dict) or not isinstance(baseline_methods, dict):
        return {**result, "values": None, "signed_mean": None, "mean_absolute": None}
    current = methods["local_chern"][4].get("result")
    baseline = baseline_methods["local_chern"][4].get("result")
    if current is None or baseline is None:
        return {**result, "values": None, "signed_mean": None, "mean_absolute": None}
    sites = np.asarray(origin["measurement_sites"], dtype=np.intp)
    delta = np.asarray(_result_field(current, "local_marker"))[sites] - np.asarray(
        _result_field(baseline, "local_marker")
    )[sites]
    return {
        **result, "values": tuple(float(value) for value in delta),
        "signed_mean": float(np.mean(delta)), "mean_absolute": float(np.mean(np.abs(delta))),
    }


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "classification": record["classification"],
        "positive_screening": record.get("positive_screening"),
        "numerical_valid": record.get("numerical_valid"),
        "methods": _compact_methods(record.get("methods")),
        "boundary": _compact_boundary(record.get("boundary_signature")),
        "error": record.get("error"),
    }


def _live_counts(outcomes: list[dict[str, Any]]) -> dict[str, int]:
    candidates = [record for record in outcomes if record["block"] == "candidate"]
    references = {
        record["n"]: record for record in outcomes
        if record["block"] == "control_start" and record["model_role"] == "topological"
    }
    flags = [_candidate_flags(record, references.get(record["n"])) for record in candidates]
    return {
        "screening_passes": sum(bool(record.get("positive_screening")) for record in candidates),
        "primary_hits": sum(primary for primary, _ in flags),
        "qualified_hits": sum(qualified for _, qualified in flags),
    }


def _candidate_flags(
    record: dict[str, Any], reference: dict[str, Any] | None
) -> tuple[bool, bool]:
    primary = _localizer_min(record, range(3))
    baseline = _localizer_min(reference, range(3))
    hit = bool(
        record.get("numerical_valid") and record.get("positive_screening")
        and primary is not None and baseline is not None
        and primary - baseline - PROXY_GAIN * baseline > DECISION_EPSILON
    )
    extra = _localizer_min(record, range(3, 9))
    baseline_extra = _localizer_min(reference, range(3, 9))
    indices = _localizer_indices(record, range(3, 9))
    qualified = bool(
        hit and len(indices) == 6 and all(value == 1 for value in indices)
        and _all_gap_differences_noninferior(record, reference)
        and extra is not None and baseline_extra is not None
        and extra - baseline_extra - PROXY_GAIN * baseline_extra > DECISION_EPSILON
    )
    return hit, qualified


def _all_finite(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value)))
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (float, complex, np.number)):
        return bool(np.isfinite(value))
    return True


def _difference(first: Any, second: Any) -> float | None:
    if not isinstance(first, (int, float)) or not isinstance(second, (int, float)):
        return None
    return float(first - second)


def _audit_plan(cells: tuple[dict[str, Any], ...], *, preflight: bool) -> None:
    expected = PREFLIGHT_BUDGET if preflight else FULL_BUDGET
    if len(cells) != expected or len({_identity_key(cell) for cell in cells}) != expected:
        raise ResearchAbort("Enumerationsplan verletzt Budget oder Zellidentität")
    if any(cell["evaluation_seed"] is not None for cell in cells):
        raise ResearchAbort("Enumerationsplan darf keine Zufallsseeds enthalten")


def _load_bound_sealed(directory: Path, expected: dict[str, Any]) -> dict[str, Any]:
    record = load_sealed(directory)
    sealed = load_record(directory / "sealed.json")
    execution = directory / sealed["execution"]
    if encode_record(load_record(execution / "input.json")) != encode_record(expected):
        raise ResearchAbort("Versiegelter Zellinput stimmt nicht mit dem Plan überein")
    if not isinstance(record, dict) or any(
        record.get(key) != value for key, value in _identity(expected).items()
    ):
        raise ResearchAbort("Versiegeltes Zellergebnis hat eine falsche Identität")
    if encode_record(load_record(execution / "outcome.json")) != encode_record(record):
        raise ResearchAbort("Versiegeltes Ergebnis widerspricht outcome.json")
    return record


def _identity(cell: dict[str, Any]) -> dict[str, Any]:
    return {key: cell.get(key) for key in (
        "n", "block", "model_role", "geometry_id", "candidate_id", "evaluation_seed"
    )}


def _identity_key(cell: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(_identity(cell).values())


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 10: vollständige lokale Geometrie-Enumeration",
        "",
        f"- Urteil: `{summary['verdict']}`",
        f"- versiegelte Slots: {summary['sealed_slots']}",
        f"- Kontrollen bestanden: {summary['controls_passed']}",
        f"- Screening bestanden: {summary['screening_passes']}",
        f"- primäre Proxytreffer: {summary['primary_hits']}",
        f"- einstellungsqualifizierte Treffer: {summary['qualified_hits']}",
        f"- größenübergreifende Proxykandidaten: {summary['paired_hits']}",
        "",
        (
            "Die Enumeration ist eine Kandidatensuche im eingefrorenen lokalen Katalog. "
            "Sie ist keine unabhängige Disorder- oder Größenbestätigung."
        ),
        "",
        "Alle Einzelwerte, Ranglisten, Kontrollen und Paarungen stehen in `summary.json`.",
    ]
    return "\n".join(lines) + "\n"


def _output_path(output: Path) -> Path:
    path = output.resolve()
    if not path.name.startswith("phase_10_local_enumeration") or path.parent.name != "results":
        raise ValueError("Output muss results/phase_10_local_enumeration* heißen")
    return path
