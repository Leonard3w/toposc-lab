"""Read-only result compatibility, candidate detail and follow-up configuration.

Historical databases are never migrated. Every detail is decoded using the
existing checked ResearchStore; a follow-up contains a lossless geometry copy.
"""

from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from toposc_lab.research.embedded import EmbeddedDomain
from toposc_lab.research.embedded_cohort import geometry_id
from toposc_lab.research.space import geometry_from_payload
from toposc_lab.research.storage import ResearchStore, atomic_text, dumps


def scalar_observables(
    result: dict[str, Any], geometry: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Expose persisted measurements without recomputing physics or replacing missing data."""
    metrics = result.get("metrics", {})
    row = {
        "quality": metrics.get("quality"),
        "success": metrics.get("success"),
        "minimum_abs_energy": metrics.get("minimum_abs_energy"),
        "boundary_weight": metrics.get("boundary_weight"),
        "interior_localizer_gap": None,
        "edge_weight": None,
        "center_weight": None,
        "chern_bulk_mean": None,
        "polarization": None,
        "self_conjugacy": None,
    }
    if result.get("status") != "completed":
        return {key: None for key in row}
    if result.get("primary_valid") is False:
        row["quality"] = row["success"] = None
    spatial = result.get("spatial", [])
    domain = EmbeddedDomain(**result["domain"]) if result.get("domain") else None
    if domain:
        inner = [
            p for p in spatial if domain.distances(np.array([p["point"]]))[0] >= domain.bulk_inset
        ]
        if inner and all(p.get("valid") and p.get("gap") is not None for p in inner):
            row["interior_localizer_gap"] = min(p["gap"] for p in inner)
    window = result.get("boundary_window", {}).get("window")
    if window is not None:
        strips = window.get("strip_weights", {})
        row["edge_weight"] = strips.get("1", strips.get("1.0"))
        if geometry and domain:
            coordinates = np.asarray(geometry["coordinates"])
            center = np.all(np.abs(coordinates - np.asarray(domain.center)) <= 1, axis=1)
            weights = np.asarray(window.get("site_probability", []))
            if weights.shape == (len(coordinates),):
                row["center_weight"] = float(weights[center].sum())
    marker = result.get("chern_marker", {})
    if marker.get("status") == "available":
        row["chern_bulk_mean"] = marker.get("bulk_mean")
    states = result.get("majorana", {}).get("states", [])
    for target, source in (
        ("polarization", "polarization_norm"),
        ("self_conjugacy", "self_conjugacy"),
    ):
        values = [s[source] for s in states if s.get(source) is not None]
        if values:
            row[target] = float(np.mean(values))
    return row


def _stage_summary(result: dict[str, Any], geometry: dict[str, Any] | None) -> dict[str, Any]:
    return {
        **{
            k: result[k]
            for k in (
                "stage",
                "status",
                "error",
                "kind",
                "seed",
                "onsite_width",
                "primary_valid",
                "primary_invalid_reason",
                "indices",
                "localizer_gaps",
                "topology",
                "adapter_id",
            )
            if k in result
        },
        "metrics": {**result.get("metrics", {}), **scalar_observables(result, geometry)},
    }


def read_candidate(directory: str | Path, identity: str) -> dict[str, Any]:
    store = ResearchStore(directory)
    candidate = store.get("candidate", identity)
    if candidate is None:
        cohort = store.get("validation_cohort", default={})
        candidate = next((r for r in cohort.get("candidates", []) if r["id"] == identity), None)
    if candidate is None:
        raise ValueError(f"Candidate not found: {identity}")
    candidate = copy.deepcopy(candidate)
    candidate["exact_results"] = {}
    prefix = identity + ":"
    with store.connect(readonly=True) as db:
        # prefix matching in Python avoids interpreting IDs as LIKE/GLOB expressions.
        for row in db.execute(
            "SELECT * FROM objects WHERE kind='exact_result' AND substr(id,1,?)=? ORDER BY rowid",
            (len(prefix), prefix),
        ):
            candidate["exact_results"][row["id"][len(prefix) :]] = store.decode(row)
    candidate["provenance"] = {
        "manifest": store.get("manifest"),
        "config": store.get("config"),
        "source_directory": str(Path(directory).resolve()),
    }
    candidate["geometry_sha256"] = geometry_id(geometry_from_payload(candidate["geometry"]))
    return candidate


def historical_snapshot(directory: str | Path, *, lightweight: bool = False) -> dict[str, Any]:
    """Translate fixed-cohort records to the existing dashboard contract, read-only."""
    store = ResearchStore(directory)
    with store.connect(readonly=True) as db:
        db.execute("BEGIN")
        objects: dict[str, Any] = {}
        for row in db.execute(
            "SELECT * FROM objects WHERE id='current' AND kind IN "
            "('config','state','manifest','validation_cohort','validation_settings','report')"
        ):
            objects[row["kind"]] = store.decode(row)
        cohort, settings = objects["validation_cohort"], objects["validation_settings"]
        candidates = {
            c["id"]: {**c, "exact_results": {}, "baseline": c.get("family") == "regular"}
            for c in cohort["candidates"]
        }
        for row in db.execute("SELECT * FROM objects WHERE kind='exact_result' ORDER BY rowid"):
            identity, _, stage = row["id"].partition(":")
            if identity in candidates:
                result = store.decode(row)
                candidates[identity]["exact_results"][stage] = _stage_summary(
                    result, candidates[identity]["geometry"]
                )
        attempts = [dict(row) for row in db.execute("SELECT * FROM attempts ORDER BY number")]
        checkpoints = [
            {"id": row["id"], **store.decode(row)}
            for row in db.execute("SELECT * FROM objects WHERE kind='checkpoint' ORDER BY rowid")
        ]
        events = [
            {**dict(row), "payload": json.loads(row["payload"])}
            for row in db.execute("SELECT * FROM events ORDER BY number")
        ]
    expected = 1 + len(settings.get("widths", [])) * len(settings.get("seeds", []))
    for candidate in candidates.values():
        results = list(candidate["exact_results"].values())
        complete = len(results) == expected and all(r.get("status") == "completed" for r in results)
        disorder = [r for r in results if r.get("stage", {}).get("kind") == "disorder"]
        valid = [
            r
            for r in disorder
            if r.get("primary_valid") and r["metrics"].get("quality") is not None
        ]
        clean = next((r for r in results if r.get("stage", {}).get("kind") == "clean"), None)
        candidate["clean"] = clean
        candidate["raw_metrics"] = dict(clean.get("metrics", {})) if clean else {}
        candidate["raw_metrics"]["robustness_quality_mean"] = (
            float(np.mean([r["metrics"]["quality"] for r in valid])) if valid else None
        )
        candidate["score"] = (
            candidate["raw_metrics"]["robustness_quality_mean"]
            if complete and len(valid) == len(disorder)
            else None
        )
        candidate.update(
            origin="exact" if complete else "partial_exact",
            observed=complete,
            validation_state="EXACT_EVALUATED" if complete else "PARTIAL_EXACT",
            details_available=True,
            source_format="fixed_cohort",
        )
        if lightweight:
            for key in ("exact_results", "clean", "geometry"):
                candidate.pop(key, None)
    state = objects["state"]
    state.update(
        exact_evaluations=len(attempts),
        completed_evaluations=sum(a["status"] == "complete" for a in attempts),
        error_count=sum(a["status"] == "failed" for a in attempts),
        generated=len(candidates),
        historical_read_only=True,
    )
    result = {
        "config": objects["config"],
        "state": state,
        "manifest": objects["manifest"],
        "candidates": list(candidates.values()),
        "archive": [],
        "history": [],
        "events": events,
        "checkpoints": checkpoints,
        "attempts": attempts,
        "model_versions": [],
        "surrogate": {},
        "baselines": [c for c in candidates.values() if c.get("baseline")],
        "allocation": {},
        "selection_counts": {},
        "report": objects.get("report", {}).get("markdown", ""),
        "warnings": [
            "Historical study: inspection is read-only. Use a follow-up to evaluate new seeds."
        ],
    }
    from toposc_lab.research.reporting import diagnostics

    result["diagnostics"] = diagnostics(result)
    state.update(
        {
            key: result["diagnostics"][key]
            for key in (
                "best_score",
                "best_validated_candidate",
                "archive_coverage",
                "duplicate_rate",
                "near_duplicate_rate",
                "invalid_rate",
                "surrogate_error",
                "diversity",
            )
        }
    )
    return result


def follow_up(
    directory: str | Path,
    candidate_ids: list[str],
    *,
    widths: list[float],
    seeds: list[int],
    output_directory: str,
    _require_fresh: bool = True,
) -> dict[str, Any]:
    """Build a new normal config; deliberately performs no writes or exact calls."""
    from toposc_lab.research.config import ExperimentConfig
    from toposc_lab.research.studio_config import preset
    from toposc_lab.research.studio_space import checked_geometry

    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Select distinct candidates")
    store = ResearchStore(directory)
    source_config, manifest = store.get("config"), store.get("manifest")
    old_physics = source_config.get("physics", {})
    old_settings = store.get("validation_settings", default={})
    old_seeds = set(old_physics.get("disorder_seeds", old_settings.get("seeds", [])))
    if _require_fresh and old_seeds.intersection(seeds):
        raise ValueError("Follow-up seeds must be fresh; overlap with source disorder seeds")
    if Path(output_directory).resolve() == Path(directory).resolve():
        raise ValueError("Follow-up output must be a new directory")
    records = []
    cohort = store.get("validation_cohort", default={})
    for identity in candidate_ids:
        record = store.get("candidate", identity)
        if record is None:
            record = next((r for r in cohort.get("candidates", []) if r["id"] == identity), None)
        if record is None:
            raise ValueError(f"Candidate not found: {identity}")
        geometry = checked_geometry(record)
        records.append(
            {
                "id": identity,
                "source_candidate_id": identity,
                "family": record.get("family", "selected"),
                "geometry": copy.deepcopy(record["geometry"]),
                "geometry_sha256": geometry_id(geometry),
                "source_geometry_sha256": geometry_id(geometry),
                "source_run": str(Path(directory).resolve()),
                "source_experiment_id": manifest["experiment_id"],
                "selection_note": "Selected from prior results; selection bias must be disclosed.",
            }
        )
    config = preset("candidate_validation")
    config.update(
        name=source_config["name"] + " — follow-up",
        algorithm="fixed_candidates",
        geometry_space="fixed_candidates",
        output_directory=output_directory,
        space={"candidates": records},
        baselines=[],
        search={},
        pool_size=len(records),
        batch_size=len(records),
        cycles=1,
        candidate_budget=len(records),
    )
    # Preserve every available physical setting. Only adapter version/ensemble change.
    from dataclasses import fields

    from toposc_lab.research.studio_physics import STUDIO_ADAPTER_ID, StudioPhysicsProtocol

    allowed = {f.name for f in fields(StudioPhysicsProtocol)} - {"objective", "adapter_id"}
    physics = {k: v for k, v in old_physics.items() if k in allowed}
    domain = old_physics.get("domain", cohort.get("domain"))
    if domain is None:
        g = geometry_from_payload(records[0]["geometry"])
        coords = g.coordinates
        domain = {
            "bounds": [
                float(coords[:, 0].min()),
                float(coords[:, 0].max()),
                float(coords[:, 1].min()),
                float(coords[:, 1].max()),
            ],
            "boundary_shell": 0.875,
            "bulk_inset": min(2.0, float(np.ptp(coords, axis=0).min()) / 3),
            "cell_padding": 0.5,
        }
    physics.pop("probe_rule", None)
    if widths:
        validators = list(physics.get("validators", StudioPhysicsProtocol().validators))
        if "robustness" not in validators:
            validators.append("robustness")
        physics["validators"] = validators
    physics.update(
        adapter_id=STUDIO_ADAPTER_ID,
        domain=domain,
        disorder_widths=list(widths),
        disorder_seeds=list(seeds),
        clean_seed=max([*seeds, *old_seeds, 0]) + 1,
        clean_reference=True,
        confirmation=False,
    )
    config["physics"] = physics
    stages = 1 + len(widths) * len(seeds)
    config["exact_budget"] = len(records) * stages + config.get("retry_limit", 1)
    config["studio"]["variables"] = {key: "locked" for key in config["studio"]["variables"]}
    resolved = ExperimentConfig.from_dict(config)
    resolved.validate_plugins()
    return resolved.to_dict()


def clone_config(directory: str | Path, *, output_directory: str) -> dict[str, Any]:
    """Clone configuration for editing, adapting historical fixed schedules losslessly."""
    from toposc_lab.research.config import ExperimentConfig

    store = ResearchStore(directory)
    config = store.get("config")
    cohort = store.get("validation_cohort")
    if cohort is not None:
        settings = store.get("validation_settings")
        cloned = follow_up(
            directory,
            [c["id"] for c in cohort["candidates"]],
            widths=settings["widths"],
            seeds=settings["seeds"],
            output_directory=output_directory,
            _require_fresh=False,
        )
        cloned["physics"]["clean_seed"] = settings["clean_seed"]
        cloned["exact_budget"] = max(cloned["exact_budget"], config["exact_budget"])
        cloned["wall_seconds"] = config["wall_seconds"]
        cloned["studio"]["description"] = (
            "Clone of a historical study using the same disorder seeds; not independent confirmation."
        )
        for candidate in cloned["space"]["candidates"]:
            candidate["selection_note"] = (
                "Complete frozen source cohort; cloned with original seeds."
            )
        config = cloned
    config["name"] = store.get("config")["name"] + " (copy)"
    config["output_directory"] = output_directory
    result = ExperimentConfig.from_dict(config)
    result.validate_plugins()
    return result.to_dict()


def export_studio(directory: Path, snapshot: dict[str, Any]) -> None:
    """Derived exports only. All mandatory evidence stays in the existing store."""
    store = ResearchStore(directory)
    output = snapshot["config"]["studio"]["output"]
    rows = []
    candidates = {c["id"]: c for c in store.all("candidate") if c.get("geometry")}
    raw_path = directory / "reports" / "realizations.jsonl"
    temporary = raw_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream, store.connect(readonly=True) as db:
        for saved in db.execute("SELECT * FROM objects WHERE kind='exact_result' ORDER BY rowid"):
            identity = saved["id"].partition(":")[0]
            result = store.decode(saved)
            stream.write(dumps({"candidate": identity, "result": result}) + "\n")
            row = {
                "candidate": identity,
                "family": candidates.get(identity, {}).get("family"),
                **result.get("stage", {}),
                "status": result.get("status"),
                "error": result.get("error"),
                "primary_valid": result.get("primary_valid"),
                "primary_invalid_reason": result.get("primary_invalid_reason"),
                **scalar_observables(result, candidates.get(identity, {}).get("geometry")),
            }
            rows.append(row)
    temporary.replace(raw_path)
    text = io.StringIO(newline="")
    columns = list(dict.fromkeys(k for r in rows for k in r)) or ["candidate", "status"]
    writer = csv.DictWriter(text, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(directory / "reports" / "realization_diagnostics.csv", text.getvalue())
    atomic_text(
        directory / "reports" / "resolved_config.json", json.dumps(snapshot["config"], indent=2)
    )
    if output.get("save_plots") and rows:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(8, 5), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        for family in dict.fromkeys(r["family"] for r in rows):
            selected = [r for r in rows if r["family"] == family and r.get("quality") is not None]
            ax.scatter(
                [r.get("width", 0) for r in selected],
                [r["quality"] for r in selected],
                label=family,
                alpha=0.65,
            )
        ax.set(
            xlabel="Disorder width W",
            ylabel="Q (exact)",
            title="Individual finite-system realizations",
        )
        ax.legend()
        fig.savefig(directory / "plots" / "quality.png", dpi=150)


def studio_report(snapshot: dict[str, Any]) -> str:
    """Report actual configured scope without legacy fixed-square claims."""
    config, state = snapshot["config"], snapshot["state"]
    candidates = snapshot["candidates"]
    completed = [c for c in candidates if c.get("observed")]
    failed = [c for c in completed if c.get("validation_state") == "FAILED"]

    def block(value: Any) -> str:
        return (
            "```json\n"
            + json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n```\n"
        )

    return "\n".join(
        [
            f"# {config['name']} — TOPOSC Research Studio",
            (
                f"Experiment {state['experiment_id']}; status {state['status']}; "
                f"{len(completed)} candidates evaluated, {len(failed)} failed; "
                f"{state.get('exact_evaluations', 0)} charged exact attempts."
            ),
            "## Question and scope",
            config["question"],
            config["studio"]["description"],
            (
                "Exact finite-system diagnostics only. Q, spatial Localizer gaps, boundary density, "
                "Chern markers and Majorana diagnostics are separate measurements. No phase, bulk gap, "
                "mobility gap or separated Majorana modes are established by this report."
            ),
            "## Resolved configuration",
            block(config),
            "## Execution",
            block(
                {
                    k: state.get(k)
                    for k in (
                        "created",
                        "finished",
                        "elapsed_seconds",
                        "exact_seconds",
                        "planned_realizations",
                        "exact_evaluations",
                        "finish_reason",
                    )
                }
            ),
            "## Candidate evidence",
            block(
                [
                    {
                        k: c.get(k)
                        for k in (
                            "id",
                            "family",
                            "geometry_sha256",
                            "score",
                            "validation_state",
                            "raw_metrics",
                            "source_run",
                            "source_candidate_id",
                        )
                    }
                    for c in completed
                ]
            ),
            "## Interpretation",
            (
                "Sampling is descriptive. An optimization objective does not replace its raw observables. "
                "Disorder uncertainty in candidate records refers to the specified finite seed ensemble; "
                "selection and shared seeds limit broader inference. Missing/failed measurements are "
                "retained explicitly. Follow-up selections retain source IDs and geometry hashes."
            ),
            "## Reproduction",
            block(snapshot["manifest"]),
            (
                "The existing SQLite journal is authoritative. reports/realizations.jsonl, "
                "realization_diagnostics.csv and resolved_config.json are derived exports. "
                "Resume requires matching archived source and runtime. No further experiment was started."
            ),
        ]
    )
