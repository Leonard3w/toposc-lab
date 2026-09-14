"""Bounded Phase-16A experiment: frozen inputs, outcome-blind edits, exact journal."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from toposc_lab.active_learning.benchmark import source_provenance
from toposc_lab.active_learning.cycle import derived_seed
from toposc_lab.data import (
    DatasetRecord,
    ExactPhysicsDataset,
    ReproducibilityMetadata,
    load_dataset,
    record_from_dict,
    record_to_dict,
    save_dataset,
)
from toposc_lab.discovery.engine import plain, scientific_record
from toposc_lab.discovery.storage import atomic_json, encode_run, read_json, writer_lease
from toposc_lab.discovery.validation import majorana_evidence
from toposc_lab.generative.physics import ExactGeometryEvaluator, candidate
from toposc_lab.generative.space import (
    GeometrySearchSpace,
    embedded_edges,
    structural_distance,
    structural_key,
)
from toposc_lab.geometry import Geometry
from toposc_lab.patterns.analysis import analyze
from toposc_lab.patterns.exact import evaluate_deletion
from toposc_lab.patterns.features import extract_pattern_features
from toposc_lab.patterns.interventions import (
    MOTIFS,
    deletion_candidates,
    matched_swaps,
    motif_counts,
    valid_swaps,
)

THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")
ARMS = ("reference", "random", "evolution", "surrogate")


@dataclass(frozen=True)
class PatternCampaignConfig:
    """The v2 scientific protocol is intentionally fixed, not a search tuner."""

    seeds: tuple[int, ...] = tuple(range(16101, 16109))
    proposals_per_seed: int = 8
    raw_cap_per_seed: int = 2000
    minimum_distance: float = 0.15
    exact_attempt_cap: int = 700
    schema_version: int = 2

    def __post_init__(self) -> None:
        if (
            tuple(self.seeds) != tuple(range(16101, 16109))
            or self.proposals_per_seed != 8
            or self.raw_cap_per_seed != 2000
            or self.minimum_distance != 0.15
            or self.exact_attempt_cap != 700
            or self.schema_version != 2
        ):
            raise ValueError("Phase-16A v2 configuration is frozen")
        if any(
            type(v) is not int
            for v in (
                *self.seeds,
                self.proposals_per_seed,
                self.raw_cap_per_seed,
                self.exact_attempt_cap,
                self.schema_version,
            )
        ):
            raise TypeError("integer configuration must use integers, not booleans or floats")
        object.__setattr__(self, "seeds", tuple(self.seeds))


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def metrics(record: DatasetRecord) -> dict[str, Any]:
    return dict(plain(record.observables[0].values))


def indices(record: DatasetRecord) -> list[Any]:
    return list(plain(record.topology[0].parameters["indices"]))


def build(edges: list[list[int]]) -> Geometry:
    if any(len(e) != 2 for e in edges):
        raise ValueError("each edge needs exactly two endpoints")
    return GeometrySearchSpace().build(frozenset((a, b) for a, b in edges))


def freeze(path: Path, payload: Any) -> Any:
    """Immutable semantic artifact: an existing, different plan is an error."""
    normalized = json.loads(json.dumps(payload, allow_nan=False))
    if path.exists():
        if read_json(path) != normalized:
            raise ValueError(f"frozen artifact mismatch: {path}")
    else:
        atomic_json(path, normalized)
    return normalized


class ExactJournal:
    """Each invocation is charged before solving; completed stages are reused."""

    def __init__(self, directory: Path, provenance: ReproducibilityMetadata, cap: int = 700):
        self.directory = directory
        self.provenance = provenance
        self.cap = cap
        self.evaluator = ExactGeometryEvaluator(provenance)

    def calculate(
        self, name: str, geometry: Geometry, seed: int, *, deletion: bool = False, mu: float = 2.0
    ) -> dict[str, Any]:
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name):
            raise ValueError("exact stage name must be a safe filename")
        if deletion and mu != 2.0:
            raise ValueError("single-edge deletion fixes chemical potential at 2")
        attempts = sorted((self.directory / "attempts").glob("*.json"))
        for number, attempt_path in enumerate(attempts, 1):
            attempt = read_json(attempt_path)
            if attempt_path.name != f"{number:04d}.json" or attempt["attempt"] != number:
                raise ValueError("noncontiguous or inconsistent exact attempt journal")
            committed = self.directory / "exact" / f"{attempt['request']['name']}.json"
            if attempt["status"] == "complete" and not committed.exists():
                raise ValueError("committed exact result is missing")
        request = {
            "name": name,
            "seed": seed,
            "deletion": deletion,
            "mu": mu,
            "candidate_id": candidate(geometry, mu).candidate_id,
        }
        path = self.directory / "exact" / f"{name}.json"
        if path.exists():
            result: dict[str, Any] = read_json(path)
            if result["request"] != request:
                raise ValueError("exact cache request mismatch")
            attempt = read_json(self.directory / "attempts" / f"{result['attempt']:04d}.json")
            if attempt["request"] != request:
                raise ValueError("exact attempt mismatch")
            if attempt["status"] == "started":
                atomic_json(
                    self.directory / "attempts" / f"{result['attempt']:04d}.json",
                    {**attempt, "status": "complete", "result_sha256": digest(path)},
                )
            elif attempt.get("result_sha256") != digest(path):
                raise ValueError("exact result checksum mismatch")
            return result
        number = len(attempts) + 1
        if number > self.cap:
            raise RuntimeError("Phase-16A exact attempt cap exhausted")
        attempt_path = self.directory / "attempts" / f"{number:04d}.json"
        entry = {"attempt": number, "request": request, "status": "started"}
        atomic_json(attempt_path, entry)
        start = time.perf_counter()
        try:
            record, run = (
                evaluate_deletion(geometry, seed, self.provenance)
                if deletion
                else self.evaluator.evaluate(candidate(geometry, mu), seed)
            )
            diagnostics = majorana_evidence(record, run)
            if diagnostics["operator_phs_residual"] > 1e-10:
                raise ValueError("operator PHS failed")
            result = {
                "request": request,
                "attempt": number,
                "record": record_to_dict(record),
                "majorana": diagnostics,
                "run": encode_run(run),
                "seconds": time.perf_counter() - start,
            }
            # Match resumed JSON exactly, including tuple-valued diagnostic fields.
            result = json.loads(json.dumps(result, allow_nan=False))
            atomic_json(path, result)
        except Exception as error:
            atomic_json(attempt_path, {**entry, "status": "failed", "error": str(error)})
            raise
        atomic_json(attempt_path, {**entry, "status": "complete", "result_sha256": digest(path)})
        return result

    def confirmed(
        self, name: str, geometry: Geometry, seed: int, *, deletion: bool = False
    ) -> tuple[DatasetRecord, dict[str, Any]]:
        base = self.calculate(name, geometry, seed, deletion=deletion)
        repeat = self.calculate(
            name + "-confirmation", geometry, derived_seed(seed, 0, 169), deletion=deletion
        )
        a, b = record_from_dict(base["record"]), record_from_dict(repeat["record"])
        if scientific_record(a) != scientific_record(b) or base["majorana"] != repeat["majorana"]:
            raise ValueError("independent exact confirmation disagrees")
        return a, base


def legacy_rows(directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[tuple[int, int], ...], dict[str, Any]] = {}
    raw_count = 0
    for arm in ARMS:
        dataset = load_dataset(directory / "inputs" / f"{arm}.json")
        for record in dataset.records:
            geometry = record.geometry.to_geometry()
            if GeometrySearchSpace().reasons(geometry):
                raise ValueError("legacy record outside frozen stratum")
            if record.model != candidate(geometry).model:
                raise ValueError("legacy model outside frozen stratum")
            raw_count += 1
            key = structural_key(geometry)
            origin = "patch" if arm == "reference" else arm
            source = {
                "arm": arm,
                "record_id": record.record_id,
                "exact_id": record.geometry.exact_id,
                "path": f"inputs/{arm}.json",
            }
            if key in groups:
                old = groups[key]
                # D4 equivalence is geometric; verify invariant endpoints before pooling.
                np.testing.assert_allclose(
                    old["spectrum"], record.spectrum.eigenvalues, rtol=0, atol=1e-10
                )
                for endpoint in ("quality", "localizer_gap", "minimum_abs_energy", "eligible"):
                    if abs(old["metrics"][endpoint] - metrics(record)[endpoint]) > 1e-10:
                        raise ValueError("symmetry-equivalent legacy endpoints disagree")
                old["origins"] = sorted(set(old["origins"] + [origin]))
                old["sources"].append(source)
                continue
            groups[key] = {
                "id": f"legacy-{len(groups):02d}",
                "exact_id": record.geometry.exact_id,
                "cohort": "legacy",
                "seed": 15201,
                "generator": origin,
                "origins": [origin],
                "sources": [source],
                "edges": [list(e) for e in embedded_edges(geometry)],
                "metrics": metrics(record),
                "indices": indices(record),
                "spectrum": list(record.spectrum.eigenvalues),
            }
    return list(groups.values()), {"raw_records": raw_count, "distinct_d4_geometries": len(groups)}


def fresh_plan(legacy: list[dict[str, Any]], config: PatternCampaignConfig) -> dict[str, Any]:
    space = GeometrySearchSpace()
    archive = [build(r["edges"]) for r in legacy]
    accepted = []
    audit = []
    for i, seed in enumerate(config.seeds):
        generator = "random" if i % 2 == 0 else "patch"
        rng = np.random.default_rng(seed)
        count = 0
        for attempt in range(config.raw_cap_per_seed):
            geometry = space.sample(rng, patch=generator == "patch")
            reasons = space.reasons(geometry)
            event: dict[str, Any] = {
                "seed": seed,
                "generator": generator,
                "attempt": attempt,
                "edges": [list(e) for e in embedded_edges(geometry)],
                "reasons": list(reasons),
            }
            if reasons:
                event["status"] = "invalid"
            else:
                distance = min(structural_distance(geometry, old) for old in archive)
                event["nearest_archive_distance"] = distance
                if distance <= config.minimum_distance:
                    event["status"] = "duplicate"
                else:
                    event["status"] = "accepted"
                    accepted.append(
                        {
                            "id": f"fresh-{seed}-{count:02d}",
                            "seed": seed,
                            "generator": generator,
                            "edges": event["edges"],
                        }
                    )
                    archive.append(geometry)
                    count += 1
            audit.append(event)
            if count == config.proposals_per_seed:
                break
        if count != config.proposals_per_seed:
            raise RuntimeError("raw proposal cap exhausted without relaxing validity")
    return {"accepted": accepted, "audit": audit, "selection": "before_any_fresh_physics"}


def enrich(row: dict[str, Any]) -> dict[str, Any]:
    feature = extract_pattern_features(build(row["edges"]))
    return {
        **row,
        "features": dict(feature.values),
        "feature_details": {
            "schema_version": feature.schema_version,
            "adjacency_spectrum": feature.adjacency_spectrum,
            "laplacian_spectrum": feature.laplacian_spectrum,
            "normalized_laplacian_spectrum": feature.normalized_laplacian_spectrum,
            "degree_sequence": feature.degree_sequence,
            "triangles": feature.triangles,
            "chordless_squares": feature.chordless_squares,
            "ball_growth": feature.ball_growth,
            "dimension_status": feature.dimension_status,
        },
    }


def intervention_plan(rows: list[dict[str, Any]], config: PatternCampaignConfig) -> dict[str, Any]:
    plans = []
    availability = []
    swaps_cache = {}
    for motif_index, motif in enumerate(MOTIFS):
        for seed in config.seeds:
            parents = sorted(
                (r for r in rows if r["cohort"] == "fresh" and r["seed"] == seed),
                key=lambda r: (-r["metrics"]["quality"], r["id"]),
            )
            selected = 0
            for parent in parents:
                geometry = build(parent["edges"])
                if parent["id"] not in swaps_cache:
                    swaps_cache[parent["id"]] = valid_swaps(geometry)
                match_seed = derived_seed(seed, int(parent["id"][-2:]), 163, motif_index)
                perturbation_seeds = tuple(derived_seed(match_seed, r, 163) for r in range(2))
                pairs = matched_swaps(
                    geometry,
                    motif,
                    swaps_cache[parent["id"]],
                    match_seed,
                    replicate_seeds=perturbation_seeds,
                )
                availability.append(
                    {
                        "motif": motif,
                        "seed": seed,
                        "parent": parent["id"],
                        "valid_swaps": len(swaps_cache[parent["id"]]),
                        "matched_pairs": len(pairs),
                        "match_seed": match_seed,
                        "perturbation_seeds": perturbation_seeds,
                    }
                )
                if not pairs:
                    continue
                selected += 1
                for replicate, (target, control) in enumerate(pairs):
                    name = f"ablation-{motif_index}-{parent['id']}-{replicate}"
                    plan = {
                        "name": name,
                        "motif": motif,
                        "seed": seed,
                        "parent": parent["id"],
                        "replicate": replicate,
                        "match_seed": match_seed,
                        "perturbation_seed": perturbation_seeds[replicate],
                        "before_counts": motif_counts(geometry),
                        "exact_seed": derived_seed(
                            seed, int(parent["id"][-2:]), 164 + motif_index, replicate
                        ),
                    }
                    for arm, swap in (("target", target), ("control", control)):
                        changed = swap.apply(geometry)
                        plan[arm] = {
                            "swap": asdict(swap),
                            "counts": motif_counts(changed),
                            "edges": [list(e) for e in embedded_edges(changed)],
                        }
                    plans.append(plan)
                if selected == 2:
                    break
            print(f"planned {motif} seed={seed}: {selected} parents", flush=True)
    simplifications = []
    for seed in config.seeds:
        parent = min(
            (r for r in rows if r["cohort"] == "fresh" and r["seed"] == seed),
            key=lambda r: (-r["metrics"]["quality"], r["id"]),
        )
        geometry = build(parent["edges"])
        choices = deletion_candidates(geometry)
        if not choices:
            simplifications.append({"parent": parent["id"], "seed": seed, "status": "unavailable"})
            continue
        deletion_seed = derived_seed(seed, 0, 168)
        edge = choices[int(np.random.default_rng(deletion_seed).integers(len(choices)))]
        simplifications.append(
            {
                "parent": parent["id"],
                "seed": seed,
                "status": "planned",
                "removed": list(edge),
                "selection_seed": deletion_seed,
                "edges": [list(e) for e in embedded_edges(geometry) if e != edge],
            }
        )
    return {
        "ablations": plans,
        "availability": availability,
        "simplifications": simplifications,
        "selection": "frozen_before_any_intervention_physics",
    }


def initialize(
    directory: Path, root: Path, legacy: Path, config: PatternCampaignConfig
) -> ReproducibilityMetadata:
    if any(os.environ.get(name) != "1" for name in THREADS):
        raise ValueError("set all four numerical thread limits to 1 before Python starts")
    provenance = source_provenance(root)
    protocol = root / "docs/decisions/phase_16a_protocol.md"
    driver = Path(__file__)
    inputs = {arm: digest(legacy / arm / "dataset.json") for arm in ARMS}
    runtime: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "threads": {name: os.environ[name] for name in THREADS},
    }
    identity: dict[str, Any] = {
        "config": asdict(config),
        "source_sha256": provenance.runtime["source_sha256"],
        "driver_sha256": digest(driver),
        "protocol_sha256": digest(protocol),
        "inputs_sha256": inputs,
        "runtime": runtime,
        "git_commit": provenance.git_commit,
        "git_dirty": provenance.git_dirty,
    }
    freeze(directory / "manifest.json", identity)
    if not (directory / "source.zip").exists():
        temporary_archive = directory / "source.zip.tmp"
        with zipfile.ZipFile(temporary_archive, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted((root / "src").rglob("*.py")):
                archive.write(path, path.relative_to(root).as_posix())
            for path in sorted((root / "tests").glob("test_pattern*.py")):
                archive.write(path, path.relative_to(root).as_posix())
            archive.write(protocol, protocol.relative_to(root).as_posix())
            archive.write(root / "pyproject.toml", "pyproject.toml")
        os.replace(temporary_archive, directory / "source.zip")
    with zipfile.ZipFile(directory / "source.zip") as archive:
        expected_source = {
            p.relative_to(root).as_posix(): p.read_bytes() for p in (root / "src").rglob("*.py")
        }
        actual_source = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.startswith("src/") and name.endswith(".py")
        }
        if (
            actual_source != expected_source
            or archive.read(protocol.relative_to(root).as_posix()) != protocol.read_bytes()
        ):
            raise ValueError("source archive mismatch")
        if archive.testzip() is not None:
            raise ValueError("source archive CRC failure")
    freeze(directory / "source-archive.json", {"sha256": digest(directory / "source.zip")})
    for arm in ARMS:
        target = directory / "inputs" / f"{arm}.json"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(legacy / arm / "dataset.json", target)
        if digest(target) != inputs[arm]:
            raise ValueError("frozen input checksum mismatch")
    return replace(
        provenance,
        runtime={
            **dict(provenance.runtime),
            **runtime,
            "benchmark": "phase16a.pattern-ablation.v2",
            "protocol_sha256": identity["protocol_sha256"],
            "driver_sha256": identity["driver_sha256"],
        },
        solver_settings={
            **dict(provenance.solver_settings),
            "kappas": (0.1, 0.2, 0.3),
            "probe": (2.5, 2.5),
        },
    )


def run_campaign(directory: Path, *, analysis_only: bool = False) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    config = PatternCampaignConfig()
    with writer_lease(directory):
        provenance = initialize(directory, root, root / "results/phase15-gate-final", config)
        if analysis_only:
            rows = read_json(directory / "features.json")
            distances = np.array(read_json(directory / "distances.json"))
            experiments = read_json(directory / "experiments.json")
            simplifications = read_json(directory / "simplifications.json")
        else:
            journal = ExactJournal(directory, provenance, config.exact_attempt_cap)
            legacy, counts = legacy_rows(directory)
            freeze(directory / "legacy-audit.json", counts)
            if (directory / "fresh-plan.json").exists():
                plan = read_json(directory / "fresh-plan.json")
            else:
                plan = freeze(directory / "fresh-plan.json", fresh_plan(legacy, config))
            space = GeometrySearchSpace()
            grid = space.build(
                frozenset(
                    e
                    for e in space.edge_pool
                    if np.linalg.norm(space.coordinates[e[0]] - space.coordinates[e[1]]) == 1
                )
            )
            for mu in (2.0, 8.0):
                result = journal.calculate(f"reference-{int(mu)}", grid, 16900 + int(mu), mu=mu)
                record = record_from_dict(result["record"])
                if bool(metrics(record)["eligible"]) != (mu == 2):
                    raise ValueError("positive/trivial reference gate failed")
            rows = [enrich(r) for r in legacy]
            records = []
            for proposed in plan["accepted"]:
                record, result = journal.confirmed(
                    proposed["id"],
                    build(proposed["edges"]),
                    derived_seed(proposed["seed"], int(proposed["id"][-2:]), 162),
                )
                records.append(record)
                rows.append(
                    enrich(
                        {
                            **proposed,
                            "cohort": "fresh",
                            "exact_id": record.geometry.exact_id,
                            "origins": [proposed["generator"]],
                            "metrics": metrics(record),
                            "indices": indices(record),
                            "exact_path": f"exact/{proposed['id']}.json",
                            "majorana": result["majorana"],
                        }
                    )
                )
                print(f"{proposed['id']}: quality={metrics(record)['quality']:.9f}", flush=True)
            dataset_path = directory / "fresh-dataset.json"
            if dataset_path.exists():
                stored = load_dataset(dataset_path).records
                if [record_to_dict(r) for r in stored] != [record_to_dict(r) for r in records]:
                    raise ValueError("frozen fresh dataset mismatch")
            else:
                save_dataset(dataset_path, ExactPhysicsDataset(tuple(records)))
            # Fresh-timestamp, new-seed independent rerun of one preselected fresh case.
            first = plan["accepted"][0]
            reproduced, _ = journal.confirmed("independent-repeat", build(first["edges"]), 16999)
            if scientific_record(reproduced) != scientific_record(records[0]):
                raise ValueError("fresh-seed independent reference reproduction failed")
            freeze(
                directory / "reproduction.json",
                {"id": first["id"], "scientific_equality": True, "repeat_seed": 16999},
            )
            freeze(directory / "features.json", rows)
            geometries = [build(r["edges"]) for r in rows]
            distances = np.zeros((len(rows), len(rows)))
            for i, a in enumerate(geometries):
                for j in range(i):
                    distances[i, j] = distances[j, i] = structural_distance(a, geometries[j])
            freeze(directory / "distances.json", distances.tolist())
            if (directory / "intervention-plan.json").exists():
                edits = read_json(directory / "intervention-plan.json")
            else:
                edits = freeze(
                    directory / "intervention-plan.json", intervention_plan(rows, config)
                )
            parents = {r["id"]: r for r in rows}
            experiments = []
            for edit in edits["ablations"]:
                parent = parents[edit["parent"]]
                experiment = {
                    k: edit[k]
                    for k in (
                        "motif",
                        "seed",
                        "parent",
                        "replicate",
                        "match_seed",
                        "perturbation_seed",
                    )
                }
                experiment.update(
                    before=parent["metrics"],
                    before_indices=parent["indices"],
                    before_counts=edit["before_counts"],
                )
                for arm in ("target", "control"):
                    name = edit["name"] + "-" + arm
                    record, result = journal.confirmed(
                        name, build(edit[arm]["edges"]), edit["exact_seed"]
                    )
                    experiment[arm] = metrics(record)
                    experiment[arm + "_indices"] = indices(record)
                    experiment[arm + "_id"] = record.geometry.exact_id
                    experiment[arm + "_counts"] = edit[arm]["counts"]
                    experiment[arm + "_path"] = f"exact/{name}.json"
                    experiment[arm + "_majorana"] = result["majorana"]
                experiments.append(experiment)
                print(
                    f"{edit['name']}: target-control={experiment['target']['quality'] - experiment['control']['quality']:+.9f}",
                    flush=True,
                )
            freeze(directory / "experiments.json", experiments)
            simplifications = []
            for edit in edits["simplifications"]:
                if edit["status"] == "unavailable":
                    simplifications.append(edit)
                    continue
                parent = parents[edit["parent"]]
                name = f"simplification-{edit['seed']}"
                record, _ = journal.confirmed(
                    name, build(edit["edges"]), edit["selection_seed"], deletion=True
                )
                before, after = parent["metrics"], metrics(record)
                ratio = after["quality"] / before["quality"] if before["quality"] > 0 else None
                retained = (
                    ratio is not None
                    and ratio >= 0.9
                    and before["eligible"] == after["eligible"]
                    and parent["indices"] == indices(record)
                )
                simplifications.append(
                    {
                        **edit,
                        "status": "complete",
                        "before": before,
                        "after": after,
                        "before_indices": parent["indices"],
                        "after_indices": indices(record),
                        "retention_ratio": ratio,
                        "retains_90_percent": retained,
                        "exact_path": f"exact/{name}.json",
                        "exact_id": record.geometry.exact_id,
                    }
                )
            freeze(directory / "simplifications.json", simplifications)
        result = analyze(rows, distances, experiments, simplifications)
        freeze(directory / "analysis.json", result)
        attempts = [read_json(p) for p in sorted((directory / "attempts").glob("*.json"))]
        accounting = {
            "attempts": len(attempts),
            "cap": config.exact_attempt_cap,
            "completed": sum(a["status"] == "complete" for a in attempts),
            "failed": sum(a["status"] == "failed" for a in attempts),
            "interrupted": sum(a["status"] == "started" for a in attempts),
            "exact_seconds": sum(
                read_json(p)["seconds"] for p in (directory / "exact").glob("*.json")
            ),
        }
        freeze(directory / "accounting.json", accounting)
        inventory = {
            p.relative_to(directory).as_posix(): digest(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name not in ("inventory.json", "writer.lock")
        }
        freeze(directory / "inventory.json", inventory)
        print(json.dumps(accounting), flush=True)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    run_campaign(args.output, analysis_only=args.analysis_only)


if __name__ == "__main__":
    main()
