"""Freeze, execute and audit the bounded Phase-16B falsification experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import zipfile
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from toposc_lab.active_learning.benchmark import source_provenance
from toposc_lab.active_learning.cycle import derived_seed
from toposc_lab.data import ReproducibilityMetadata, record_from_dict, validate_dataset_record
from toposc_lab.discovery.engine import scientific_record
from toposc_lab.discovery.storage import decode_run, read_json, writer_lease
from toposc_lab.discovery.validation import majorana_evidence, model_for
from toposc_lab.generative.space import GeometrySearchSpace, embedded_edges
from toposc_lab.patterns.campaign import (
    THREADS,
    ExactJournal,
    build,
    digest,
    freeze,
    indices,
    metrics,
)
from toposc_lab.patterns.interventions import deletion_candidates, valid_swaps
from toposc_lab.patterns.validation import (
    HYPOTHESES,
    ExclusionIndex,
    orbit_masks,
    pair_eligible,
    select_pair,
    summarize,
    swap_description,
)


def edge_sets(value: Any) -> list[list[list[int]]]:
    result = []
    if isinstance(value, dict):
        edges = value.get("edges")
        if (
            isinstance(edges, list)
            and edges
            and all(
                isinstance(e, list) and len(e) == 2 and all(type(i) is int for i in e)
                for e in edges
            )
        ):
            result.append(edges)
        for k, v in value.items():
            if k != "edges":
                result.extend(edge_sets(v))
    elif isinstance(value, list):
        for v in value:
            result.extend(edge_sets(v))
    return result


def exclusions(root: Path, directory: Path) -> dict[str, Any]:
    path = directory / "exclusions.json"
    if path.exists():
        existing: dict[str, Any] = read_json(path)
        return existing
    old = root / "results/phase16a"
    inventory = read_json(old / "inventory.json")
    for name, checksum in inventory.items():
        if digest(old / name) != checksum:
            raise ValueError(f"Phase16A preservation failure: {name}")
    files = [old / name for name in ("features.json", "intervention-plan.json", "fresh-plan.json")]
    # Every extant production discovery pool is a conservative seen-data exclusion.
    for p in sorted((root / "results").rglob("plan.json")):
        if (
            p.parent.name.startswith("cycle-")
            and (p.parent.parent / "manifest.json").exists()
            and not any(part.startswith("pytest") for part in p.parts)
        ):
            files.append(p)
    unique = {}
    checksums = {}
    for p in files:
        checksum = digest(p)
        payload = read_json(p)
        if digest(p) != checksum:
            raise ValueError(f"input changed while reading: {p}")
        checksums[p.relative_to(root).as_posix()] = checksum
        for edges in edge_sets(payload):
            unique[min(orbit_masks(edges))] = edges
    value = {
        "inputs_sha256": checksums,
        "edges": [unique[k] for k in sorted(unique)],
        "phase16a_inventory_sha256": digest(old / "inventory.json"),
        "cutoff": "snapshot_before_any_phase16b_proposal_or_physics",
    }
    print(f"exclusion snapshot: {len(files)} inputs, {len(unique)} D4 structures", flush=True)
    frozen: dict[str, Any] = freeze(path, value)
    return frozen


def plan_validation(excluded: dict[str, Any]) -> dict[str, Any]:
    index = ExclusionIndex(excluded["edges"])
    space = GeometrySearchSpace()
    parents, proposals = [], []
    for ordinal, seed in enumerate(range(16201, 16217)):
        rng = np.random.default_rng(seed)
        generator = "random" if ordinal % 2 == 0 else "patch"
        for attempt in range(2000):
            geometry = space.sample(rng, patch=generator == "patch")
            edges = [list(e) for e in embedded_edges(geometry)]
            reasons = list(space.reasons(geometry))
            distance = index.distance(edges) if not reasons else None
            accepted = not reasons and distance is not None and distance > 0.15
            proposals.append(
                {
                    "seed": seed,
                    "attempt": attempt,
                    "edges": edges,
                    "reasons": reasons,
                    "distance": distance,
                    "accepted": accepted,
                }
            )
            if accepted:
                break
        else:
            raise RuntimeError(f"raw proposal cap exhausted for {seed}")
        edits = []
        rejected_edits = 0
        for swap in valid_swaps(geometry):
            row = swap_description(geometry, swap)
            row["distance"] = index.distance(row["edges"])
            if row["distance"] > 0.15:
                edits.append(row)
            else:
                rejected_edits += 1
        pairs = {
            h: select_pair(h, edits, derived_seed(seed, 0, 162, i))
            for i, h in enumerate(("h1", "h2", "h4", "h5"))
        }
        choices: list[dict[str, Any]] = []
        for edge in deletion_candidates(geometry):
            changed = [e for e in edges if tuple(e) != edge]
            distance = index.distance(changed)
            if distance > 0.15:
                choices.append({"removed": list(edge), "edges": changed, "distance": distance})
        selection = np.random.default_rng(derived_seed(seed, 0, 163)).permutation(len(choices))[:2]
        deletions = [choices[int(i)] for i in selection]
        parent = {
            "seed": seed,
            "generator": generator,
            "edges": edges,
            "distance": index.distance(edges),
            "pairs": pairs,
            "deletions": deletions,
            "valid_separated_swaps": len(edits),
            "near_archive_swaps": rejected_edits,
            "eligible_deletions": len(choices),
        }
        parents.append(parent)
        # Subsequent independent families are separated from ALL selected arms.
        index.add(edges)
        for pair in pairs.values():
            if pair:
                for arm in ("a", "b"):
                    index.add(pair[arm]["edges"])
        for deletion in deletions:
            index.add(deletion["edges"])
        print(
            f"planned seed {seed}: pairs {[h for h, p in pairs.items() if p]}, "
            f"deletions {len(deletions)}",
            flush=True,
        )
    return {
        "parents": parents,
        "proposals": proposals,
        "selection": "outcome_blind",
        "seeds": list(range(16201, 16217)),
        "exact_attempt_cap": 700,
    }


def initialize(root: Path, directory: Path) -> ReproducibilityMetadata:
    if any(os.environ.get(k) != "1" for k in THREADS):
        raise ValueError("all four numerical thread limits must be 1")
    provenance = source_provenance(root)
    protocol = root / "docs/decisions/phase_16b_protocol.md"
    identity: dict[str, Any] = {
        "source_sha256": provenance.runtime["source_sha256"],
        "git_commit": provenance.git_commit,
        "git_dirty": provenance.git_dirty,
        "driver_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(protocol),
        "exclusions_sha256": digest(directory / "exclusions.json"),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": version("scipy"),
            "threads": {k: os.environ[k] for k in THREADS},
        },
        "attempt_cap": 700,
    }
    freeze(directory / "manifest.json", identity)
    archive_path = directory / "source.zip"
    paths = sorted((root / "src").rglob("*.py")) + [
        Path(__file__),
        protocol,
        root / "pyproject.toml",
    ]
    paths += sorted((root / "tests").glob("test_pattern_validation*.py"))
    if not archive_path.exists():
        temporary = directory / "source.zip.tmp"
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for p in paths:
                archive.write(p, p.relative_to(root).as_posix())
        os.replace(temporary, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None or any(
            archive.read(p.relative_to(root).as_posix()) != p.read_bytes() for p in paths
        ):
            raise ValueError("frozen source archive mismatch")
    freeze(directory / "source-archive.json", {"sha256": digest(archive_path)})
    return replace(
        provenance,
        runtime={
            **dict(provenance.runtime),
            **identity["runtime"],
            "benchmark": "phase16b.falsification.v1",
            "protocol_sha256": identity["protocol_sha256"],
            "driver_sha256": identity["driver_sha256"],
        },
    )


def evaluate_plan(
    directory: Path, plan: dict[str, Any], provenance: ReproducibilityMetadata
) -> dict[str, Any]:
    journal = ExactJournal(directory, provenance, cap=700)
    rows = []

    def evaluate(
        name: str, edges: list[list[int]], seed: int, deletion: bool = False
    ) -> dict[str, Any]:
        geometry = build(edges)
        record, _base = journal.confirmed(name, geometry, seed, deletion=deletion)
        result: dict[str, Any] = {
            "stage": name,
            "metrics": metrics(record),
            "indices": indices(record),
            "record_id": record.record_id,
            "sensitivity": {},
        }
        if not deletion:
            for mu in (1.9, 2.1):
                raw = journal.calculate(name + f"-mu{int(mu * 10)}", geometry, seed, mu=mu)
                other = record_from_dict(raw["record"])
                result["sensitivity"][str(mu)] = {
                    "metrics": metrics(other),
                    "indices": indices(other),
                }
        return result

    for parent in plan["parents"]:
        seed = parent["seed"]
        row: dict[str, Any] = {
            "seed": seed,
            "generator": parent["generator"],
            "parent": evaluate(f"parent-{seed}", parent["edges"], seed),
            "pairs": {},
            "deletions": [],
        }
        for i, (h, pair) in enumerate(parent["pairs"].items()):
            if pair:
                row["pairs"][h] = {
                    arm: evaluate(
                        f"{h}-{seed}-{arm}", pair[arm]["edges"], derived_seed(seed, i, 164, j)
                    )
                    for j, arm in enumerate(("a", "b"))
                }
        for i, deletion in enumerate(parent["deletions"]):
            row["deletions"].append(
                evaluate(
                    f"deletion-{seed}-{i}",
                    deletion["edges"],
                    derived_seed(seed, i, 165),
                    deletion=True,
                )
            )
        rows.append(row)
        print(
            f"evaluated seed {seed}, quality {row['parent']['metrics']['quality']:.9f}", flush=True
        )
    space = GeometrySearchSpace()
    grid = space.build(
        frozenset(
            e
            for e in space.edge_pool
            if np.linalg.norm(space.coordinates[e[0]] - space.coordinates[e[1]]) == 1
        )
    )
    references = {}
    for mu in (2.0, 8.0):
        name = f"reference-mu{int(mu)}"
        first = journal.calculate(name, grid, 16297, mu=mu)
        second = journal.calculate(name + "-confirmation", grid, 16298, mu=mu)
        a, b = record_from_dict(first["record"]), record_from_dict(second["record"])
        if scientific_record(a) != scientific_record(b) or first["majorana"] != second["majorana"]:
            raise ValueError("reference repeat disagrees")
        if indices(a) != ([1, 1, 1] if mu == 2 else [0, 0, 0]):
            raise ValueError("known positive/negative reference failed")
        references[str(mu)] = {"metrics": metrics(a), "indices": indices(a)}
    return {"rows": rows, "references": references}


def analyze(results: dict[str, Any]) -> dict[str, Any]:
    effects: dict[str, list[dict[str, Any]]] = {h: [] for h in HYPOTHESES}
    for row in results["rows"]:
        parent = row["parent"]
        for h, pair in row["pairs"].items():
            a, b = pair["a"], pair["b"]
            sign = 1 if h == "h1" else -1
            effect = sign * (a["metrics"]["quality"] - b["metrics"]["quality"])
            effects[h].append(
                {
                    "seed": row["seed"],
                    "effect": effect,
                    "a_minus_parent": a["metrics"]["quality"] - parent["metrics"]["quality"],
                    "b_minus_parent": b["metrics"]["quality"] - parent["metrics"]["quality"],
                    "sensitivity": {
                        mu: sign
                        * (
                            a["sensitivity"][mu]["metrics"]["quality"]
                            - b["sensitivity"][mu]["metrics"]["quality"]
                        )
                        for mu in ("1.9", "2.1")
                    },
                }
            )
        if row["deletions"]:
            q = parent["metrics"]["quality"]
            retained = [
                q > 0
                and d["metrics"]["quality"] >= 0.9 * q
                and d["indices"] == parent["indices"]
                and bool(d["metrics"]["eligible"])
                for d in row["deletions"]
            ]
            effects["h3"].append(
                {
                    "seed": row["seed"],
                    "effect": float(np.mean(retained)),
                    "retained": retained,
                    "parent_quality": q,
                    "topology_changed": [
                        d["indices"] != parent["indices"] for d in row["deletions"]
                    ],
                }
            )
    tests = summarize(effects)
    records = [r["parent"] for r in results["rows"]]
    for r in results["rows"]:
        records += r["deletions"]
        records += [p[a] for p in r["pairs"].values() for a in ("a", "b")]
    return {
        "tests": tests,
        "main_model_record_roles": len(records),
        "main_model_success_roles": sum(r["metrics"]["quality"] >= 0.2 for r in records),
        "best_main_quality": max(r["metrics"]["quality"] for r in records),
        "cross_model_validation": "unavailable: no validated second-model wiring adapter",
        "parameter_sensitivity": "same chiral-p-wave model, mu=1.9 and 2.1",
        "mechanism_gate": "NOT_ESTABLISHED",
        "reason": "Frozen 16A recurrent/intervention chain not established; cross-model generality unavailable.",
        "phase_claim": False,
        "majorana_claim": False,
        "success_threshold": 0.2,
    }


def audit(directory: Path) -> dict[str, Any]:
    manifest = read_json(directory / "manifest.json")
    if digest(directory / "exclusions.json") != manifest["exclusions_sha256"]:
        raise ValueError("exclusion snapshot changed")
    if digest(directory / "source.zip") != read_json(directory / "source-archive.json")["sha256"]:
        raise ValueError("source archive changed")
    if digest(directory / "plan.json") != read_json(directory / "plan-identity.json")["sha256"]:
        raise ValueError("plan changed")
    inventory_path = directory / "inventory.json"
    if inventory_path.exists():
        inventory = read_json(inventory_path)
        actual = {
            p.relative_to(directory).as_posix()
            for p in directory.rglob("*")
            if p.is_file() and p.name not in ("writer.lock", "inventory.json")
        }
        if set(inventory) != actual or any(
            digest(directory / k) != v for k, v in inventory.items()
        ):
            raise ValueError("inventory mismatch")
    plan = read_json(directory / "plan.json")
    index = ExclusionIndex(read_json(directory / "exclusions.json")["edges"])
    expected_stages = 4
    for parent in plan["parents"]:
        family = [parent["edges"]]
        expected_stages += 4 + 2 * len(parent["deletions"])
        for h, pair in parent["pairs"].items():
            if pair:
                if not pair_eligible(h, pair["a"], pair["b"]):
                    raise ValueError("strict match violation")
                expected_stages += 8
                family.extend(pair[arm]["edges"] for arm in ("a", "b"))
        family.extend(d["edges"] for d in parent["deletions"])
        if any(index.distance(edges) <= 0.15 for edges in family):
            raise ValueError("independent family separation failed")
        for edges in family:
            index.add(edges)
    attempts = sorted((directory / "attempts").glob("*.json"))
    if len(attempts) != expected_stages or len(list((directory / "exact").glob("*.json"))) != len(
        attempts
    ):
        raise ValueError("planned exact stages missing or unexpected")
    confirmations = 0
    for number, p in enumerate(attempts, 1):
        entry = read_json(p)
        if entry["attempt"] != number or entry["status"] != "complete":
            raise ValueError("incomplete or noncontiguous attempt journal")
        target = directory / "exact" / (entry["request"]["name"] + ".json")
        if digest(target) != entry["result_sha256"]:
            raise ValueError("exact checksum mismatch")
        raw = read_json(target)
        record = record_from_dict(raw["record"])
        validate_dataset_record(record).raise_for_errors()
        if raw["request"] != entry["request"] or raw["attempt"] != number:
            raise ValueError("request journal mismatch")
        if raw["majorana"]["operator_phs_residual"] > 1e-10:
            raise ValueError("PHS failure")
        run = decode_run(raw["run"])
        # Normalize tuple-valued fields to the stored JSON representation.
        if json.loads(json.dumps(majorana_evidence(record, run))) != raw["majorana"]:
            raise ValueError("stored diagnostics do not reproduce")
        simulation = run.simulation_result
        assert simulation is not None
        matrix = model_for(record).hamiltonian()
        vectors, energies = simulation.eigenvectors, simulation.eigenvalues
        if (
            np.max(np.abs(matrix @ vectors - vectors * energies)) > 1e-10
            or np.max(np.abs(vectors.conj().T @ vectors - np.eye(len(energies)))) > 1e-10
            or not np.allclose(energies, record.spectrum.eigenvalues, atol=1e-10, rtol=0)
        ):
            raise ValueError("stored eigensystem residual failed")
        if target.stem.endswith("-confirmation"):
            base = read_json(target.with_name(target.stem.removesuffix("-confirmation") + ".json"))
            if scientific_record(record) != scientific_record(record_from_dict(base["record"])):
                raise ValueError("independent confirmation mismatch")
            if (
                raw["majorana"] != base["majorana"]
                or raw["request"]["seed"] == base["request"]["seed"]
            ):
                raise ValueError("diagnostics/seed confirmation mismatch")
            confirmations += 1
    if len(attempts) > 700:
        raise ValueError("attempt cap exceeded")
    if analyze(read_json(directory / "results.json")) != read_json(directory / "analysis.json"):
        raise ValueError("stored analysis does not reproduce")
    return {
        "status": "PASS",
        "attempts": len(attempts),
        "confirmations": confirmations,
        "audit_adds_numerical_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("results/phase16b"))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with writer_lease(directory):
        if args.audit_only:
            print(json.dumps(audit(directory)), flush=True)
            return
        excluded = exclusions(root, directory)
        provenance = initialize(root, directory)
        path = directory / "plan.json"
        plan = read_json(path) if path.exists() else freeze(path, plan_validation(excluded))
        freeze(directory / "plan-identity.json", {"sha256": digest(path)})
        if args.plan_only:
            return
        result = freeze(directory / "results.json", evaluate_plan(directory, plan, provenance))
        analysis = freeze(directory / "analysis.json", analyze(result))
        checked = freeze(directory / "audit.json", audit(directory))
        inventory = {
            p.relative_to(directory).as_posix(): digest(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file() and p.name not in ("writer.lock", "inventory.json")
        }
        freeze(directory / "inventory.json", inventory)
        print(json.dumps({"gate": analysis["mechanism_gate"], **checked}), flush=True)


if __name__ == "__main__":
    main()
