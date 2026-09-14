"""Audit frozen Phase-16A records using stored eigensystems; never solve physics."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from toposc_lab.data import load_dataset, record_from_dict, validate_dataset_record
from toposc_lab.discovery.engine import plain, scientific_record
from toposc_lab.discovery.storage import atomic_json, decode_run, read_json
from toposc_lab.discovery.validation import majorana_evidence, model_for
from toposc_lab.generative.physics import candidate
from toposc_lab.generative.space import (
    GeometrySearchSpace,
    embedded_edges,
    structural_distance,
    structural_key,
)
from toposc_lab.patterns.interventions import (
    MOTIFS,
    EdgeSwap,
    deletion_reasons,
    motif_counts,
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def normalize(value: Any) -> Any:
    return json.loads(json.dumps(plain(value), allow_nan=False))


def topology_label(metrics: dict[str, Any], indices: list[Any]) -> str:
    if bool(metrics["eligible"]):
        return "eligible_nonzero"
    if indices == [0, 0, 0]:
        return "resolved_trivial"
    return "unresolved_or_ineligible"


def audit(directory: Path, root: Path) -> dict[str, Any]:
    inventory = read_json(directory / "inventory.json")
    paths = {
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file() and p.name not in ("inventory.json", "writer.lock")
    }
    check(set(inventory) == paths, "inventory file set mismatch")
    for name, checksum in inventory.items():
        check(digest(directory / name) == checksum, f"inventory checksum: {name}")
    print(f"verified {len(inventory)} inventory checksums", flush=True)
    manifest = read_json(directory / "manifest.json")
    archive_digest = read_json(directory / "source-archive.json")["sha256"]
    check(digest(directory / "source.zip") == archive_digest, "source archive digest")
    source_hash = sha256()
    current_source = {}
    for path in sorted((root / "src").rglob("*.py")):
        name = path.relative_to(root).as_posix()
        content = path.read_bytes()
        source_hash.update(name.encode())
        source_hash.update(content)
        current_source[name] = content
    check(source_hash.hexdigest() == manifest["source_sha256"], "current source digest")
    with zipfile.ZipFile(directory / "source.zip") as archive:
        check(archive.testzip() is None, "source archive CRC")
        archived_source = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.startswith("src/") and name.endswith(".py")
        }
        check(archived_source == current_source, "archived source differs from current source")
        for name in archive.namelist():
            if not name.startswith("src/"):
                check(archive.read(name) == (root / name).read_bytes(), f"archive file: {name}")
    check(
        manifest["driver_sha256"] == digest(root / "src/toposc_lab/patterns/campaign.py"),
        "driver digest",
    )
    check(
        manifest["protocol_sha256"] == digest(root / "docs/decisions/phase_16a_protocol.md"),
        "protocol digest",
    )
    legacy_archives = []
    for path in sorted((root / "results/phase15-gate-final").glob("*/source.zip")):
        with zipfile.ZipFile(path) as archive:
            checked = 0
            for name in archive.namelist():
                if name.startswith("src/") and name.endswith(".py"):
                    check(current_source[name] == archive.read(name), f"Phase-15 source: {name}")
                    checked += 1
            check(archive.testzip() is None, f"Phase-15 archive CRC: {path}")
        legacy_archives.append({"path": path.relative_to(root).as_posix(), "files": checked})

    rows = read_json(directory / "features.json")
    space = GeometrySearchSpace()
    build = lambda edges: space.build(frozenset(tuple(e) for e in edges))
    parents = {r["id"]: r for r in rows}
    geometries = [build(row["edges"]) for row in rows]
    check(all(not space.reasons(g) for g in geometries), "cohort geometry validity")
    legacy = [r for r in rows if r["cohort"] == "legacy"]
    fresh = [r for r in rows if r["cohort"] == "fresh"]
    cohort_keys = [structural_key(g) for g in geometries]
    check(len(set(cohort_keys)) == len(rows), "cohort D4 duplicates")
    distances = np.asarray(read_json(directory / "distances.json"))
    recomputed = np.zeros((len(rows), len(rows)))
    for i, geometry in enumerate(geometries):
        for j in range(i):
            recomputed[i, j] = recomputed[j, i] = structural_distance(geometry, geometries[j])
    np.testing.assert_array_equal(distances, recomputed)
    fresh_positions = [i for i, row in enumerate(rows) if row["cohort"] == "fresh"]
    minimum_fresh_distance = min(
        distances[i, j] for i in fresh_positions for j in range(len(rows)) if i != j
    )
    check(minimum_fresh_distance > 0.15, "fresh cohort separation")
    plan = read_json(directory / "fresh-plan.json")
    check(
        [(r["id"], r["edges"]) for r in fresh] == [(r["id"], r["edges"]) for r in plan["accepted"]],
        "fresh rows do not match the outcome-blind plan",
    )
    generator_counts = {
        str(seed): dict(Counter(r["generator"] for r in fresh if r["seed"] == seed))
        for seed in manifest["config"]["seeds"]
    }
    for i, seed in enumerate(manifest["config"]["seeds"]):
        expected = "random" if i % 2 == 0 else "patch"
        check(generator_counts[str(seed)] == {expected: 8}, f"seed allocation: {seed}")
    legacy_by_key = {structural_key(build(row["edges"])): row for row in legacy}
    legacy_source_count = 0
    for arm, checksum in manifest["inputs_sha256"].items():
        path = directory / "inputs" / f"{arm}.json"
        check(digest(path) == checksum, f"legacy input digest: {arm}")
        check(
            digest(root / "results/phase15-gate-final" / arm / "dataset.json") == checksum,
            f"original legacy dataset changed: {arm}",
        )
        for record in load_dataset(path).records:
            validate_dataset_record(record).raise_for_errors()
            row = legacy_by_key[structural_key(record.geometry.to_geometry())]
            np.testing.assert_allclose(
                row["spectrum"], record.spectrum.eigenvalues, rtol=0, atol=1e-10
            )
            observed = plain(record.observables[0].values)
            for endpoint in ("quality", "eligible", "localizer_gap", "minimum_abs_energy"):
                np.testing.assert_allclose(
                    row["metrics"][endpoint], observed[endpoint], rtol=0, atol=1e-10
                )
            check(
                len({abs(v) if v is not None else None for v in row["indices"]})
                == len(
                    {
                        abs(v) if v is not None else None
                        for v in record.topology[0].parameters["indices"]
                    }
                ),
                "legacy topology resolution differs across pooled origins",
            )
            legacy_source_count += 1

    attempts = [read_json(p) for p in sorted((directory / "attempts").glob("*.json"))]
    for number, attempt in enumerate(attempts, 1):
        check(attempt["attempt"] == number, "attempt journal order")
        path = directory / "exact" / f"{attempt['request']['name']}.json"
        if attempt["status"] == "complete":
            check(path.exists(), f"missing committed exact stage: {path}")
            check(attempt["result_sha256"] == digest(path), f"exact attempt checksum: {path}")
    exact = {}
    raw_keys = set()
    maximum_eigen_residual = 0.0
    maximum_orthogonality_residual = 0.0
    exact_seconds = 0.0
    for number, path in enumerate(sorted((directory / "exact").glob("*.json")), 1):
        payload = read_json(path)
        record = record_from_dict(payload["record"])
        validate_dataset_record(record).raise_for_errors()
        request = payload["request"]
        attempt = attempts[payload["attempt"] - 1]
        check(
            attempt["request"] == request and attempt["status"] == "complete",
            "exact request journal",
        )
        check(request["name"] == path.stem, "exact filename request")
        check(record.provenance.seed == request["seed"], "exact seed provenance")
        check(
            record.provenance.runtime["source_sha256"] == manifest["source_sha256"], "exact source"
        )
        check(
            record.provenance.runtime["protocol_sha256"] == manifest["protocol_sha256"],
            "exact protocol",
        )
        geometry = record.geometry.to_geometry()
        check(
            candidate(geometry, request["mu"]).candidate_id == request["candidate_id"],
            "candidate identity mismatch",
        )
        check(
            not (deletion_reasons(geometry) if request["deletion"] else space.reasons(geometry)),
            "exact geometry outside declared stratum",
        )
        run = decode_run(payload["run"])
        simulation = run.simulation_result
        check(simulation is not None, "missing stored simulation")
        np.testing.assert_array_equal(record.spectrum.eigenvalues, simulation.eigenvalues)
        diagnostics = normalize(majorana_evidence(record, run))
        check(diagnostics == payload["majorana"], "stored Majorana diagnostic mismatch")
        check(diagnostics["majorana_claim"] is False, "unsupported Majorana claim")
        hamiltonian = model_for(record).hamiltonian()
        vectors = simulation.eigenvectors
        eigen_residual = float(
            np.max(np.abs(hamiltonian @ vectors - vectors * simulation.eigenvalues))
        )
        orthogonality = float(np.max(np.abs(vectors.conj().T @ vectors - np.eye(len(vectors)))))
        check(
            eigen_residual < 1e-10 and orthogonality < 1e-10,
            "stored eigensystem failed arithmetic check",
        )
        maximum_eigen_residual = max(maximum_eigen_residual, eigen_residual)
        maximum_orthogonality_residual = max(maximum_orthogonality_residual, orthogonality)
        values = normalize(record.observables[0].values)
        topology = normalize(record.topology[0].parameters)
        gap = min(topology["localizer_gaps"])
        check(values["localizer_gap"] == gap, "minimum localizer gap mismatch")
        check(values["quality"] == values["eligible"] * gap, "quality gate mismatch")
        check(bool(values["success"]) == (values["quality"] >= 0.2), "frozen success mismatch")
        weights = [state["boundary_weight"] for state in diagnostics["states"]]
        np.testing.assert_array_equal(values["boundary_weights"], weights)
        exact[path.stem] = {
            "scientific": scientific_record(record),
            "majorana": diagnostics,
            "metrics": values,
            "indices": topology["indices"],
            "key": structural_key(geometry),
            "exact_id": record.geometry.exact_id,
        }
        raw_keys.add(structural_key(geometry))
        exact_seconds += payload["seconds"]
        if number % 100 == 0:
            print(f"audited {number} stored exact eigensystems without solvers", flush=True)
    confirmations = 0
    for name, result in exact.items():
        if name.endswith("-confirmation"):
            base = exact[name.removesuffix("-confirmation")]
            check(result == base, f"exact confirmation differs: {name}")
            confirmations += 1
        elif not name.startswith("reference-"):
            check(name + "-confirmation" in exact, f"unconfirmed exact stage: {name}")
    for row in fresh:
        check(row["metrics"] == exact[row["id"]]["metrics"], "fresh metric extraction")
        check(row["indices"] == exact[row["id"]]["indices"], "fresh topology extraction")
    check(exact["reference-2"]["metrics"]["eligible"] == 1, "positive reference")
    check(exact["reference-8"]["metrics"]["eligible"] == 0, "trivial reference")
    reproduced = read_json(directory / "reproduction.json")
    check(exact["independent-repeat"] == exact[reproduced["id"]], "independent repeat mismatch")

    edits = read_json(directory / "intervention-plan.json")
    experiments = read_json(directory / "experiments.json")
    check(len(edits["ablations"]) == len(experiments), "experiment count mismatch")
    used = defaultdict(set)
    intervention_keys = set()
    transitions = defaultdict(Counter)
    motif_parent_counts = defaultdict(set)
    for edit, experiment in zip(edits["ablations"], experiments, strict=True):
        parent = parents[edit["parent"]]
        geometry = build(parent["edges"])
        before_counts = motif_counts(geometry)
        check(edit["before_counts"] == before_counts, "parent motif count mismatch")
        check(experiment["before"] == parent["metrics"], "experiment parent metrics")
        check(experiment["parent"] == edit["parent"], "experiment parent ordering")
        motif = edit["motif"]
        motif_parent_counts[motif].add(edit["parent"])
        group = (edit["parent"], motif)
        parent_label = topology_label(parent["metrics"], parent["indices"])
        for arm in ("target", "control"):
            swap_data = edit[arm]["swap"]
            swap = EdgeSwap(
                tuple(tuple(e) for e in swap_data["removed"]),
                tuple(tuple(e) for e in swap_data["added"]),
                tuple(swap_data["counts"]),
            )
            changed = swap.apply(geometry)
            check(
                embedded_edges(changed) == embedded_edges(build(edit[arm]["edges"])), "swap edges"
            )
            check(motif_counts(changed) == edit[arm]["counts"], "changed motif counts")
            check(
                [changed.degree(i) for i in range(36)] == [geometry.degree(i) for i in range(36)],
                "intervention degree sequence mismatch",
            )
            key = structural_key(changed)
            # Distinct physical edits are required; symmetry duplicates are also flagged.
            edit_key = tuple(embedded_edges(changed))
            check(edit_key not in used[group], "reused edit within parent/motif")
            used[group].add(edit_key)
            intervention_keys.add(key)
            result = exact[edit["name"] + "-" + arm]
            check(key == result["key"], "intervention exact geometry")
            check(experiment[arm] == result["metrics"], "intervention metrics extraction")
            check(
                experiment[arm + "_indices"] == result["indices"],
                "intervention topology extraction",
            )
            transitions[motif + "/" + arm][
                parent_label + " -> " + topology_label(result["metrics"], result["indices"])
            ] += 1
        check(edit["target"]["counts"][motif] < before_counts[motif], "target did not remove motif")
        check(
            edit["control"]["counts"][motif] == before_counts[motif], "control changed target motif"
        )
        for nuisance in ("diagonal_count", "boundary_bulk_edges"):
            if nuisance != motif:
                check(
                    edit["target"]["counts"][nuisance] == edit["control"]["counts"][nuisance],
                    "unmatched nuisance",
                )
    simplifications = read_json(directory / "simplifications.json")
    simplified_keys = set()
    for row in simplifications:
        if row["status"] == "unavailable":
            continue
        geometry = build(row["edges"])
        check(not deletion_reasons(geometry), "invalid simplification")
        before = {tuple(e) for e in parents[row["parent"]]["edges"]}
        after = set(embedded_edges(geometry))
        check(
            before - after == {tuple(row["removed"])} and after <= before,
            "single deletion mismatch",
        )
        check(
            exact[f"simplification-{row['seed']}"]["metrics"] == row["after"],
            "simplification extraction",
        )
        simplified_keys.add(structural_key(geometry))
    accounting = read_json(directory / "accounting.json")
    count_status = Counter(a["status"] for a in attempts)
    check(accounting["attempts"] == len(attempts) <= accounting["cap"], "attempt cap accounting")
    check(accounting["completed"] == count_status["complete"] == len(exact), "completed accounting")
    check(accounting["failed"] == count_status["failed"], "failure accounting")
    check(accounting["interrupted"] == count_status["started"], "interruption accounting")
    np.testing.assert_allclose(accounting["exact_seconds"], exact_seconds, rtol=0, atol=1e-8)
    all_study_keys = set(cohort_keys) | intervention_keys | simplified_keys
    return {
        "status": "PASS",
        "non_test_numerical_attempts_added": 0,
        "inventory_files_verified": len(inventory),
        "inventory_sha256": digest(directory / "inventory.json"),
        "source_sha256": manifest["source_sha256"],
        "source_archive_sha256": archive_digest,
        "source_files_verified": len(current_source),
        "phase15_source_archives_unchanged": legacy_archives,
        "stored_exact_records_verified": len(exact),
        "confirmations_verified": confirmations,
        "maximum_eigen_equation_residual": maximum_eigen_residual,
        "maximum_eigenvector_orthogonality_residual": maximum_orthogonality_residual,
        "majorana_diagnostics_recomputed_from_stored_vectors": len(exact),
        "legacy_source_records_verified": legacy_source_count,
        "cohort_rows": len(rows),
        "legacy_d4_geometries": len(legacy),
        "fresh_d4_geometries": len(fresh),
        "minimum_fresh_to_other_cohort_distance": minimum_fresh_distance,
        "fresh_counts_by_seed_and_generator": generator_counts,
        "fresh_proposal_status_counts": dict(Counter(e["status"] for e in plan["audit"])),
        "intervention_pairs_verified": len(experiments),
        "intervention_distinct_parents_by_motif": {m: len(motif_parent_counts[m]) for m in MOTIFS},
        "intervention_distinct_d4_geometries": len(intervention_keys),
        "simplified_distinct_d4_geometries": len(simplified_keys),
        "study_distinct_d4_geometries": len(all_study_keys),
        "all_raw_exact_distinct_d4_geometries_including_reference": len(raw_keys),
        "study_plus_reference_distinct_d4_geometries": len(all_study_keys | raw_keys),
        "topology_transitions": {key: dict(value) for key, value in transitions.items()},
        "accounting": accounting,
        "limitations": [
            "Localizer spectra were not recomputed; raw gaps, indices and signatures were retained and confirmation equality checked.",
            "Stored BdG eigenpairs and Majorana diagnostics were checked arithmetically without any eigensolver.",
            "No independent Phase-16B geometries or robustness/scaling ensemble was generated.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("results/phase16a"))
    parser.add_argument("--output", type=Path, default=Path("results/phase16a-audit.json"))
    args = parser.parse_args()
    directory = args.directory.resolve()
    output = args.output.resolve()
    check(directory not in output.parents, "audit output must remain outside immutable campaign")
    with ExitStack() as stack:
        for namespace in ("numpy.linalg", "scipy.linalg"):
            for name in ("eig", "eigh", "eigvals", "eigvalsh"):
                stack.enter_context(
                    patch(
                        namespace + "." + name,
                        side_effect=RuntimeError("audit forbids new eigensolver calls"),
                    )
                )
        result = audit(directory, Path(__file__).resolve().parents[1])
    result["audit_script_sha256"] = digest(Path(__file__))
    atomic_json(output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "inventory_files_verified",
                    "stored_exact_records_verified",
                    "confirmations_verified",
                    "study_distinct_d4_geometries",
                    "non_test_numerical_attempts_added",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
