"""Read-only Phase-18 artifact audit and geometry-only intervention preparation.

No diagonalizations, no historical writes. Output must be outside the study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from toposc_lab.data import record_from_dict, validate_dataset_record
from toposc_lab.models.chiral_p_wave import ChiralPWaveModel, ChiralPWaveParameters
from toposc_lab.research.space import (
    FixedConnectivitySpace,
    geometry_from_payload,
    geometry_to_payload,
)
from toposc_lab.research.storage import ResearchStore, atomic_text
from toposc_lab.research.validation_cohort import digest, structural_descriptors, validate_cohort
from toposc_lab.research.validation_reporting import analyze, load_snapshot
from toposc_lab.robustness.disorder import exact_hamiltonian_id
from toposc_lab.robustness.onsite import apply_uniform_onsite_disorder


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit(directory: Path) -> dict:
    store = ResearchStore(directory)
    store.integrity_check()
    snap = load_snapshot(directory)
    cohort = snap["cohort"]
    validate_cohort(cohort)
    candidates = {c["id"]: c for c in cohort["candidates"]}
    for c in candidates.values():
        actual = structural_descriptors(geometry_from_payload(c["geometry"]))
        require(digest(actual) == digest(c["descriptors"]), "Cohort descriptor mismatch")
    checked = 0
    for key, r in snap["results"].items():
        if r.get("status") != "completed":
            continue
        identity, _ = key.split(":", 1)
        g = geometry_from_payload(candidates[identity]["geometry"])
        model = ChiralPWaveModel(
            g, ChiralPWaveParameters(hopping=1, chemical_potential=2, pairing=1, chirality=1)
        )
        clean = model.hamiltonian()
        matrix = clean
        stage = r["stage"]
        if stage["kind"] == "disorder":
            matrix = np.asarray(
                apply_uniform_onsite_disorder(
                    g,
                    clean,
                    width=stage["width"],
                    seed=stage["seed"],
                    nambu_basis=model.nambu_basis,
                ).state
            )
        require(
            exact_hamiltonian_id(matrix) == r["hamiltonian_id"], "Hamiltonian identity mismatch"
        )
        require(
            np.allclose(
                np.real(np.diag(matrix - clean)[: g.n_sites]),
                r["onsite_offsets"],
                atol=1e-12,
                rtol=0,
            ),
            "Onsite field mismatch",
        )
        for i, kappa in enumerate(r["kappas"]):
            center = next(
                p for p in r["spatial"] if p["point"] == r["probe"] and p["kappa"] == kappa
            )
            require(
                center["index"] == r["indices"][i] and center["gap"] == r["localizer_gaps"][i],
                "Center/diagnostic mismatch",
            )
        eligible = (
            all(i is not None for i in r["indices"])
            and len(set(r["indices"])) == 1
            and r["indices"][0] != 0
            and min(r["localizer_gaps"]) > 1e-10
            and max(r["metrics"]["operator_phs_residual"], r["metrics"]["spectral_phs_residual"])
            <= 1e-10
        )
        quality = min(r["localizer_gaps"]) if eligible else 0.0
        require(quality == r["metrics"]["quality"], "Primary quality mismatch")
        require(r["metrics"]["success"] == (eligible and quality >= 0.2), "Success mismatch")
        # Phase-17 embeds a DatasetRecord; validate its scientific schema as saved.
        record_key = "dataset_record"
        if record_key in r:
            validate_dataset_record(record_from_dict(r[record_key])).raise_for_errors()
        boundary = r["boundary_window"]
        distance = np.array(boundary["distance_to_boundary"])
        for group in (
            boundary["groups"]
            + boundary["states"]
            + ([boundary["window"]] if boundary["window"] else [])
        ):
            weights = np.array(group["site_probability"])
            require(np.isclose(weights.sum(), 1, atol=1e-10), "Unnormalized boundary density")
            for width in (1, 2, 3):
                require(
                    np.isclose(
                        weights[distance < width].sum(),
                        group["strip_weights"][str(width)],
                        atol=1e-10,
                    ),
                    "Boundary strip mismatch",
                )
            require(
                np.isclose(weights @ distance, group["mean_boundary_distance"], atol=1e-10),
                "Boundary depth mismatch",
            )
            require(np.isclose(sum(group["shell_weights"]), 1, atol=1e-10), "Shell normalization")
        if boundary["state_count"]:
            projected = np.mean([s["site_probability"] for s in boundary["states"]], axis=0)
            require(
                np.allclose(projected, boundary["window"]["site_probability"], atol=1e-10),
                "Window projector density mismatch",
            )
        checked += 1
    summary = analyze(snap)
    require(summary["complete_schedule"], "Pilot schedule incomplete")
    require(
        all(r["all_present"] and r["matched"] for r in summary["paired_field_checks"]),
        "Unmatched paired fields",
    )
    exact_calls = 0
    return {
        "gate": "PASS",
        "verified_realizations": checked,
        "additional_exact_calls": exact_calls,
        "source_sha256": snap["manifest"]["source_sha256"],
        "cohort_sha256": cohort["sha256"],
        "attempt_digest": digest(snap["attempts"]),
        "result_digest": digest(snap["results"]),
        "checks": [
            "SQLite integrity and every object checksum",
            "cohort descriptors",
            "seed/geometry reconstructed Hamiltonian identities",
            "center/spatial consistency",
            "quality/success arithmetic",
            "boundary normalization/strips/depth/projector",
            "complete paired seed coverage",
        ],
        "scope": "No fresh diagonalization or independent proof of stored eigenvectors",
    }


def interventions(cohort: dict) -> dict:
    """Freeze reversible forward/back edits in three backgrounds per hypothesis."""
    space = FixedConnectivitySpace(**cohort["space"])
    rng = np.random.default_rng(180050)
    designs = []
    for c in cohort["candidates"][:3]:
        parent = geometry_from_payload(c["geometry"])
        base = structural_descriptors(parent)
        pool = {}
        for _ in range(24):
            child, _ = space.mutate(parent, rng, "multi_rewire_local")
            if not space.validate(child) and space.hash(child) != space.hash(parent):
                pool.setdefault(space.hash(child), child)
        measured = [(g, structural_descriptors(g)) for g in pool.values()]
        for feature, direction in (
            ("balanced_cut_min_conductance", 1),
            ("bulk_degree_variance", -1),
            ("long_bond_fraction", 1),
        ):
            qualifying = [
                (g, d) for g, d in measured if direction * (d[feature] - base[feature]) > 1e-12
            ]
            if not qualifying:
                designs.append(
                    {
                        "parent": c["id"],
                        "feature": feature,
                        "status": "unavailable_under_bounded_search",
                        "valid_unique_proposals": len(measured),
                    }
                )
                continue
            g, d = min(
                qualifying,
                key=lambda item: (
                    -direction * (item[1][feature] - base[feature]),
                    space.hash(item[0]),
                ),
            )
            before, after = set(space.edges(parent)), set(space.edges(g))
            removed, added = before - after, after - before
            require((before - removed) | added == after, "Forward intervention mismatch")
            require((after - added) | removed == before, "Reverse intervention mismatch")
            designs.append(
                {
                    "parent": c["id"],
                    "feature": feature,
                    "direction": direction,
                    "status": "geometry_prepared_not_physics_tested",
                    "before": base,
                    "after": d,
                    "remove": sorted(removed),
                    "add": sorted(added),
                    "geometry": geometry_to_payload(g),
                    "reverse": "Remove added edges and restore removed edges exactly",
                    "confounders": {
                        k: d[k] - base[k]
                        for k in base
                        if isinstance(base[k], (int, float))
                        and isinstance(d[k], (int, float))
                        and abs(d[k] - base[k]) > 1e-12
                    },
                    "causal_claim": False,
                }
            )
    return {
        "cohort_sha256": cohort["sha256"],
        "seed": 180050,
        "proposals_per_parent": 24,
        "designs": designs,
        "exact_calls": 0,
        "scope": "Three fixed backgrounds per hypothesis; null feasibility is retained; no confirmation data used",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.study.resolve(), args.output.resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Audit output must be outside the study")
    output.mkdir(parents=True, exist_ok=True)
    review = audit(source)
    review["review_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic_text(output / "audit.json", json.dumps(review, indent=2))
    cohort = load_snapshot(source)["cohort"]
    atomic_text(output / "interventions.json", json.dumps(interventions(cohort), indent=2))
    print(json.dumps(review, indent=2))


if __name__ == "__main__":
    main()
