"""Outcome-blind controls and a frozen historical shortlist for Phase 18."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from toposc_lab.geometry import Geometry
from toposc_lab.research.descriptors import compute_descriptors
from toposc_lab.research.space import (
    FixedConnectivitySpace,
    geometry_from_payload,
    geometry_to_payload,
)
from toposc_lab.research.storage import ResearchStore, dumps
from toposc_lab.research.validation_diagnostics import boundary_distance

SPACE: dict[str, Any] = {
    "side": 10,
    "min_degree": 2,
    "max_degree": 6,
    "max_bond_length": 2.0,
    "bond_tolerance": 0.0,
    "forbid_crossings": True,
    "minimum_distance": 0.002,
    "site_crossings": "unconnected",
    "initialization_rewires": [5, 10, 20, 30, 40],
}


def digest(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def structural_descriptors(geometry: Geometry) -> dict[str, Any]:
    """Reuse existing metrics; add explicitly geometric cuts and bridge detection."""
    result: dict[str, Any] = compute_descriptors(geometry)
    coordinates = geometry.coordinates
    assert coordinates is not None
    n = geometry.n_sites
    adjacency: list[list[int]] = [[] for _ in range(n)]
    edges = [(e.source, e.target) for e in geometry.edges]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    degree = np.array([len(a) for a in adjacency])
    seen, low, clock, bridges = [-1] * n, [0] * n, [0], []

    def visit(u: int, parent: int) -> None:
        seen[u] = low[u] = clock[0]
        clock[0] += 1
        for v in adjacency[u]:
            if v == parent:
                continue
            if seen[v] < 0:
                visit(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > seen[u]:
                    bridges.append([min(u, v), max(u, v)])
            else:
                low[u] = min(low[u], seen[v])

    for u in range(n):
        if seen[u] < 0:
            visit(u, -1)
    boundary = boundary_distance(coordinates) < 1
    cuts = []
    side = round(n**0.5)
    for axis in (0, 1):
        for cut in np.arange(0.5, side - 1, 1):
            left = coordinates[:, axis] < cut
            fraction = float(left.mean())
            if 0.25 <= fraction <= 0.75:
                count = sum(bool(left[a]) != bool(left[b]) for a, b in edges)
                cuts.append(
                    {
                        "axis": axis,
                        "position": float(cut),
                        "edges": count,
                        "conductance": count / min(degree[left].sum(), degree[~left].sum()),
                    }
                )
    result.update(
        bridge_count=len(bridges),
        bridges=sorted(bridges),
        degree_histogram={str(k): int(np.sum(degree == k)) for k in sorted(set(degree))},
        boundary_weak_fraction=float(np.mean(degree[boundary] <= 2)),
        bulk_weak_fraction=float(np.mean(degree[~boundary] <= 2)),
        balanced_cut_min_edges=min(c["edges"] for c in cuts),
        balanced_cut_min_conductance=float(min(c["conductance"] for c in cuts)),
        balanced_axis_cuts=cuts,
    )
    for name, region in (("boundary", boundary), ("bulk", ~boundary)):
        vectors = np.array(
            [coordinates[b] - coordinates[a] for a, b in edges if region[a] and region[b]]
        )
        if len(vectors):
            lengths = np.linalg.norm(vectors, axis=1)
            directions = vectors / lengths[:, None]
            spectrum = np.linalg.eigvalsh(directions.T @ directions / len(vectors))
            result[name + "_anisotropy"] = float(spectrum[-1] - spectrum[0])
            result[name + "_long_bond_fraction"] = float(np.mean(lengths > 2**0.5 + 1e-12))
        else:
            result[name + "_anisotropy"] = None
            result[name + "_long_bond_fraction"] = None
    return result


def freeze_cohort(historical: Path, *, pool_size: int = 96, seed: int = 180001) -> dict[str, Any]:
    """No evaluator is called; only historical outcomes and new geometry are read."""
    if pool_size < 12:
        raise ValueError("Control pool needs at least twelve proposals")
    space = FixedConnectivitySpace(**SPACE)
    reference = space.reference()
    candidates: list[dict[str, Any]] = []

    def add(identity: str, geometry: Geometry, source: dict[str, Any]) -> None:
        reasons = space.validate(geometry)
        if reasons:
            raise ValueError(f"Invalid cohort member {identity}: {reasons}")
        candidates.append(
            {
                "id": identity,
                "geometry": geometry_to_payload(geometry),
                "symmetry_hash": space.hash(geometry),
                "source": source,
                "descriptors": structural_descriptors(geometry),
            }
        )

    add("regular", reference, {"kind": "reference"})
    historical_info: dict[str, Any] = {"directory": str(historical), "available": False}
    if (historical / "research.sqlite3").exists():
        store = ResearchStore(historical)
        rows = []
        with store.connect(readonly=True) as db:
            db.execute("BEGIN")
            manifest_row = db.execute(
                "SELECT * FROM objects WHERE kind='manifest' AND id='current'"
            ).fetchone()
            manifest = store.decode(manifest_row)
            for row in db.execute(
                "SELECT * FROM objects WHERE kind='candidate' "
                "AND json_extract(payload,'$.observed')=1 "
                "AND json_extract(payload,'$.score') IS NOT NULL ORDER BY id"
            ):
                c = store.decode(row)
                g = geometry_from_payload(c["geometry"])
                if space.validate(g):
                    continue
                rows.append(
                    {
                        "id": c["id"],
                        "geometry": g,
                        "baseline": c["baseline"],
                        "score": c["score"],
                        "boundary": c["raw_metrics"]["boundary_weight"],
                        "checksum": row["checksum"],
                    }
                )
        regular = next((r for r in rows if r["id"] == "baseline-regular"), None)
        searched = [r for r in rows if not r["baseline"]]
        historical_info = {
            "directory": str(historical),
            "available": True,
            "experiment_id": manifest["experiment_id"],
            "source_sha256": manifest["source_sha256"],
            "eligible_complete_records": len(rows),
        }
        selected = []
        if searched and regular:
            best = min(searched, key=lambda r: (-r["score"], r["id"]))
            selected.append(("historical_best", best))
            edge_preserving = [
                r
                for r in searched
                if r["score"] > regular["score"] and r["boundary"] >= 0.97 and r["id"] != best["id"]
            ]
            if edge_preserving:
                second = min(edge_preserving, key=lambda r: (-r["boundary"], r["id"]))
                selected.append(("historical_boundary", second))
            high = [
                r
                for r in searched
                if r["score"] >= 0.95 * best["score"]
                and r["id"] not in {s["id"] for _, s in selected}
            ]
            if high:
                third = min(
                    high,
                    key=lambda r: (
                        -min(space.distance(r["geometry"], s["geometry"]) for _, s in selected),
                        r["id"],
                    ),
                )
                selected.append(("historical_diverse", third))
            for label, row in selected:
                add(
                    label,
                    row["geometry"],
                    {
                        "kind": "historical",
                        "candidate_id": row["id"],
                        "payload_sha256": row["checksum"],
                        "old_quality": row["score"],
                        "old_boundary": row["boundary"],
                    },
                )

    rng = np.random.default_rng(seed)
    pool: dict[str, Geometry] = {}
    for _ in range(pool_size):
        geometry, _ = space.sample_with_metadata(rng)
        if space.validate(geometry):
            raise ValueError("Control generator produced invalid geometry")
        pool.setdefault(space.hash(geometry), geometry)
    available = [
        g for key, g in sorted(pool.items()) if key not in {c["symmetry_hash"] for c in candidates}
    ]
    while len(candidates) < 4:
        if not available:
            raise ValueError("Insufficient distinct fallback candidates")
        add(
            f"replacement_{len(candidates)}",
            available.pop(0),
            {"kind": "structural_replacement", "reason": "Historical role missing or unsuitable"},
        )
    evaluated = [(g, structural_descriptors(g)) for g in available]
    pairs = []
    for feature, label in (
        ("balanced_cut_min_conductance", "cut"),
        ("bulk_degree_variance", "degree"),
        ("long_bond_fraction", "long"),
    ):
        evaluated.sort(key=lambda item: (item[1][feature], space.hash(item[0])))
        low_g, low_d = evaluated.pop(0)
        high_g, high_d = evaluated.pop(-1)
        add(
            label + "_low",
            low_g,
            {"kind": "structural_control", "feature": feature, "extreme": "min"},
        )
        add(
            label + "_high",
            high_g,
            {"kind": "structural_control", "feature": feature, "extreme": "max"},
        )
        a, b = set(space.edges(low_g)), set(space.edges(high_g))
        differences = {
            k: high_d[k] - low_d[k]
            for k in low_d
            if isinstance(low_d[k], (int, float)) and isinstance(high_d[k], (int, float))
        }
        pairs.append(
            {
                "low": label + "_low",
                "high": label + "_high",
                "feature": feature,
                "descriptor_differences_high_minus_low": differences,
                "remove_from_low": sorted(a - b),
                "add_to_low": sorted(b - a),
                "reverse_intervention": "Exchange remove and add to recover low exactly",
                "causal_claim": False,
            }
        )
    distances = [
        {
            "first": a["id"],
            "second": b["id"],
            "D4_edge_jaccard": space.distance(
                geometry_from_payload(a["geometry"]), geometry_from_payload(b["geometry"])
            ),
        }
        for i, a in enumerate(candidates)
        for b in candidates[i + 1 :]
    ]
    cohort = {
        "version": "phase18.cohort.v1",
        "space": SPACE,
        "historical": historical_info,
        "geometry_seed": seed,
        "pool_proposals": pool_size,
        "pool_unique": len(pool),
        "candidates": candidates,
        "control_pairs": pairs,
        "pairwise_distances": distances,
        "new_exact_calls": 0,
        "selection_protocol": "docs/decisions/phase18_validation_protocol.md",
    }
    return {**cohort, "sha256": digest(cohort)}


def validate_cohort(cohort: dict[str, Any]) -> None:
    if cohort.get("version") != "phase18.cohort.v1":
        raise ValueError("Unsupported cohort")
    if digest({k: v for k, v in cohort.items() if k != "sha256"}) != cohort["sha256"]:
        raise ValueError("Cohort checksum mismatch")
    space = FixedConnectivitySpace(**cohort["space"])
    seen = set()
    seen_geometries = set()
    for c in cohort["candidates"]:
        geometry = geometry_from_payload(c["geometry"])
        if c["id"] in seen or space.validate(geometry):
            raise ValueError("Duplicate identity or invalid cohort geometry")
        seen.add(c["id"])
        if space.hash(geometry) != c["symmetry_hash"]:
            raise ValueError("Cohort geometry identity mismatch")
        if c["symmetry_hash"] in seen_geometries:
            raise ValueError("Symmetry-duplicate cohort geometries")
        seen_geometries.add(c["symmetry_hash"])
        # Preview fields must agree with the authoritative serialized geometry.
        rebuilt = geometry_to_payload(geometry)
        for key in ("coordinates", "edges", "n_sites", "boundary_sites"):
            if c["geometry"][key] != rebuilt[key]:
                raise ValueError("Cohort geometry preview mismatch")
    if "regular" not in seen:
        raise ValueError("Regular reference required")
