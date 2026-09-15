"""Predeclared Phase-16B matching and parent-level falsification statistics."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from toposc_lab.generative.space import GeometrySearchSpace, embedded_edges
from toposc_lab.geometry import Geometry
from toposc_lab.patterns.features import adjacency, motif_instances
from toposc_lab.patterns.interventions import MOTIFS, EdgeSwap, motif_counts
from toposc_lab.patterns.statistics import grouped_mean_interval, holm, sign_flip_pvalue

HYPOTHESES = ("h1", "h2", "h3", "h4", "h5")


def edge_mask(edges: list[list[int]] | tuple[tuple[int, int], ...]) -> int:
    """Canonical fixed-grid edge bitset, including explicit validation."""
    value = 0
    for edge in edges:
        if len(edge) != 2 or any(type(i) is not int or not 0 <= i < 36 for i in edge):
            raise ValueError("expected two integer lattice endpoints")
        a, b = sorted(edge)
        if a == b:
            raise ValueError("self edge")
        bit = 1 << (36 * a + b)
        if value & bit:
            raise ValueError("duplicate edge")
        value |= bit
    return value


def orbit_masks(edges: list[list[int]] | tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    edge_mask(edges)
    masks = []
    for mirror in (False, True):
        for turns in range(4):
            labels = []
            for i in range(36):
                x, y = i % 6, i // 6
                if mirror:
                    x = 5 - x
                for _ in range(turns):
                    x, y = 5 - y, x
                labels.append(y * 6 + x)
            masks.append(edge_mask(tuple((labels[a], labels[b]) for a, b in edges)))
    return tuple(sorted(set(masks)))


class ExclusionIndex:
    """Exact D4 Jaccard check using integer bitsets; no approximate neighbor search."""

    def __init__(self, edge_sets: list[list[list[int]]]):
        self.masks: set[int] = set()
        for edges in edge_sets:
            self.add(edges)

    def add(self, edges: list[list[int]]) -> None:
        self.masks.update(orbit_masks(edges))

    def distance(self, edges: list[list[int]]) -> float:
        mask = edge_mask(edges)
        return min(
            (
                (mask ^ other).bit_count() / max(1, (mask | other).bit_count())
                for other in self.masks
            ),
            default=1.0,
        )


def swap_description(geometry: Geometry, swap: EdgeSwap) -> dict[str, Any]:
    changed = swap.apply(geometry)
    before, after = motif_counts(geometry), motif_counts(changed)
    triangles = set(motif_instances(adjacency(geometry))[0])
    new_triangles = set(motif_instances(adjacency(changed))[0])
    boundary = geometry.boundary_sites
    lost, gained = triangles - new_triangles, new_triangles - triangles
    vertices = sorted({i for e in swap.removed for i in e})
    xy = GeometrySearchSpace().coordinates[vertices]
    return {
        "edges": [list(e) for e in embedded_edges(changed)],
        "removed": [list(e) for e in swap.removed],
        "added": [list(e) for e in swap.added],
        "delta": {k: after[k] - before[k] for k in (*MOTIFS, "diagonal_count")},
        "lost_boundary_triangles": sum(bool(set(t) & boundary) for t in lost),
        "lost_bulk_triangles": sum(not bool(set(t) & boundary) for t in lost),
        "gained_triangles": len(gained),
        "centroid": xy.mean(axis=0).tolist(),
        "radius": float(np.linalg.norm(xy - (2.5, 2.5), axis=1).mean()),
    }


def pair_eligible(hypothesis: str, a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Arm ordering: H1 decrease/increase; H2 boundary/bulk; H4/5 decrease/increase."""
    da, db = a["delta"], b["delta"]
    keys = {*MOTIFS, "diagonal_count"}
    targets = {
        "h1": "boundary_bulk_edges",
        "h4": "chordless_square_count",
        "h5": "central_diagonal_count",
    }
    if hypothesis in targets:
        key = targets[hypothesis]
        if not da[key] < 0 < db[key]:
            return False
        keys.remove(key)
        if hypothesis == "h5" and (da["diagonal_count"] != 0 or db["diagonal_count"] != 0):
            return False
    elif hypothesis == "h2":
        if not (
            da["triangle_count"] == db["triangle_count"] == -1
            and a["lost_boundary_triangles"] == b["lost_bulk_triangles"] == 1
            and a["lost_bulk_triangles"] == b["lost_boundary_triangles"] == 0
            and a["gained_triangles"] == b["gained_triangles"] == 0
        ):
            return False
    else:
        raise ValueError("unsupported swap hypothesis")
    return all(da[k] == db[k] for k in keys)


def select_pair(hypothesis: str, edits: list[dict[str, Any]], seed: int) -> dict[str, Any] | None:
    matches = []
    for first, second in combinations(range(len(edits)), 2):
        for i, j in ((first, second), (second, first)):
            a, b = edits[i], edits[j]
            if pair_eligible(hypothesis, a, b):
                cost = float(
                    np.sum((np.array(a["centroid"]) - b["centroid"]) ** 2)
                    + (a["radius"] - b["radius"]) ** 2
                )
                matches.append((cost, i, j))
    if not matches:
        return None
    minimum = min(row[0] for row in matches)
    ties = [row for row in matches if row[0] == minimum]
    cost, i, j = ties[int(np.random.default_rng(seed).integers(len(ties)))]
    return {"a": edits[i], "b": edits[j], "spatial_cost": cost, "eligible_pairs": len(matches)}


def summarize(effects: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Five fixed tests, one observation per independent seed, no edit pseudoreplication."""
    tests: dict[str, Any] = {}
    for h in HYPOTHESES:
        rows = effects[h]
        seeds = np.array([row["seed"] for row in rows])
        if len(set(seeds)) != len(rows):
            raise ValueError("one effect per independent seed required")
        values = np.array([row["effect"] for row in rows], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("finite effects required")
        centered = values - 0.75 if h == "h3" else values
        tests[h] = {
            "available_parents": len(rows),
            "rows": rows,
            "summary": grouped_mean_interval(values, seeds, 16299) if rows else None,
            "positive_parents": int(np.sum(centered > 0)),
            "p": sign_flip_pvalue(centered) if len(rows) >= 8 else 1.0,
            "status": "tested" if len(rows) >= 8 else "insufficient_matched_parents",
        }
    for h, adjusted in zip(HYPOTHESES, holm([tests[h]["p"] for h in HYPOTHESES]), strict=True):
        t = tests[h]
        t["holm_p"] = adjusted
        t["supported"] = False
        if t["status"] != "tested" or adjusted > 0.05:
            continue
        mean = t["summary"]["mean"]
        rows = t["rows"]
        if h == "h3":
            t["supported"] = mean > 0.75
        elif h in ("h1", "h2"):
            sensitivity = all(
                np.mean([r["sensitivity"][m] for r in rows]) > 0 for m in ("1.9", "2.1")
            )
            reciprocal = h != "h1" or (
                np.mean([r["a_minus_parent"] for r in rows]) > 0
                and np.mean([r["b_minus_parent"] for r in rows]) < 0
            )
            t["supported"] = bool(
                mean > 0
                and t["positive_parents"] >= 0.75 * len(rows)
                and sensitivity
                and reciprocal
            )
        else:
            t["status"] = "significant_effect_challenges_provisional_explanation"
    return tests
