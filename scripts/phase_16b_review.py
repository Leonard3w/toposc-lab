"""Independent read-only mapping, intervention and preservation review for Phase 16B."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from toposc_lab.data import record_from_dict
from toposc_lab.discovery.storage import atomic_json, read_json
from toposc_lab.generative.space import embedded_edges
from toposc_lab.patterns.campaign import build, digest, indices, metrics
from toposc_lab.patterns.interventions import MOTIFS, EdgeSwap, deletion_reasons, motif_counts
from toposc_lab.patterns.validation import pair_eligible, swap_description


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = root / "results/phase16b"
    plan = read_json(directory / "plan.json")
    results = read_json(directory / "results.json")
    rows = {r["seed"]: r for r in results["rows"]}
    assert [p["seed"] for p in plan["parents"]] == list(range(16201, 16217))
    role_count = 0
    swap_count = 0
    for parent in plan["parents"]:
        g = build(parent["edges"])
        row = rows[parent["seed"]]
        roles = [(row["parent"], parent["edges"])]
        for h, pair in parent["pairs"].items():
            if pair is None:
                assert h not in row["pairs"]
                continue
            rebuilt = {}
            for arm in ("a", "b"):
                edit = pair[arm]
                after = motif_counts(build(edit["edges"]))
                swap = EdgeSwap(
                    tuple(tuple(e) for e in edit["removed"]),
                    tuple(tuple(e) for e in edit["added"]),
                    tuple(after[k] for k in (*MOTIFS, "diagonal_count")),
                )
                actual = swap_description(g, swap)
                assert all(actual[k] == v for k, v in edit.items() if k != "distance")
                rebuilt[arm] = actual
                roles.append((row["pairs"][h][arm], edit["edges"]))
                swap_count += 1
            assert pair_eligible(h, rebuilt["a"], rebuilt["b"])
        for actual, deletion in zip(row["deletions"], parent["deletions"], strict=True):
            edges = {tuple(e) for e in parent["edges"]} - {tuple(deletion["removed"])}
            assert edges == {tuple(e) for e in deletion["edges"]}
            assert not deletion_reasons(build(deletion["edges"]))
            roles.append((actual, deletion["edges"]))
        for role, edges in roles:
            raw = read_json(directory / "exact" / (role["stage"] + ".json"))
            record = record_from_dict(raw["record"])
            assert list(embedded_edges(record.geometry.to_geometry())) == [tuple(e) for e in edges]
            assert metrics(record) == role["metrics"] and indices(record) == role["indices"]
            assert record.record_id == role["record_id"]
            for mu, sensitivity in role["sensitivity"].items():
                other = record_from_dict(
                    read_json(
                        directory / "exact" / (role["stage"] + f"-mu{int(float(mu) * 10)}.json")
                    )["record"]
                )
                chemical_potential = other.model.parameters["chemical_potential"]
                assert isinstance(chemical_potential, (int, float))
                assert not isinstance(chemical_potential, bool)
                assert float(chemical_potential) == float(mu)
                assert metrics(other) == sensitivity["metrics"]
                assert indices(other) == sensitivity["indices"]
                assert other.geometry.exact_id == record.geometry.exact_id
            role_count += 1
    old = root / "results/phase16a"
    inventory = read_json(old / "inventory.json")
    assert all(digest(old / k) == v for k, v in inventory.items())
    retained: list[bool] = []
    for row in results["rows"]:
        q = row["parent"]["metrics"]["quality"]
        retained.extend(
            q > 0
            and d["metrics"]["quality"] >= 0.9 * q
            and bool(d["metrics"]["eligible"])
            and d["indices"] == row["parent"]["indices"]
            for d in row["deletions"]
        )
    report = {
        "status": "PASS",
        "mapped_roles": role_count,
        "checked_swaps": swap_count,
        "deletions": len(retained),
        "retained": sum(retained),
        "retention_fraction": float(np.mean(retained)),
        "phase16a_files_preserved": len(inventory),
        "new_numerical_calls": 0,
        "review_driver_sha256": digest(Path(__file__)),
    }
    atomic_json(root / "results/phase16b-independent-review.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
