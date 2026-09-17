"""Seeded geometry-only reachability audit; never runs exact physics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from toposc_lab.research.config import ExperimentConfig
from toposc_lab.research.descriptors import compute_descriptors
from toposc_lab.research.space import FixedConnectivitySpace


def sample_ranges(seed: int = 17102, per_band: int = 20) -> dict:
    rows = []
    rng = np.random.default_rng(seed)
    space = FixedConnectivitySpace(**{**ExperimentConfig().space,
                                     "max_bond_length": 2.0, "site_crossings": "unconnected"})
    for band in (5, 10, 20, 30, 40):
        seeded = FixedConnectivitySpace(**{**ExperimentConfig().space,
                                          "max_bond_length": 2.0,
                                          "site_crossings": "unconnected",
                                          "initialization_rewires": (band,)})
        for sample in range(per_band):
            geometry = seeded.sample(rng)
            for depth in range(11):
                reasons = space.validate(geometry)
                if reasons:
                    raise ValueError(reasons)
                descriptors = compute_descriptors(geometry)
                rows.append({"band": band, "sample": sample, "depth": depth,
                             **{key: descriptors[key] for key in (
                                 "regular_edge_distance", "long_bond_fraction", "mean_bond_length")}})
                if depth < 10:
                    geometry, _ = space.mutate(geometry, rng, "multi_rewire_explore")
    return {"seed": seed, "per_band": per_band, "exact_attempts": 0,
            "validated_geometries": len(rows), "rows": rows,
            "ranges": {key: [min(row[key] for row in rows), max(row[key] for row in rows)]
                       for key in ("regular_edge_distance", "long_bond_fraction", "mean_bond_length")},
            "scope": "Seeded reachable examples; not extrema or scientific evidence."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-band", type=int, default=20)
    args = parser.parse_args()
    if args.per_band < 1:
        parser.error("per-band must be positive")
    result = sample_ranges(per_band=args.per_band)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
