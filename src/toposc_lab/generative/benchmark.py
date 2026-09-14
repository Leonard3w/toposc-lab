"""Frozen equal-exact-budget generator gate; fresh outputs, paid labels and full ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np

from toposc_lab.active_learning.benchmark import source_provenance
from toposc_lab.active_learning.cycle import derived_seed
from toposc_lab.data import DatasetRecord, ExactPhysicsDataset, record_to_dict, save_dataset
from toposc_lab.generative.generators import METHODS, Evidence, GeometryGenerator, ProposalAudit
from toposc_lab.generative.physics import PROTOCOL_ID, ExactGeometryEvaluator, candidate
from toposc_lab.generative.space import GeometrySearchSpace, structural_distance, structural_key
from toposc_lab.geometry import Geometry, extract_geometry_descriptors
from toposc_lab.ml import FeatureOODDetector, HandcraftedFeatureSchema, extract_handcrafted_features
from toposc_lab.ml.features import FeatureInput

GATE_SEEDS = tuple(range(14101, 14121))
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")

    def encode(item: object) -> dict[str, Any]:
        if isinstance(item, Mapping):
            return dict(item)
        raise TypeError(f"unsupported JSON value: {type(item).__name__}")

    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=encode), encoding="utf-8"
    )
    temporary.replace(path)


def _metrics(record: DatasetRecord) -> dict[str, Any]:
    return dict(record.observables[0].values)


def summarize(
    records: Sequence[DatasetRecord],
    timings: Sequence[float],
    wall: float,
    audit: ProposalAudit,
    warm_count: int = 4,
) -> dict[str, Any]:
    geometries = [r.geometry.to_geometry() for r in records]
    metrics = [_metrics(r) for r in records]
    qualities = [float(m["quality"]) for m in metrics]
    success = [m["success"] == 1 for m in metrics]
    distances = [
        structural_distance(a, b) for i, a in enumerate(geometries) for b in geometries[i + 1 :]
    ]
    novelty = [
        min(structural_distance(g, w) for w in geometries[:warm_count])
        for g in geometries[warm_count:]
    ]
    schema = HandcraftedFeatureSchema.fit(records[:warm_count])
    features = extract_handcrafted_features(records, schema=schema).values
    ood = (
        FeatureOODDetector(flag_extrapolation=True)
        .fit(features[:warm_count])
        .assess(features, record_ids=tuple(r.record_id for r in records))
    )
    later = np.arange(len(records)) >= warm_count
    descriptors = [extract_geometry_descriptors(g) for g in geometries]
    cells = {
        (
            int(cast(int, d["triangle_count"])) // 4,
            int(cast(int, d["maximum_finite_shortest_path_length"])),
        )
        for d in descriptors
    }
    best = max(range(len(records)), key=lambda i: qualities[i])
    alternate = max(range(len(records)), key=lambda i: metrics[i]["alternate_quality"])
    return {
        "exact_calls": len(records),
        "best_quality": qualities[best],
        "best_record_id": records[best].record_id,
        "best_candidate_id": candidate(geometries[best]).candidate_id,
        "best_metrics": metrics[best],
        "qualities": qualities,
        "best_so_far": np.maximum.accumulate(qualities).tolist(),
        "success_rate": float(np.mean(success)),
        "any_success": any(success),
        "first_hit": next((i + 1 for i, hit in enumerate(success) if hit), None),
        "warm_improvement": qualities[best] - max(qualities[:warm_count]),
        "alternate_winner_same": best == alternate,
        "alternate_best": metrics[alternate]["alternate_quality"],
        "mean_pairwise_distance": float(np.mean(distances)),
        "minimum_pairwise_distance": min(distances),
        "mean_novelty_to_warm": float(np.mean(novelty)),
        "descriptor_cells": len(cells),
        "unique_structures": len({structural_key(g) for g in geometries}),
        "accepted_duplicate_rate": 1
        - len({structural_key(g) for g in geometries}) / len(geometries),
        "proposal_attempts": audit.attempted,
        "invalid_proposals": audit.invalid,
        "duplicate_proposals": audit.duplicate,
        "invalid_rate": audit.invalid / max(1, audit.attempted),
        "duplicate_rate": audit.duplicate / max(1, audit.attempted),
        "invalid_reasons": dict(audit.reasons),
        "ood_flags": ood.is_ood.tolist(),
        "ood_scores": ood.scores.tolist(),
        "ood_threshold": ood.threshold,
        "ood_rate_after_warm": float(np.mean(ood.is_ood[later])),
        "ood_quality_mean": float(np.mean(np.array(qualities)[ood.is_ood]))
        if any(ood.is_ood)
        else None,
        "id_quality_mean": float(np.mean(np.array(qualities)[~ood.is_ood])),
        "wall_seconds": wall,
        "exact_seconds": sum(timings),
        "matrix_dimension": 72,
        "localizer_dimension": 144,
        "cubic_work_proxy": len(records) * (4 * 72**3 + 3 * 144**3),
        "record_ids": [r.record_id for r in records],
        "pareto_record_ids": [
            records[i].record_id
            for i, m in enumerate(metrics)
            if not any(
                other["quality"] >= m["quality"]
                and other["boundary_weight"] >= m["boundary_weight"]
                and (
                    other["quality"] > m["quality"]
                    or other["boundary_weight"] > m["boundary_weight"]
                )
                for other in metrics
            )
        ],
    }


def paired_comparisons(trials: list[dict[str, Any]]) -> dict[str, Any]:
    rng = np.random.default_rng(14999)
    draws = rng.integers(0, len(trials), size=(50000, len(trials)))
    comparisons: dict[str, Any] = {}
    for method in ("patch", "coverage", "active"):
        comparisons[method] = {}
        for baseline in ("random", "evolution"):
            differences = np.array(
                [t[method]["best_quality"] - t[baseline]["best_quality"] for t in trials]
            )
            means = np.mean(differences[draws], axis=1)
            interval = np.quantile(means, [0.05 / 12, 1 - 0.05 / 12]).tolist()
            wins = int(np.sum(differences > 1e-10))
            comparisons[method][baseline] = {
                "mean_difference": float(np.mean(differences)),
                "simultaneous_interval": interval,
                "wins": wins,
                "ties": int(np.sum(np.abs(differences) <= 1e-10)),
                "advantage": bool(len(trials) == 20 and interval[0] > 0 and wins >= 16),
            }
        comparisons[method]["reproducible_advantage"] = all(
            comparisons[method][b]["advantage"] for b in ("random", "evolution")
        )
    return comparisons


def run_benchmark(
    destination: Path,
    *,
    seeds: tuple[int, ...] = GATE_SEEDS,
    budget: int = 24,
    references: bool = True,
) -> dict[str, Any]:
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("distinct nonnegative seeds required")
    if budget < 8 or budget % 4:
        raise ValueError("budget must be >=8 and divisible by four")
    if any(os.environ.get(name) != "1" for name in THREADS):
        raise ValueError("set all numerical thread variables to 1 before starting Python")
    root = Path(__file__).resolve().parents[3]
    base = source_provenance(root)
    protocol_path = root / "docs/decisions/phase_14_protocol.md"
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    provenance = replace(
        base,
        runtime={
            **dict(base.runtime),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "scipy": version("scipy"),
            "benchmark": PROTOCOL_ID,
            "protocol_sha256": protocol_hash,
            "threads": {n: os.environ[n] for n in THREADS},
        },
        solver_settings={
            **dict(base.solver_settings),
            "kappas": (0.1, 0.2, 0.3),
            "probe": (2.5, 2.5),
        },
    )
    destination.mkdir(parents=True, exist_ok=False)
    write_json(
        destination / "manifest.json",
        {
            "protocol": PROTOCOL_ID,
            "protocol_sha256": protocol_hash,
            "seeds": seeds,
            "budget": budget,
            "methods": METHODS,
            "source_sha256": base.runtime["source_sha256"],
            "git_commit": base.git_commit,
            "git_dirty": base.git_dirty,
            "timestamp_utc": base.timestamp_utc,
            "runtime": dict(provenance.runtime),
            "expected_exact_calls": len(seeds) * len(METHODS) * (budget + 3)
            + (4 if references else 0),
        },
    )
    with zipfile.ZipFile(destination / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted((root / "src").rglob("*.py")):
            archive.write(p, p.relative_to(root).as_posix())
        archive.write(protocol_path, protocol_path.relative_to(root).as_posix())
        for p in sorted((root / "tests").glob("test_generative*.py")):
            archive.write(p, p.relative_to(root).as_posix())
        audit_script = root / "scripts/phase_14_audit.py"
        archive.write(audit_script, audit_script.relative_to(root).as_posix())
    evaluator = ExactGeometryEvaluator(provenance)
    ledger_path = destination / "exact-attempts.jsonl"
    call_count = 0

    def exact(
        geometry: Geometry, seed: int, directory: Path, name: str, mu: float = 2.0
    ) -> tuple[Evidence, float]:
        nonlocal call_count
        proposal = candidate(geometry, mu)
        call_count += 1
        started = time.perf_counter()
        entry = {
            "call": call_count,
            "seed": seed,
            "candidate_id": proposal.candidate_id,
            "path": (directory / f"{name}.json").relative_to(destination).as_posix(),
        }
        with ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**entry, "status": "started"}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            record, run = evaluator.evaluate(proposal, seed)
        except Exception as error:
            with ledger_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**entry, "status": "failed", "error": str(error)}) + "\n")
            # Hard-stop retains paid failure and prevents an under-budget comparison.
            raise RuntimeError(f"exact attempt {call_count} failed; inspect ledger") from error
        elapsed = time.perf_counter() - started
        save_dataset(directory / f"{name}.json", ExactPhysicsDataset((record,)))
        with ledger_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        **entry,
                        "status": "complete",
                        "record_id": record.record_id,
                        "seconds": elapsed,
                    }
                )
                + "\n"
            )
        return Evidence(record, run), elapsed

    reference_results = []
    if references:
        space = GeometrySearchSpace()
        xy = space.coordinates
        grid = space.build(
            frozenset(e for e in space.edge_pool if np.linalg.norm(xy[e[0]] - xy[e[1]]) == 1)
        )
        for mu in (2.0, 8.0):
            pair = [
                exact(grid, 14900 + i, destination, f"reference-{mu}-{i}", mu)[0].record
                for i in range(2)
            ]
            if _metrics(pair[0])["eligible"] != float(mu == 2.0) or not np.allclose(
                pair[0].spectrum.eigenvalues, pair[1].spectrum.eigenvalues, atol=1e-10, rtol=0
            ):
                raise RuntimeError("physics reference gate failed")
            reference_results.append({"mu": mu, "metrics": _metrics(pair[0]), "agrees": True})
    trials = []
    for seed in seeds:
        trial: dict[str, Any] = {"seed": seed}
        warm_generator = GeometryGenerator("random")
        warm = warm_generator.propose((), seed=derived_seed(seed, 0, 140))
        for method in METHODS:
            directory = destination / str(seed) / method
            directory.mkdir(parents=True)
            generator = GeometryGenerator(method)
            # Include the same warm proposal accounting independently in every arm.
            generator.audit.attempted = warm_generator.audit.attempted
            generator.audit.invalid = warm_generator.audit.invalid
            generator.audit.duplicate = warm_generator.audit.duplicate
            generator.audit.reasons.update(warm_generator.audit.reasons)
            generator.audit.events.extend(warm_generator.audit.events)
            history: list[Evidence] = []
            timings = []
            started = time.perf_counter()
            for batch in range(budget // 4):
                geometries = (
                    warm
                    if batch == 0
                    else generator.propose(tuple(history), seed=derived_seed(seed, batch, 140))
                )
                if history:
                    warm_records = tuple(e.record for e in history[:4])
                    schema = HandcraftedFeatureSchema.fit(warm_records)
                    training = extract_handcrafted_features(warm_records, schema=schema).values
                    proposals = tuple(candidate(g) for g in geometries)
                    inputs = tuple(
                        FeatureInput(c.candidate_id, c.geometry, c.model) for c in proposals
                    )
                    matrix = extract_handcrafted_features(inputs, schema=schema)
                    assessment = (
                        FeatureOODDetector(flag_extrapolation=True)
                        .fit(training)
                        .assess(matrix.values, record_ids=matrix.record_ids)
                    )
                    # Persist before exact evaluation; a warning cannot reject valid physics.
                    write_json(
                        directory / f"selected-ood-{batch}.json",
                        {
                            "candidate_ids": list(matrix.record_ids),
                            "is_ood": assessment.is_ood.tolist(),
                            "scores": assessment.scores.tolist(),
                            "threshold": assessment.threshold,
                            "reference_record_ids": [r.record_id for r in warm_records],
                            "caution": assessment.caution,
                            "stage": "before_exact",
                        },
                    )
                for g in geometries:
                    item, elapsed = exact(
                        g,
                        derived_seed(seed, len(history), 141),
                        directory,
                        f"search-{len(history):02d}",
                    )
                    history.append(item)
                    timings.append(elapsed)
            records = [e.record for e in history]
            summary = summarize(records, timings, time.perf_counter() - started, generator.audit)
            winner = max(records, key=lambda r: _metrics(r)["quality"])
            confirmed = [
                exact(
                    winner.geometry.to_geometry(),
                    derived_seed(seed, i, 142),
                    directory,
                    f"confirmation-{i}",
                    mu,
                )[0].record
                for i, mu in enumerate((2.0, 1.9, 2.1))
            ]
            agrees = bool(
                np.allclose(
                    winner.spectrum.eigenvalues,
                    confirmed[0].spectrum.eigenvalues,
                    atol=1e-10,
                    rtol=0,
                )
                and abs(_metrics(winner)["quality"] - _metrics(confirmed[0])["quality"]) <= 1e-10
            )
            if not agrees:
                raise RuntimeError("winner exact revalidation disagreement")
            summary["confirmation"] = {
                "agrees": agrees,
                "metrics": [_metrics(r) for r in confirmed],
                "record_ids": [r.record_id for r in confirmed],
                "parameter_success_rate": float(
                    np.mean([_metrics(r)["success"] for r in confirmed[1:]])
                ),
            }
            write_json(
                directory / "proposal-audit.json",
                {"events": generator.audit.events, "predictions": generator.audit.predictions},
            )
            write_json(directory / "summary.json", summary)
            trial[method] = summary
            print(
                f"seed {seed} {method}: best={summary['best_quality']:.6f} "
                f"success={summary['success_rate']:.3f} seconds={summary['wall_seconds']:.1f}",
                flush=True,
            )
        trials.append(trial)
        write_json(
            destination / "progress.json",
            {"completed_seeds": [t["seed"] for t in trials], "exact_calls": call_count},
        )
    report = {
        "protocol": PROTOCOL_ID,
        "seeds": seeds,
        "budget": budget,
        "references": reference_results,
        "trials": trials,
        "comparisons": paired_comparisons(trials),
        "exact_calls": call_count,
        "search_exact_calls": len(seeds) * len(METHODS) * budget,
        "confirmation_exact_calls": len(seeds) * len(METHODS) * 3,
        "source_sha256": base.runtime["source_sha256"],
        "protocol_sha256": protocol_hash,
    }
    write_json(destination / "report.json", report)
    inventory = {
        p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in destination.rglob("*")
        if p.is_file()
    }
    write_json(destination / "inventory.json", inventory)
    return report


def scientific_record(record: DatasetRecord) -> dict[str, Any]:
    """Strip volatile provenance solely for independent reproduction comparisons."""
    payload = record_to_dict(record)
    payload.pop("record_id")
    payload.pop("provenance")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("development", "gate", "repeat"), required=True)
    args = parser.parse_args()
    seeds = (
        (14001, 14002)
        if args.mode == "development"
        else ((14101,) if args.mode == "repeat" else GATE_SEEDS)
    )
    run_benchmark(args.output, seeds=seeds, references=args.mode != "repeat")


if __name__ == "__main__":
    main()
