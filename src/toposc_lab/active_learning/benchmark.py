"""Predeclared, matched exact-budget comparison with the existing Phase-10 loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from toposc_lab.active_learning.acquisition import AcquisitionConfig
from toposc_lab.active_learning.campaign import CampaignConfig, run_campaign
from toposc_lab.active_learning.cycle import CycleConfig, derived_seed
from toposc_lab.active_learning.pool import Candidate, CandidateSpace, generate_candidate_pool
from toposc_lab.active_learning.reference import (
    REFERENCE_ID,
    KitaevReferenceEvaluator,
    quality,
    reference_candidate,
)
from toposc_lab.active_learning.training import TrainingConfig, retrain
from toposc_lab.data import ExactPhysicsDataset, ReproducibilityMetadata, save_dataset
from toposc_lab.evaluation import ObjectiveDirection
from toposc_lab.ml import RegressionTargetDefinition, RegressionTargetSource
from toposc_lab.search import (
    ElitismConfig,
    GenerationLoopConfig,
    LexicographicFitnessDefinition,
    MutationValidityPolicy,
    OffspringProposal,
    TournamentSelectionConfig,
    create_initial_population,
    geometry_to_genome,
    run_generation_loop,
)

GATE_SEEDS = (13101, 13102, 13103, 13104, 13105)


def reference_training_config() -> TrainingConfig:
    return TrainingConfig(
        RegressionTargetDefinition(
            "finite_chain_quality",
            RegressionTargetSource.OBSERVABLE,
            "finite_chain_quality",
            "1",
            "quality",
        )
    )


def _metrics(record) -> dict:
    return dict(record.observables[0].values)


def _arm_summary(records, exact_seconds, wall_seconds) -> dict:
    scores = np.array([_metrics(r)["quality"] for r in records])
    best_index = int(np.argmax(scores))
    hits = np.flatnonzero(scores >= 0.02)
    return {
        "exact_simulations": len(records),
        "unique_candidates": len({Candidate(r.geometry, r.model).candidate_id for r in records}),
        "best_score": float(scores[best_index]),
        "median_score": float(np.median(scores)),
        "best_so_far": np.maximum.accumulate(scores).tolist(),
        "best_so_far_auc": float(np.mean(np.maximum.accumulate(scores))),
        "first_hit_0_02": None if not len(hits) else int(hits[0]) + 1,
        "winner_id": records[best_index].record_id,
        "winner_sites": records[best_index].geometry.to_geometry().n_sites,
        "winner_metrics": _metrics(records[best_index]),
        "dimension_cubed_work": sum(r.spectrum.basis_size**3 for r in records),
        "exact_seconds": exact_seconds,
        "wall_seconds": wall_seconds,
        "alternate_splitting_scale_best": {
            str(scale): max(quality(_metrics(r), splitting_scale=scale) for r in records)
            for scale in (0.005, 0.02)
        },
    }


def _paired_evidence(trials, baseline) -> dict:
    differences = np.array(
        [t["arms"]["active"]["best_score"] - t["arms"][baseline]["best_score"] for t in trials]
    )
    rng = np.random.default_rng(13000)
    samples = rng.choice(differences, size=(10000, len(differences))).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "differences": differences.tolist(),
        "mean": float(differences.mean()),
        "ci95": [float(low), float(high)],
        "positive_seeds": int(sum(differences > 1e-12)),
        "reproducible_advantage": bool(
            len(trials) >= 5
            and low > 1e-12
            and sum(differences > 1e-12) >= np.ceil(0.8 * len(trials))
        ),
    }


def run_discovery_benchmark(
    output: str | Path,
    *,
    provenance: ReproducibilityMetadata,
    seeds: tuple[int, ...] = GATE_SEEDS,
    cycle_config: CycleConfig | None = None,
) -> dict:
    """No adaptive tuning, cached labels, or seed replacement in this benchmark."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("benchmark seeds must be nonempty and unique")
    if cycle_config is None:
        cycle_config = CycleConfig(reference_training_config(), AcquisitionConfig(), 40)
    if cycle_config.acquisition.batch_size != 4:
        raise ValueError("frozen benchmark requires acquisition batch_size=4")
    if cycle_config.training.target != reference_training_config().target:
        raise ValueError("benchmark target must match the frozen exact quality")
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("benchmark output must be fresh; campaign resume is a separate API")
    destination.mkdir(parents=True, exist_ok=True)
    space = CandidateSpace(tuple(reference_candidate(n) for n in range(6, 66)))
    evaluator = KitaevReferenceEvaluator(provenance)
    training = cycle_config.training
    trials = []
    for seed in seeds:
        trial_path = destination / str(seed)
        initial_candidates = generate_candidate_pool(space, count=8, seed=seed)
        initial_ids = frozenset(c.candidate_id for c in initial_candidates)
        reserved = tuple(evaluator(reference_candidate(n), seed + 50000 + n) for n in (68, 70, 72))
        save_dataset(trial_path / "heldout.json", ExactPhysicsDataset(reserved))
        arm_records = {}
        arms = {}
        active_result = None
        for arm in ("active", "random", "evolutionary"):
            records = []
            timings = []
            started = time.perf_counter()

            def exact(candidate, exact_seed, timings=timings, records=records):
                before = time.perf_counter()
                record, run = evaluator.evaluate(candidate, exact_seed)
                timings.append(time.perf_counter() - before)
                records.append(record)
                return record, run

            if arm in ("active", "random"):
                for i, c in enumerate(initial_candidates):
                    exact(c, derived_seed(seed, 0, 90, i))
                initial = ExactPhysicsDataset(tuple(records))
                if arm == "active":
                    initial_surrogate = retrain(
                        initial, config=training, seed=seed, reserved_records=reserved
                    )
                    active_result = run_campaign(
                        space,
                        initial,
                        config=CampaignConfig(
                            cycle_config,
                            seed,
                            4,
                            16,
                            REFERENCE_ID,
                            provenance.runtime["source_sha256"],
                        ),
                        evaluator=lambda c, s: exact(c, s)[0],
                        checkpoint_path=trial_path / "active-checkpoint.json",
                        reserved_records=reserved,
                    )
                else:
                    pool = generate_candidate_pool(
                        space, count=16, seed=derived_seed(seed, 0, 91), excluded_ids=initial_ids
                    )
                    for i, c in enumerate(pool):
                        exact(c, derived_seed(seed, 1, 90, i))
            else:
                initial_population = create_initial_population(
                    tuple(geometry_to_genome(c.geometry.to_geometry()) for c in initial_candidates),
                    validity_policy=MutationValidityPolicy(
                        minimum_site_count=6, maximum_site_count=65
                    ),
                )

                def evaluate_member(member, seed=seed, exact=exact):
                    c = reference_candidate(member.genome.to_geometry().n_sites)
                    return exact(
                        c, derived_seed(seed, member.generation_index, 90, member.member_index)
                    )[1]

                def produce(request):
                    rng = np.random.default_rng(request.seed)
                    proposals = []
                    for i in range(request.required_offspring_count):
                        parent = i % len(request.selected_members)
                        n = (
                            request.selected_members[parent]
                            .population_member.genome.to_geometry()
                            .n_sites
                        )
                        if rng.random() < 0.2:
                            n = int(rng.integers(6, 66))
                        else:
                            step = int(rng.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]))
                            n = int(np.clip(n + step, 6, 65))
                        proposals.append(
                            OffspringProposal(
                                geometry_to_genome(reference_candidate(n).geometry.to_geometry()),
                                (parent,),
                                parent,
                                "phase13.bounded-chain-mutation.v1",
                                request.seed,
                            )
                        )
                    return tuple(proposals)

                run_generation_loop(
                    initial_population,
                    definition=LexicographicFitnessDefinition(
                        ("quality",), (ObjectiveDirection.MAXIMIZE,), REFERENCE_ID
                    ),
                    evaluator=evaluate_member,
                    config=GenerationLoopConfig(
                        2, TournamentSelectionConfig(8, 3), ElitismConfig(1)
                    ),
                    seed=seed,
                    offspring_producer=produce,
                    producer_identifier="phase13.bounded-chain-mutation.v1",
                )
            if len(records) != 24:
                raise RuntimeError(f"unmatched exact budget for {arm}: {len(records)}")
            arms[arm] = _arm_summary(records, sum(timings), time.perf_counter() - started)
            arm_records[arm] = tuple(records)
            # Evolution can evaluate identical slots with different provenance; preserve all calls.
            unique_records = {r.record_id: r for r in records}
            save_dataset(
                trial_path / f"{arm}.json", ExactPhysicsDataset(tuple(unique_records.values()))
            )
        revalidation = {}
        for arm, records in arm_records.items():
            winner = max(records, key=lambda r: _metrics(r)["quality"])
            n = winner.geometry.to_geometry().n_sites
            confirmation = tuple(
                evaluator(reference_candidate(n, mu), seed + 90000 + i)
                for i, mu in enumerate((1.8, 1.75, 1.85))
            )
            save_dataset(trial_path / f"{arm}-confirmation.json", ExactPhysicsDataset(confirmation))
            agrees = np.allclose(
                winner.spectrum.eigenvalues,
                confirmation[0].spectrum.eigenvalues,
                rtol=0,
                atol=1e-10,
            )
            agrees = agrees and all(
                abs(_metrics(winner)[k] - _metrics(confirmation[0])[k]) <= 1e-10
                for k in _metrics(winner)
            )
            revalidation[arm] = {
                "agrees": bool(agrees),
                "exact_simulations": 3,
                "mu_sensitivity_scores": [_metrics(r)["quality"] for r in confirmation],
            }
        final_surrogate = retrain(
            active_result.dataset, config=training, seed=seed, reserved_records=reserved
        )
        holdout_pool = tuple(Candidate(r.geometry, r.model) for r in reserved)
        truth = np.array([_metrics(r)["quality"] for r in reserved])
        initial_error = truth - np.array([p.value for p in initial_surrogate.predict(holdout_pool)])
        final_error = truth - np.array([p.value for p in final_surrogate.predict(holdout_pool)])
        by_candidate = {
            Candidate(r.geometry, r.model).candidate_id: r for r in arm_records["active"]
        }
        errors = {"ood": [], "in_distribution": []}
        strategies = {"exploration": 0, "exploitation": 0}
        relaxed_count = 0
        for history in active_result.history:
            predictions = {p["candidate_id"]: p for p in history["predictions"]}
            for selected in history["selected"]:
                p = predictions[selected["candidate_id"]]
                strategy = selected["strategy"]
                strategies[
                    "exploitation" if strategy.startswith("exploitation") else "exploration"
                ] += 1
                relaxed_count += int(strategy.endswith("diversity_relaxed"))
                r = by_candidate[selected["candidate_id"]]
                errors["ood" if p["is_ood"] else "in_distribution"].append(
                    abs(p["value"] - _metrics(r)["quality"])
                )
        trials.append(
            {
                "seed": seed,
                "arms": arms,
                "revalidation": revalidation,
                "initial_candidate_ids": sorted(initial_ids),
                "strategies": strategies,
                "diversity_relaxed_selections": relaxed_count,
                "heldout_rmse_initial": float(np.sqrt(np.mean(initial_error**2))),
                "heldout_rmse_final": float(np.sqrt(np.mean(final_error**2))),
                "prequential_errors": {
                    k: {"count": len(v), "mae": float(np.mean(v)) if v else None}
                    for k, v in errors.items()
                },
                "failures": [
                    v for h in active_result.history for v in h["verifications"] if v["failure"]
                ],
            }
        )
    comparisons = {b: _paired_evidence(trials, b) for b in ("random", "evolutionary")}
    report = {
        "schema_version": 1,
        "protocol": REFERENCE_ID,
        "seeds": list(seeds),
        "cycle_config": asdict(cycle_config),
        "provenance": {
            "git_commit": provenance.git_commit,
            "git_dirty": provenance.git_dirty,
            "timestamp_utc": provenance.timestamp_utc,
            "runtime": dict(provenance.runtime),
        },
        "search_simulations": 72 * len(seeds),
        "heldout_simulations": 3 * len(seeds),
        "confirmation_simulations": 9 * len(seeds),
        "total_exact_simulations": 84 * len(seeds),
        "trials": trials,
        "comparisons": comparisons,
        "empirical_gate_passed": all(c["reproducible_advantage"] for c in comparisons.values())
        and all(r["agrees"] for t in trials for r in t["revalidation"].values())
        and not any(t["failures"] for t in trials),
        "full_suite_required": True,
    }
    (destination / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def source_provenance(root: Path) -> ReproducibilityMetadata:
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    )
    return ReproducibilityMetadata(
        seed=0,
        git_commit=commit,
        git_dirty=dirty,
        package_version="0.1.0",
        solver_name="exact_diagonalization",
        solver_version="numpy.linalg.eigh",
        solver_settings={"basis_ordering": "component_major", "eigenvectors": True},
        tolerances={"hermiticity": 1e-10, "particle_hole": 1e-10},
        timestamp_utc=datetime.now(UTC).isoformat(),
        runtime={
            "benchmark": REFERENCE_ID,
            "source_sha256": digest.hexdigest(),
            "numpy": np.__version__,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_discovery_benchmark(args.output, provenance=source_provenance(Path.cwd()))
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("total_exact_simulations", "comparisons", "empirical_gate_passed")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
