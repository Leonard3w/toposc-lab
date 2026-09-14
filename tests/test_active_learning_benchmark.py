from __future__ import annotations

from dataclasses import replace

import numpy as np

from toposc_lab.active_learning.benchmark import _paired_evidence, run_discovery_benchmark
from toposc_lab.active_learning.reference import KitaevReferenceEvaluator, reference_candidate
from toposc_lab.data import ReproducibilityMetadata, validate_dataset_record


def provenance():
    return ReproducibilityMetadata(
        seed=0,
        git_commit="test-source",
        git_dirty=True,
        package_version="0.1.0",
        solver_name="exact_diagonalization",
        solver_version="numpy.linalg.eigh",
        solver_settings={"basis_ordering": "component_major"},
        tolerances={"particle_hole": 1e-10},
        timestamp_utc="2026-09-11T00:00:00Z",
        runtime={"source_sha256": "test-source"},
    )


def test_reference_physics_positive_negative_and_independent_seed():
    evaluator = KitaevReferenceEvaluator(provenance())
    positive = evaluator(reference_candidate(12, 0.0), 1)
    negative = evaluator(reference_candidate(12, 3.0), 1)
    assert positive.topology[0].is_topological
    assert not negative.topology[0].is_topological
    assert positive.observables[0].values["quality"] > 0
    assert negative.observables[0].values["quality"] == 0
    assert validate_dataset_record(positive).is_valid
    repeated = evaluator(reference_candidate(12, 0.0), 2)
    np.testing.assert_allclose(
        positive.spectrum.eigenvalues, repeated.spectrum.eigenvalues, rtol=0, atol=1e-10
    )
    assert positive.observables == repeated.observables


def test_benchmark_budget_existing_evolution_and_reproduction(tmp_path):
    # Development seed is disjoint from frozen gate seeds; outcomes are never asserted favorable.
    first = run_discovery_benchmark(tmp_path / "first", provenance=provenance(), seeds=(13001,))
    second = run_discovery_benchmark(
        tmp_path / "second",
        provenance=replace(provenance(), timestamp_utc="2026-09-12T00:00:00Z"),
        seeds=(13001,),
    )
    assert first["total_exact_simulations"] == 84
    assert not first["empirical_gate_passed"]  # one seed cannot establish stability
    for result in (first, second):
        for trial in result["trials"]:
            assert all(arm["exact_simulations"] == 24 for arm in trial["arms"].values())
            assert trial["arms"]["active"]["unique_candidates"] == 24
            assert trial["arms"]["random"]["unique_candidates"] == 24
            assert all(r["agrees"] for r in trial["revalidation"].values())
            assert not trial["failures"]
            for arm in trial["arms"].values():
                arm.pop("wall_seconds")
                arm.pop("exact_seconds")
                arm.pop("winner_id")  # record identity legitimately includes provenance
        result["provenance"].pop("timestamp_utc")
    assert first == second


def test_gate_does_not_pass_ties_or_single_favorable_seed():
    trials = [
        {"arms": {"active": {"best_score": a}, "random": {"best_score": b}}}
        for a, b in ((1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 0.0))
    ]
    assert not _paired_evidence(trials, "random")["reproducible_advantage"]
