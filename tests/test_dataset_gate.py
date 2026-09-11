from __future__ import annotations

from toposc_lab.data.dataset_benchmark import (
    DATASET_GATE_BENCHMARK_SEED,
    run_dataset_foundation_benchmark,
)


def test_dataset_foundation_gate_fixed_seed_exact_physics(tmp_path) -> None:
    first = run_dataset_foundation_benchmark(tmp_path / "first.json")
    second = run_dataset_foundation_benchmark(tmp_path / "second.json")

    assert first.seed == DATASET_GATE_BENCHMARK_SEED
    assert first.record_count == 3
    assert first.topological_classifications == (True, False, True)
    assert first.split_record_counts == (1, 1, 1)
    assert first.validation_passed
    assert first.round_trip_identical
    assert first == second
