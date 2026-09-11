from __future__ import annotations

from toposc_lab.ml import SURROGATE_GATE_BENCHMARK_SEED, run_surrogate_gate_benchmark


def test_surrogate_ready_gate_on_exact_held_out_labels() -> None:
    first = run_surrogate_gate_benchmark()
    second = run_surrogate_gate_benchmark()

    assert first == second
    assert first.seed == SURROGATE_GATE_BENCHMARK_SEED
    assert first.exact_record_count == 30
    assert first.beats_trivial_predictor
    assert first.selected_test_rmse < first.trivial_test_rmse
    assert first.selected_default in {"ridge", "stump_boosting"}
    assert not first.gnn_meaningful_improvement
    assert first.uncertainty_diagnostics.sample_count == len(first.exact_test_label_ids)
    assert 0.0 <= first.calibration_metrics.empirical_coverage <= 1.0
    assert first.unusual_graph_flagged_ood
    assert first.predictions_link_exact_labels
    assert first.dataset_fingerprint.startswith("exact-dataset-v1-sha256:")
    assert first.passed
