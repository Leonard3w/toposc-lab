"""Real tiny budget, pre-exact OOD, fail-closed accounting and preserved artifacts."""

import json

import pytest

from toposc_lab.generative.benchmark import THREADS, run_benchmark


@pytest.fixture
def threads(monkeypatch):
    for name in THREADS:
        monkeypatch.setenv(name, "1")


def test_equal_paid_budgets_and_pre_exact_ood(tmp_path, threads):
    destination = tmp_path / "gate"
    report = run_benchmark(destination, seeds=(14001,), budget=8, references=False)
    assert report["exact_calls"] == 55
    assert report["search_exact_calls"] == 40
    assert all(not c["reproducible_advantage"] for c in report["comparisons"].values())
    for method in ("random", "evolution", "patch", "coverage", "active"):
        summary = report["trials"][0][method]
        assert summary["exact_calls"] == 8
        assert summary["accepted_duplicate_rate"] == 0
        ood = json.loads((destination / "14001" / method / "selected-ood-1.json").read_text())
        assert ood["stage"] == "before_exact"
        assert ood["is_ood"] == summary["ood_flags"][4:]
        assert ood["reference_record_ids"] == summary["record_ids"][:4]
    with pytest.raises(FileExistsError):
        run_benchmark(destination, seeds=(14001,), budget=8, references=False)


def test_exact_failure_is_charged_and_never_published(tmp_path, threads, monkeypatch):
    from toposc_lab.generative.physics import ExactGeometryEvaluator

    def fail(*args):
        raise ArithmeticError("injected solver failure")

    monkeypatch.setattr(ExactGeometryEvaluator, "evaluate", fail)
    destination = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="exact attempt 1 failed"):
        run_benchmark(destination, seeds=(14001,), budget=8, references=False)
    ledger = [
        json.loads(line) for line in (destination / "exact-attempts.jsonl").read_text().splitlines()
    ]
    assert [entry["status"] for entry in ledger] == ["started", "failed"]
    assert not (destination / "report.json").exists()
    assert not list(destination.rglob("search-*.json"))
