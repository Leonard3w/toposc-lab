from pathlib import Path

import numpy as np
import pytest

from toposc_lab.active_learning.benchmark import source_provenance
from toposc_lab.discovery.storage import atomic_json, read_json
from toposc_lab.generative.space import GeometrySearchSpace
from toposc_lab.patterns.campaign import ExactJournal, PatternCampaignConfig, freeze


def grid():
    space = GeometrySearchSpace()
    return space.build(
        frozenset(
            e
            for e in space.edge_pool
            if np.linalg.norm(space.coordinates[e[0]] - space.coordinates[e[1]]) == 1
        )
    )


def test_protocol_config_is_fixed_and_immutable():
    seeds = list(range(16101, 16109))
    config = PatternCampaignConfig(seeds=seeds)
    seeds[0] = 0
    assert config.seeds == tuple(range(16101, 16109))
    with pytest.raises(ValueError):
        PatternCampaignConfig(exact_attempt_cap=701)
    with pytest.raises(ValueError):
        PatternCampaignConfig(seeds=tuple(range(16101, 16107)))


def test_frozen_semantics_reject_changes_and_detect_corruption(tmp_path):
    path = tmp_path / "frozen.json"
    assert freeze(path, {"v": (1, 2)}) == {"v": [1, 2]}
    freeze(path, {"v": [1, 2]})
    with pytest.raises(ValueError, match="frozen artifact mismatch"):
        freeze(path, {"v": [3]})
    path.write_text(path.read_text().replace('"v"', '"w"'))
    with pytest.raises(ValueError, match="corrupt"):
        freeze(path, {"w": [1, 2]})


def test_exact_journal_resume_confirmation_and_cap(tmp_path, monkeypatch):
    journal = ExactJournal(tmp_path, source_provenance(Path.cwd()), cap=2)
    record, first = journal.confirmed("grid", grid(), 16000)
    assert first["majorana"]["operator_phs_residual"] == 0
    assert record.observables[0].values["eligible"] == 1
    monkeypatch.setattr(type(journal.evaluator), "evaluate", lambda *a: pytest.fail("cache missed"))
    assert journal.confirmed("grid", grid(), 16000)[1] == first
    with pytest.raises(ValueError, match="cache request mismatch"):
        journal.calculate("grid", grid(), 16001)
    with pytest.raises(RuntimeError, match="cap exhausted"):
        journal.calculate("another", grid(), 16002)
    attempt = tmp_path / "attempts/0001.json"
    payload = read_json(attempt)
    payload["status"] = "started"
    payload.pop("result_sha256")
    atomic_json(attempt, payload)
    journal.calculate("grid", grid(), 16000)
    assert read_json(attempt)["status"] == "complete"
    assert len(list((tmp_path / "attempts").glob("*.json"))) == 2


def test_failed_numerical_attempt_is_charged(tmp_path, monkeypatch):
    journal = ExactJournal(tmp_path, source_provenance(Path.cwd()), cap=1)

    def fail(*args):
        raise ArithmeticError("synthetic solver failure")

    monkeypatch.setattr(type(journal.evaluator), "evaluate", fail)
    with pytest.raises(ArithmeticError):
        journal.calculate("failure", grid(), 1)
    assert read_json(tmp_path / "attempts/0001.json")["status"] == "failed"
    with pytest.raises(RuntimeError, match="cap exhausted"):
        journal.calculate("failure", grid(), 1)


def test_interrupted_attempt_is_never_free(tmp_path, monkeypatch):
    journal = ExactJournal(tmp_path, source_provenance(Path.cwd()), cap=1)

    def interrupted(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(type(journal.evaluator), "evaluate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        journal.calculate("interruption", grid(), 1)
    assert read_json(tmp_path / "attempts/0001.json")["status"] == "started"
    with pytest.raises(RuntimeError, match="cap exhausted"):
        journal.calculate("interruption", grid(), 1)


def test_missing_committed_result_and_attempt_gap_fail_closed(tmp_path):
    journal = ExactJournal(tmp_path, source_provenance(Path.cwd()), cap=3)
    journal.confirmed("grid", grid(), 16000)
    exact_path = tmp_path / "exact/grid.json"
    data = exact_path.read_bytes()
    exact_path.unlink()
    with pytest.raises(ValueError, match="committed exact result is missing"):
        journal.calculate("grid", grid(), 16000)
    exact_path.write_bytes(data)
    (tmp_path / "attempts/0001.json").unlink()
    with pytest.raises(ValueError, match="noncontiguous"):
        journal.calculate("grid", grid(), 16000)
