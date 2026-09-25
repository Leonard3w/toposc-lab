"""Independent validation: physics equivalence, statistical units and recovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest

from toposc_lab.research.engine import ResearchEngine
from toposc_lab.research.physics import FiniteSystemEvaluator, PhysicsProtocol
from toposc_lab.research.space import (
    FixedConnectivitySpace,
    geometry_from_payload,
    geometry_to_payload,
)
from toposc_lab.research.storage import ResearchStore
from toposc_lab.research.validation import ValidationStudy, make_config, pilot_settings
from toposc_lab.research.validation_cohort import digest, structural_descriptors, validate_cohort
from toposc_lab.research.validation_diagnostics import (
    ValidationEvaluator,
    boundary_diagnostics,
    probe_positions,
)
from toposc_lab.research.validation_reporting import analyze, load_snapshot
from toposc_lab.research.validation_statistics import confirmation_grid, summarize, wilson


@pytest.fixture
def small_cohort():
    settings = {
        "side": 4,
        "min_degree": 2,
        "max_degree": 6,
        "max_bond_length": 2.0,
        "bond_tolerance": 0.0,
        "forbid_crossings": True,
        "site_crossings": "unconnected",
    }
    space = FixedConnectivitySpace(**settings)
    geometry = space.reference()
    candidate = {
        "id": "regular",
        "geometry": geometry_to_payload(geometry),
        "symmetry_hash": space.hash(geometry),
        "source": {"kind": "reference"},
        "descriptors": structural_descriptors(geometry),
    }
    c = {"version": "phase18.cohort.v1", "space": settings, "candidates": [candidate]}
    return {**c, "sha256": digest(c)}


def engineering(cohort):
    return {
        **pilot_settings(cohort),
        "mode": "engineering",
        "seeds": [189001, 189002],
        "widths": [0.5],
        "exact_budget": 5,
        "checkpoint_every": 2,
    }


def test_probe_count_and_interior_coordinates_are_frozen():
    points = probe_positions(10)
    assert len(points) == 17
    assert points.count([4.5, 4.5]) == 1
    assert all([x, y] in points for x in (2, 4.5, 7) for y in (2, 4.5, 7))
    assert [0, 4.5] in points and [4.5, 9] in points


def test_boundary_projector_invariant_under_degenerate_unitary_rotation():
    xy = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    energies = np.array([-2, -0.4, -0.4, -0.2, 0.2, 0.4, 0.4, 2.0])
    vectors = np.eye(8, dtype=complex)
    before = boundary_diagnostics(energies, vectors, xy)
    rotation = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    vectors[:, 1:3] = vectors[:, 1:3] @ rotation
    after = boundary_diagnostics(energies, vectors, xy)
    np.testing.assert_allclose(
        before["window"]["site_probability"], after["window"]["site_probability"]
    )
    np.testing.assert_allclose(
        before["groups"][0]["site_probability"], after["groups"][0]["site_probability"]
    )
    assert after["state_count"] == 6
    assert sum(after["window"]["shell_weights"]) == pytest.approx(1)


def test_window_boundary_includes_whole_near_degenerate_group():
    xy = np.array([[0, 0], [0, 1]], float)
    r = boundary_diagnostics(np.array([-2, 0.5 - 1e-9, 0.5 + 1e-9, 2]), np.eye(4), xy)
    assert r["state_count"] == 2
    empty = boundary_diagnostics(np.array([-2.0, -1.0, 1.0, 2.0]), np.eye(4), xy)
    assert empty["window"] is None and empty["status"] == "empty_window"


def test_additional_diagnostics_preserve_historical_primary_and_match_disorder_fields(small_cohort):
    space = FixedConnectivitySpace(**small_cohort["space"])
    geometry = space.reference()
    protocol = PhysicsProtocol(
        disorder_widths=(0.5,), disorder_seeds=(189001, 189002), confirmation=False
    )
    evaluator = ValidationEvaluator(protocol)
    stage = evaluator.plan()[1]
    old = FiniteSystemEvaluator(protocol).evaluate(geometry, stage)
    new = evaluator.evaluate(geometry, stage)
    assert old["metrics"] == new["metrics"]
    assert old["hamiltonian_id"] == new["hamiltonian_id"]
    assert old["indices"] == new["indices"]
    rng = np.random.default_rng(189010)
    changed, _ = space.mutate(geometry, rng, "local_rewiring")
    other = evaluator.evaluate(changed, stage)
    np.testing.assert_allclose(new["onsite_offsets"], other["onsite_offsets"], atol=1e-12, rtol=0)
    center = [p for p in new["spatial"] if p["point"] == new["probe"]]
    assert [p["gap"] for p in center] == old["localizer_gaps"]
    assert new["operations"]["localizer_diagonalizations"] == 3 * len(probe_positions(4))


def test_chern_bulk_and_boundary_profiles_have_consistent_coordinates():
    geometry = FixedConnectivitySpace(side=6).reference()
    protocol = PhysicsProtocol(
        disorder_widths=(0.5,), disorder_seeds=(189001, 189002), confirmation=False
    )
    evaluator = ValidationEvaluator(protocol)
    result = evaluator.evaluate(geometry, evaluator.plan()[0])
    marker = result["chern_marker"]
    assert marker["status"] == "available"
    assert sum(marker["bulk_mask"]) == 4
    assert marker["trace_residual"] < 1e-8
    assert marker["projector_residual"] < 1e-10
    w = result["boundary_window"]["window"]
    assert w["strip_weights"]["1"] <= w["strip_weights"]["2"] <= w["strip_weights"]["3"] + 1e-14
    assert sum(w["shell_weights"]) == pytest.approx(1)


def test_geometry_descriptors_separate_boundary_and_bulk(small_cohort):
    d = small_cohort["candidates"][0]["descriptors"]
    assert d["bridge_count"] == 0
    assert d["bulk_weak_fraction"] == 0
    assert d["boundary_weak_fraction"] > 0
    assert d["bulk_degree_variance"] == 0
    assert d["anisotropy"] == pytest.approx(0)
    assert d["balanced_cut_min_edges"] == 4


def test_frozen_cohort_rejects_corruption_and_preview_mismatch(small_cohort):
    bad = deepcopy(small_cohort)
    bad["candidates"][0]["geometry"]["edges"].pop()
    with pytest.raises(ValueError, match="checksum"):
        validate_cohort(bad)
    bad["sha256"] = digest({k: v for k, v in bad.items() if k != "sha256"})
    with pytest.raises(ValueError, match="preview"):
        validate_cohort(bad)


def fake_result(q, valid=True, *, success=None):
    return {
        "status": "completed",
        "primary_valid": valid,
        "metrics": {"quality": q, "success": q >= 0.2 if success is None else success},
    }


def test_statistics_keep_invalid_missing_and_seed_blocks_separate():
    records = {}
    for w in (1.0, 2.0):
        records["regular", w, 1] = fake_result(0.3)
        records["regular", w, 2] = fake_result(0.4)
        records["historical_best", w, 1] = fake_result(0.35)
        records["historical_best", w, 2] = fake_result(0.5)
    records["historical_best", 2.0, 2] = fake_result(0, valid=False)
    summary = summarize(
        ["regular", "historical_best"], [1.0, 2.0], [1, 2, 3], records, confirmation=True
    )
    row = next(
        r for r in summary["curves"] if r["candidate"] == "historical_best" and r["width"] == 2
    )
    assert (row["valid"], row["invalid"], row["missing"], row["planned"]) == (1, 1, 1, 3)
    assert row["success_fraction_valid"] == 1
    assert row["success_fraction_bounds_all_planned"] == [1 / 3, 1]
    primary = summary["primary_seed_block_contrasts"][0]
    assert primary["pairs"] == 1  # Not 2 correlated widths, not 6 row observations.
    assert primary["mean"] == pytest.approx(0.05)
    assert primary["interval"] is None and not primary["claim_eligible"]
    assert wilson(0, 0) is None
    assert wilson(1, 2) == pytest.approx([0.09453120573423074, 0.9054687942657693])


def test_grid_rule_and_nonmonotonicity_are_explicit():
    curves = [
        {
            "candidate": "regular",
            "width": w,
            "success_fraction_valid": p,
            "invalid": 0,
            "missing": 0,
        }
        for w, p in ((1.2, 1), (3, 0.5), (6, 1), (9, 0), (12, 0))
    ]
    grid = confirmation_grid({"curves": curves}, [1.2, 3.0, 6.0, 9.0, 12.0])
    assert grid["widths"] == pytest.approx([1.2, 1.65, 2.1, 2.55, 3])
    assert grid["pilot_nonmonotone"]
    curves[0]["invalid"] = 1
    assert not confirmation_grid({"curves": curves}, [1.2, 3.0, 6.0, 9.0, 12.0])["bracketed"]


@pytest.fixture
def no_plots(monkeypatch):
    monkeypatch.setattr("toposc_lab.research.validation_reporting.plot_results", lambda *a: None)


def test_small_end_to_end_pause_resume_and_completed_replay(tmp_path, small_cohort, no_plots):
    path = ValidationStudy.create_study(tmp_path / "study", engineering(small_cohort), small_cohort)
    with pytest.raises(ValueError, match="Fixed-cohort"):
        ResearchEngine(path)
    store = ResearchStore(path)
    store.request("pause")
    assert ValidationStudy(path).run()["status"] == "PAUSED"
    assert not store.attempts()
    assert ValidationStudy(path).run(max_stages=1)["status"] == "PAUSED"
    assert len(store.attempts()) == 1
    assert ValidationStudy(path).run()["status"] == "COMPLETED"
    snapshot = load_snapshot(path)
    summary = analyze(snapshot)
    assert summary["complete_schedule"] and summary["charged_attempts"] == 3
    assert all(r["matched"] for r in summary["paired_field_checks"])
    assert (path / "reports" / "realizations.jsonl").exists()
    before = store.attempts()
    assert ValidationStudy(path).run()["status"] == "COMPLETED"
    assert store.attempts() == before


def test_interrupted_stage_remains_charged_then_identical_retry(tmp_path, small_cohort, no_plots):
    path = ValidationStudy.create_study(tmp_path / "crash", engineering(small_cohort), small_cohort)

    def crash(name, payload):
        if name == "after_simulation":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        ValidationStudy(path, hook=crash).run()
    store = ResearchStore(path)
    assert len(store.attempts()) == 1
    assert ValidationStudy(path).run()["status"] == "COMPLETED"
    assert len(store.attempts()) == 4
    assert store.attempts()[0]["status"] == "interrupted"
    path2 = ValidationStudy.create_study(
        tmp_path / "normal", engineering(small_cohort), small_cohort
    )
    ValidationStudy(path2).run()
    for a, b in zip(store.all("exact_result"), ResearchStore(path2).all("exact_result")):
        assert a["hamiltonian_id"] == b["hamiltonian_id"]
        assert a["metrics"] == b["metrics"]
        assert a["spatial"] == b["spatial"]
        assert a["boundary_window"] == b["boundary_window"]


def test_explicit_attempt_cap_stops_incomplete_run(tmp_path, small_cohort, no_plots):
    settings = {**engineering(small_cohort), "exact_budget": 1}
    path = ValidationStudy.create_study(tmp_path / "cap", settings, small_cohort)
    state = ValidationStudy(path).run()
    assert state["status"] == "STOPPED"
    assert state["exact_evaluations"] == 1
    assert not analyze(load_snapshot(path))["complete_schedule"]


def test_numerical_failures_are_persisted_not_converted_to_trivial(
    tmp_path, small_cohort, no_plots
):
    class Broken:
        def plan(self):
            return [{"key": "clean", "kind": "clean", "width": 0.0, "seed": 180100}]

        def evaluate(self, *args):
            raise np.linalg.LinAlgError("test failure")

    path = ValidationStudy.create_study(
        tmp_path / "failed", engineering(small_cohort), small_cohort
    )
    ValidationStudy(path, evaluator=Broken()).run()
    record = ResearchStore(path).all("exact_result")[0]
    assert record["status"] == "failed" and "metrics" not in record
    assert ResearchStore(path).attempts()[0]["status"] == "failed"


def test_changed_config_and_source_cannot_resume(tmp_path, small_cohort, no_plots, monkeypatch):
    path = ValidationStudy.create_study(tmp_path / "guard", engineering(small_cohort), small_cohort)
    config = json.loads((path / "study.json").read_text())
    config["widths"] = [99]
    (path / "study.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="study.json"):
        ValidationStudy(path).run()
    (path / "study.json").write_text(json.dumps(engineering(small_cohort)))
    monkeypatch.setattr("toposc_lab.research.provenance.source_digest", lambda: "changed")
    with pytest.raises(ValueError, match="source mismatch"):
        ValidationStudy(path).run()


def test_confirmation_requires_explicit_budgets(small_cohort):
    settings = {**engineering(small_cohort), "exact_budget": None, "wall_seconds": None}
    with pytest.raises(ValueError, match="Explicit"):
        make_config(settings, small_cohort)


def test_zero_width_is_not_a_repeated_disorder_sample(small_cohort):
    with pytest.raises(ValueError, match="one clean"):
        make_config({**engineering(small_cohort), "widths": [0]}, small_cohort)


def test_real_process_exit_and_recovery(tmp_path, small_cohort, no_plots):
    path = ValidationStudy.create_study(
        tmp_path / "hard_exit", engineering(small_cohort), small_cohort
    )
    code = (
        "import os,sys; from toposc_lab.research.validation import ValidationStudy; "
        "ValidationStudy(sys.argv[1], hook=lambda name,payload: "
        "os._exit(73) if name=='after_simulation' else None).run()"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        check=False,
    )
    assert result.returncode == 73, result.stderr
    store = ResearchStore(path)
    assert store.attempts()[0]["status"] == "started"
    assert ValidationStudy(path).run()["status"] == "COMPLETED"
    assert [a["status"] for a in store.attempts()] == [
        "interrupted",
        "complete",
        "complete",
        "complete",
    ]


def test_wall_clock_budget_pauses_before_a_stage(tmp_path, small_cohort, no_plots):
    settings = {**engineering(small_cohort), "wall_seconds": 1e-9}
    path = ValidationStudy.create_study(tmp_path / "wall", settings, small_cohort)
    state = ValidationStudy(path).run()
    assert state["status"] == "PAUSED"
    assert state["pause_reason"] == "wall_clock_budget"
    assert state["exact_evaluations"] == 0


def test_safe_stop_is_terminal_and_does_not_add_results(tmp_path, small_cohort, no_plots):
    path = ValidationStudy.create_study(tmp_path / "stop", engineering(small_cohort), small_cohort)
    ResearchStore(path).request("stop")
    assert ValidationStudy(path).run()["status"] == "STOPPED"
    assert not ResearchStore(path).attempts()
    assert ValidationStudy(path).run()["status"] == "STOPPED"


def test_retry_exhaustion_is_visible_and_bounded(tmp_path, small_cohort, no_plots):
    settings = {**engineering(small_cohort), "retry_limit": 0}
    path = ValidationStudy.create_study(tmp_path / "no_retry", settings, small_cohort)

    def crash(name, payload):
        if name == "attempt_started":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        ValidationStudy(path, hook=crash).run()
    ValidationStudy(path).run()
    store = ResearchStore(path)
    assert len(store.attempts()) == 3
    failed = store.get("exact_result", "regular:clean")
    assert failed["status"] == "failed" and "retry limit" in failed["error"].lower()


def test_singular_center_does_not_become_a_valid_trivial_result(small_cohort, monkeypatch):
    protocol = make_config(engineering(small_cohort), small_cohort).physics_protocol()
    evaluator = ValidationEvaluator(protocol)
    original = FiniteSystemEvaluator.evaluate

    def singular(self, geometry, stage):
        result = original(self, geometry, stage)
        result["indices"][0] = None
        result["localizer_gaps"][0] = 0
        return result

    monkeypatch.setattr(FiniteSystemEvaluator, "evaluate", singular)
    geometry = geometry_from_payload(small_cohort["candidates"][0]["geometry"])
    result = evaluator.evaluate(geometry, evaluator.plan()[0])
    assert not result["primary_valid"]
    summary = summarize(["regular"], [0.5], [1], {("regular", 0.5, 1): result}, confirmation=False)
    assert summary["curves"][0]["invalid"] == 1
    assert summary["curves"][0]["success_fraction_valid"] is None
