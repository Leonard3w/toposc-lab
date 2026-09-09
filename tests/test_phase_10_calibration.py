"""Contract tests for the frozen square-neighborhood calibration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from toposc_lab.search import phase_10_calibration_campaign as campaign
from toposc_lab.search.geometry_genome import GeometryGenome
from toposc_lab.search.phase_9_8_evaluation import (
    Phase98GeometryApplicability,
    validate_phase_9_8_geometry,
)
from toposc_lab.search.phase_10_calibration import (
    AMPLITUDES,
    EDGE_STEPS,
    FULL_ROOTS,
    PREFLIGHT_ROOTS,
    build_calibration_geometry,
    build_calibration_trajectory,
    calibration_edge_pool,
    calibration_position_field,
    derive_calibration_schedule,
    ordered_calibration_cells,
)
from toposc_lab.search.phase_10_research import square_reference


def test_frozen_grid_seed_roles_and_edge_pool() -> None:
    assert AMPLITUDES == (0.0, 0.005, 0.025, 0.05, 0.1)
    assert EDGE_STEPS == (0, 1, 2, 4, 8)
    assert PREFLIGHT_ROOTS == (10_829_900, 10_829_901)
    assert len(FULL_ROOTS) == 16
    assert len(calibration_edge_pool()) == 210
    schedule = derive_calibration_schedule(PREFLIGHT_ROOTS[0])
    assert schedule == derive_calibration_schedule(PREFLIGHT_ROOTS[0])
    assert len(schedule["step_seeds"]) == 8
    assert len({schedule["position_seed"], *schedule["step_seeds"]}) == 9


def test_position_field_pins_boundary_and_rebuilds_displacements() -> None:
    base = square_reference()
    assert base.coordinates is not None
    field = calibration_position_field(123)
    assert field.shape == (64, 2)
    assert not field.flags.writeable
    assert np.all(field[sorted(base.boundary_sites)] == 0.0)
    interior = sorted(set(range(64)) - set(base.boundary_sites))
    assert np.all(np.abs(field[interior]) <= 1.0)
    coordinates = base.coordinates + 0.1 * field
    pairs = tuple(sorted((edge.source, edge.target) for edge in base.edges))
    geometry = build_calibration_geometry(coordinates, pairs)
    assert geometry.dimension_records == ()
    assert all(
        np.array_equal(
            edge.displacement,
            geometry.coordinates[edge.target] - geometry.coordinates[edge.source],
        )
        for edge in geometry.edges
    )
    report = validate_phase_9_8_geometry(
        geometry, applicability=Phase98GeometryApplicability.CLEAN_PRIMARY
    )
    assert report.is_applicable


def test_one_complete_trajectory_is_valid_and_has_all_cells() -> None:
    plan = build_calibration_trajectory(123)
    assert len(plan["operations"]) == 8
    assert len(plan["cells"]) == 25
    assert {
        (cell["amplitude"], cell["edge_step"]) for cell in plan["cells"]
    } == {(a, k) for a in AMPLITUDES for k in EDGE_STEPS}
    assert all(operation["legal_swap_count"] > 0 for operation in plan["operations"])
    assert all(cell["measurements"]["base_valid"] for cell in plan["cells"])
    cells = ordered_calibration_cells((plan, plan))
    assert len(cells) == 50
    assert cells[0]["block"] == "control_start"
    assert cells[-1]["block"] == "control_end"
    assert sum(cell["block"] == "positions" for cell in cells) == 8
    assert sum(cell["block"] == "connectivity" for cell in cells) == 8
    assert sum(cell["block"] == "combined" for cell in cells) == 32


def test_summary_uses_trajectory_denominators_and_checks_controls() -> None:
    scientific = _toy_scientific()
    outcomes = [
        _toy_outcome("control_start", None, 0.0, 0, scientific),
        _toy_outcome("positions", 1, 0.005, 0, scientific),
        _toy_outcome("positions", 2, 0.005, 0, scientific),
        _toy_outcome("control_end", None, 0.0, 0, scientific),
    ]
    summary = campaign.calibration_summary(tuple(outcomes), preflight=False)
    assert summary["control_valid"]
    assert summary["all_evaluations_available"]
    assert summary["cells"][0]["trajectory_count"] == 2
    assert summary["cells"][0]["clean_eligible_count"] == 2

    failed = dict(outcomes[2])
    failed.update(scientific=None, error={"type": "ArithmeticError", "message": "test"})
    summary = campaign.calibration_summary(
        (outcomes[0], outcomes[1], failed, outcomes[-1]), preflight=True
    )
    assert not summary["all_evaluations_available"]
    assert summary["cells"][0]["operational_failure_count"] == 1
    assert summary["cells"][0]["clean_eligible_fraction_all"] == 0.5


def test_cli_requires_explicit_calibration_mode() -> None:
    from toposc_lab.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["phase-10-calibration", "--preflight"])
    assert args.calibration_mode == "preflight"
    with pytest.raises(SystemExit):
        parser.parse_args(["phase-10-calibration"])
    with pytest.raises(SystemExit):
        parser.parse_args(["phase-10-calibration", "--preflight", "--full"])


def test_output_path_is_restricted_to_named_results_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="phase_10_calibration"):
        campaign._output_path(tmp_path / "wrong")


def test_preflight_is_sealed_and_completed_resume_does_not_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = {"code_commit": "test-implementation", "threads": {}}
    monkeypatch.setattr(campaign, "calibration_environment", lambda: environment)
    monkeypatch.setattr(campaign, "_output_path", lambda path: path.resolve())
    monkeypatch.setattr(campaign, "build_calibration_trajectory", _toy_plan)
    monkeypatch.setattr(
        campaign,
        "evaluate_research_geometry",
        lambda geometry, code: SimpleNamespace(run=None, scientific=_toy_scientific()),
    )
    root = tmp_path / "phase_10_calibration_test"
    report = campaign.run_calibration_campaign(root, mode="preflight")
    assert report.is_file()
    complete = campaign.load_record(report.parent / "complete.json")
    assert complete["preflight_passed"] is True
    summary = campaign.load_record(report.parent / "analysis.json")
    assert summary["total_evaluation_attempts"] == 50
    assert len(list((report.parent / "evaluations").glob("cell_*/sealed.json"))) == 50
    monkeypatch.setattr(
        campaign,
        "evaluate_research_geometry",
        lambda *args, **kwargs: pytest.fail("completed calibration was repeated"),
    )
    assert campaign.run_calibration_campaign(root, mode="resume") == report


def _toy_outcome(
    block: str, root: int | None, amplitude: float, edge_step: int,
    scientific: dict[str, Any]
) -> dict[str, Any]:
    return {
        "block": block,
        "root": root,
        "amplitude": amplitude,
        "edge_step": edge_step,
        "edge_distance": 0,
        "run": None,
        "scientific": scientific,
        "error": None,
    }


def _toy_scientific() -> dict[str, Any]:
    return {
        "clean_eligible": True,
        "gate_reasons": [],
        "localizer_protection_proxy": 0.23,
        "minimum_boundary_weight_first_four": 0.9,
        "topology_grid": {
            "bott": [{"index": 1, "estimate": 1.0, "is_quantized": True}] * 3,
            "local_chern": [
                {
                    "bulk_site_count": 4,
                    "chern_number": 1,
                    "bulk_chern_estimate": 1.0,
                    "is_quantized": True,
                }
            ] * 2,
            "localizer": [
                {"kappa": 0.2, "local_chern_number": 1, "localizer_gap": 0.23}
            ] * 3,
        },
        "boundary_signature": {
            "states": [{"energy": float(index)} for index in range(16)],
            "boundary_localized_count": 8,
            "maximum_pair_residual": 0.0,
        },
    }


def _toy_plan(root: int) -> dict[str, Any]:
    base = square_reference()
    assert base.coordinates is not None
    pairs = tuple(sorted((edge.source, edge.target) for edge in base.edges))
    genome = GeometryGenome.from_geometry(build_calibration_geometry(base.coordinates, pairs))
    cells = [
        {
            "amplitude": amplitude,
            "edge_step": edge_step,
            "edge_distance": 0,
            "genome": genome,
            "measurements": {"base_valid": True},
        }
        for amplitude in AMPLITUDES
        for edge_step in EDGE_STEPS
    ]
    return {
        "schedule": derive_calibration_schedule(root),
        "position_field": np.zeros((64, 2)),
        "operations": [],
        "cells": cells,
    }
