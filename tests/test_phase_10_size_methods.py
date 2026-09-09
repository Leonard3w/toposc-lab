"""Contract tests for the frozen finite-size and bulk-region comparison."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from toposc_lab.search import phase_10_size_methods_campaign as campaign
from toposc_lab.search._research_storage import AttemptLedger
from toposc_lab.search.geometry_genome import GeometryGenome
from toposc_lab.search.phase_9_8_evaluation import (
    build_phase_9_8_boundary_signature,
    evaluate_phase_9_8_descriptive_geometry,
)
from toposc_lab.search.phase_10_size_methods import (
    FULL_ROOTS,
    MASK_NAMES,
    PREFLIGHT_ROOTS,
    SIZES,
    build_size_topology_inputs,
    build_size_trajectory,
    derive_size_schedule,
    edge_step_count,
    ordered_size_cells,
    size_edge_pool,
    size_position_field,
    size_reference,
    validate_size_geometry,
)


def test_frozen_sizes_budgets_seed_roles_and_pools() -> None:
    assert SIZES == (8, 10, 12)
    assert tuple(edge_step_count(n) for n in SIZES) == (4, 7, 10)
    assert tuple(len(size_edge_pool(n)) for n in SIZES) == (210, 342, 506)
    assert PREFLIGHT_ROOTS == (10_839_900, 10_839_901)
    assert FULL_ROOTS == tuple(range(10_840_000, 10_840_016))
    schedule = derive_size_schedule(PREFLIGHT_ROOTS[0])
    assert schedule == derive_size_schedule(PREFLIGHT_ROOTS[0])
    assert tuple(item["n"] for item in schedule["sizes"]) == SIZES
    assert tuple(len(item["step_seeds"]) for item in schedule["sizes"]) == (4, 7, 10)
    roles = [schedule["root"]]
    for item in schedule["sizes"]:
        roles.extend((item["size_seed"], item["position_seed"], *item["step_seeds"]))
    assert len(roles) == len(set(roles))


@pytest.mark.parametrize("n", SIZES)
def test_reference_position_field_and_variable_validator(n: int) -> None:
    geometry = size_reference(n)
    assert geometry.n_sites == n * n
    assert geometry.n_edges == 2 * n * (n - 1)
    assert len(geometry.boundary_sites) == 4 * n - 4
    assert validate_size_geometry(geometry, n)["is_valid"]
    field = size_position_field(n, 123)
    assert field.shape == (n * n, 2)
    assert not field.flags.writeable
    assert np.all(field[sorted(geometry.boundary_sites)] == 0.0)
    assert np.all(np.abs(field) <= 1.0)


def test_complete_trajectory_and_slot_order_match_frozen_budget() -> None:
    plan = build_size_trajectory(123)
    assert tuple(item["n"] for item in plan["sizes"]) == SIZES
    assert tuple(len(item["operations"]) for item in plan["sizes"]) == (4, 7, 10)
    assert all(len(item["cells"]) == 3 for item in plan["sizes"])
    assert all(
        operation["legal_swap_count"] > 0
        for item in plan["sizes"]
        for operation in item["operations"]
    )
    assert all(
        validate_size_geometry(cell["genome"].to_geometry(), item["n"])["is_valid"]
        for item in plan["sizes"]
        for cell in item["cells"]
    )
    cells = ordered_size_cells((plan, plan))
    assert len(cells) == 24
    for n in SIZES:
        selected = [cell for cell in cells if cell["n"] == n]
        assert [cell["block"] for cell in selected] == [
            "control_start",
            "positions",
            "positions",
            "connectivity",
            "connectivity",
            "combined",
            "combined",
            "control_end",
        ]


@pytest.mark.parametrize(
    ("n", "counts"),
    ((8, (16, 4, 16, 4, 16)), (10, (36, 16, 36, 16, 16)), (12, (64, 36, 64, 36, 36))),
)
def test_topology_input_masks_have_frozen_names_areas_and_counts(
    n: int, counts: tuple[int, ...]
) -> None:
    inputs, evidence = build_size_topology_inputs(size_reference(n), n)
    assert evidence["mask_names"] == MASK_NAMES
    assert tuple(int(np.count_nonzero(mask)) for mask in inputs.bulk_masks) == counts
    assert np.isclose(np.sum(inputs.position_areas), n * n, rtol=0.0, atol=1.0e-10)
    assert inputs.bott_periods == ((0.95 * n, 0.95 * n), (float(n), float(n)),
                                   (1.05 * n, 1.05 * n))
    assert inputs.localizer_probe == ((n - 1) / 2, (n - 1) / 2)
    assert inputs.localizer_kappas == (0.1, 0.2, 0.3)


@pytest.mark.parametrize("n", SIZES)
def test_real_square_controls_pass_every_frozen_method(n: int) -> None:
    geometry = size_reference(n)
    inputs, _ = build_size_topology_inputs(geometry, n)
    timings: dict[str, float] = {}
    run, grid = evaluate_phase_9_8_descriptive_geometry(
        geometry,
        inputs=inputs,
        code_version="size-methods-control-test",
        timing_callback=lambda category, seconds: timings.__setitem__(category, seconds),
    )
    assert run.is_valid and run.evaluation is not None
    boundary = build_phase_9_8_boundary_signature(run.evaluation)
    scientific = campaign._encode_size_scientific(run, grid, boundary)
    assert scientific["available"] is True
    assert scientific["control_pass"] is True
    assert scientific["screening_pass"] is True
    assert set(timings) == {"hamiltonian", "solver", "diagnostics", "topology"}
    assert all(seconds >= 0.0 for seconds in timings.values())


def test_availability_checks_real_pipeline_and_grid_shapes() -> None:
    scientific = _toy_scientific()
    assert campaign._scientific_available(scientific)
    malformed = dict(scientific)
    malformed["topology_grid"] = dict(scientific["topology_grid"])
    malformed["topology_grid"]["local_chern"] = [scientific["topology_grid"]["local_chern"][0]]
    assert not campaign._scientific_available(malformed)
    invalid_pipeline = dict(scientific)
    invalid_pipeline["pipeline"] = {"is_valid": False}
    assert not campaign._scientific_available(invalid_pipeline)


def test_summary_keeps_failures_in_denominator_and_paired_differences() -> None:
    outcomes = []
    for n in SIZES:
        outcomes.append(_toy_record(n, "control_start", None, _toy_scientific()))
        for root in (1, 2):
            outcomes.append(_toy_record(n, "positions", root, _toy_scientific(0.99)))
        for root in (1, 2):
            scientific = None if n == 8 and root == 2 else _toy_scientific(0.98)
            outcomes.append(_toy_record(n, "connectivity", root, scientific))
        for root in (1, 2):
            outcomes.append(_toy_record(n, "combined", root, _toy_scientific(0.97)))
        outcomes.append(_toy_record(n, "control_end", None, _toy_scientific()))
    summary = campaign.size_methods_summary(tuple(outcomes), preflight=True)
    target = next(cell for cell in summary["cells"] if cell["n"] == 8 and cell["block"] == "connectivity")
    assert target["attempt_count"] == 2
    assert target["available_count"] == 1
    assert target["operational_failure_count"] == 1
    assert target["screening_fraction_all"] == 0.5
    paired = next(item for item in summary["paired_central_differences"] if item["n"] == 8)
    assert paired["paired_available_count"] == 1


def test_sealed_resume_binding_rejects_changed_input(tmp_path: Path) -> None:
    directory = tmp_path / "cell"
    expected = {
        "n": 8,
        "block": "positions",
        "root": 1,
        "amplitude": 0.05,
        "edge_step": 0,
        "edge_distance": 0,
        "geometry_id": "geometry",
    }
    record = {**campaign._identity(expected), "run": None, "scientific": None, "error": None}
    ledger = AttemptLedger(directory)
    ledger.record("input.json", expected)
    ledger.record("outcome.json", record)
    ledger.seal(record)
    assert campaign._load_bound_sealed(directory, expected) == record
    changed = dict(expected, geometry_id="changed")
    with pytest.raises(BaseException, match="Zellinput"):
        campaign._load_bound_sealed(directory, changed)


def test_preflight_budget_completion_and_completed_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = {"code_commit": "test-implementation", "threads": {}}
    monkeypatch.setattr(campaign, "size_methods_environment", lambda: environment)
    monkeypatch.setattr(campaign, "_output_path", lambda path: path.resolve())
    monkeypatch.setattr(campaign, "build_size_trajectory", _toy_plan)

    def prepare(cell: dict[str, Any]) -> dict[str, Any]:
        return {
            **cell,
            "geometry_id": f"{cell['n']}-{cell['block']}-{cell['root']}",
            "validation": {"is_valid": True},
            "topology_input_evidence": {"mask_names": MASK_NAMES},
            "evaluation_seed": None,
        }

    monkeypatch.setattr(campaign, "_prepare_cell", prepare)

    def evaluate(
        directory: Path,
        cell: dict[str, Any],
        code: str,
        monitor: Any,
        mode: str,
        index: int,
        total: int,
        *,
        estimated_seconds: float | None,
    ) -> dict[str, Any]:
        del directory, code, estimated_seconds
        monitor.emit(
            "evaluated",
            stage=mode,
            size=cell["n"],
            slot=index + 1,
            completed=index + 1,
            total=total,
            category="physics",
            operation_seconds=0.01,
        )
        return _toy_record(cell["n"], cell["block"], cell["root"], _toy_scientific())

    monkeypatch.setattr(campaign, "_evaluate_cell", evaluate)
    root = tmp_path / "phase_10_size_methods_test"
    report = campaign.run_size_methods_campaign(root, mode="preflight")
    assert report.is_file()
    complete = campaign.load_record(report.parent / "complete.json")
    assert complete["preflight_passed"] is True
    assert complete["seconds_per_size"] == {"8": 0.01, "10": 0.01, "12": 0.01}
    analysis = campaign.load_record(report.parent / "analysis.json")
    assert analysis["total_evaluation_attempts"] == 24
    monkeypatch.setattr(
        campaign,
        "_evaluate_cell",
        lambda *args, **kwargs: pytest.fail("completed stage was repeated"),
    )
    assert campaign.run_size_methods_campaign(root, mode="resume") == report


def test_cli_requires_explicit_size_methods_mode() -> None:
    from toposc_lab.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["phase-10-size-methods", "--preflight"])
    assert args.size_methods_mode == "preflight"
    with pytest.raises(SystemExit):
        parser.parse_args(["phase-10-size-methods"])
    with pytest.raises(SystemExit):
        parser.parse_args(["phase-10-size-methods", "--preflight", "--full"])


def test_output_path_is_restricted_to_named_results_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="phase_10_size_methods"):
        campaign._output_path(tmp_path / "wrong")


def _toy_scientific(central: float = 1.0) -> dict[str, Any]:
    chern = [
        {
            "bulk_chern_estimate": central,
            "chern_number": 1,
            "is_quantized": True,
        }
        for _ in MASK_NAMES
    ]
    return {
        "available": True,
        "screening_pass": True,
        "control_pass": True,
        "gate_reasons": [],
        "pipeline": {"is_valid": True},
        "topology_grid": {
            "bott": [{"estimate": 1.0, "index": 1}] * 3,
            "local_chern": chern,
            "localizer": [{"local_chern_number": 1, "localizer_gap": 0.23}] * 3,
        },
        "marker_maps": [{}] * 5,
        "boundary_signature": {
            "boundary_localized_count": 8,
            "maximum_pair_residual": 0.0,
            "states": [{"energy": float(index)} for index in range(8)],
        },
        "localizer_protection_proxy": 0.23,
    }


def _toy_record(
    n: int, block: str, root: int | None, scientific: dict[str, Any] | None
) -> dict[str, Any]:
    run = SimpleNamespace(
        simulation_result=SimpleNamespace(eigenvalues=np.arange(2 * n * n, dtype=float))
    )
    return {
        "n": n,
        "block": block,
        "root": root,
        "amplitude": 0.0 if block != "positions" else 0.05,
        "edge_step": 0,
        "edge_distance": 0,
        "geometry_id": f"{n}-{block}-{root}",
        "run": run,
        "scientific": scientific,
        "error": None if scientific is not None else {"type": "TestError", "message": "test"},
    }


def _toy_plan(root: int) -> dict[str, Any]:
    schedule = derive_size_schedule(root)
    size_plans = []
    for n in SIZES:
        geometry = size_reference(n)
        genome = GeometryGenome.from_geometry(geometry)
        measurements = validate_size_geometry(geometry, n)["measurements"]
        cells = [
            {
                "block": block,
                "amplitude": 0.05 if block in ("positions", "combined") else 0.0,
                "edge_step": edge_step_count(n) if block in ("connectivity", "combined") else 0,
                "edge_distance": 0,
                "genome": genome,
                "measurements": measurements,
                "degree_sequence": tuple(
                    len(geometry.neighbors(site)) for site in range(geometry.n_sites)
                ),
                "boundary_fraction": len(geometry.boundary_sites) / geometry.n_sites,
                "edge_endpoint_class_counts": {},
            }
            for block in ("positions", "connectivity", "combined")
        ]
        size_schedule = next(item for item in schedule["sizes"] if item["n"] == n)
        size_plans.append(
            {
                "n": n,
                "schedule": size_schedule,
                "position_field": np.zeros((n * n, 2)),
                "operations": [],
                "cells": cells,
            }
        )
    return {"root": root, "schedule": schedule, "sizes": size_plans}
