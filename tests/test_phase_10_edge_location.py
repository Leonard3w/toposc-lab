"""Contract tests for the frozen Phase-10 edge-location comparison."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from toposc_lab.search import phase_10_edge_location_campaign as campaign
from toposc_lab.search._research_storage import AttemptLedger, load_record
from toposc_lab.search.phase_10_edge_location import (
    ANCHORS,
    ARMS,
    FULL_PAIR_IDS,
    PREFLIGHT_PAIR_IDS,
    N,
    build_edge_location_cell,
    build_edge_location_plan,
    pair_definition,
)
from toposc_lab.search.phase_10_size_methods import (
    MASK_NAMES,
    size_reference,
    validate_size_geometry,
)


def test_frozen_pair_space_and_stage_budgets() -> None:
    assert N == 12
    assert ANCHORS == (3, 4, 5, 6, 7)
    assert ARMS == ("boundary", "interior")
    assert PREFLIGHT_PAIR_IDS == (2, 7)
    assert FULL_PAIR_IDS == tuple(range(40))
    assert len(build_edge_location_plan(preflight=True)) == 6
    assert 2 + 2 * len(FULL_PAIR_IDS) == 82


def test_all_frozen_interventions_are_unique_valid_and_resource_matched() -> None:
    reference = size_reference(N)
    reference_degrees = tuple(
        len(reference.neighbors(site)) for site in range(reference.n_sites)
    )
    geometry_ids = set()
    for pair_id in FULL_PAIR_IDS:
        paired = [build_edge_location_cell(pair_id, arm) for arm in ARMS]
        assert [cell["intervention_depth"] for cell in paired] == [0, 3]
        for cell in paired:
            geometry = cell["genome"].to_geometry()
            assert validate_size_geometry(geometry, N)["is_valid"]
            assert cell["degree_sequence"] == reference_degrees
            assert cell["edge_distance"] == 2
            assert len(cell["measurement_sites"]) == 16
            assert len(set(cell["measurement_sites"])) == 16
            assert cell["removed_edge_lengths"] == (1.0, 1.0)
            assert np.allclose(
                cell["added_edge_lengths"], (np.sqrt(2.0), np.sqrt(2.0)),
                rtol=0.0, atol=1.0e-12,
            )
            geometry_ids.add(cell["geometry_id"])
        for key in (
            "mirror", "rotation", "anchor_y", "edge_distance", "degree_sequence",
            "removed_edge_lengths", "added_edge_lengths",
            "removed_edge_displacements", "added_edge_displacements",
        ):
            assert paired[0][key] == paired[1][key]
    assert len(geometry_ids) == 80


def test_transform_index_and_alternating_arm_order_are_exact() -> None:
    definition = pair_definition(27, "interior")
    assert (definition["mirror"], definition["rotation"], definition["anchor_y"]) == (
        1, 1, 5
    )
    plan = build_edge_location_plan(preflight=True)
    assert [(cell["pair_id"], cell["arm"]) for cell in plan[1:-1]] == [
        (2, "boundary"), (2, "interior"),
        (7, "interior"), (7, "boundary"),
    ]


def test_prepare_cell_audits_identity_order_and_unit_areas() -> None:
    prepared = campaign._prepare_cell(build_edge_location_cell(0, "boundary"))
    evidence = prepared["topology_input_evidence"]
    assert np.array_equal(evidence["unique_coordinate_order"], np.arange(N * N))
    assert np.allclose(
        evidence["position_areas_in_coordinate_order"], 1.0, rtol=0.0, atol=1.0e-10
    )
    assert tuple(int(np.count_nonzero(mask)) for mask in evidence["bulk_masks_in_coordinate_order"]) == (
        65, 38, 64, 36, 36
    )


def test_summary_keeps_failures_and_computes_frozen_pair_contrasts() -> None:
    reference = _toy_record("control_start", None, None, _toy_scientific(1.0, 0.0))
    outcomes = [reference]
    for pair_id in PREFLIGHT_PAIR_IDS:
        outcomes.append(
            _toy_record("edge_location", pair_id, "boundary", _toy_scientific(0.99, 0.1))
        )
        outcomes.append(
            _toy_record(
                "edge_location", pair_id, "interior",
                None if pair_id == 7 else _toy_scientific(0.97, 0.3),
            )
        )
    outcomes.append(_toy_record("control_end", None, None, _toy_scientific(1.0, 0.0)))
    summary = campaign.edge_location_summary(tuple(outcomes), preflight=True)
    assert summary["control_valid"] is True
    assert summary["all_evaluations_available"] is False
    assert summary["arms"]["boundary"]["attempt_count"] == 2
    assert summary["arms"]["interior"]["available_count"] == 1
    assert summary["primary_paired_difference"]["count"] == 1
    assert summary["primary_paired_difference"]["values"] == pytest.approx([0.02])
    assert summary["paired_patch_absolute_difference"]["values"] == pytest.approx([0.2])
    assert [row["available"] for row in summary["pair_results"]] == [True, False]


def test_sealed_resume_binding_rejects_changed_pair(tmp_path: Path) -> None:
    expected = {
        "n": N,
        "block": "edge_location",
        "pair_id": 2,
        "arm": "boundary",
        "mirror": 0,
        "rotation": 0,
        "anchor_y": 5,
        "geometry_id": "geometry",
        "evaluation_seed": None,
    }
    directory = tmp_path / "cell"
    record = {**campaign._identity(expected), "run": None, "scientific": None, "error": None}
    ledger = AttemptLedger(directory)
    ledger.record("input.json", expected)
    ledger.record("outcome.json", record)
    ledger.seal(record)
    assert campaign._load_bound_sealed(directory, expected) == record
    with pytest.raises(BaseException, match="Zellinput"):
        campaign._load_bound_sealed(directory, dict(expected, pair_id=7))


def test_preflight_completion_and_completed_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = {"code_commit": "test-implementation", "threads": {}}
    monkeypatch.setattr(campaign, "edge_location_environment", lambda: environment)
    monkeypatch.setattr(campaign, "_output_path", lambda path: path.resolve())

    def prepare(cell: dict[str, Any]) -> dict[str, Any]:
        return {**cell, "topology_input_evidence": {"mask_names": MASK_NAMES}}

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
            "evaluated", stage=mode, size=N, slot=index + 1, completed=index + 1,
            total=total, category="physics", operation_seconds=0.01,
        )
        return _toy_record(
            cell["block"], cell["pair_id"], cell["arm"], _toy_scientific(1.0, 0.0),
            cell=cell,
        )

    monkeypatch.setattr(campaign, "_evaluate_cell", evaluate)
    root = tmp_path / "phase_10_edge_location_test"
    report = campaign.run_edge_location_campaign(root, mode="preflight")
    complete = load_record(report.parent / "complete.json")
    assert complete["preflight_passed"] is True
    assert load_record(report.parent / "analysis.json")[
        "total_evaluation_attempts"
    ] == 6
    monkeypatch.setattr(
        campaign, "_evaluate_cell",
        lambda *args, **kwargs: pytest.fail("completed stage was repeated"),
    )
    assert campaign.run_edge_location_campaign(root, mode="resume") == report


def test_cli_requires_explicit_edge_location_mode() -> None:
    from toposc_lab.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["phase-10-edge-location", "--preflight"])
    assert args.edge_location_mode == "preflight"
    with pytest.raises(SystemExit):
        parser.parse_args(["phase-10-edge-location"])
    with pytest.raises(SystemExit):
        parser.parse_args(["phase-10-edge-location", "--preflight", "--full"])


def test_output_path_is_restricted_to_named_results_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="phase_10_edge_location"):
        campaign._output_path(tmp_path / "wrong")


def _toy_scientific(central: float, marker_value: float) -> dict[str, Any]:
    marker = np.full(N * N, marker_value, dtype=float)
    local_chern = [
        {
            "bulk_chern_estimate": central,
            "chern_number": 1,
            "is_quantized": True,
            "bulk_site_count": 36,
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
            "bott": [{"estimate": 1.0, "index": 1, "is_quantized": True}] * 3,
            "local_chern": local_chern,
            "localizer": [
                {
                    "local_chern_number": 1,
                    "localizer_gap": 0.23,
                    "is_invertible": True,
                    "signature": 2,
                }
            ] * 3,
        },
        "marker_maps": [
            {
                "local_marker": marker.copy(),
                "bulk_mask": np.ones(N * N, dtype=bool),
            }
            for _ in MASK_NAMES
        ],
        "boundary_signature": {
            "boundary_localized_count": 8,
            "minimum_boundary_weight_first_four": 0.9,
            "maximum_pair_residual": 0.0,
            "pairing_cost": 0.0,
            "particle_hole_pairs_by_state_index": [[0, 7], [1, 6], [2, 5], [3, 4]],
            "reasons": [],
            "states": [{"energy": float(index)} for index in range(8)],
        },
        "localizer_protection_proxy": 0.23,
    }


def _toy_record(
    block: str,
    pair_id: int | None,
    arm: str | None,
    scientific: dict[str, Any] | None,
    *,
    cell: dict[str, Any] | None = None,
) -> dict[str, Any]:
    definition = {} if pair_id is None else pair_definition(pair_id, arm)  # type: ignore[arg-type]
    identity = {
        "n": N,
        "block": block,
        "pair_id": pair_id,
        "arm": arm,
        "mirror": definition.get("mirror"),
        "rotation": definition.get("rotation"),
        "anchor_y": definition.get("anchor_y"),
        "geometry_id": "control" if pair_id is None else f"{pair_id}-{arm}",
        "evaluation_seed": None,
    }
    if cell is not None:
        identity = campaign._identity(cell)
    metrics = {
        "measurement_sites": definition.get("measurement_sites", ()),
    }
    if cell is not None:
        metrics = campaign._construction_metrics(cell)
    return {
        **identity,
        "construction_metrics": metrics,
        "run": SimpleNamespace(
            simulation_result=SimpleNamespace(eigenvalues=np.arange(2 * N * N, dtype=float))
        ),
        "scientific": scientific,
        "error": None if scientific is not None else {"type": "TestError", "message": "test"},
    }
