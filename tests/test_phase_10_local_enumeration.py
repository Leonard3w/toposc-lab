from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from toposc_lab.search import phase_10_local_enumeration_campaign as campaign
from toposc_lab.search._research_storage import (
    AttemptLedger,
    decode_record,
    encode_record,
    json_bytes,
)
from toposc_lab.search.phase_10_local_enumeration import (
    ALL_KAPPAS,
    CATALOG_IDS_SHA256,
    EXTRA_KAPPAS,
    FULL_BUDGET,
    PREFLIGHT_BUDGET,
    PRIMARY_KAPPAS,
    build_local_enumeration_plan,
    paired_candidate_ids,
)


def test_frozen_kappas_budgets_catalog_and_size_pairing() -> None:
    assert PRIMARY_KAPPAS == (0.1, 0.2, 0.3)
    assert EXTRA_KAPPAS == (0.09, 0.11, 0.18, 0.22, 0.27, 0.33)
    assert len(ALL_KAPPAS) == 9
    assert (PREFLIGHT_BUDGET, FULL_BUDGET) == (8, 268)
    preflight = build_local_enumeration_plan(preflight=True)
    full = build_local_enumeration_plan(preflight=False)
    assert len(preflight) == 8
    assert [cell["block"] for cell in preflight] == [
        "control_start", "control_start", "control_start", "intervention",
        "intervention", "control_end", "control_end", "control_end",
    ]
    assert len(full) == 268
    assert sum(cell["block"] == "candidate" for cell in full) == 256
    assert sum(len(cell["origins"]) for cell in full if cell["block"] == "candidate") == 288
    assert len(paired_candidate_ids()) == 128
    assert full[3]["n"] == 16 and full[134 + 3]["n"] == 20
    assert CATALOG_IDS_SHA256 == (
        "13c9e2c750af250f20fd3d245d48b6abc9b2936176e5187dd76ba245ff1c6be0"
    )


def test_cli_requires_exactly_one_enumeration_mode() -> None:
    from toposc_lab.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["phase-10-local-enumeration", "--preflight"])
    assert args.local_enumeration_mode == "preflight"
    with pytest.raises(SystemExit):
        parser.parse_args(["phase-10-local-enumeration"])
    with pytest.raises(SystemExit):
        parser.parse_args(["phase-10-local-enumeration", "--preflight", "--full"])


def test_strict_gain_threshold_and_decision_zone() -> None:
    reference = _record(16, "control_start", "topological", 0.25)
    scale = campaign.PROXY_GAIN * 0.25
    below = _record(16, "candidate", "topological", 0.25 + scale - 2e-8)
    border = _record(16, "candidate", "topological", 0.25 + scale)
    above = _record(16, "candidate", "topological", 0.25 + scale + 2e-8)
    assert campaign._candidate_row(below, reference)["threshold_status"] == "below_threshold"
    assert campaign._candidate_row(border, reference)["threshold_status"] == "threshold_borderline"
    row = campaign._candidate_row(above, reference)
    assert row["threshold_status"] == "primary_proxy_hit"
    assert row["sensitivity_qualified_hit"]


def test_noninvertible_extra_localizer_is_scientific_nonqualification() -> None:
    reference = _record(16, "control_start", "topological", 0.25)
    candidate = _record(16, "candidate", "topological", 0.30)
    candidate["methods"]["localizer"][4] = _available(
        {"local_chern_number": None, "localizer_gap": 0.0, "is_invertible": False}
    )
    row = campaign._candidate_row(candidate, reference)
    assert row["complete"]
    assert row["primary_proxy_hit"]
    assert not row["sensitivity_qualified_hit"]


def test_method_dictionary_roundtrip_preserves_summary_inputs() -> None:
    record = _record(16, "candidate", "topological", 0.30)
    encoded = decode_record(encode_record(record["methods"]))
    record["methods"] = encoded
    reference = _record(16, "control_start", "topological", 0.25)
    row = campaign._candidate_row(record, reference)
    assert row["central_chern"] == 1.0
    json_bytes(row)


def test_control_panel_requires_all_nine_localizers_and_repeat_agreement() -> None:
    first = _record(16, "control_start", "topological", 0.25)
    second = _record(16, "control_end", "topological", 0.25)
    assert campaign._control_is_valid(first)
    assert campaign._controls_agree(first, second)
    second["methods"]["localizer"][8]["result"]["localizer_gap"] += 2e-10
    assert not campaign._controls_agree(first, second)
    first["methods"]["localizer"].pop()
    assert not campaign._control_is_valid(first)


def test_undefined_control_has_expected_projector_rejections_and_nine_gaps() -> None:
    record = _record(12, "control_start", "undefined", 0.0)
    assert campaign._control_is_valid(record)
    record["methods"]["bott"][0] = _available({"bott_index": 0, "bott_estimate": 0.0})
    assert not campaign._control_is_valid(record)


def test_actual_n12_undefined_control_records_all_methods_and_solver_calls() -> None:
    cell = campaign._prepare_cell(build_local_enumeration_plan(preflight=True)[2])
    record = campaign._calculate_cell(cell, "test-code", cast(Any, _Monitor()), {}, {})
    assert [len(record["methods"][name]) for name in ("bott", "local_chern", "localizer")] == [
        3, 5, 9
    ]
    assert record["method_call_count"] == 17
    assert record["linear_algebra_calls"]["eigvalsh"] >= 18
    assert campaign._control_is_valid(record)


def test_sealed_resume_is_bound_to_the_exact_candidate_input() -> None:
    expected = {
        "n": 16, "block": "candidate", "model_role": "topological",
        "geometry_id": "geometry-a", "candidate_id": "geometry-a", "evaluation_seed": None,
    }
    record = {**campaign._identity(expected), "classification": "operational_failure"}
    with TemporaryDirectory(dir=".") as name:
        directory = Path(name) / "cell"
        ledger = AttemptLedger(directory)
        ledger.record("input.json", expected)
        ledger.record("outcome.json", record)
        ledger.seal(record)
        assert campaign._load_bound_sealed(directory, expected) == record
        with pytest.raises(BaseException, match="Zellinput"):
            campaign._load_bound_sealed(directory, {**expected, "candidate_id": "geometry-b"})


def test_incomplete_inventory_has_no_null_result_claim() -> None:
    controls = []
    for n in (16, 20):
        for block in ("control_start", "control_end"):
            for role in ("topological", "trivial", "undefined"):
                controls.append(_record(n, block, role, 0.25 if role == "topological" else 0.0))
    summary = campaign.local_enumeration_summary(tuple(controls), preflight=False)
    assert not summary["inventory_complete"]
    assert summary["verdict"] == "incomplete_or_unresolved"


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("paired", "paired_proxy_candidates_found"),
        ("primary_only", "single_size_or_setting_dependent_hits"),
        ("borderline", "threshold_borderline_only"),
        ("null", "no_relevant_proxy_gain_in_catalog"),
        ("incomplete", "incomplete_or_unresolved"),
        ("control", "control_validation_failed"),
    ),
)
def test_all_frozen_full_verdict_priorities(
    case: str, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    controls = [
        {"n": n, "block": block, "model_role": role}
        for n in (16, 20)
        for block in ("control_start", "control_end")
        for role in ("topological", "trivial", "undefined")
    ]
    candidates = [
        {
            "n": n, "block": "candidate", "model_role": "topological",
            "candidate_id": f"{n}-{index:03d}", "complete": True,
            "positive_screening": True, "delta_L": 0.0,
            "primary_proxy_hit": False, "sensitivity_qualified_hit": False,
            "threshold_status": "below_threshold",
        }
        for n in (16, 20)
        for index in range(128)
    ]
    if case == "paired":
        candidates[0]["primary_proxy_hit"] = True
        candidates[0]["sensitivity_qualified_hit"] = True
    elif case == "primary_only":
        candidates[0]["primary_proxy_hit"] = True
    elif case == "borderline":
        candidates[0]["threshold_status"] = "threshold_borderline"
    elif case == "incomplete":
        candidates[0]["complete"] = False
    monkeypatch.setattr(campaign, "_candidate_row", lambda record, reference: record)
    monkeypatch.setattr(campaign, "_control_is_valid", lambda record: case != "control")
    monkeypatch.setattr(campaign, "_controls_agree", lambda first, second: True)
    monkeypatch.setattr(campaign, "_compact_record", lambda record: {})
    monkeypatch.setattr(
        campaign,
        "_pair_rows",
        lambda rows: [{"paired_proxy_candidate": case == "paired"}],
    )
    summary = campaign.local_enumeration_summary(tuple(controls + candidates), preflight=False)
    assert summary["verdict"] == expected


def test_output_path_is_restricted_to_named_results_directory() -> None:
    with pytest.raises(ValueError, match="phase_10_local_enumeration"):
        campaign._output_path(Path("wrong"))


def _available(result: dict[str, Any]) -> dict[str, Any]:
    return {"status": "available", "result": result, "error": None}


def _rejected() -> dict[str, Any]:
    return {
        "status": "rejected", "result": None,
        "error": {"type": "ValueError", "message": "undefined at a Fermi-level eigenstate"},
    }


class _Monitor:
    def emit(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


def _record(n: int, block: str, role: str, gap: float) -> dict[str, Any]:
    if role == "undefined":
        bott = [_rejected() for _ in range(3)]
        chern = [_rejected() for _ in range(5)]
        localizer = [
            _available({
                "local_chern_number": 0,
                "localizer_gap": kappa / np.sqrt(2.0),
                "is_invertible": True,
            })
            for kappa in ALL_KAPPAS
        ]
        classification = "undefined_fermi_projector"
        positive = False
        minimum = 0.0
    else:
        index = 1 if role == "topological" else 0
        bott = [_available({"bott_index": index, "bott_estimate": float(index)}) for _ in range(3)]
        chern = [
            _available({
                "chern_number": index,
                "bulk_chern_estimate": float(index),
                "local_marker": np.full(n * n, float(index)),
                "bulk_mask": np.ones(n * n, dtype=bool),
            })
            for _ in range(5)
        ]
        localizer = [
            _available({"local_chern_number": index, "localizer_gap": gap, "is_invertible": True})
            for _ in ALL_KAPPAS
        ]
        classification = "primary_topological" if index else "primary_trivial"
        positive = bool(index)
        minimum = 0.5
    boundary = None if role == "undefined" else {
        "particle_hole_pairs_by_state_index": (), "pairing_cost": 0.0,
        "maximum_pair_residual": 0.0, "boundary_localized_count": 4,
        "minimum_boundary_weight_first_four": 0.9, "reasons": (),
        "states": [
            {"state_index": index, "energy": float(index), "boundary_weight": 0.9}
            for index in range(8)
        ],
    }
    candidate_id = f"{n}-{block}-{role}"
    return {
        "n": n, "block": block, "model_role": role, "geometry_id": candidate_id,
        "candidate_id": candidate_id if block in ("candidate", "intervention") else None,
        "evaluation_seed": None, "origins": (), "known_measurement_variant": False,
        "rotation_signature": None, "dihedral_signature": None,
        "run": SimpleNamespace(simulation_result=SimpleNamespace(eigenvalues=np.asarray([-1.0, 1.0]))),
        "methods": {"bott": bott, "local_chern": chern, "localizer": localizer},
        "classification": classification, "positive_screening": positive,
        "minimum_fermi_distance": minimum, "numerical_valid": True,
        "boundary_signature": boundary, "error": None,
    }
