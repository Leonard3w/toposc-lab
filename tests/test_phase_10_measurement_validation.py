"""Contract tests for the bounded Phase-10 measurement validation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from toposc_lab.models.chiral_p_wave import ChiralPWaveModel
from toposc_lab.search import phase_10_measurement_validation_campaign as campaign
from toposc_lab.search._research_storage import (
    AttemptLedger,
    decode_record,
    encode_record,
    json_bytes,
    load_record,
    load_sealed,
)
from toposc_lab.search.phase_10_measurement_validation import (
    ARMS,
    FULL_SIZES,
    MASK_NAMES,
    MODEL_ROLES,
    OFFSETS,
    PREFLIGHT_SIZE,
    build_measurement_cell,
    build_measurement_plan,
    build_measurement_topology_inputs,
    intervention_definition,
    measurement_reference,
    model_parameters,
    validate_measurement_geometry,
)


def test_frozen_roles_sizes_order_and_budgets() -> None:
    assert PREFLIGHT_SIZE == 12
    assert FULL_SIZES == (16, 20)
    assert MODEL_ROLES == ("topological", "trivial", "undefined")
    assert OFFSETS == (-2, 0, 2)
    assert ARMS == ("boundary", "interior")
    assert len(build_measurement_plan(preflight=True)) == 6
    full = build_measurement_plan(preflight=False)
    assert len(full) == 36
    for n in FULL_SIZES:
        cells = [cell for cell in full if cell["n"] == n]
        assert [cell["model_role"] for cell in cells[:3]] == list(MODEL_ROLES)
        assert [cell["block"] for cell in cells[:3]] == ["control_start"] * 3
        assert [
            (cell["model_role"], cell["offset"], cell["arm"])
            for cell in cells[3:15]
        ] == [
            (role, offset, arm)
            for role in ("topological", "trivial")
            for offset in OFFSETS
            for arm in ARMS
        ]
        assert [cell["block"] for cell in cells[-3:]] == ["control_end"] * 3


def test_model_roles_have_exact_explicit_parameters() -> None:
    assert model_parameters("topological").model_dump() == {
        "hopping": 1.0,
        "chemical_potential": 2.0,
        "pairing": 1.0,
        "chirality": 1,
        "plane_axes": (0, 1),
    }
    assert model_parameters("trivial").chemical_potential == 12.0
    undefined = model_parameters("undefined")
    assert (undefined.hopping, undefined.chemical_potential, undefined.pairing) == (
        0.0, 0.0, 0.0
    )


def test_all_frozen_interventions_are_valid_degree_preserving_and_local() -> None:
    geometry_ids = set()
    for n in (PREFLIGHT_SIZE, *FULL_SIZES):
        reference = measurement_reference(n)
        reference_degrees = tuple(
            len(reference.neighbors(site)) for site in range(reference.n_sites)
        )
        for offset in OFFSETS:
            for arm in ARMS:
                definition = intervention_definition(n, offset, arm)
                cell = build_measurement_cell(
                    n, "topological", block="intervention", offset=offset, arm=arm
                )
                geometry = cell["genome"].to_geometry()
                assert validate_measurement_geometry(geometry, n)["is_valid"]
                assert cell["degree_sequence"] == reference_degrees
                assert cell["edge_distance"] == 2
                assert len(definition["measurement_sites"]) == 16
                expected_depth = 0 if arm == "boundary" else n // 4
                assert definition["intervention_depth"] == expected_depth
                geometry_ids.add((n, cell["geometry_id"]))
    assert len(geometry_ids) == 18


@pytest.mark.parametrize(
    ("n", "counts"),
    ((12, (64, 36, 64, 36, 36)), (16, (144, 100, 144, 100, 64)),
     (20, (256, 196, 256, 196, 100))),
)
def test_topology_inputs_fix_order_areas_and_mask_counts(
    n: int, counts: tuple[int, ...]
) -> None:
    inputs, evidence = build_measurement_topology_inputs(measurement_reference(n), n)
    assert evidence["mask_names"] == MASK_NAMES
    assert np.array_equal(evidence["unique_coordinate_order"], np.arange(n * n))
    assert np.allclose(inputs.position_areas, 1.0, rtol=0.0, atol=1.0e-10)
    assert tuple(int(np.count_nonzero(mask)) for mask in inputs.bulk_masks) == counts
    assert inputs.localizer_probe == ((n - 1) / 2, (n - 1) / 2)


def test_classification_uses_only_central_marker_in_primary_panel() -> None:
    methods = _methods(index=1, central=1, supplemental=(0, 1, 1, 1))
    assert campaign._classify(methods, 0.1) == ("primary_topological", True)
    assert campaign._classify(_methods(index=0, central=0), 0.1) == (
        "primary_trivial", False
    )
    assert campaign._classify(methods, 0.0) == ("undefined_fermi_projector", False)
    methods["localizer"][0]["status"] = "rejected"
    methods["localizer"][0]["result"] = None
    assert campaign._classify(methods, 0.1)[0] == "mixed_or_unresolved"


def test_undefined_control_requires_method_rejections_and_analytic_localizer_gap() -> None:
    record = _record(12, "undefined", "control_start", _undefined_methods())
    assert campaign._control_is_valid(record)
    record["methods"]["bott"][0]["status"] = "available"
    assert not campaign._control_is_valid(record)


def test_zero_hamiltonian_methods_are_recorded_individually(tmp_path: Path) -> None:
    cell = build_measurement_cell(12, "undefined", block="control_start")
    geometry = cell["genome"].to_geometry()
    hamiltonian = ChiralPWaveModel(
        geometry, model_parameters("undefined")
    ).hamiltonian()
    assert np.count_nonzero(hamiltonian) == 0
    inputs, _ = build_measurement_topology_inputs(geometry, 12)
    methods = campaign._evaluate_methods(
        hamiltonian, inputs, _Monitor(), {}, {}
    )
    record = _record(12, "undefined", "control_start", methods)
    assert campaign._control_is_valid(record)
    record["run"] = None
    ledger = AttemptLedger(tmp_path / "method_record")
    ledger.record("outcome.json", record)
    ledger.seal(record)
    loaded = load_sealed(tmp_path / "method_record")
    assert loaded["methods"]["bott"][0]["status"] == "rejected"
    assert loaded["methods"]["localizer"][0]["result"]["local_chern_number"] == 0


def test_trivial_intervention_obeys_frozen_norm_bound() -> None:
    cell = build_measurement_cell(
        12, "trivial", block="intervention", offset=0, arm="interior"
    )
    geometry = cell["genome"].to_geometry()
    hamiltonian = ChiralPWaveModel(geometry, model_parameters("trivial")).hamiltonian()
    atomic = np.diag(np.concatenate((-12.0 * np.ones(144), 12.0 * np.ones(144))))
    perturbation = hamiltonian - atomic
    assert np.max(np.sum(np.abs(perturbation), axis=1)) <= 8.0 + 1.0e-12


def test_summary_applies_all_three_frozen_full_verdicts() -> None:
    outcomes = _full_toy_outcomes()
    accepted = campaign.measurement_validation_summary(tuple(outcomes), preflight=False)
    assert accepted["verdict"] == "bounded_measurement_contract_accepted"
    outcomes[3]["positive_screening"] = False
    unstable = campaign.measurement_validation_summary(tuple(outcomes), preflight=False)
    assert unstable["verdict"] == "controls_passed_intervention_panel_not_stable"
    outcomes = _full_toy_outcomes()
    trivial = next(
        record for record in outcomes
        if record["block"] == "intervention" and record["model_role"] == "trivial"
    )
    trivial["classification"] = "mixed_or_unresolved"
    failed = campaign.measurement_validation_summary(tuple(outcomes), preflight=False)
    assert failed["verdict"] == "control_validation_failed"


def test_full_report_after_dictionary_storage_roundtrip() -> None:
    outcomes = _full_toy_outcomes()
    for record in outcomes:
        for group in record["methods"].values():
            for item in group:
                if item["result"] is not None:
                    item["result"] = vars(item["result"])
        # Pipeline fixtures are not serializable; methods exercise the actual codec.
        record["methods"] = decode_record(encode_record(record["methods"]))
    summary = campaign.measurement_validation_summary(tuple(outcomes), preflight=False)
    assert summary["verdict"] == "bounded_measurement_contract_accepted"
    assert len(summary["variants"]) == 24
    assert len(summary["paired_differences"]) == 12
    assert len(summary["size_stability"]) == 6
    assert summary["variants"][0]["patch_marker_change"]["site_ids"] == tuple(range(16))
    assert summary["variants"][0]["depth_marker_change"] is not None
    assert summary["variants"][0]["graph_mask_change"] is not None
    json_bytes(summary)
    assert "bounded_measurement_contract_accepted" in campaign._render_report(summary)


def test_sealed_resume_is_bound_to_exact_cell(tmp_path: Path) -> None:
    expected = build_measurement_cell(12, "topological", block="control_start")
    expected = {**expected, "topology_input_evidence": {"mask_names": MASK_NAMES}}
    record = {
        **campaign._identity(expected),
        "run": None,
        "methods": None,
        "classification": "operational_failure",
    }
    directory = tmp_path / "cell"
    ledger = AttemptLedger(directory)
    ledger.record("input.json", expected)
    ledger.record("outcome.json", record)
    ledger.seal(record)
    assert campaign._load_bound_sealed(directory, expected) == {
        **record, "measurement_sites": expected["measurement_sites"]
    }
    with pytest.raises(BaseException, match="Zellinput"):
        campaign._load_bound_sealed(directory, dict(expected, model_role="trivial"))


def test_legacy_patch_is_restored_from_bound_input_without_rewriting(tmp_path: Path) -> None:
    expected = {
        "n": 16, "block": "intervention", "model_role": "topological",
        "offset": -2, "arm": "boundary", "geometry_id": "test-geometry",
        "evaluation_seed": None, "measurement_sites": tuple(range(16)),
    }
    directory = tmp_path / "cell"
    record = {**campaign._identity(expected), "methods": None}
    ledger = AttemptLedger(directory)
    ledger.record("input.json", expected)
    ledger.record("outcome.json", record)
    ledger.seal(record)
    original = (ledger.execution / "outcome.json").read_bytes()
    loaded = campaign._load_bound_sealed(directory, expected)
    assert loaded["measurement_sites"] == tuple(range(16))
    assert (ledger.execution / "outcome.json").read_bytes() == original


def test_conflicting_archived_patch_is_rejected(tmp_path: Path) -> None:
    expected = {
        "n": 16, "block": "intervention", "model_role": "topological",
        "offset": -2, "arm": "boundary", "geometry_id": "test-geometry",
        "evaluation_seed": None, "measurement_sites": tuple(range(16)),
    }
    directory = tmp_path / "cell"
    record = {**campaign._identity(expected), "measurement_sites": (99,)}
    ledger = AttemptLedger(directory)
    ledger.record("input.json", expected)
    ledger.record("outcome.json", record)
    ledger.seal(record)
    with pytest.raises(BaseException, match="Messregion"):
        campaign._load_bound_sealed(directory, expected)


def test_preflight_completion_and_completed_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = {"code_commit": "implementation-test", "threads": {}}
    monkeypatch.setattr(
        campaign, "measurement_validation_environment", lambda: environment
    )
    monkeypatch.setattr(campaign, "_output_path", lambda path: path.resolve())
    cells = tuple(
        {
            "n": 12,
            "block": block,
            "model_role": role,
            "offset": None,
            "arm": None,
            "geometry_id": f"12-{block}-{role}",
            "evaluation_seed": None,
        }
        for block in ("control_start", "control_end")
        for role in MODEL_ROLES
    )
    monkeypatch.setattr(
        campaign, "build_measurement_plan", lambda *, preflight: cells
    )
    monkeypatch.setattr(campaign, "_prepare_cell", lambda cell: cell)

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
            "evaluated", stage=mode, size=12, slot=index + 1,
            completed=index + 1, total=total, category="physics",
            operation_seconds=0.01,
        )
        role = cell["model_role"]
        methods = _undefined_methods() if role == "undefined" else _methods(
            index=1 if role == "topological" else 0,
            central=1 if role == "topological" else 0,
        )
        return _record(12, role, cell["block"], methods)

    monkeypatch.setattr(campaign, "_evaluate_cell", evaluate)
    root = tmp_path / "phase_10_measurement_validation_test"
    report = campaign.run_measurement_validation_campaign(root, mode="preflight")
    assert load_record(report.parent / "complete.json")["preflight_passed"] is True
    monkeypatch.setattr(
        campaign,
        "_evaluate_cell",
        lambda *args, **kwargs: pytest.fail("completed stage was repeated"),
    )
    assert campaign.run_measurement_validation_campaign(root, mode="resume") == report


def test_cli_requires_explicit_measurement_validation_mode() -> None:
    from toposc_lab.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["phase-10-measurement-validation", "--preflight"])
    assert args.measurement_mode == "preflight"
    with pytest.raises(SystemExit):
        parser.parse_args(["phase-10-measurement-validation"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["phase-10-measurement-validation", "--preflight", "--full"]
        )


def test_output_path_is_restricted_to_named_results_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="phase_10_measurement_validation"):
        campaign._output_path(tmp_path / "wrong")


def _available(result: Any) -> dict[str, Any]:
    return {"status": "available", "result": result, "error": None}


class _Monitor:
    def emit(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


def _rejected(message: str) -> dict[str, Any]:
    return {
        "status": "rejected",
        "result": None,
        "error": {"type": "ValueError", "message": message},
    }


def _methods(
    *, index: int, central: int, supplemental: tuple[int, ...] | None = None,
    central_estimate: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if supplemental is None:
        supplemental = (central,) * 4
    estimates = (*[float(value) for value in supplemental],
                 float(central if central_estimate is None else central_estimate))
    return {
        "bott": [
            _available(SimpleNamespace(bott_index=index, bott_estimate=float(index)))
            for _ in range(3)
        ],
        "local_chern": [
            _available(
                SimpleNamespace(
                    chern_number=value,
                    bulk_chern_estimate=estimate,
                    local_marker=np.zeros(400),
                    bulk_mask=np.ones(400, dtype=bool),
                )
            )
            for value, estimate in zip((*supplemental, central), estimates, strict=True)
        ],
        "localizer": [
            _available(SimpleNamespace(local_chern_number=index, localizer_gap=0.25))
            for _ in range(3)
        ],
    }


def _undefined_methods() -> dict[str, list[dict[str, Any]]]:
    return {
        "bott": [_rejected("undefined at Fermi level") for _ in range(3)],
        "local_chern": [_rejected("undefined at Fermi level") for _ in range(5)],
        "localizer": [
            _available(
                SimpleNamespace(
                    local_chern_number=0, localizer_gap=kappa / np.sqrt(2.0)
                )
            )
            for kappa in (0.1, 0.2, 0.3)
        ],
    }


def _record(
    n: int,
    role: str,
    block: str,
    methods: dict[str, list[dict[str, Any]]],
    *,
    offset: int | None = None,
    arm: str | None = None,
    central_estimate: float | None = None,
) -> dict[str, Any]:
    classification = {
        "topological": "primary_topological",
        "trivial": "primary_trivial",
        "undefined": "undefined_fermi_projector",
    }[role]
    if central_estimate is not None and role != "undefined":
        methods = _methods(
            index=1 if role == "topological" else 0,
            central=1 if role == "topological" else 0,
            central_estimate=central_estimate,
        )
    eigenvalues = np.asarray([-1.0, 1.0])
    return {
        "n": n,
        "block": block,
        "model_role": role,
        "offset": offset,
        "arm": arm,
        "geometry_id": f"{n}-{block}-{role}-{offset}-{arm}",
        "evaluation_seed": None,
        "measurement_sites": () if block != "intervention" else tuple(range(16)),
        "run": SimpleNamespace(
            simulation_result=SimpleNamespace(eigenvalues=eigenvalues)
        ),
        "methods": methods,
        "classification": classification,
        "positive_screening": role == "topological",
        "regional_sensitivity": False,
        "error": None,
    }


def _full_toy_outcomes() -> list[dict[str, Any]]:
    outcomes = []
    for n in FULL_SIZES:
        for role in MODEL_ROLES:
            methods = _undefined_methods() if role == "undefined" else _methods(
                index=1 if role == "topological" else 0,
                central=1 if role == "topological" else 0,
            )
            outcomes.append(_record(n, role, "control_start", methods))
        for role in ("topological", "trivial"):
            for offset in OFFSETS:
                for arm in ARMS:
                    index = 1 if role == "topological" else 0
                    estimate = float(index) + (0.001 if n == 20 and role == "topological" else 0.0)
                    outcomes.append(
                        _record(
                            n, role, "intervention", _methods(index=index, central=index),
                            offset=offset, arm=arm, central_estimate=estimate,
                        )
                    )
        for role in MODEL_ROLES:
            methods = _undefined_methods() if role == "undefined" else _methods(
                index=1 if role == "topological" else 0,
                central=1 if role == "topological" else 0,
            )
            outcomes.append(_record(n, role, "control_end", methods))
    return outcomes
