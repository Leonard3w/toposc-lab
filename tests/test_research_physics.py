"""Exact adapter contracts and evidence gates, including the existing reference."""

import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from toposc_lab.active_learning.benchmark import source_provenance
from toposc_lab.data import record_from_dict, validate_dataset_record
from toposc_lab.discovery import DiscoveryConfig
from toposc_lab.discovery.validation import disorder_evidence
from toposc_lab.generative.physics import ExactGeometryEvaluator, candidate
from toposc_lab.generative.space import GeometrySearchSpace
from toposc_lab.geometry import GeometryEdge
from toposc_lab.geometry.generators.square import square
from toposc_lab.research.physics import (
    ADAPTER_ID,
    OBJECTIVE_REGISTRY,
    VALIDATOR_REGISTRY,
    FiniteSystemEvaluator,
    ObjectiveDefinition,
    PhysicsProtocol,
    ValidatorDefinition,
    register_objective,
    register_validator,
)


@pytest.fixture
def small_ensemble():
    evaluator = FiniteSystemEvaluator(
        PhysicsProtocol(disorder_widths=(0.0, 0.2), disorder_seeds=(17, 18))
    )
    geometry = square(3, 3)
    results = {stage["key"]: evaluator.evaluate(geometry, stage) for stage in evaluator.plan()}
    return evaluator, results


def test_protocol_round_trip_and_matched_seed_stage_plan():
    protocol = PhysicsProtocol()
    assert PhysicsProtocol(**json.loads(json.dumps(asdict(protocol)))) == protocol
    evaluator = FiniteSystemEvaluator(protocol)
    assert len(evaluator.plan()) == 14
    for width in protocol.disorder_widths:
        assert [
            stage["seed"]
            for stage in evaluator.plan()
            if stage["kind"] == "disorder" and stage["width"] == width
        ] == list(protocol.disorder_seeds)
    assert evaluator.plan() == FiniteSystemEvaluator().plan()


@pytest.mark.parametrize(
    "changes",
    [
        {"hopping": 2},
        {"chemical_potential": 2.1},
        {"pairing": 0.5},
        {"chirality": -1},
        {"basis": "site_major"},
        {"tolerance": 1e-8},
        {"success_threshold": 0.1},
        {"kappas": (0.1,)},
        {"adapter_id": "future"},
        {"probe_rule": "random"},
        {"hopping": True},
        {"disorder_seeds": (1, 1)},
        {"disorder_seeds": (1,)},
        {"disorder_widths": (-0.1,)},
        {"disorder_widths": (0.2, 0.1)},
        {"disorder_widths": (float("nan"),)},
        {"confirmation": 1},
        {"validators": ("majorana",)},
        {"objective": "W_c"},
    ],
)
def test_frozen_science_and_malformed_protocol_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        PhysicsProtocol(**changes)


def test_exact_reference_equivalence_to_phase14_and_phase15():
    space = GeometrySearchSpace()
    xy = space.coordinates
    geometry = space.build(
        frozenset(
            edge for edge in space.edge_pool if np.linalg.norm(xy[edge[0]] - xy[edge[1]]) == 1
        )
    )
    provenance = source_provenance(Path.cwd())
    old, _ = ExactGeometryEvaluator(provenance).evaluate(candidate(geometry), 15001)
    evaluator = FiniteSystemEvaluator(
        PhysicsProtocol(clean_seed=15001, disorder_widths=(0.2,), disorder_seeds=(15002, 15003)),
        provenance,
    )
    clean = evaluator.evaluate(geometry, evaluator.plan()[0])
    np.testing.assert_allclose(clean["spectrum"], old.spectrum.eigenvalues, atol=1e-10, rtol=0)
    for key in (
        "quality",
        "eligible",
        "success",
        "localizer_gap",
        "boundary_weight",
        "minimum_abs_energy",
    ):
        assert clean["metrics"][key] == pytest.approx(old.observables[0].values[key], abs=1e-10)
    assert clean["probe"] == [2.5, 2.5]
    stage = next(stage for stage in evaluator.plan() if stage["kind"] == "disorder")
    disorder = evaluator.evaluate(geometry, stage)
    legacy_disorder = disorder_evidence(old, stage["seed"], DiscoveryConfig())
    np.testing.assert_allclose(
        disorder["spectrum"], legacy_disorder["spectrum"], atol=1e-10, rtol=0
    )
    assert disorder["hamiltonian_id"] == legacy_disorder["hamiltonian_id"]
    assert disorder["metrics"]["quality"] == pytest.approx(legacy_disorder["quality"], abs=1e-10)
    assert disorder["indices"] == legacy_disorder["indices"]


def test_default_100_site_clean_adapter_is_finite_and_dataset_valid():
    evaluator = FiniteSystemEvaluator()
    clean = evaluator.evaluate(square(10, 10), evaluator.plan()[0])
    assert len(clean["spectrum"]) == 200
    assert clean["probe"] == [4.5, 4.5]
    assert clean["metrics"]["spectral_phs_residual"] < 1e-10
    assert clean["metrics"]["eigensystem_residual"] < 1e-10
    assert clean["majorana"]["status"] == "diagnostics_only"
    assert not clean["majorana_claim"] and not clean["phase_claim"]
    assert clean["metrics"]["bulk_gap"] is None
    record = record_from_dict(clean["dataset_record"])
    validate_dataset_record(record).raise_for_errors()
    assert record.spectrum.basis_size == 200
    assert record.provenance.solver_settings["adapter_id"] == ADAPTER_ID
    json.dumps(clean, allow_nan=False)


def test_ensemble_summary_wilson_scope_and_no_unsupported_claims(small_ensemble):
    evaluator, results = small_ensemble
    summary = evaluator.summarize(results)
    assert summary["complete"] and summary["score_ready"]
    assert summary["reproducibility"]["status"] == "validated"
    assert summary["score"] == np.mean([g["success_fraction"] for g in summary["robustness"]])
    for group in summary["robustness"]:
        assert group["completed_samples"] == 2
        assert group["success_wilson_lower"] <= group["success_fraction"]
        assert group["success_wilson_upper"] >= group["success_fraction"]
        assert group["success_wilson_upper"] > group["success_wilson_lower"]
    assert not summary["validation_results"]["majorana"]["passed"]
    assert not summary["validation_results"]["finite_size"]["passed"]
    assert summary["critical_disorder_strength"]["value"] is None
    assert "MAJORANA_VALIDATED" not in summary["validation_states"]
    assert "FINITE_SIZE_VALIDATED" not in summary["validation_states"]
    json.dumps(summary, allow_nan=False)


def test_partial_and_failed_members_cannot_create_high_score(small_ensemble):
    evaluator, results = small_ensemble
    partial = dict(results)
    partial.pop("disorder_1_0")
    summary = evaluator.summarize(partial)
    assert not summary["complete"]
    assert summary["score"] is None
    assert summary["robustness"][1]["success_fraction"] is None
    assert summary["missing_stages"] == ["disorder_1_0"]
    partial["disorder_1_0"] = {"status": "failed", "error": "numerical failure"}
    summary = evaluator.summarize(partial)
    assert summary["score"] is None
    assert summary["validation_state"] == "FAILED"
    assert summary["robustness"][1]["execution_failure_indices"] == [0]
    assert summary["robustness"][1]["successes"][0] is False
    assert summary["robustness"][1]["requested_samples"] == 2
    assert summary["robustness"][1]["success_fraction"] <= 0.5


def test_reproducibility_failure_blocks_score(small_ensemble):
    evaluator, results = small_ensemble
    results = deepcopy(results)
    results["confirmation"]["spectrum"][0] += 1e-6
    summary = evaluator.summarize(results)
    assert summary["reproducibility"]["status"] == "failed"
    assert summary["validation_state"] == "FAILED"
    assert summary["score"] is None


def test_zero_width_is_clean_and_same_seed_offsets_match_across_connectivities():
    evaluator = FiniteSystemEvaluator(
        PhysicsProtocol(disorder_widths=(0.0, 0.2), disorder_seeds=(17, 18))
    )
    geometry = square(3, 3)
    clean = evaluator.evaluate(geometry, evaluator.plan()[0])
    zero_stage = next(stage for stage in evaluator.plan() if stage["kind"] == "disorder")
    zero = evaluator.evaluate(geometry, zero_stage)
    assert zero["hamiltonian_id"] == clean["hamiltonian_id"]
    assert zero["metrics"]["quality"] == clean["metrics"]["quality"]
    mutated = replace(geometry, edges=(*geometry.edges, GeometryEdge(0, 4)))
    stage = next(stage for stage in evaluator.plan() if stage["width"] == 0.2)
    first = evaluator.evaluate(geometry, stage)
    second = evaluator.evaluate(mutated, stage)
    assert first["onsite_offsets"] == second["onsite_offsets"]
    assert first["hamiltonian_id"] != second["hamiltonian_id"]


def test_invalid_stage_positions_boundaries_and_displacements_are_rejected():
    evaluator = FiniteSystemEvaluator()
    geometry = square(3, 3)
    stage = evaluator.plan()[0]
    with pytest.raises(ValueError, match="stage"):
        evaluator.evaluate(geometry, {**stage, "seed": 0})
    for invalid in (
        replace(geometry, coordinates=geometry.coordinates + 1),
        replace(geometry, boundary_sites=frozenset()),
        square(3, 3, boundary_x="periodic"),
        square(2, 3),
    ):
        with pytest.raises(ValueError):
            evaluator.evaluate(invalid, stage)


def test_clean_only_config_and_registry_extension(small_ensemble):
    _evaluator, results = small_ensemble
    clean_only = FiniteSystemEvaluator(
        PhysicsProtocol(
            objective="clean_quality", validators=("spectrum", "topology"), confirmation=False
        )
    )
    assert len(clean_only.plan()) == 1
    summary = clean_only.summarize({"clean": results["clean"]})
    assert summary["score"] == results["clean"]["metrics"]["quality"]
    objective = ObjectiveDefinition("test_observable", "Test extension", lambda value: 0.5)
    validator = ValidatorDefinition(
        "test_gate",
        "Test future gate",
        "EXACT_EVALUATED",
        lambda summary, protocol: {"passed": False},
    )
    try:
        register_objective(objective)
        register_validator(validator)
        protocol = PhysicsProtocol(
            objective="test_observable", validators=("spectrum", "test_gate")
        )
        assert protocol.objective == "test_observable"
        with pytest.raises(ValueError):
            register_objective(objective)
        with pytest.raises(ValueError):
            register_validator(validator)
    finally:
        OBJECTIVE_REGISTRY.pop("test_observable", None)
        VALIDATOR_REGISTRY.pop("test_gate", None)


def test_experiment_config_expands_defaults_and_keeps_one_objective():
    from toposc_lab.research.config import ExperimentConfig

    default = ExperimentConfig()
    assert default.objective == "robustness_success_fraction"
    assert default.physics["tolerance"] == 1e-10
    assert default.physics["disorder_seeds"] == [17001, 17002, 17003, 17004]
    assert default.surrogate["ensemble_size"] == 8
    assert "objective" not in default.physics
    assert ExperimentConfig(**default.to_dict()).fingerprint == default.fingerprint
    configured = ExperimentConfig(
        objective="clean_quality",
        physics={"confirmation": False, "validators": ["spectrum", "topology"]},
    )
    assert configured.physics_protocol().objective == "clean_quality"
    assert len(FiniteSystemEvaluator(configured.physics_protocol()).plan()) == 1
    with pytest.raises(ValueError, match="conflicts"):
        ExperimentConfig(
            objective="clean_quality", physics={"objective": "robustness_quality_mean"}
        )


def test_engine_uses_declared_objective_and_launch_provenance_in_exact_dataset(tmp_path):
    from toposc_lab.research.config import ExperimentConfig
    from toposc_lab.research.engine import ResearchEngine

    config = ExperimentConfig(
        objective="clean_quality",
        physics={"confirmation": False, "validators": ["spectrum", "topology"]},
    )
    directory = ResearchEngine.create(tmp_path / "research-provenance", config)
    engine = ResearchEngine(directory)
    assert engine.evaluator.protocol.objective == "clean_quality"
    result = engine.evaluator.evaluate(square(2, 2), engine.evaluator.plan()[0])
    record = record_from_dict(result["dataset_record"])
    assert record.provenance.git_commit == engine.manifest["git_commit"]
    assert record.provenance.git_dirty == engine.manifest["git_dirty"]
    assert record.provenance.runtime["source_sha256"] == engine.manifest["source_sha256"]
    assert record.provenance.runtime["experiment_id"] == engine.manifest["experiment_id"]
    assert record.provenance.solver_settings["protocol"]["objective"] == "clean_quality"
    assert record.provenance.timestamp_utc == engine.manifest["timestamp"]


def test_config_validates_registered_space_and_surrogate_extensions():
    from toposc_lab.research.config import ExperimentConfig
    from toposc_lab.research.space import SPACE_REGISTRY, FixedConnectivitySpace
    from toposc_lab.research.surrogate import SURROGATE_REGISTRY, GraphSurrogate

    try:
        SPACE_REGISTRY["test_space"] = FixedConnectivitySpace
        SURROGATE_REGISTRY["test_surrogate"] = GraphSurrogate
        config = ExperimentConfig(geometry_space="test_space", surrogate_model="test_surrogate")
        config.validate_plugins()
    finally:
        SPACE_REGISTRY.pop("test_space", None)
        SURROGATE_REGISTRY.pop("test_surrogate", None)
