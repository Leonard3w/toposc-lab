"""Studio parameters reach the validated physics without changing its formulas."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from toposc_lab.geometry import square
from toposc_lab.models.chiral_p_wave import ChiralPWaveModel, ChiralPWaveParameters
from toposc_lab.research.embedded import (
    EmbeddedDomain,
    EmbeddedPhysicsProtocol,
    EmbeddedValidationEvaluator,
)
from toposc_lab.research.physics import PhysicsProtocol, create_physics_protocol
from toposc_lab.research.studio_physics import ADAPTER_ID, StudioEvaluator, StudioPhysicsProtocol


def protocol(**changes):
    return StudioPhysicsProtocol(
        domain=EmbeddedDomain(bounds=(0, 2, 0, 2), boundary_shell=.2, bulk_inset=.5),
        disorder_widths=(.3,), disorder_seeds=(5, 6), confirmation=False, **changes,
    )


def test_protocol_roundtrip_and_registered_factory():
    original = protocol(hopping=.8, chemical_potential=1.7, pairing=.6, chirality=-1,
                        kappas=(.15,), tolerance=1e-9, energy_cutoff=.8, boundary_widths=(.5, 1.5))
    payload = json.loads(json.dumps(asdict(original)))
    assert StudioPhysicsProtocol(**payload) == original
    assert create_physics_protocol(payload, objective=original.objective) == original
    with pytest.raises(ValueError, match="frozen"):
        PhysicsProtocol(hopping=.8)


@pytest.mark.parametrize("settings", [
    {"hopping": float("nan")}, {"chemical_potential": True}, {"chirality": 0},
    {"kappas": ()}, {"kappas": (0,)}, {"solver": "sparse"},
    {"tolerance": 0}, {"boundary_widths": (1, 1)}, {"energy_cutoff": -1},
    {"clean_reference": False, "objective": "clean_quality"},
    {"near_zero_count": 0}, {"near_zero_count": True}, {"confidence_level": 1},
    {"majorana_zero_tolerance": -1},
    {"majorana_zero_tolerance": .1, "majorana_splitting_tolerance": .01},
])
def test_invalid_or_unimplemented_physics_rejected(settings):
    with pytest.raises((TypeError, ValueError)):
        protocol(**settings)


def test_default_science_matches_frozen_embedded_adapter():
    p = protocol()
    old = EmbeddedValidationEvaluator(EmbeddedPhysicsProtocol(
        domain=p.domain, disorder_widths=p.disorder_widths,
        disorder_seeds=p.disorder_seeds, confirmation=False,
    ))
    new = StudioEvaluator(p)
    geometry = square(3, 3)
    stage = new.plan()[1]
    left, right = old.evaluate(geometry, stage), new.evaluate(geometry, stage)
    for key in ("spectrum", "onsite_offsets", "localizer_gaps", "indices", "metrics",
                "majorana", "spatial", "boundary_window", "chern_marker"):
        assert left[key] == right[key]
    assert right["adapter_id"] == ADAPTER_ID


def test_changed_parameters_reach_primary_and_diagnostic_hamiltonians():
    p = protocol(hopping=.7, chemical_potential=1.2, pairing=.4, chirality=-1,
                 kappas=(.15,), energy_cutoff=.7, boundary_widths=(.5, 1.5))
    geometry = square(3, 3)
    evaluator = StudioEvaluator(p)
    result = evaluator.evaluate(geometry, evaluator.plan()[0])
    matrix = ChiralPWaveModel(geometry, ChiralPWaveParameters(
        hopping=p.hopping, chemical_potential=p.chemical_potential,
        pairing=p.pairing, chirality=p.chirality,
    )).hamiltonian()
    np.testing.assert_allclose(result["spectrum"], np.linalg.eigvalsh(matrix), atol=p.tolerance)
    assert result["boundary_window"]["energy_cutoff"] == .7
    if result["boundary_window"]["window"]:
        assert set(result["boundary_window"]["window"]["strip_weights"]) == {"0.5", "1.5"}


def test_independent_seeds_are_deterministic_and_actual_seeds_preserved():
    evaluator = StudioEvaluator(protocol(paired_seeds=False, kappas=(.1,)))
    geometry = square(3, 3)
    stage = evaluator.plan()[1]
    first = evaluator.evaluate(geometry, stage)
    second = evaluator.evaluate(replace(geometry, metadata={"unrelated": True}), stage)
    assert first["stage"]["seed"] != stage["seed"]
    assert first["stage"] == second["stage"]
    assert first["onsite_offsets"] == second["onsite_offsets"]
    assert first["requested_stage"] == stage
    changed = replace(geometry, edges=geometry.edges[:-1])
    third = evaluator.evaluate(changed, stage)
    assert third["stage"]["seed"] != first["stage"]["seed"]


def test_disorder_only_plan_and_summary_dont_invent_clean_evidence():
    evaluator = StudioEvaluator(protocol(clean_reference=False, paired_seeds=False, kappas=(.1,)))
    geometry = square(3, 3)
    assert len(evaluator.plan()) == 2
    assert all(s["kind"] == "disorder" for s in evaluator.plan())
    results = {s["key"]: evaluator.evaluate(geometry, s) for s in evaluator.plan()}
    summary = evaluator.summarize(results)
    assert summary["clean"] is None
    assert summary["score_ready"] and summary["complete"]
    assert summary["score"] is not None
    assert summary["adapter_id"] == ADAPTER_ID
    assert summary["validation_state"] == "EXACT_EVALUATED"
    assert summary["robustness"][0]["seeds"] == [r["stage"]["seed"] for r in results.values()]
    assert summary["robustness"][0]["requested_seeds"] == [5, 6]
    assert evaluator.summarize({})["score"] is None


def test_diagnostic_controls_and_wilson_confidence_use_existing_algorithms():
    geometry = square(3, 3)
    changed = StudioEvaluator(protocol(
        near_zero_count=2, majorana_zero_tolerance=1e-7,
        majorana_splitting_tolerance=.02, majorana_splitting_phs_tolerance=1e-6,
        confidence_level=.8, kappas=(.1,),
    ))
    results = {s["key"]: changed.evaluate(geometry, s) for s in changed.plan()}
    majorana = results["clean"]["majorana"]
    assert len(majorana["states"]) == 2
    assert majorana["zero_tolerance"] == 1e-7
    assert majorana["splitting_tolerance"] == .02
    assert majorana["splitting_phs_tolerance"] == 1e-6
    summary = changed.summarize(results)
    assert summary["robustness"][0]["confidence_level"] == .8
    conventional = StudioEvaluator(replace(changed.protocol, confidence_level=.95)).summarize(results)
    assert summary["score"] == conventional["score"]
    assert summary["robustness"][0]["success_wilson_lower"] >= conventional["robustness"][0]["success_wilson_lower"]
    assert summary["robustness"][0]["success_wilson_upper"] <= conventional["robustness"][0]["success_wilson_upper"]
