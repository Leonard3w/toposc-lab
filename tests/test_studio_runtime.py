"""Studio adapters must preserve exact geometry, durable state and old evidence."""

import json
from dataclasses import asdict

import pytest

from toposc_lab.geometry import square
from toposc_lab.research.embedded import EmbeddedDomain
from toposc_lab.research.embedded_cohort import geometry_id
from toposc_lab.research.space import geometry_from_payload, geometry_to_payload
from toposc_lab.research.studio_space import (
    EmbeddedSamplingSpace,
    FixedCandidateSpace,
    StudioSampling,
)


def test_fixed_candidates_preserve_geometry_and_resume_order():
    geometry = square(6, 6)
    record = {"id": "source-id", "family": "regular", "geometry": geometry_to_payload(geometry)}
    space = FixedCandidateSpace(
        candidates=[record], domain=asdict(EmbeddedDomain(bounds=(0, 5, 0, 5)))
    )
    from toposc_lab.research.studio_space import FixedCandidateStrategy

    strategy = FixedCandidateStrategy(space=space, seed=5)
    first = strategy.propose(1)
    assert len(first) == 1 and first[0]["source_candidate_id"] == "source-id"
    assert geometry_id(geometry_from_payload(first[0]["geometry"])) == geometry_id(geometry)
    resumed = FixedCandidateStrategy(space=space, seed=5).resume(strategy.checkpoint())
    assert resumed.propose(1) == []
    assert strategy.checkpoint() == resumed.checkpoint()


def test_fixed_candidate_preview_tampering_rejected():
    payload = geometry_to_payload(square(6, 6))
    payload["coordinates"][0][0] = 0.4
    with pytest.raises(ValueError, match="preview"):
        FixedCandidateSpace(
            candidates=[{"id": "x", "geometry": payload}],
            domain=asdict(EmbeddedDomain(bounds=(0, 5, 0, 5))),
        )


def test_sampling_regular_once_and_deterministic_resume():
    space = EmbeddedSamplingSpace(
        families=["regular"], side=6, domain=asdict(EmbeddedDomain(bounds=(0, 5, 0, 5)))
    )
    sampler = StudioSampling(space=space, seed=52)
    first = sampler.propose(1)
    assert len(first) == 1 and first[0]["family"] == "regular"
    restored = StudioSampling(space=space, seed=52).resume(sampler.checkpoint())
    assert restored.propose(1) == []
    assert sampler.checkpoint() == restored.checkpoint()


def test_follow_up_copies_science_without_touching_source(tmp_path):
    from toposc_lab.research.service import ResearchService
    from toposc_lab.research.storage import ResearchStore

    path = tmp_path / "old"
    store = ResearchStore(path, create=True)
    g = square(6, 6)
    domain = asdict(EmbeddedDomain(bounds=(0, 5, 0, 5)))
    store.save(
        "validation_cohort",
        {
            "domain": domain,
            "candidates": [
                {
                    "id": "regular",
                    "family": "regular",
                    "geometry": geometry_to_payload(g),
                    "geometry_sha256": geometry_id(g),
                }
            ],
        },
    )
    store.save(
        "validation_settings", {"widths": [3], "seeds": [191001, 191002], "clean_seed": 191000}
    )
    store.save("config", {"name": "Old", "physics": {"domain": domain}, "schema_version": 1})
    store.save("manifest", {"experiment_id": "old-experiment"})
    before = store.path.read_bytes()
    result = ResearchService.follow_up(
        path,
        ["regular"],
        widths=[3, 6],
        seeds=[9001, 9002],
        output_directory=str(tmp_path / "follow-up"),
    )
    assert result["schema_version"] == 2
    assert result["algorithm"] == "fixed_candidates"
    assert result["physics"]["disorder_seeds"] == [9001, 9002]
    assert result["space"]["candidates"][0]["source_candidate_id"] == "regular"
    assert store.path.read_bytes() == before
    assert not (tmp_path / "follow-up").exists()
    with pytest.raises(ValueError, match="fresh|overlap"):
        ResearchService.follow_up(
            path,
            ["regular"],
            widths=[3],
            seeds=[191001, 9002],
            output_directory=str(tmp_path / "bad"),
        )


def test_strategy_does_not_accept_unknown_settings():
    space = EmbeddedSamplingSpace(
        families=["regular"], side=6, domain=asdict(EmbeddedDomain(bounds=(0, 5, 0, 5)))
    )
    with pytest.raises(TypeError):
        StudioSampling(space=space, unknown_mutation=True)


def test_shared_runner_pause_resume_exact_export_and_followup(tmp_path, monkeypatch):
    from toposc_lab.research.service import ResearchService
    from toposc_lab.research.storage import ResearchStore
    from toposc_lab.research.studio_config import preset, preview_config

    # Isolate this lifecycle test from other workers editing the source tree.
    # Dedicated provenance regression tests exercise the real source guard.
    monkeypatch.setattr("toposc_lab.research.engine.verify", lambda *a: None)
    config = preset("quick_test")
    config["output_directory"] = str(tmp_path / "exact")
    preview = preview_config(config)
    assert not preview["errors"]
    path = ResearchService.create(preview["config"], config["output_directory"])
    store = ResearchStore(path)
    assert store.get("manifest")["config_sha256"] == preview["config_sha256"]
    requested = False

    def pause(name, payload):
        nonlocal requested
        if name == "stage_saved" and not requested:
            requested = True
            store.request("pause")

    first = ResearchService.run_experiment(path, hook=pause)
    assert first["status"] == "PAUSED" and first["exact_evaluations"] == 1
    final = ResearchService.run_experiment(path)
    assert final["status"] == "COMPLETED" and final["exact_evaluations"] == 3
    assert store.count("exact_result") == 3 and not store.all("error")
    snap = ResearchService.snapshot(path)
    assert len(snap["candidates"]) == 1
    candidate = snap["candidates"][0]
    assert candidate["observed"] and candidate["score"] is not None
    assert all(r["status"] == "completed" for r in candidate["exact_results"].values())
    assert len((path / "reports" / "realizations.jsonl").read_text().splitlines()) == 3
    fresh = ResearchService.follow_up(
        path,
        [candidate["id"]],
        widths=[2],
        seeds=[22001, 22002],
        output_directory=str(tmp_path / "new"),
    )
    assert fresh["physics"]["chemical_potential"] == config["physics"]["chemical_potential"]
    new = ResearchService.create(fresh, fresh["output_directory"])
    state = ResearchService.run_experiment(new)
    assert state["status"] == "COMPLETED" and state["exact_evaluations"] == 3
    assert (
        ResearchService.candidate(new, candidate["id"])["geometry_sha256"]
        == candidate["geometry_sha256"]
    )
    assert ResearchService.run_experiment(new)["exact_evaluations"] == 3


def test_cli_preview_uses_same_config_and_creates_no_run(tmp_path, capsys):
    from toposc_lab.research.__main__ import main
    from toposc_lab.research.studio_config import preset, preview_config

    data = preset("quick_test")
    data["output_directory"] = str(tmp_path / "not-created")
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")
    main(["preview", "--config", str(config_path)])
    result = json.loads(capsys.readouterr().out)
    assert result["config_sha256"] == preview_config(data)["config_sha256"]
    assert result["config"] == preview_config(data)["config"]
    assert not (tmp_path / "not-created").exists()


def test_generation_failures_are_charged_and_saved(tmp_path, monkeypatch):
    from toposc_lab.research.service import ResearchService
    from toposc_lab.research.storage import ResearchStore
    from toposc_lab.research.studio_config import preset

    monkeypatch.setattr("toposc_lab.research.engine.verify", lambda *a: None)

    def failure(*args):
        raise RuntimeError("construction exhausted")

    monkeypatch.setattr(EmbeddedSamplingSpace, "generate", failure)
    config = preset("quick_test")
    config["output_directory"] = str(tmp_path / "failed")
    path = ResearchService.create(config, config["output_directory"])
    state = ResearchService.run_experiment(path)
    assert state["status"] == "COMPLETED" and state["exact_evaluations"] == 0
    records = ResearchStore(path).all("candidate")
    assert len(records) == 1 and records[0]["validation_state"] == "REJECTED"


def test_rewired_recipe_rejects_fallback_violating_constraints(monkeypatch):
    from dataclasses import replace

    from toposc_lab.research.space import FixedConnectivitySpace

    base = square(6, 6)
    invalid = replace(base, edges=base.edges[:-1])
    monkeypatch.setattr(FixedConnectivitySpace, "sample", lambda *a: invalid)
    space = EmbeddedSamplingSpace(
        families=["rewired_square"], side=6, domain=asdict(EmbeddedDomain(bounds=(0, 5, 0, 5)))
    )
    with pytest.raises(ValueError, match="constraints"):
        space.generate("rewired_square", 17)


def test_invalid_primary_evidence_keeps_raw_values_but_not_valid_scalar_score():
    from toposc_lab.research.studio_results import scalar_observables

    raw = {
        "status": "completed",
        "primary_valid": False,
        "primary_invalid_reason": "singular_or_phs",
        "metrics": {"quality": 0.0, "success": False},
    }
    values = scalar_observables(raw)
    assert values["quality"] is None and values["success"] is None
    assert raw["metrics"]["quality"] == 0.0


def test_small_clean_only_source_followup_enables_disorder(tmp_path):
    from toposc_lab.research.config import ExperimentConfig
    from toposc_lab.research.service import ResearchService
    from toposc_lab.research.storage import ResearchStore

    old = ExperimentConfig(
        objective="clean_quality", space={"side": 3}, physics={"validators": ["spectrum"]}
    ).to_dict()
    store = ResearchStore(tmp_path / "source", create=True)
    store.save("config", old)
    store.save("manifest", {"experiment_id": "old-clean"})
    store.save(
        "candidate",
        {"id": "small", "family": "regular", "geometry": geometry_to_payload(square(3, 3))},
        "small",
    )
    config = ResearchService.follow_up(
        store.directory,
        ["small"],
        widths=[2],
        seeds=[22201, 22202],
        output_directory=str(tmp_path / "new"),
    )
    assert "robustness" in config["physics"]["validators"]
    assert config["physics"]["domain"]["bulk_inset"] < 1
    from toposc_lab.research.physics import create_evaluator

    evaluator = create_evaluator(ExperimentConfig.from_dict(config).physics_protocol(), provenance=None)
    assert len(evaluator.plan()) == 3
