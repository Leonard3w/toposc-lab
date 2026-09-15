"""Group analysis uses stored fixtures only, never scientific evaluations."""

import csv
import json
from dataclasses import asdict, replace
from hashlib import sha256

import pytest
from dataset_fixtures import representative_dataset_record

from toposc_lab.data import create_dataset_record, record_to_dict
from toposc_lab.data.dataset_schema import ModelParametersRecord, ObservableResultRecord
from toposc_lab.discovery.config import DiscoveryConfig
from toposc_lab.discovery.storage import atomic_json, read_json
from toposc_live.groups import (
    GroupDefinition,
    GroupMember,
    export_group,
    load_group,
    summarize,
)
from toposc_live.models import CampaignSnapshot, CandidateSnapshot
from toposc_live.processes import ProcessService
from toposc_live.reader import CampaignReader


def snapshot(path, seed, qualities=(0.1, 0.2), status="Completed"):
    config = asdict(DiscoveryConfig(seed=seed, cycles=2, generator="patch"))
    candidates = tuple(
        sorted(
            (
                CandidateSnapshot(
                    f"record-{i}",
                    seed=seed * 100 + i,
                    cycle=i,
                    quality=q,
                    campaign_seed=seed,
                    generator="random" if i == 0 else "patch",
                    topology="valid",
                    indices=[1, 1, 1],
                    minimum_abs_energy=0.03,
                )
                for i, q in enumerate(qualities)
            ),
            key=lambda c: -c.quality,
        )
    )
    return CampaignSnapshot(
        path,
        str(seed),
        status,
        config,
        {
            "Completed cycles": len(qualities),
            "Generated": 12,
            "Valid pool": 8,
            "Invalid": 3,
            "Duplicates / near-duplicates": 1,
            "Exact completed": 4,
            "Exact attempts charged": 5,
        },
        candidates=candidates,
    )


def aggregate(*snapshots, roles=None):
    roles = roles or ["included"] * len(snapshots)
    definition = GroupDefinition(
        "Fixture",
        tuple(
            GroupMember(
                s.directory,
                role,
                "Windows persistence failure" if role != "included" else "",
            )
            for s, role in zip(snapshots, roles)
        ),
    )
    manifest = {
        "environment": {"python": "fixture"},
        "external_record_ids": [],
        "source_sha256": "source",
        "provenance": {"runtime": {"benchmark": "fixture"}},
    }
    return summarize(definition, tuple((s, manifest) for s in snapshots))


def test_compatible_group_maxima_success_and_seed_identity(tmp_path):
    a = snapshot(tmp_path / "a", 10, (0.1, 0.2))
    b = snapshot(tmp_path / "b", 11, (0.3, 0.15))
    group = aggregate(a, b)
    stats = group.statistics
    assert stats["global_best_exact_quality"] == 0.3
    assert [m["best_exact_quality"] for m in group.members] == [0.2, 0.3]
    assert stats["candidate_success_count"] == 2  # Inclusive frozen threshold.
    assert stats["candidate_success_fraction"] == 0.5
    assert stats["seed_success_count"] == 2
    assert stats["per_seed_best_mean"] == 0.25
    assert stats["per_seed_best_median"] == 0.25
    assert stats["per_seed_best_sample_sd"] == pytest.approx(0.070710678)
    assert stats["Generated"] == 24
    assert [c.quality for c in group.candidates] == [0.3, 0.2, 0.15, 0.1]
    assert group.candidates[0].campaign_seed == 11
    assert group.candidates[0].seed == 1100
    assert len({c.selection_key for c in group.candidates}) == 4
    assert group.curves["11"] == ((1, 0.3), (2, 0.3))
    assert group.curves["Group mean"] == ((1, 0.2), (2, 0.25))


@pytest.mark.parametrize(
    "field,value",
    [
        ("stratum", "other"),
        ("generator", "random"),
        ("model", "other"),
        ("success_threshold", 0.21),
        ("batch_size", 8),
        ("pool_size", 16),
        ("cycles", 4),
        ("tolerance", 1e-8),
        ("disorder_samples", 8),
        ("minimum_distance", 0.1),
        ("retry_reserve", 9),
    ],
)
def test_incompatible_fields_rejected(tmp_path, field, value):
    a, b = snapshot(tmp_path / "a", 1), snapshot(tmp_path / "b", 2)
    b = replace(b, config={**b.config, field: value})
    with pytest.raises(ValueError, match="Incompatible"):
        aggregate(a, b)


def test_complete_failed_incomplete_and_technical_dropout(tmp_path):
    a = snapshot(tmp_path / "a", 1)
    b = snapshot(tmp_path / "b", 2, (0.15,), "Failed")
    c = snapshot(tmp_path / "c", 3, (0.1,), "Running")
    d = snapshot(tmp_path / "d", 17101, (0.99,), "Failed")
    group = aggregate(a, b, c, d, roles=["included"] * 3 + ["technical_dropout"])
    assert [
        group.statistics[k]
        for k in ("complete_seeds", "failed_seeds", "incomplete_seeds", "technical_dropouts")
    ] == [1, 1, 1, 1]
    assert group.statistics["all_member_failed_seeds"] == 2
    assert group.statistics["global_best_exact_quality"] == 0.2
    assert group.statistics["exact_candidates"] == 4
    assert "17101" not in group.curves
    assert len(group.curves["Group mean"]) == 1
    assert all(c.campaign_seed != 17101 for c in group.candidates)
    assert any("provisional" in w for w in group.warnings)


def test_unknown_values_are_not_zero(tmp_path):
    a = snapshot(tmp_path / "a", 1, (), "Unknown")
    a = replace(a, progress={"Completed cycles": 0})
    group = aggregate(a)
    assert group.statistics["Generated"] is None
    assert group.statistics["global_best_exact_quality"] is None
    assert group.statistics["candidate_success_fraction"] is None
    assert group.curves["Group mean"] == ()


def test_duplicate_seed_and_expected_seed_rejected(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        aggregate(snapshot(tmp_path / "a", 1), snapshot(tmp_path / "b", 1))
    a = snapshot(tmp_path / "a", 1)
    definition = GroupDefinition("wrong", (GroupMember(a.directory, expected_seed=2),))
    with pytest.raises(ValueError, match="Expected campaign seed"):
        summarize(definition, ((a, {}),))


def test_export_json_csv_readonly_and_no_overwrite(tmp_path):
    group = aggregate(snapshot(tmp_path / "campaign", 1))
    group.definition.members[0].directory.mkdir()
    original = tmp_path / "campaign/evidence.json"
    original.write_text("stored scientific evidence")
    export_group(group, tmp_path / "summary.json", 1)
    data = json.loads((tmp_path / "summary.json").read_text())
    assert len(data["top_candidates"]) == 1
    assert data["top_candidates"][0]["Campaign Seed"] == 1
    assert data["top_candidates"][0]["Exact Eval Seed"] == 101
    export_group(group, tmp_path / "summary.csv")
    with (tmp_path / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {r["row_type"] for r in rows} == {"summary", "member", "candidate"}
    with pytest.raises(ValueError, match="outside"):
        export_group(group, original)
    with pytest.raises(FileExistsError):
        export_group(group, tmp_path / "summary.json")
    assert original.read_text() == "stored scientific evidence"


@pytest.fixture
def stored_group(tmp_path):
    members = []
    for seed in (100, 101):
        root = tmp_path / f"campaign-{seed}"
        config = DiscoveryConfig(seed=seed, cycles=1, generator="patch")
        atomic_json(
            root / "manifest.json",
            {
                "config": asdict(config),
                "config_sha256": config.fingerprint,
                "environment": {"python": "fixture"},
                "source_sha256": "source",
                "external_record_ids": [],
            },
        )
        atomic_json(
            root / "checkpoint.json", {"completed_cycles": 1, "config_sha256": config.fingerprint}
        )
        cycle = root / "cycle-0000"
        atomic_json(
            cycle / "plan.json",
            {
                "method": "random",
                "audit": {"attempted": 10, "invalid": 1, "duplicate": 1},
                "geometries": ["fixture"] * 8,
            },
        )
        for index in range(4):
            record = representative_dataset_record(seed=seed * 10 + index)
            record = create_dataset_record(
                geometry=record.geometry,
                model=ModelParametersRecord(
                    model_name="chiral_p_wave",
                    model_version="1",
                    parameters={
                        "hopping": 1.0,
                        "pairing": 1.0,
                        "chemical_potential": 2.0,
                        "chirality": 1,
                    },
                ),
                spectrum=record.spectrum,
                topology=record.topology,
                robustness=record.robustness,
                provenance=record.provenance,
                observables=(
                    ObservableResultRecord(
                        kind="finite_geometry_quality", version="1", values={"quality": index / 10}
                    ),
                ),
            )
            atomic_json(
                cycle / f"candidate-{index:04d}.json",
                {
                    "status": "complete",
                    "record": record_to_dict(record),
                    "exact_confirmation": True,
                },
            )
            atomic_json(
                root / f"attempts/{index:06d}.json",
                {"status": "complete", "stage": f"cycle-0000/base-{index:04d}.json"},
            )
        inventory = {p.name: sha256(p.read_bytes()).hexdigest() for p in cycle.glob("*.json")}
        atomic_json(cycle / "commit.json", {"cycle": 0, "inventory": inventory})
        members.append(GroupMember(root, expected_seed=seed))
    return GroupDefinition("Stored fixture", tuple(members)), CampaignReader(
        ProcessService(tmp_path / "app-state")
    )


def test_validated_stored_group_repeated_read_no_mutation(stored_group):
    definition, reader = stored_group
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for m in definition.members
        for p in m.directory.rglob("*")
        if p.is_file()
    }
    first = load_group(reader, definition)
    reads = reader.cache.reads
    second = load_group(reader, definition)
    assert first == second
    assert reader.cache.reads == reads
    assert first.statistics["exact_candidates"] == 8
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


@pytest.mark.parametrize(
    "damage", ["checksum", "inventory", "config", "record", "missing", "extra"]
)
def test_group_fails_closed_on_invalid_evidence(stored_group, damage):
    definition, reader = stored_group
    root = definition.members[0].directory
    path = root / "cycle-0000/candidate-0000.json"
    if damage == "checksum":
        path.write_text("{}")
    elif damage == "inventory":
        atomic_json(path, {"status": "complete"})
    elif damage == "config":
        manifest = read_json(root / "manifest.json")
        manifest["config"]["seed"] = 999
        atomic_json(root / "manifest.json", manifest)
    elif damage == "record":
        outcome = read_json(path)
        outcome["record"]["record_id"] = "bad"
        atomic_json(path, outcome)
        commit = read_json(root / "cycle-0000/commit.json")
        commit["inventory"][path.name] = sha256(path.read_bytes()).hexdigest()
        atomic_json(root / "cycle-0000/commit.json", commit)
    elif damage == "missing":
        path.unlink()
    else:
        atomic_json(root / "cycle-0000/candidate-9999.json", read_json(path))
    with pytest.raises(ValueError):
        load_group(reader, definition)


def test_definition_relative_paths_and_dropout_reason(tmp_path):
    path = tmp_path / "group.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "Saved",
                "members": [
                    {
                        "directory": "campaign",
                        "expected_seed": 17101,
                        "role": "technical_dropout",
                        "reason": "Windows write failure",
                    }
                ],
            }
        )
    )
    definition = GroupDefinition.read(path)
    assert definition.members[0].directory == (tmp_path / "campaign").resolve()
    assert definition.members[0].role == "technical_dropout"


@pytest.mark.parametrize(
    "payload",
    [[], None, {"schema_version": 1}, {"schema_version": 1, "name": "bad", "members": [None]}],
)
def test_malformed_definition_rejected_cleanly(tmp_path, payload):
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        GroupDefinition.read(path)


@pytest.mark.parametrize("key", ["environment", "external_record_ids", "provenance"])
def test_incompatible_provenance_scope_rejected(tmp_path, key):
    a, b = snapshot(tmp_path / "a", 1), snapshot(tmp_path / "b", 2)
    definition = GroupDefinition("fixture", (GroupMember(a.directory), GroupMember(b.directory)))
    one = {
        "environment": {"python": "same"},
        "external_record_ids": [],
        "provenance": {"runtime": {"benchmark": "one"}},
    }
    two = {**one, key: {"runtime": {"benchmark": "two"}} if key == "provenance" else ["different"]}
    with pytest.raises(ValueError, match="Incompatible stored"):
        summarize(definition, ((a, one), (b, two)))


def test_source_difference_is_visible_and_does_not_change_resume_policy(tmp_path):
    a, b = snapshot(tmp_path / "a", 1), snapshot(tmp_path / "b", 2)
    definition = GroupDefinition("fixture", (GroupMember(a.directory), GroupMember(b.directory)))
    group = summarize(definition, ((a, {"source_sha256": "old"}), (b, {"source_sha256": "new"})))
    assert any("Source fingerprints differ" in warning for warning in group.warnings)
    assert [m["source_sha256"] for m in group.members] == ["old", "new"]


def test_plots_setting_is_not_a_scientific_incompatibility(tmp_path):
    a, b = snapshot(tmp_path / "a", 1), snapshot(tmp_path / "b", 2)
    b = replace(b, config={**b.config, "plots": False})
    assert aggregate(a, b).statistics["included_seeds"] == 2


def test_checkpoint_change_during_group_read_is_rejected(stored_group, monkeypatch):
    definition, reader = stored_group
    path = definition.members[0].directory / "checkpoint.json"
    original = reader.cache.read
    reads = 0

    def changing(target, warnings, **kwargs):
        nonlocal reads
        if target == path:
            reads += 1
            if reads == 2:
                data = read_json(path)
                data["completed_cycles"] = 0
                atomic_json(path, data)
        return original(target, warnings, **kwargs)

    monkeypatch.setattr(reader.cache, "read", changing)
    with pytest.raises(ValueError, match="changed during group preflight"):
        load_group(reader, definition)


def test_group_repeated_refresh_during_atomic_attempt_writes(stored_group):
    from concurrent.futures import ThreadPoolExecutor

    definition, reader = stored_group
    root = definition.members[0].directory
    path = root / "attempts/000000.json"
    payload = read_json(path)

    def write():
        for i in range(25):
            atomic_json(path, {**payload, "fixture_refresh": i})

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(write)
        for _ in range(4):
            try:
                group = load_group(reader, definition)
                assert group.statistics["exact_candidates"] == 8
            except ValueError as error:
                assert any(
                    message in str(error)
                    for message in ("changed while being read", "Permission denied")
                )
        future.result()
    assert load_group(reader, definition).statistics["Exact completed"] == 8
