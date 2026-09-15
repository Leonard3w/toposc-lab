"""Stored fixture tests only: no exact engine, generator or eigensolver calls."""

import base64
import json
from dataclasses import asdict
from hashlib import sha256

import pytest

from toposc_lab.discovery.config import DiscoveryConfig
from toposc_lab.discovery.storage import atomic_json
from toposc_live.models import majorana_label
from toposc_live.processes import ProcessService
from toposc_live.reader import CampaignReader, discover_campaigns, geometry_snapshot


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "campaign"
    config = DiscoveryConfig(cycles=2)
    atomic_json(
        root / "manifest.json", {"config": asdict(config), "config_sha256": config.fingerprint}
    )
    atomic_json(
        root / "checkpoint.json", {"completed_cycles": 1, "config_sha256": config.fingerprint}
    )
    record = {
        "schema_version": 1,
        "result_kind": "exact",
        "record_id": "record-a",
        "geometry": {"exact_id": "fixture-geometry"},
        "observables": [
            {
                "kind": "finite_geometry_quality",
                "values": {"quality": 0.125, "minimum_abs_energy": 0.03, "boundary_weight": 0.7},
            },
            {
                "kind": "discovery_majorana_diagnostics",
                "values": {"status": "diagnostics_only", "majorana_claim": False},
            },
        ],
        "provenance": {"seed": 101},
        "topology": [{"validity": "unresolved", "parameters": {"indices": [1, None, 1]}}],
    }
    atomic_json(
        root / "cycle-0000/candidate-0000.json",
        {
            "status": "complete",
            "record": record,
            "exact_confirmation": True,
            "common_ood": {"is_ood": True},
        },
    )
    atomic_json(
        root / "cycle-0000/plan.json",
        {
            "audit": {"attempted": 6, "invalid": 1, "duplicate": 1},
            "geometries": ["archived"] * 4,
            "selected_indices": [0],
            "candidate_ids": ["a"],
            "common_ood": {"a": {"is_ood": True}},
        },
    )
    atomic_json(
        root / "attempts/000000.json",
        {"number": 0, "status": "complete", "stage": "cycle-0000/base-0000.json"},
    )
    atomic_json(
        root / "attempts/000001.json",
        {"number": 1, "status": "started", "stage": "cycle-0001/base-0000.json"},
    )
    return root


@pytest.fixture
def reader(tmp_path):
    return CampaignReader(ProcessService(tmp_path / "app-state"))


def test_read_only_cached_repeated_refresh(campaign, reader):
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in campaign.rglob("*") if p.is_file()}
    first = reader.load(campaign)
    reads = reader.cache.reads
    second = reader.load(campaign)
    assert first == second
    assert reader.cache.reads == reads
    assert first.status == "Unknown"  # External liveness is not inferred from mtimes.
    assert first.best.quality == 0.125
    assert first.best.topology == "unresolved"
    assert first.best.indices == [1, None, 1]
    assert first.best.confirmation is True
    assert "majorana_claim=false" in first.best.details()
    assert first.progress["Exact completed"] == 1
    assert first.progress["Exact attempts charged"] == 2
    assert first.progress["Invalid"] == 1
    assert first.curves["auto (resolved generator unavailable until plan/report)"] == ((1, 0.125),)
    assert before == {
        p: (p.read_bytes(), p.stat().st_mtime_ns) for p in campaign.rglob("*") if p.is_file()
    }


def test_completed_and_configuration_mismatch(campaign, reader):
    atomic_json(campaign / "checkpoint.json", {"completed_cycles": 2, "config_sha256": "wrong"})
    snapshot = reader.load(campaign)
    assert any("mismatch" in warning for warning in snapshot.warnings)


def test_completed_campaign(campaign, reader):
    config = DiscoveryConfig(cycles=2)
    atomic_json(
        campaign / "checkpoint.json", {"completed_cycles": 2, "config_sha256": config.fingerprint}
    )
    assert reader.load(campaign).status == "Completed"


@pytest.mark.parametrize(
    "contents", ["{", "null", "[]", '{"payload":{},"sha256":"bad"}', '{"payload":null,"sha256":0}']
)
def test_corrupt_optional_artifacts(campaign, reader, contents):
    (campaign / "summary.json").write_text(contents)
    snapshot = reader.load(campaign)
    assert snapshot.best.quality == 0.125
    assert snapshot.warnings


def test_incomplete_record_recovers(campaign, reader):
    path = campaign / "cycle-0000/candidate-0000.json"
    saved = path.read_bytes()
    path.write_text("{")
    assert reader.load(campaign).best is None
    path.write_bytes(saved)
    assert reader.load(campaign).best.quality == 0.125


def test_missing_optional_records(campaign, reader):
    (campaign / "checkpoint.json").unlink()
    snapshot = reader.load(campaign)
    assert snapshot.best.robustness is None
    assert snapshot.best.geometry.message == "unavailable"
    assert snapshot.elapsed is None


def test_future_schema_fails_closed(campaign, reader):
    atomic_json(campaign / "manifest.json", {"config": {"schema_version": 99}})
    snapshot = reader.load(campaign)
    assert not snapshot.candidates
    assert "Unsupported" in snapshot.warnings[0]


def test_generic_generator_legacy_report_only(tmp_path, reader):
    root = tmp_path / "legacy"
    atomic_json(
        root / "manifest.json", {"config": {"generator": "future-generator/omega", "cycles": 1}}
    )
    atomic_json(root / "summary.json", {"completed_cycles": 1})
    atomic_json(
        root / "leaderboard.json",
        [
            {
                "kind": "exact",
                "record_id": "x",
                "geometry_id": "legacy",
                "quality": 0.11,
                "majorana_claim": False,
            },
            {"kind": "predicted", "quality": 10},
            {"kind": "exact", "record_id": "y", "quality": 0.15},
        ],
    )
    snapshot = reader.load(root)
    assert snapshot.status == "Completed"
    assert len(snapshot.candidates) == 2
    assert snapshot.best.record_id == "y"
    assert snapshot.best.generator == "future-generator/omega"
    assert snapshot.best.minimum_abs_energy is None
    assert snapshot.threshold is None
    assert snapshot.best.confirmation is None
    assert snapshot.curves == {}


def test_discovery_nested_recent_missing_and_bounded(campaign, tmp_path):
    deep = tmp_path / "a/b/c/d/e"
    atomic_json(deep / "manifest.json", {})
    found = discover_campaigns((tmp_path,), (tmp_path / "missing",))
    assert campaign in found
    assert deep not in found
    assert deep in discover_campaigns((), (deep,))


@pytest.mark.parametrize(
    "data",
    [
        {"majorana_claim": False},
        {"status": "diagnostics_only"},
        {"observables": [{"conventions": {"majorana_claim": False}}]},
    ],
)
def test_conservative_claim_labels(data):
    assert "majorana_claim=false" in majorana_label(data)


def test_no_inferred_claim():
    assert "unavailable" in majorana_label({"minimum_abs_energy": 0.0, "indices": [1, 1, 1]})


def test_geometry_generic_dimensions_and_bad_optional_data():
    geometry = geometry_snapshot({}, {"coordinates": [[0, 0, 3], [1, 0, 4]], "edges": [[0, 1]]})
    assert geometry.coordinates == ((0, 0, 3), (1, 0, 4))
    assert "projected" in geometry.message
    assert (
        "unavailable" in geometry_snapshot({}, {"coordinates": [[0, 0]], "edges": [[0, 9]]}).message
    )
    assert "unavailable" in geometry_snapshot({"geometry": {"archive": "corrupt"}}).message


def test_geometry_archive_public_decoder():
    from toposc_lab.geometry import chain, geometry_to_bytes

    archive = geometry_to_bytes(chain(4))
    result = geometry_snapshot(
        {
            "geometry": {
                "archive": base64.b64encode(archive).decode(),
                "archive_schema_version": 1,
                "exact_id": "geometry-archive-v1-sha256:" + sha256(archive).hexdigest(),
            }
        }
    )
    assert len(result.coordinates) == 4
    assert len(result.edges) == 3


def test_running_failed_interrupted_reconnect(campaign, reader, monkeypatch):
    for status in ("Running", "Failed", "Interrupted"):
        monkeypatch.setattr(
            reader.processes, "inspect", lambda directory, s=status: {"status": s, "state": {}}
        )
        assert reader.load(campaign).status == status


def test_updated_attempt_and_base_during_validation(campaign, reader):
    raw = json.loads((campaign / "cycle-0000/candidate-0000.json").read_text())["payload"]
    (campaign / "cycle-0000/candidate-0000.json").unlink()
    atomic_json(
        campaign / "cycle-0000/base-0000.json",
        {"status": "complete", "payload": {"record": raw["record"]}},
    )
    assert reader.load(campaign).best.confirmation is None
    atomic_json(
        campaign / "attempts/000001.json",
        {"number": 1, "status": "complete", "stage": "cycle-0001/base-0000.json"},
    )
    assert reader.load(campaign).progress["Exact completed"] == 2


def test_inspection_never_runs_science(campaign, reader, monkeypatch):
    import numpy as np

    from toposc_lab.discovery import DiscoveryEngine

    def forbidden(*args, **kwargs):
        raise AssertionError("Scientific work forbidden in a campaign reader")

    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    monkeypatch.setattr(np.linalg, "eigvalsh", forbidden)
    monkeypatch.setattr(DiscoveryEngine, "run", forbidden)
    assert reader.load(campaign).best.quality == 0.125


def test_malformed_optional_nested_fields(campaign, reader):
    path = campaign / "cycle-0000/candidate-0000.json"
    outcome = json.loads(path.read_text())["payload"]
    outcome["record"]["observables"].append({"kind": [], "values": "broken"})
    outcome["record"]["robustness"] = [None, "bad"]
    atomic_json(path, outcome)
    assert reader.load(campaign).best.quality == 0.125
