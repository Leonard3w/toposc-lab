import json
import os
import platform
from dataclasses import asdict, replace
from hashlib import sha256
from importlib.metadata import version

import pytest

from toposc_lab.discovery.config import DiscoveryConfig
from toposc_lab.discovery.storage import atomic_json
from toposc_live.configuration import ConfigurationAdapter


def test_official_mapping_preserves_every_other_field(tmp_path):
    adapter = ConfigurationAdapter()
    config = adapter.build(generator="random", seed=42, cycles=5, batch_size=8, pool_size=12)
    expected = replace(
        DiscoveryConfig(), generator="random", seed=42, cycles=5, batch_size=8, pool_size=12
    )
    assert asdict(config) == asdict(expected)
    plans = adapter.plans("test", tmp_path / "new", config, seed_count=3, one_cycle=True)
    assert [p.config.seed for p in plans] == [42, 43, 44]
    assert all(p.iterations == 1 for p in plans)
    for p in plans:
        assert asdict(replace(p.config, seed=42)) == asdict(config)
    assert not (tmp_path / "new").exists()
    summary = adapter.summary(plans)
    assert str(config.exact_attempt_cap * 3) in summary
    assert "0.2" in summary and "No Majorana" in summary


@pytest.mark.parametrize("generator", ["patch", "random", "evolution", "coverage", "auto"])
def test_engine_supported_generators(generator):
    assert (
        ConfigurationAdapter()
        .build(generator=generator, seed=1, cycles=1, batch_size=4, pool_size=8)
        .generator
        == generator
    )


@pytest.mark.parametrize(
    "changes", [{"generator": "invented"}, {"cycles": 0}, {"pool_size": 3}, {"seed": -1}]
)
def test_unsupported_configs_rejected(changes):
    values = {"generator": "auto", "seed": 1, "cycles": 1, "batch_size": 4, "pool_size": 8}
    values.update(changes)
    with pytest.raises(ValueError):
        ConfigurationAdapter().build(**values)


def test_existing_directory_and_blank_name_rejected(tmp_path):
    adapter = ConfigurationAdapter()
    with pytest.raises(ValueError, match="must not exist"):
        adapter.plans("name", tmp_path, DiscoveryConfig())
    with pytest.raises(ValueError, match="name"):
        adapter.plans(" ", tmp_path / "new", DiscoveryConfig())


@pytest.fixture
def resumable(tmp_path, monkeypatch):
    from toposc_lab.active_learning import benchmark
    from toposc_lab.discovery import engine

    class Provenance:
        def __init__(self):
            self.runtime = {"source_sha256": "fixture-source"}

    monkeypatch.setattr(benchmark, "source_provenance", lambda root: Provenance())
    config = DiscoveryConfig(cycles=2)
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"source fixture")
    manifest = {
        "config": asdict(config),
        "config_sha256": config.fingerprint,
        "source_sha256": "fixture-source",
        "external_record_ids": [],
        "source_zip_sha256": sha256(archive.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "platform": platform.platform(),
            "threads": {n: os.environ.get(n) for n in engine.THREADS},
        },
    }
    atomic_json(tmp_path / "manifest.json", manifest)
    atomic_json(
        tmp_path / "checkpoint.json", {"completed_cycles": 1, "config_sha256": config.fingerprint}
    )
    return tmp_path, manifest


def test_resume_compatible_read_only(resumable):
    root, _ = resumable
    before = {p: p.read_bytes() for p in root.iterdir()}
    plan = ConfigurationAdapter().resume_plan(root)
    assert plan.resume
    assert plan.config == DiscoveryConfig(cycles=2)
    assert before == {p: p.read_bytes() for p in root.iterdir()}


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("source_sha256", "other", "source mismatch"),
        ("environment", {}, "runtime"),
        ("external_record_ids", ["external"], "exclusion"),
        ("config_sha256", "bad", "fingerprint"),
        ("source_zip_sha256", "bad", "integrity"),
    ],
)
def test_resume_rejects_incompatible(resumable, key, value, match):
    root, manifest = resumable
    manifest[key] = value
    atomic_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match=match):
        ConfigurationAdapter().resume_plan(root)


def test_scientific_mutation_rejected_by_official_constructor(resumable):
    root, manifest = resumable
    manifest["config"]["success_threshold"] = 0.1
    atomic_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="frozen"):
        ConfigurationAdapter().resume_plan(root)


def test_no_new_config_schema_serialization(tmp_path):
    config = ConfigurationAdapter().defaults()
    assert DiscoveryConfig(**json.loads(json.dumps(asdict(config)))) == config
