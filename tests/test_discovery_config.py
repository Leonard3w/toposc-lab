from dataclasses import replace

import pytest

from toposc_lab.discovery import DiscoveryConfig, default_generator
from toposc_lab.discovery.storage import atomic_json, read_json, writer_lease


def test_scoped_policy_and_frozen_definitions():
    config = DiscoveryConfig()
    assert config.resolved_generator == "patch"
    assert default_generator("new_space") == "evolution"
    for options in (
        {"stratum": "new_space"},
        {"success_threshold": 0.1},
        {"minimum_distance": 0.0},
        {"generator": "rl"},
        {"seed": True},
        {"pool_size": 2},
        {"disorder_samples": 1},
    ):
        with pytest.raises(ValueError):
            replace(config, **options)
    assert replace(config, generator="random").resolved_generator == "random"
    assert config.exact_attempt_cap == 80


def test_atomic_storage_corruption_and_lease(tmp_path):
    path = tmp_path / "state.json"
    atomic_json(path, {"cycle": 1})
    assert read_json(path) == {"cycle": 1}
    path.write_text(path.read_text().replace('"cycle": 1', '"cycle": 2'))
    with pytest.raises(ValueError, match="corrupt"):
        read_json(path)
    with writer_lease(tmp_path), pytest.raises(OSError), writer_lease(tmp_path):
        pass


def test_json_config_is_immutable_and_has_identical_fingerprint():
    import json
    from dataclasses import asdict

    config = DiscoveryConfig()
    payload = json.loads(json.dumps(asdict(config)))
    loaded = DiscoveryConfig(**payload)
    payload["probe"][0] = 0.0
    payload["kappas"][0] = 100.0
    assert loaded.probe == (2.5, 2.5)
    assert loaded.kappas == (0.1, 0.2, 0.3)
    assert loaded.fingerprint == config.fingerprint
    for name in ("hopping", "pairing", "chirality", "schema_version", "onsite_width"):
        with pytest.raises((TypeError, ValueError)):
            replace(config, **{name: True})
