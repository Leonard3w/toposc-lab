"""Configuration contracts without a research run, worker or geometry search."""

from __future__ import annotations

import copy
import json

import pytest

from toposc_lab.research.config import ExperimentConfig
from toposc_lab.research.studio_config import (
    PRESETS,
    field_specs,
    planned_evaluations,
    preset,
    preview_config,
    switch_geometry_space,
)


def test_historical_default_fingerprint_is_unchanged():
    config = ExperimentConfig()
    assert config.fingerprint == "513fe0b990081133e181d4fb558a58d9154de407b6b5193a37c6f123b25204c4"
    assert "studio" not in config.to_dict()
    assert ExperimentConfig.from_dict(config.to_dict()).fingerprint == config.fingerprint


def test_studio_requires_explicit_version():
    with pytest.raises(ValueError, match="schema_version 2"):
        ExperimentConfig(studio={"description": "Must not affect legacy hashes"})
    with pytest.raises(ValueError, match="unsupported configuration version"):
        ExperimentConfig(schema_version=3)


@pytest.mark.parametrize("name", PRESETS)
def test_presets_are_resolved_roundtrippable_configs(name, tmp_path, monkeypatch):
    from toposc_lab.research.studio_space import EmbeddedSamplingSpace

    def fail_generate(*args, **kwargs):
        raise AssertionError("configuration must never generate a geometry")

    monkeypatch.setattr(EmbeddedSamplingSpace, "generate", fail_generate)
    raw = preset(name)
    config = ExperimentConfig.from_dict(raw)
    config.validate_plugins()
    target = tmp_path / "config.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    assert ExperimentConfig.from_file(target).to_dict() == raw
    assert ExperimentConfig.from_file(target).fingerprint == config.fingerprint
    assert preview_config(raw, check_output=False)["errors"] == []


def test_presets_are_independent_and_include_existing_space_defaults():
    first = preset("phase19_like")
    first["space"]["generator"]["n_sites"] = 12
    assert preset("phase19_like")["space"]["generator"]["n_sites"] == 64
    assert set(preset("phase19_like")["space"]) >= {
        "min_degree",
        "max_degree",
        "max_bond_length",
        "bond_tolerance",
        "forbid_crossings",
        "site_crossings",
        "rewires",
    }


def test_schema_two_direct_defaults_are_sampling_and_locked():
    config = ExperimentConfig(schema_version=2)
    assert config.algorithm == "embedded_random"
    assert config.objective == "robustness_quality_mean"
    assert config.baselines == []
    assert set(config.studio["variables"].values()) == {"locked"}
    assert config.physics["adapter_id"] == "phase20.configurable-chiral-p-wave.v1"


@pytest.mark.parametrize("key", ["hopping", "chemical_potential", "pairing"])
def test_physics_values_editable_but_search_variables_rejected(key):
    raw = preset("quick_test")
    raw["physics"][key] = 0.75
    config = ExperimentConfig.from_dict(raw)
    assert getattr(config.physics_protocol(), key) == 0.75
    raw["studio"]["variables"][key] = "variable"
    with pytest.raises(ValueError, match="cannot vary physics"):
        ExperimentConfig.from_dict(raw)


@pytest.mark.parametrize("variable", ["coordinates", "connectivity"])
def test_free_generators_cannot_change_locks(variable):
    raw = preset("phase19_like")
    raw["studio"]["variables"][variable] = "locked"
    with pytest.raises(ValueError, match=f"locked {variable}"):
        ExperimentConfig.from_dict(raw)


def test_regular_square_rejects_illusory_variability():
    raw = preset("quick_test")
    raw["studio"]["variables"]["coordinates"] = "variable"
    with pytest.raises(ValueError, match="fixed coordinates"):
        ExperimentConfig.from_dict(raw)


@pytest.mark.parametrize("mode", ["weighted", "pareto"])
def test_unsupported_objective_modes_fail(mode):
    raw = preset("quick_test")
    raw["studio"]["objective_mode"] = mode
    with pytest.raises(ValueError, match="not supported"):
        ExperimentConfig.from_dict(raw)


def test_unsupported_space_strategy_combination():
    raw = preset("quick_test")
    raw["algorithm"] = "map_elites"
    with pytest.raises(ValueError, match="embedded_random only"):
        ExperimentConfig.from_dict(raw)


def test_optimization_requires_explicit_objective_and_resolves_search_defaults():
    raw = {"schema_version": 2, "geometry_space": "fixed_connectivity", "algorithm": "evolution"}
    with pytest.raises(ValueError, match="single objective"):
        ExperimentConfig.from_dict(raw)
    raw["studio"] = {"objective_mode": "single"}
    config = ExperimentConfig.from_dict(raw)
    config.validate_plugins()
    assert config.search["quality_parent_probability"] == 0.0
    assert config.search["quality_parent_fraction"] == 0.25
    assert config.search["initialization_probability"] == 0.0
    assert config.search["mutation_rates"]
    assert config.search["batch_novelty"] is False


def test_preview_freezes_exact_resolved_config_and_checks_output(tmp_path):
    raw = preset("quick_test")
    raw["output_directory"] = str(tmp_path / "new")
    preview = preview_config(raw)
    config = ExperimentConfig.from_dict(preview["config"])
    assert not preview["errors"]
    assert preview["planned_evaluations"] == 3
    assert preview["config_sha256"] == config.fingerprint
    (tmp_path / "new").mkdir()
    (tmp_path / "new" / "existing.txt").write_text("historical", encoding="utf-8")
    assert preview_config(raw)["errors"]
    assert not preview_config(raw, check_output=False)["errors"]


def test_preview_separates_stages_from_attempt_cap():
    raw = preset("phase19_like")
    raw["exact_budget"] = 29  # ten stages per candidate: only two complete candidates
    preview = preview_config(raw, check_output=False)
    assert preview["planned_evaluations"] == 20
    assert "upper bound" in preview["summary"]
    assert any("truncate" in warning for warning in preview["warnings"])


def test_preview_regular_geometry_once_and_too_small_budget():
    raw = preset("quick_test")
    raw.update(cycles=3, candidate_budget=3)
    config = ExperimentConfig.from_dict(raw)
    assert planned_evaluations(config) == 3
    raw["exact_budget"] = 2
    assert any("complete candidate" in message for message in preview_config(raw, False)["errors"])


def test_duplicate_seeds_and_missing_reproducibility_fail():
    raw = preset("quick_test")
    raw["physics"]["disorder_seeds"] = [1, 1]
    with pytest.raises(ValueError, match="unique"):
        ExperimentConfig.from_dict(raw)
    raw = preset("quick_test")
    raw["studio"]["output"]["save_raw_realizations"] = False
    with pytest.raises(ValueError, match="saves are required"):
        ExperimentConfig.from_dict(raw)
    raw = preset("quick_test")
    raw["studio"]["output"]["save_plots"] = True
    assert ExperimentConfig.from_dict(raw).studio["output"]["save_plots"] is True


def test_physical_domain_has_one_source_and_reaches_space():
    raw = preset("quick_test")
    raw["physics"]["domain"]["boundary_shell"] = 0.5
    config = ExperimentConfig.from_dict(raw)
    assert "domain" not in config.space
    assert config.geometry_space_instance().physical_domain.boundary_shell == 0.5
    raw["space"]["domain"] = copy.deepcopy(raw["physics"]["domain"])
    with pytest.raises(ValueError, match="canonical physical domain"):
        ExperimentConfig.from_dict(raw)


def test_metadata_has_scientific_and_inherited_controls_with_dynamic_visibility():
    specs = {item["path"]: item for item in field_specs(preset("phase19_like"))}
    for path in (
        "physics.hopping",
        "physics.disorder_widths",
        "physics.energy_cutoff",
        "physics.domain.bulk_inset",
        "space.generator.minimum_separation",
        "space.min_degree",
        "studio.variables.connectivity",
        "blas_threads",
    ):
        assert path in specs
        assert set(specs[path]) >= {
            "label",
            "type",
            "default",
            "description",
            "category",
            "expert",
            "editable",
            "searchable",
        }
    assert specs["physics.domain.bulk_inset"]["editable"]
    assert specs["studio.variables.connectivity"]["choices"] == ["locked", "variable"]
    assert not specs["physics.solver"]["editable"]
    regular = {item["path"] for item in field_specs(preset("quick_test"))}
    assert "space.generator.minimum_separation" not in regular
    assert "space.rewires" not in regular
    assert "surrogate.n_estimators" not in regular


def test_bad_unknown_config_and_finite_size_extension_fail():
    raw = preset("quick_test")
    raw["studio"]["unknown_science"] = 4
    with pytest.raises(ValueError, match="unknown studio"):
        ExperimentConfig.from_dict(raw)
    raw = preset("quick_test")
    raw["physics"]["finite_size"] = True
    with pytest.raises(ValueError, match="finite-size extension"):
        ExperimentConfig.from_dict(raw)


def test_invalid_draft_keeps_family_controls_and_explicit_locks():
    raw = preset("quick_test")
    raw["space"]["families"] = ["amorphous_planar"]
    raw["physics"]["disorder_seeds"] = [4, 4]
    before = copy.deepcopy(raw)
    specs = {item["path"]: item for item in field_specs(raw)}
    assert "space.generator.minimum_separation" in specs
    assert specs["studio.variables.coordinates"]["default"] == "locked"
    assert raw == before
    assert preview_config(raw, False)["errors"]


def test_space_switch_is_explicit_and_creates_compatible_draft():
    raw = preset("quick_test")
    switched = switch_geometry_space(raw, "fixed_connectivity")
    assert switched["physics"] == raw["physics"]
    assert switched["seed"] == raw["seed"]
    assert switched["algorithm"] == "random"
    assert switched["studio"]["variables"]["connectivity"] == "variable"
    ExperimentConfig.from_dict(switched).validate_plugins()
    returned = switch_geometry_space(switched, "embedded_sampling")
    assert returned["algorithm"] == "embedded_random"
    assert returned["space"]["families"] == ["regular"]
    ExperimentConfig.from_dict(returned).validate_plugins()
    follow_up = switch_geometry_space(raw, "fixed_candidates")
    assert follow_up["space"]["candidates"] == []
    assert "space.candidates" in {item["path"] for item in field_specs(follow_up)}
    assert preview_config(follow_up, False)["errors"]
    assert raw == preset("quick_test")


def test_frozen_numerical_definitions_recorded_and_read_only():
    from toposc_lab.research.studio_physics import FROZEN_NUMERICAL_SETTINGS

    raw = preset("quick_test")
    assert raw["studio"]["definitions"] == FROZEN_NUMERICAL_SETTINGS
    specs = [item for item in field_specs(raw) if item["path"].startswith("studio.definitions.")]
    assert specs and all(not item["editable"] and item["expert"] for item in specs)
    raw["studio"]["definitions"]["success_threshold"] = 0.8
    with pytest.raises(ValueError, match="immutable"):
        ExperimentConfig.from_dict(raw)


def test_resource_and_complete_candidate_guards_apply_outside_preview():
    raw = preset("quick_test")
    raw["exact_budget"] = 2
    with pytest.raises(ValueError, match="complete candidate"):
        ExperimentConfig.from_dict(raw)
    raw = preset("quick_test")
    raw["space"]["side"] = 64
    raw["physics"]["domain"]["bounds"] = [0, 63, 0, 63]
    with pytest.raises(ValueError, match="8 GiB"):
        ExperimentConfig.from_dict(raw)


def test_fixed_connectivity_requires_matching_domain_and_boundary():
    raw = switch_geometry_space(preset("quick_test"), "fixed_connectivity")
    raw["physics"]["domain"]["bounds"] = [0, 6, 0, 6]
    with pytest.raises(ValueError, match="domain bounds"):
        ExperimentConfig.from_dict(raw)
    raw["physics"]["domain"]["bounds"] = [0, 7, 0, 7]
    raw["physics"]["domain"]["boundary_shell"] = 1.0
    with pytest.raises(ValueError, match="below unit"):
        ExperimentConfig.from_dict(raw)
