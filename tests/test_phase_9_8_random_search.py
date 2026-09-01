from __future__ import annotations

from dataclasses import replace
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    GeometryBoundaryComponent,
    ammann_beenker_patch,
    geometry_to_bytes,
    sierpinski_carpet,
    square,
)
from toposc_lab.search import (
    PHASE_9_8_CANDIDATES_PER_TRIAL,
    PHASE_9_8_CONFIRMATION_SEEDS,
    PHASE_9_8_DRY_RUN_SEEDS,
    PHASE_9_8_PROTOCOL_COMMIT,
    PHASE_9_8_PROTOCOL_IDENTIFIER,
    PHASE_9_8_SEARCH_TRIAL_SEEDS,
    PHASE_9_8_VALIDATION_SEEDS,
    GeometrySamplingRecipe,
    Phase98DisorderChannel,
    Phase98GeometryApplicability,
    RandomGeometrySamplingConfig,
    build_phase_9_8_primary_topology_inputs,
    build_phase_9_8_ammann_beenker_topology_inputs,
    build_phase_9_8_sierpinski_topology_inputs,
    evaluate_phase_9_8_primary_geometry,
    run_phase_9_8_dry_run,
    run_phase_9_8_random_search,
    sample_random_geometries,
    validate_phase_9_8_geometry,
)
from toposc_lab.search import random_search_experiment as experiment_module


def _square_reference():  # type: ignore[no-untyped-def]
    geometry = square(
        8,
        8,
        spacing=1.0,
        boundary_x="open",
        boundary_y="open",
    )
    return replace(
        geometry,
        boundary_components=(
            GeometryBoundaryComponent("outer", 0, geometry.boundary_sites),
        ),
    )


def test_protocol_revision_budgets_seed_roles_and_channels_are_frozen() -> None:
    assert PHASE_9_8_PROTOCOL_IDENTIFIER == "TOPOSC-P9.8-RS-001"
    assert PHASE_9_8_PROTOCOL_COMMIT == "dc967ec2876f221d7b4f362f6224d7d3716f395e"
    assert PHASE_9_8_CANDIDATES_PER_TRIAL == 32
    assert PHASE_9_8_DRY_RUN_SEEDS == tuple(range(9_799_900, 9_799_910))
    assert PHASE_9_8_SEARCH_TRIAL_SEEDS == tuple(range(9_800_000, 9_800_032))
    assert PHASE_9_8_VALIDATION_SEEDS == tuple(range(9_810_000, 9_810_064))
    assert PHASE_9_8_CONFIRMATION_SEEDS == tuple(range(9_820_000, 9_820_128))
    assert tuple(channel.value for channel in Phase98DisorderChannel) == (
        "onsite",
        "hopping",
        "pairing",
        "coordinate",
        "edge_removal",
        "node_removal",
    )


def test_frozen_generators_are_registered_stochastic_version_one() -> None:
    candidate = BUILTIN_GEOMETRY_GENERATORS.get("hard_core_planar_graph")
    reference = BUILTIN_GEOMETRY_GENERATORS.get("hard_core_planar_reference")

    assert candidate.stochastic and candidate.version == 1
    assert reference.stochastic and reference.version == 1


def test_reserved_dry_run_is_exact_reproducible_and_contract_valid() -> None:
    records = run_phase_9_8_dry_run()

    assert tuple(record.seed for record in records) == PHASE_9_8_DRY_RUN_SEEDS
    assert len({record.exact_geometry_id for record in records}) == len(records)
    assert all(record.complete_attempt_count >= 1 for record in records)
    assert all(record.proposal_count >= 64 for record in records)

    for record in records:
        geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
            "hard_core_planar_graph",
            seed=record.seed,
        )
        report = validate_phase_9_8_geometry(
            geometry,
            applicability=Phase98GeometryApplicability.CLEAN_PRIMARY,
        )
        assert report.is_applicable
        assert geometry.n_sites == 64
        assert geometry.n_edges == 112
        assert min(len(geometry.neighbors(site)) for site in geometry.site_indices) == 2
        assert max(len(geometry.neighbors(site)) for site in geometry.site_indices) == 4
        assert 24 <= len(geometry.boundary_sites) <= 32


def test_phase_9_1_sampler_remains_the_only_derived_seed_api() -> None:
    config = RandomGeometrySamplingConfig(
        recipes=(GeometrySamplingRecipe("hard_core_planar_graph"),),
        sample_count=2,
    )

    first = sample_random_geometries(config, seed=PHASE_9_8_DRY_RUN_SEEDS[0])
    second = sample_random_geometries(config, seed=PHASE_9_8_DRY_RUN_SEEDS[0])

    assert tuple(
        sample.generation_request.seed for sample in first.samples
    ) == tuple(sample.generation_request.seed for sample in second.samples)
    assert tuple(geometry_to_bytes(sample.geometry) for sample in first.samples) == tuple(
        geometry_to_bytes(sample.geometry) for sample in second.samples
    )
    assert all(
        sample.generation_request.seed
        == sample.geometry.metadata["generation"]["seed"]
        for sample in first.samples
    )


def test_candidate_and_reference_completion_modes_are_explicit() -> None:
    seed = PHASE_9_8_DRY_RUN_SEEDS[1]
    candidate = BUILTIN_GEOMETRY_GENERATORS.generate(
        "hard_core_planar_graph",
        seed=seed,
    )
    reference = BUILTIN_GEOMETRY_GENERATORS.generate(
        "hard_core_planar_reference",
        seed=seed,
    )

    assert candidate.metadata["construction_mode"] == "candidate"
    assert reference.metadata["construction_mode"] == "reference"
    assert candidate.metadata["completion_priority"] == (
        "pcg64_raw_word_then_source_target"
    )
    assert reference.metadata["completion_priority"] == (
        "edge_length_then_source_target"
    )
    assert geometry_to_bytes(candidate) != geometry_to_bytes(reference)


def test_clean_geometry_report_requires_explicit_outer_component() -> None:
    without_component = square(8, 8)
    invalid = validate_phase_9_8_geometry(
        without_component,
        applicability=Phase98GeometryApplicability.CLEAN_PRIMARY,
    )
    valid = validate_phase_9_8_geometry(
        _square_reference(),
        applicability=Phase98GeometryApplicability.CLEAN_PRIMARY,
    )

    assert not invalid.is_applicable
    assert tuple(issue.code for issue in invalid.issues) == (
        "boundary_component_count",
    )
    assert valid.is_applicable
    assert valid.issues == ()


def test_primary_topology_inputs_follow_component_major_and_voronoi_contract() -> None:
    geometry = _square_reference()
    inputs = build_phase_9_8_primary_topology_inputs(geometry)

    assert geometry.coordinates is not None
    assert inputs.basis_coordinates.shape == (128, 2)
    assert np.array_equal(inputs.basis_coordinates[:64], geometry.coordinates)
    assert np.array_equal(inputs.basis_coordinates[64:], geometry.coordinates)
    assert inputs.position_areas is not None
    assert np.all(inputs.position_areas > 0.0)
    assert np.isclose(np.sum(inputs.position_areas), 64.0, rtol=0.0, atol=1.0e-10)
    assert tuple(np.count_nonzero(mask) for mask in inputs.bulk_masks) == (16, 4)
    assert inputs.bott_periods == ((7.6, 7.6), (8.0, 8.0), (8.4, 8.4))
    assert inputs.localizer_probe == (3.5, 3.5)
    assert inputs.localizer_kappas == (0.1, 0.2, 0.3)


def test_square_calibration_passes_all_frozen_scientific_gates(tmp_path: Path) -> None:
    result = evaluate_phase_9_8_primary_geometry(
        _square_reference(),
        code_version="phase-9.8-square-calibration",
    )

    assert result.run.is_valid
    assert result.clean_eligible
    assert result.gate_reasons == ()
    assert result.topology_grid is not None
    assert [item.bott_index for item in result.topology_grid.bott] == [1, 1, 1]
    assert [item.chern_number for item in result.topology_grid.local_chern] == [1, 1]
    assert [item.local_chern_number for item in result.topology_grid.localizer] == [
        1,
        1,
        1,
    ]
    assert all(
        item.confidence.convergence_checked
        for item in result.topology_grid.representatives
    )
    assert result.localizer_protection_proxy == pytest.approx(0.231474243629927)
    assert result.boundary_signature is not None
    assert sorted(
        state
        for pair in result.boundary_signature.particle_hole_pairs
        for state in pair
    ) == sorted(
        state.state_index for state in result.boundary_signature.states
    )
    assert result.boundary_signature.maximum_pair_residual <= 1.0e-8
    assert result.boundary_signature.boundary_localized_count >= 4
    assert result.minimum_boundary_weight_first_four >= 0.80

    scientific_manifest = tmp_path / "square_scientific.json"
    experiment_module._write_json_exclusive(
        scientific_manifest,
        experiment_module._encode_scientific(result),
    )
    encoded = json.loads(scientific_manifest.read_text(encoding="utf-8"))
    assert encoded["clean_eligible"] is True
    assert encoded["topology_grid"]["localizer_protection_proxy"] == pytest.approx(
        0.231474243629927
    )
    assert len(encoded["boundary_signature"]["states"]) == 8


def test_sierpinski_inputs_do_not_fabricate_a_local_chern_bulk() -> None:
    geometry = sierpinski_carpet(order=2, spacing=1.0)
    inputs = build_phase_9_8_sierpinski_topology_inputs(geometry)

    assert inputs.bulk_masks == ()
    assert inputs.bott_periods == ((8.55, 8.55), (9.0, 9.0), (9.45, 9.45))
    assert inputs.localizer_probe == (4.5, 4.5)


def test_ammann_beenker_inputs_use_native_tile_areas_and_cut_boundary() -> None:
    geometry = ammann_beenker_patch(radius=4.0, spacing=1.0)
    inputs = build_phase_9_8_ammann_beenker_topology_inputs(geometry)

    assert geometry.n_sites == 57
    assert geometry.n_edges == 96
    assert inputs.position_areas is not None
    assert inputs.position_areas.shape == (57,)
    assert np.all(inputs.position_areas > 0.0)
    assert len(inputs.bulk_masks) == 2
    assert all(np.any(mask) for mask in inputs.bulk_masks)
    assert inputs.localizer_probe == (0.0, 0.0)


def test_all_six_disorder_channels_execute_with_one_reserved_dry_seed(
    tmp_path: Path,
) -> None:
    target = experiment_module._DisorderTarget(
        role_key="candidate_dry_run",
        geometry=_square_reference(),
        is_reference=False,
    )

    summaries = tuple(
        experiment_module._execute_disorder_channel(
            tmp_path,
            target=target,
            channel=channel,
            seed_role="dry_run",
            seeds=(PHASE_9_8_DRY_RUN_SEEDS[0],),
            code_commit="phase-9.8-disorder-dry-run",
            minimum_successes=1,
            minimum_wilson_lower=0.0,
        )
        for channel in Phase98DisorderChannel
    )

    assert tuple(item.channel for item in summaries) == tuple(Phase98DisorderChannel)
    assert all(item.total_count == 1 for item in summaries)
    assert all(item.manifest.is_file() for item in summaries)
    for summary in summaries:
        payload = json.loads(summary.manifest.read_text(encoding="utf-8"))
        assert payload["seed_role"] == "dry_run"
        assert payload["requested_seeds"] == [PHASE_9_8_DRY_RUN_SEEDS[0]]
        assert len(payload["members"]) == 1
        assert payload["members"][0]["seed"] == PHASE_9_8_DRY_RUN_SEEDS[0]


def test_two_candidate_trial_pipeline_uses_only_a_reserved_dry_master_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "candidate_archives").mkdir()
    (tmp_path / "trial_manifests").mkdir()
    monkeypatch.setattr(experiment_module, "PHASE_9_8_CANDIDATES_PER_TRIAL", 2)

    trial = experiment_module._execute_search_trial(
        tmp_path,
        trial_index=0,
        master_seed=PHASE_9_8_DRY_RUN_SEEDS[2],
        reference_threshold=0.231474243629927,
        code_commit="phase-9.8-trial-dry-run",
    )

    assert trial.master_seed == PHASE_9_8_DRY_RUN_SEEDS[2]
    assert len(trial.sampling.samples) == 2
    assert len(trial.ranking.entries) == 2
    assert trial.candidate_archive.is_file()
    assert trial.trial_manifest.is_file()
    payload = json.loads(trial.trial_manifest.read_text(encoding="utf-8"))
    assert payload["master_seed"] == PHASE_9_8_DRY_RUN_SEEDS[2]
    assert len(payload["sampling"]) == 2
    assert len(payload["candidates"]) == 2
    assert sorted(payload["rank_order"]) == [0, 1]


def test_full_run_rejects_protocol_only_commit_without_creating_output(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "forbidden"

    with pytest.raises(ValueError, match="include the later Phase-9.8 implementation"):
        run_phase_9_8_random_search(
            destination,
            code_commit=PHASE_9_8_PROTOCOL_COMMIT,
        )

    assert not destination.exists()


def test_full_run_rejects_a_code_commit_that_is_not_head(tmp_path: Path) -> None:
    destination = tmp_path / "mislabeled"

    with pytest.raises(RuntimeError, match="does not match the exact Git HEAD"):
        run_phase_9_8_random_search(destination, code_commit="a" * 40)

    assert not destination.exists()


def test_git_preflight_preserves_porcelain_leading_status_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code_commit = "b" * 40
    outputs = iter(
        (
            f"{code_commit}\n",
            (
                " M src/toposc_lab/observables/__pycache__/"
                "__init__.cpython-314.pyc\n"
                " M src/toposc_lab/observables/__pycache__/"
                "spectrum.cpython-314.pyc\n"
                "?? geometry_demo.npz\n"
            ),
        )
    )

    def fake_run(
        *_args: object,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=(),
            returncode=0,
            stdout=next(outputs),
            stderr="",
        )

    monkeypatch.setattr(experiment_module.subprocess, "run", fake_run)

    experiment_module._verify_committed_worktree(code_commit)


def test_full_run_never_overwrites_an_existing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    monkeypatch.setattr(
        experiment_module,
        "_validate_full_run_environment",
        lambda code_commit: code_commit,
    )

    with pytest.raises(FileExistsError, match="never overwrite"):
        run_phase_9_8_random_search(destination, code_commit="a" * 40)


def test_phase_9_8_json_artifacts_are_exclusive_and_canonical(tmp_path: Path) -> None:
    destination = tmp_path / "record.json"
    payload = {"z": 2, "a": [True, 1.5]}

    experiment_module._write_json_exclusive(destination, payload)

    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert destination.read_bytes() == b'{"a":[true,1.5],"z":2}'
    with pytest.raises(FileExistsError, match="already exists"):
        experiment_module._write_json_exclusive(destination, payload)
