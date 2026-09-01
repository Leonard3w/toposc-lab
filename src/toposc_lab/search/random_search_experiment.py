"""Reproducible Phase-9.8 random-search benchmark ``TOPOSC-P9.8-RS-001``.

The public full-run function has no configurable scientific values.  Its only
inputs are a fresh output directory and the exact committed code revision.
Sampling, numerical evaluation, ranking, disorder, and persistence continue to
use their existing layers and meet only in this experiment-specific runner.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any, TypeAlias

import numpy as np

from toposc_lab.evaluation import ObjectiveDirection
from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    Geometry,
    GeometryBoundaryComponent,
    canonical_graph_hash,
    geometry_to_bytes,
)
from toposc_lab.models.chiral_p_wave import ChiralPWaveModel
from toposc_lab.robustness import (
    DisorderEnsembleFailureStage,
    DisorderEnsembleRequest,
    DisorderOutcome,
    PairingDisorderChannel,
    RobustnessFractionMetric,
    apply_random_edge_removal,
    apply_random_node_removal,
    apply_uniform_coordinate_perturbation,
    apply_uniform_hopping_disorder,
    apply_uniform_onsite_disorder,
    apply_uniform_pairing_disorder,
    estimate_robustness_uncertainty,
    exact_model_parameter_set_id,
    execute_disorder_ensemble,
)
from toposc_lab.search.baseline_statistics import (
    BaselineSuccessCriterion,
    SearchBaselineStatistics,
    SearchBaselineTrial,
    compute_search_baseline_statistics,
)
from toposc_lab.search.batch_evaluation import (
    BatchEvaluationCandidate,
    BatchEvaluationMember,
    BatchEvaluationRequest,
    execute_evaluation_batch,
)
from toposc_lab.search.candidate_ranking import (
    CandidateRankingConfig,
    CandidateRankingCriterion,
    CandidateRankingEntry,
    CandidateRankingResult,
    CandidateRankingValueKind,
    rank_evaluated_candidates,
)
from toposc_lab.search.candidate_storage import save_evaluated_candidate_batch
from toposc_lab.search.phase_9_8_evaluation import (
    PHASE_9_8_MODEL_PARAMETER_SET,
    PHASE_9_8_MODEL_PARAMETERS,
    PHASE_9_8_PROTOCOL_COMMIT,
    PHASE_9_8_PROTOCOL_IDENTIFIER,
    Phase98GeometryApplicability,
    Phase98ScientificEvaluation,
    build_phase_9_8_ammann_beenker_topology_inputs,
    build_phase_9_8_sierpinski_topology_inputs,
    evaluate_phase_9_8_descriptive_geometry,
    evaluate_phase_9_8_primary_geometry,
    validate_phase_9_8_geometry,
)
from toposc_lab.search.random_geometry import (
    GeometrySamplingRecipe,
    RandomGeometrySamplingConfig,
    RandomGeometrySamplingResult,
    sample_random_geometries,
)

PHASE_9_8_EXPERIMENT_VERSION = 1
PHASE_9_8_SUMMARY_SCHEMA_VERSION = 1
PHASE_9_8_DRY_RUN_SEEDS = tuple(range(9_799_900, 9_799_910))
PHASE_9_8_SEARCH_TRIAL_SEEDS = tuple(range(9_800_000, 9_800_032))
PHASE_9_8_AMORPHOUS_REFERENCE_SEEDS = tuple(range(9_801_000, 9_801_032))
PHASE_9_8_VALIDATION_SEEDS = tuple(range(9_810_000, 9_810_064))
PHASE_9_8_CONFIRMATION_SEEDS = tuple(range(9_820_000, 9_820_128))
PHASE_9_8_CANDIDATES_PER_TRIAL = 32
PHASE_9_8_MAXIMUM_SELECTION_COUNT = 8
PHASE_9_8_SCREENING_REFERENCE_FACTOR = 1.10
PHASE_9_8_VALIDATION_MINIMUM_SUCCESSES = 52
PHASE_9_8_VALIDATION_MINIMUM_WILSON_LOWER = 0.70
PHASE_9_8_CONFIRMATION_MINIMUM_SUCCESSES = 104
PHASE_9_8_CONFIRMATION_MINIMUM_WILSON_LOWER = 0.73

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_WORKTREE_ARTIFACTS = frozenset(
    {
        "geometry_demo.npz",
        "src/toposc_lab/observables/__pycache__/__init__.cpython-314.pyc",
        "src/toposc_lab/observables/__pycache__/spectrum.cpython-314.pyc",
    }
)
_JSONScalar: TypeAlias = None | bool | int | float | str
_JSONValue: TypeAlias = _JSONScalar | list["_JSONValue"] | dict[str, "_JSONValue"]


class Phase98DisorderChannel(str, Enum):
    """The six non-composed validation channels frozen by the protocol."""

    ONSITE = "onsite"
    HOPPING = "hopping"
    PAIRING = "pairing"
    COORDINATE = "coordinate"
    EDGE_REMOVAL = "edge_removal"
    NODE_REMOVAL = "node_removal"


@dataclass(frozen=True, slots=True)
class Phase98DryRunRecord:
    seed: int
    exact_geometry_id: str
    exact_serialization_sha256: str
    complete_attempt_count: int
    proposal_count: int


@dataclass(frozen=True, slots=True)
class Phase98ReferenceRecord:
    key: str
    geometry: Geometry
    scientific: Phase98ScientificEvaluation
    generation_seed: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.isidentifier():
            raise ValueError("reference key must be a Python-style identifier")
        if not isinstance(self.geometry, Geometry):
            raise TypeError("reference geometry must be Geometry")
        if not isinstance(self.scientific, Phase98ScientificEvaluation):
            raise TypeError("reference scientific result has the wrong type")


@dataclass(frozen=True, slots=True)
class Phase98TrialRecord:
    trial_index: int
    master_seed: int
    sampling: RandomGeometrySamplingResult
    scientific: tuple[Phase98ScientificEvaluation | None, ...]
    ranking: CandidateRankingResult
    candidate_archive: Path
    trial_manifest: Path


@dataclass(frozen=True, slots=True)
class Phase98SelectedCandidate:
    trial_index: int
    candidate_index: int
    geometry: Geometry
    scientific: Phase98ScientificEvaluation
    ranking_entry: CandidateRankingEntry
    geometry_id: str


@dataclass(frozen=True, slots=True)
class Phase98DisorderChannelSummary:
    role_key: str
    channel: Phase98DisorderChannel
    seed_role: str
    successful_count: int
    total_count: int
    observed_fraction: float
    wilson_interval: tuple[float, float]
    passes: bool
    operational_failure_count: int
    invalid_geometry_count: int
    invalid_evaluation_count: int
    manifest: Path


@dataclass(frozen=True, slots=True)
class Phase98ExperimentResult:
    """Compact result surface; exhaustive records remain in sealed artifacts."""

    output_directory: Path
    protocol_commit: str
    code_commit: str
    reference_threshold: float
    dry_run: tuple[Phase98DryRunRecord, ...]
    references: tuple[Phase98ReferenceRecord, ...]
    trials: tuple[Phase98TrialRecord, ...]
    baseline_statistics: SearchBaselineStatistics
    selected_candidates: tuple[Phase98SelectedCandidate, ...]
    validation: tuple[Phase98DisorderChannelSummary, ...]
    confirmation_triggered: bool
    confirmation: tuple[Phase98DisorderChannelSummary, ...]
    summary_manifest: Path
    experiment_version: int = field(default=PHASE_9_8_EXPERIMENT_VERSION, init=False)


def run_phase_9_8_dry_run() -> tuple[Phase98DryRunRecord, ...]:
    """Exercise only the ten reserved direct generator seeds, without physics or files."""
    records: list[Phase98DryRunRecord] = []
    for seed in PHASE_9_8_DRY_RUN_SEEDS:
        first = BUILTIN_GEOMETRY_GENERATORS.generate(
            "hard_core_planar_graph",
            seed=seed,
        )
        second = BUILTIN_GEOMETRY_GENERATORS.generate(
            "hard_core_planar_graph",
            seed=seed,
        )
        first_payload = geometry_to_bytes(first)
        if first_payload != geometry_to_bytes(second):
            raise RuntimeError("a dry-run seed did not reproduce exact geometry bytes")
        report = validate_phase_9_8_geometry(
            first,
            applicability=Phase98GeometryApplicability.CLEAN_PRIMARY,
        )
        if not report.is_applicable:
            reasons = ", ".join(issue.code for issue in report.issues)
            raise RuntimeError(f"dry-run geometry failed Phase-9.8 constraints: {reasons}")
        records.append(
            Phase98DryRunRecord(
                seed=seed,
                exact_geometry_id=exact_geometry_id(first),
                exact_serialization_sha256=hashlib.sha256(first_payload).hexdigest(),
                complete_attempt_count=_metadata_integer(
                    first,
                    "complete_attempt_count",
                ),
                proposal_count=_metadata_integer(first, "proposal_count"),
            )
        )
    return tuple(records)


def run_phase_9_8_random_search(
    output_directory: str | Path,
    *,
    code_commit: str,
) -> Phase98ExperimentResult:
    """Execute the complete frozen random-search, validation, and confirmation policy."""
    prepared_code_commit = _validate_full_run_environment(code_commit)
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError(
            "Phase-9.8 output directory already exists; accepted runs never overwrite"
        )

    dry_run = run_phase_9_8_dry_run()
    destination.mkdir(parents=True)
    (destination / "candidate_archives").mkdir()
    (destination / "trial_manifests").mkdir()
    (destination / "disorder_manifests").mkdir()

    references, descriptive_payloads = _evaluate_references(
        code_commit=prepared_code_commit
    )
    eligible_references = tuple(
        record for record in references if record.scientific.clean_eligible
    )
    if not eligible_references:
        raise RuntimeError(
            "no primary reference passed the frozen clean eligibility gates"
        )
    reference_threshold = max(
        record.scientific.localizer_protection_proxy
        for record in eligible_references
    )
    reference_manifest = destination / "references_v1.json"
    _write_json_exclusive(
        reference_manifest,
        {
            "schema": "toposc_phase_9_8_references",
            "schema_version": PHASE_9_8_SUMMARY_SCHEMA_VERSION,
            "protocol_identifier": PHASE_9_8_PROTOCOL_IDENTIFIER,
            "protocol_commit": PHASE_9_8_PROTOCOL_COMMIT,
            "code_commit": prepared_code_commit,
            "reference_threshold_R": reference_threshold,
            "primary_references": [
                _encode_reference(record) for record in references
            ],
            "descriptive_references": descriptive_payloads,
        },
    )

    trials: list[Phase98TrialRecord] = []
    baseline_trials: list[SearchBaselineTrial] = []
    criterion = _screening_criterion(reference_threshold)
    for trial_index, master_seed in enumerate(PHASE_9_8_SEARCH_TRIAL_SEEDS):
        trial = _execute_search_trial(
            destination,
            trial_index=trial_index,
            master_seed=master_seed,
            reference_threshold=reference_threshold,
            code_commit=prepared_code_commit,
        )
        trials.append(trial)
        baseline_trials.append(
            SearchBaselineTrial(
                trial_key=f"search_trial_{trial_index:02d}",
                ranking=trial.ranking,
            )
        )

    baseline_statistics = compute_search_baseline_statistics(
        baseline_trials,
        criterion=criterion,
        confidence_level=0.95,
    )
    selected = _select_validation_candidates(
        tuple(trials),
        criterion=criterion,
    )
    selection_manifest = destination / "validation_selection_v1.json"
    _write_json_exclusive(
        selection_manifest,
        {
            "schema": "toposc_phase_9_8_selection",
            "schema_version": 1,
            "protocol_identifier": PHASE_9_8_PROTOCOL_IDENTIFIER,
            "protocol_commit": PHASE_9_8_PROTOCOL_COMMIT,
            "code_commit": prepared_code_commit,
            "selection_rule": (
                "rank1_strong_per_successful_trial_then_frozen_ranking_then_geometry_id"
            ),
            "selected": [_encode_selection(item) for item in selected],
        },
    )

    validation: tuple[Phase98DisorderChannelSummary, ...] = ()
    confirmation: tuple[Phase98DisorderChannelSummary, ...] = ()
    confirmation_triggered = False
    validation_comparisons: list[dict[str, _JSONValue]] = []
    confirmation_comparisons: list[dict[str, _JSONValue]] = []
    if selected:
        validation_targets = _validation_targets(
            selected,
            eligible_references=eligible_references,
        )
        validation = _execute_disorder_stage(
            destination,
            targets=validation_targets,
            seed_role="validation",
            seeds=PHASE_9_8_VALIDATION_SEEDS,
            code_commit=prepared_code_commit,
            minimum_successes=PHASE_9_8_VALIDATION_MINIMUM_SUCCESSES,
            minimum_wilson_lower=PHASE_9_8_VALIDATION_MINIMUM_WILSON_LOWER,
        )
        validation_comparisons = _reference_channel_comparisons(validation)
        passing_candidate_keys = _passing_candidate_keys(validation)
        confirmation_triggered = bool(passing_candidate_keys)
        if confirmation_triggered:
            confirmation_targets = tuple(
                target
                for target in validation_targets
                if target.is_reference or target.role_key in passing_candidate_keys
            )
            confirmation = _execute_disorder_stage(
                destination,
                targets=confirmation_targets,
                seed_role="confirmation",
                seeds=PHASE_9_8_CONFIRMATION_SEEDS,
                code_commit=prepared_code_commit,
                minimum_successes=PHASE_9_8_CONFIRMATION_MINIMUM_SUCCESSES,
                minimum_wilson_lower=PHASE_9_8_CONFIRMATION_MINIMUM_WILSON_LOWER,
            )
            confirmation_comparisons = _reference_channel_comparisons(confirmation)

    summary_manifest = destination / "phase_9_8_summary_v1.json"
    artifact_paths = (
        (reference_manifest, selection_manifest)
        + tuple(item.candidate_archive for item in trials)
        + tuple(item.trial_manifest for item in trials)
        + tuple(item.manifest for item in validation)
        + tuple(item.manifest for item in confirmation)
    )
    _write_json_exclusive(
        summary_manifest,
        {
            "schema": "toposc_phase_9_8_random_search_summary",
            "schema_version": PHASE_9_8_SUMMARY_SCHEMA_VERSION,
            "claim_boundary": "finite_size_engineering_benchmark_and_screen_only",
            "protocol_identifier": PHASE_9_8_PROTOCOL_IDENTIFIER,
            "protocol_commit": PHASE_9_8_PROTOCOL_COMMIT,
            "code_commit": prepared_code_commit,
            "environment": _environment_record(),
            "solver": "numpy.linalg.eigh_full_spectrum",
            "seed_partitions": {
                "dry_run": list(PHASE_9_8_DRY_RUN_SEEDS),
                "search_trial": list(PHASE_9_8_SEARCH_TRIAL_SEEDS),
                "amorphous_reference": list(PHASE_9_8_AMORPHOUS_REFERENCE_SEEDS),
                "validation": list(PHASE_9_8_VALIDATION_SEEDS),
                "confirmation": list(PHASE_9_8_CONFIRMATION_SEEDS),
            },
            "dry_run": [_encode_dry_run(item) for item in dry_run],
            "reference_threshold_R": reference_threshold,
            "references_manifest": _artifact_link(destination, reference_manifest),
            "trials": [
                {
                    "trial_index": item.trial_index,
                    "master_seed": item.master_seed,
                    "candidate_archive": _artifact_link(destination, item.candidate_archive),
                    "trial_manifest": _artifact_link(destination, item.trial_manifest),
                }
                for item in trials
            ],
            "baseline_statistics": _encode_baseline_statistics(baseline_statistics),
            "selection_manifest": _artifact_link(destination, selection_manifest),
            "validation": [_encode_channel_summary(item, destination) for item in validation],
            "validation_reference_comparisons": validation_comparisons,
            "confirmation_triggered": confirmation_triggered,
            "confirmation": [_encode_channel_summary(item, destination) for item in confirmation],
            "confirmation_reference_comparisons": confirmation_comparisons,
            "artifact_inventory_before_summary": [
                _artifact_link(destination, path) for path in artifact_paths
            ],
            "warnings": [
                "This finite-64-site engineering screen is not a thermodynamic, causal, novelty, priority, or material claim.",
                "The Phase-7 gap remains the full finite-spectrum separation and is not relabeled as a bulk, mobility, or topological gap.",
                "A high rank or a passed screen is not a scientific discovery.",
            ],
        },
    )
    return Phase98ExperimentResult(
        output_directory=destination,
        protocol_commit=PHASE_9_8_PROTOCOL_COMMIT,
        code_commit=prepared_code_commit,
        reference_threshold=reference_threshold,
        dry_run=dry_run,
        references=references,
        trials=tuple(trials),
        baseline_statistics=baseline_statistics,
        selected_candidates=selected,
        validation=validation,
        confirmation_triggered=confirmation_triggered,
        confirmation=confirmation,
        summary_manifest=summary_manifest,
    )


@dataclass(frozen=True, slots=True)
class _DisorderTarget:
    role_key: str
    geometry: Geometry
    is_reference: bool


def _evaluate_references(
    *,
    code_commit: str,
) -> tuple[tuple[Phase98ReferenceRecord, ...], list[dict[str, _JSONValue]]]:
    square_geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "square",
        parameters={
            "n_x": 8,
            "n_y": 8,
            "spacing": 1.0,
            "boundary_x": "open",
            "boundary_y": "open",
        },
    )
    square_metadata = dict(square_geometry.metadata)
    square_metadata["phase_9_8_boundary_component_annotation"] = (
        "existing_boundary_sites_as_outer_component_zero"
    )
    square_geometry = replace(
        square_geometry,
        boundary_components=(
            GeometryBoundaryComponent("outer", 0, square_geometry.boundary_sites),
        ),
        metadata=square_metadata,
    )
    references: list[Phase98ReferenceRecord] = [
        Phase98ReferenceRecord(
            key="square_8x8",
            geometry=square_geometry,
            scientific=evaluate_phase_9_8_primary_geometry(
                square_geometry,
                code_version=code_commit,
            ),
            generation_seed=None,
        )
    ]
    for reference_index, seed in enumerate(PHASE_9_8_AMORPHOUS_REFERENCE_SEEDS):
        geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
            "hard_core_planar_reference",
            seed=seed,
        )
        references.append(
            Phase98ReferenceRecord(
                key=f"amorphous_reference_{reference_index:02d}",
                geometry=geometry,
                scientific=evaluate_phase_9_8_primary_geometry(
                    geometry,
                    code_version=code_commit,
                ),
                generation_seed=seed,
            )
        )

    descriptive: list[dict[str, _JSONValue]] = []
    ammann = BUILTIN_GEOMETRY_GENERATORS.generate(
        "ammann_beenker_patch",
        parameters={"radius": 4.0, "spacing": 1.0},
    )
    ammann_run, ammann_grid = evaluate_phase_9_8_descriptive_geometry(
        ammann,
        inputs=build_phase_9_8_ammann_beenker_topology_inputs(ammann),
        code_version=code_commit,
    )
    descriptive.append(
        _encode_descriptive_reference(
            "ammann_beenker_radius_4",
            ammann,
            ammann_run,
            ammann_grid,
        )
    )
    sierpinski = BUILTIN_GEOMETRY_GENERATORS.generate(
        "sierpinski_carpet",
        parameters={"order": 2, "spacing": 1.0},
    )
    sierpinski_run, sierpinski_grid = evaluate_phase_9_8_descriptive_geometry(
        sierpinski,
        inputs=build_phase_9_8_sierpinski_topology_inputs(sierpinski),
        code_version=code_commit,
    )
    descriptive.append(
        _encode_descriptive_reference(
            "sierpinski_carpet_order_2",
            sierpinski,
            sierpinski_run,
            sierpinski_grid,
        )
    )
    return tuple(references), descriptive


def _execute_search_trial(
    destination: Path,
    *,
    trial_index: int,
    master_seed: int,
    reference_threshold: float,
    code_commit: str,
) -> Phase98TrialRecord:
    sampling = sample_random_geometries(
        RandomGeometrySamplingConfig(
            recipes=(GeometrySamplingRecipe("hard_core_planar_graph"),),
            sample_count=PHASE_9_8_CANDIDATES_PER_TRIAL,
        ),
        seed=master_seed,
    )
    candidates = tuple(
        BatchEvaluationCandidate(
            geometry=sample.geometry,
            model_parameters=PHASE_9_8_MODEL_PARAMETER_SET,
            evaluation_seed=None,
        )
        for sample in sampling.samples
    )
    scientific: list[Phase98ScientificEvaluation | None] = [None] * len(candidates)
    callback_index = 0

    def evaluator(candidate: BatchEvaluationCandidate):  # type: ignore[no-untyped-def]
        nonlocal callback_index
        candidate_index = callback_index
        callback_index += 1
        result = evaluate_phase_9_8_primary_geometry(
            candidate.geometry,
            code_version=code_commit,
            evaluation_seed=None,
        )
        scientific[candidate_index] = result
        return result.run

    batch = execute_evaluation_batch(
        BatchEvaluationRequest(candidates),
        evaluator=evaluator,
    )
    ranking = rank_evaluated_candidates(
        batch,
        config=_ranking_config(),
        value_factory=lambda member: _ranking_values(member, scientific),
    )
    archive = save_evaluated_candidate_batch(
        destination / "candidate_archives" / f"trial_{trial_index:02d}.zip",
        batch,
    )
    manifest = destination / "trial_manifests" / f"trial_{trial_index:02d}_v1.json"
    _write_json_exclusive(
        manifest,
        {
            "schema": "toposc_phase_9_8_search_trial",
            "schema_version": 1,
            "protocol_identifier": PHASE_9_8_PROTOCOL_IDENTIFIER,
            "protocol_commit": PHASE_9_8_PROTOCOL_COMMIT,
            "code_commit": code_commit,
            "trial_index": trial_index,
            "master_seed": master_seed,
            "reference_threshold_R": reference_threshold,
            "candidate_archive": _artifact_link(destination, archive),
            "sampling": [
                {
                    "candidate_index": sample.sample_index,
                    "recipe_index": sample.recipe_index,
                    "generator_key": sample.geometry.metadata["generation"]["generator_key"],
                    "derived_generator_seed": sample.generation_request.seed,
                    "generator_provenance": sample.geometry.metadata["generation"],
                    "generator_audit": sample.geometry.metadata,
                    "geometry_id": exact_geometry_id(sample.geometry),
                    "graph_fingerprint": canonical_graph_hash(sample.geometry),
                    "model_parameter_set_id": exact_model_parameter_set_id(
                        PHASE_9_8_MODEL_PARAMETER_SET
                    ),
                }
                for sample in sampling.samples
            ],
            "candidates": [
                _encode_trial_candidate(index, item, ranking.entries[index])
                for index, item in enumerate(scientific)
            ],
            "rank_order": [entry.candidate_index for entry in ranking.ranked_entries],
        },
    )
    return Phase98TrialRecord(
        trial_index=trial_index,
        master_seed=master_seed,
        sampling=sampling,
        scientific=tuple(scientific),
        ranking=ranking,
        candidate_archive=archive,
        trial_manifest=manifest,
    )


def _ranking_config() -> CandidateRankingConfig:
    return CandidateRankingConfig(
        (
            CandidateRankingCriterion(
                "clean_eligible",
                ObjectiveDirection.MAXIMIZE,
                CandidateRankingValueKind.BOOLEAN,
            ),
            CandidateRankingCriterion(
                "localizer_protection_proxy",
                ObjectiveDirection.MAXIMIZE,
                CandidateRankingValueKind.REAL,
            ),
            CandidateRankingCriterion(
                "minimum_boundary_weight_first_four",
                ObjectiveDirection.MAXIMIZE,
                CandidateRankingValueKind.REAL,
            ),
        )
    )


def _ranking_values(
    member: BatchEvaluationMember,
    scientific: list[Phase98ScientificEvaluation | None],
) -> Mapping[str, object]:
    result = scientific[member.candidate_index]
    if result is None:
        raise ValueError("a valid batch member has no Phase-9.8 scientific record")
    return {
        "clean_eligible": result.clean_eligible,
        "localizer_protection_proxy": result.localizer_protection_proxy,
        "minimum_boundary_weight_first_four": (
            result.minimum_boundary_weight_first_four
        ),
    }


def _screening_criterion(reference_threshold: float) -> BaselineSuccessCriterion:
    def predicate(entry: CandidateRankingEntry) -> bool:
        return bool(
            entry.values["clean_eligible"] is True
            and float(entry.values["localizer_protection_proxy"])
            >= PHASE_9_8_SCREENING_REFERENCE_FACTOR * reference_threshold
        )

    return BaselineSuccessCriterion(
        key="screening_strong_candidate_v1",
        description=(
            "clean eligible and localizer protection proxy at least 1.10 times "
            "the strongest eligible primary reference"
        ),
        predicate=predicate,
    )


def _select_validation_candidates(
    trials: tuple[Phase98TrialRecord, ...],
    *,
    criterion: BaselineSuccessCriterion,
) -> tuple[Phase98SelectedCandidate, ...]:
    per_trial: list[Phase98SelectedCandidate] = []
    for trial in trials:
        strong = tuple(
            entry
            for entry in trial.ranking.ranked_entries
            if criterion.evaluate(entry)
        )
        if not strong:
            continue
        rank_one = next(entry for entry in strong if entry.rank == 1)
        scientific = trial.scientific[rank_one.candidate_index]
        if scientific is None:
            raise RuntimeError("selected candidate has no scientific record")
        geometry = rank_one.member.candidate.geometry
        per_trial.append(
            Phase98SelectedCandidate(
                trial_index=trial.trial_index,
                candidate_index=rank_one.candidate_index,
                geometry=geometry,
                scientific=scientific,
                ranking_entry=rank_one,
                geometry_id=exact_geometry_id(geometry),
            )
        )
    per_trial.sort(
        key=lambda item: (
            -int(bool(item.ranking_entry.values["clean_eligible"])),
            -float(item.ranking_entry.values["localizer_protection_proxy"]),
            -float(item.ranking_entry.values["minimum_boundary_weight_first_four"]),
            item.geometry_id.encode("ascii"),
        )
    )
    selected: list[Phase98SelectedCandidate] = []
    seen_ids: set[str] = set()
    for item in per_trial:
        if item.geometry_id in seen_ids:
            continue
        seen_ids.add(item.geometry_id)
        selected.append(item)
        if len(selected) == PHASE_9_8_MAXIMUM_SELECTION_COUNT:
            break
    return tuple(selected)


def _validation_targets(
    selected: tuple[Phase98SelectedCandidate, ...],
    *,
    eligible_references: tuple[Phase98ReferenceRecord, ...],
) -> tuple[_DisorderTarget, ...]:
    candidates = tuple(
        _DisorderTarget(
            role_key=f"candidate_trial_{item.trial_index:02d}_index_{item.candidate_index:02d}",
            geometry=item.geometry,
            is_reference=False,
        )
        for item in selected
    )
    references = tuple(
        _DisorderTarget(
            role_key=f"reference_{item.key}",
            geometry=item.geometry,
            is_reference=True,
        )
        for item in eligible_references
    )
    return candidates + references


def _execute_disorder_stage(
    destination: Path,
    *,
    targets: tuple[_DisorderTarget, ...],
    seed_role: str,
    seeds: tuple[int, ...],
    code_commit: str,
    minimum_successes: int,
    minimum_wilson_lower: float,
) -> tuple[Phase98DisorderChannelSummary, ...]:
    summaries: list[Phase98DisorderChannelSummary] = []
    for target in targets:
        for channel in Phase98DisorderChannel:
            summaries.append(
                _execute_disorder_channel(
                    destination,
                    target=target,
                    channel=channel,
                    seed_role=seed_role,
                    seeds=seeds,
                    code_commit=code_commit,
                    minimum_successes=minimum_successes,
                    minimum_wilson_lower=minimum_wilson_lower,
                )
            )
    return tuple(summaries)


def _execute_disorder_channel(
    destination: Path,
    *,
    target: _DisorderTarget,
    channel: Phase98DisorderChannel,
    seed_role: str,
    seeds: tuple[int, ...],
    code_commit: str,
    minimum_successes: int,
    minimum_wilson_lower: float,
) -> Phase98DisorderChannelSummary:
    clean_model = ChiralPWaveModel(target.geometry, PHASE_9_8_MODEL_PARAMETERS)
    clean_hamiltonian = clean_model.hamiltonian()
    ensemble = execute_disorder_ensemble(
        DisorderEnsembleRequest(seeds),
        realization_factory=_disorder_factory(
            channel,
            geometry=target.geometry,
            clean_hamiltonian=clean_hamiltonian,
        ),
    )
    successes: list[bool] = []
    operational_failures: list[int] = []
    invalid_geometry: list[int] = []
    invalid_evaluation: list[int] = []
    member_payloads: list[dict[str, _JSONValue]] = []
    for member in ensemble.members:
        if member.failure is not None:
            successes.append(False)
            operational_failures.append(member.sample_index)
            member_payloads.append(
                {
                    "sample_index": member.sample_index,
                    "seed": member.seed,
                    "success": False,
                    "classification": "operational_disorder_failure",
                    "failure": {
                        "stage": member.failure.stage.value,
                        "error_type": member.failure.error_type,
                        "message": member.failure.message,
                    },
                }
            )
            continue
        disorder = member.disorder
        assert disorder is not None
        geometry, hamiltonian, applicability = _disordered_evaluation_inputs(
            channel,
            clean_geometry=target.geometry,
            disorder=disorder,
        )
        report = validate_phase_9_8_geometry(
            geometry,
            applicability=applicability,
        )
        provenance = _encode_disorder_provenance(disorder)
        if not report.is_applicable:
            successes.append(False)
            invalid_geometry.append(member.sample_index)
            member_payloads.append(
                {
                    "sample_index": member.sample_index,
                    "seed": member.seed,
                    "success": False,
                    "classification": "invalid_transformed_geometry",
                    "provenance": provenance,
                    "geometry_constraints": _encode_constraint_report(report),
                }
            )
            continue
        try:
            scientific = evaluate_phase_9_8_primary_geometry(
                geometry,
                code_version=code_commit,
                applicability=applicability,
                hamiltonian=hamiltonian,
                evaluation_seed=member.seed,
            )
        except Exception as error:
            successes.append(False)
            operational_failures.append(member.sample_index)
            member_payloads.append(
                {
                    "sample_index": member.sample_index,
                    "seed": member.seed,
                    "success": False,
                    "classification": "operational_evaluation_failure",
                    "provenance": provenance,
                    "failure": {
                        "stage": DisorderEnsembleFailureStage.EVALUATION_CALLBACK.value,
                        "error_type": type(error).__name__,
                        "message": str(error).strip() or "evaluation raised without a message",
                    },
                }
            )
            continue
        success = scientific.clean_eligible
        successes.append(success)
        if not scientific.run.is_valid:
            if scientific.run.failure is not None:
                operational_failures.append(member.sample_index)
                classification = "operational_pipeline_failure"
            else:
                invalid_evaluation.append(member.sample_index)
                classification = "invalid_evaluation"
        else:
            classification = "passed_all_gates" if success else "scientific_gate_failure"
        member_payloads.append(
            {
                "sample_index": member.sample_index,
                "seed": member.seed,
                "success": success,
                "classification": classification,
                "provenance": provenance,
                "scientific": _encode_scientific(scientific),
            }
        )

    metric = RobustnessFractionMetric(
        criterion_key="phase_9_8_all_channel_appropriate_gates_v1",
        criterion_description=(
            "channel-appropriate geometry applicability and frozen topology, "
            "localizer-protection, and boundary gates"
        ),
        request=ensemble.request,
        successes=tuple(successes),
        execution_failure_indices=tuple(sorted(set(operational_failures))),
    )
    uncertainty = estimate_robustness_uncertainty(metric, confidence_level=0.95)
    passes = bool(
        metric.successful_count >= minimum_successes
        and uncertainty.lower_bound >= minimum_wilson_lower
    )
    manifest = (
        destination
        / "disorder_manifests"
        / f"{seed_role}_{target.role_key}_{channel.value}_v1.json"
    )
    _write_json_exclusive(
        manifest,
        {
            "schema": "toposc_phase_9_8_disorder_channel",
            "schema_version": 1,
            "protocol_identifier": PHASE_9_8_PROTOCOL_IDENTIFIER,
            "protocol_commit": PHASE_9_8_PROTOCOL_COMMIT,
            "code_commit": code_commit,
            "seed_role": seed_role,
            "role_key": target.role_key,
            "is_reference": target.is_reference,
            "source_geometry_id": exact_geometry_id(target.geometry),
            "channel": channel.value,
            "stress_definition": _channel_stress_definition(channel),
            "requested_seeds": list(seeds),
            "successes": list(metric.successes),
            "successful_count": metric.successful_count,
            "total_count": metric.total_count,
            "observed_fraction": metric.value,
            "wilson_interval_95": list(uncertainty.confidence_interval),
            "minimum_successes": minimum_successes,
            "minimum_wilson_lower": minimum_wilson_lower,
            "passes": passes,
            "operational_failure_indices": sorted(set(operational_failures)),
            "invalid_geometry_indices": invalid_geometry,
            "invalid_evaluation_indices": invalid_evaluation,
            "members": member_payloads,
        },
    )
    return Phase98DisorderChannelSummary(
        role_key=target.role_key,
        channel=channel,
        seed_role=seed_role,
        successful_count=metric.successful_count,
        total_count=metric.total_count,
        observed_fraction=metric.value,
        wilson_interval=uncertainty.confidence_interval,
        passes=passes,
        operational_failure_count=len(set(operational_failures)),
        invalid_geometry_count=len(invalid_geometry),
        invalid_evaluation_count=len(invalid_evaluation),
        manifest=manifest,
    )


def _disorder_factory(
    channel: Phase98DisorderChannel,
    *,
    geometry: Geometry,
    clean_hamiltonian: np.ndarray,
) -> Callable[[int], DisorderOutcome]:
    nambu_basis = ChiralPWaveModel(geometry, PHASE_9_8_MODEL_PARAMETERS).nambu_basis
    if channel is Phase98DisorderChannel.ONSITE:
        return lambda seed: apply_uniform_onsite_disorder(
            geometry,
            clean_hamiltonian,
            width=1.0,
            seed=seed,
            nambu_basis=nambu_basis,
        )
    if channel is Phase98DisorderChannel.HOPPING:
        return lambda seed: apply_uniform_hopping_disorder(
            geometry,
            clean_hamiltonian,
            width=0.5,
            seed=seed,
            nambu_basis=nambu_basis,
        )
    if channel is Phase98DisorderChannel.PAIRING:
        return lambda seed: apply_uniform_pairing_disorder(
            geometry,
            clean_hamiltonian,
            width=0.5,
            seed=seed,
            nambu_basis=nambu_basis,
            channel=PairingDisorderChannel.CHIRAL_P_WAVE,
            chirality=1,
            plane_axes=(0, 1),
        )
    if channel is Phase98DisorderChannel.COORDINATE:
        return lambda seed: apply_uniform_coordinate_perturbation(
            geometry,
            width=0.20,
            seed=seed,
        )
    if channel is Phase98DisorderChannel.EDGE_REMOVAL:
        return lambda seed: apply_random_edge_removal(
            geometry,
            removal_probability=0.05,
            seed=seed,
        )
    return lambda seed: apply_random_node_removal(
        geometry,
        removal_probability=0.02,
        seed=seed,
    )


def _disordered_evaluation_inputs(
    channel: Phase98DisorderChannel,
    *,
    clean_geometry: Geometry,
    disorder: DisorderOutcome,
) -> tuple[Geometry, np.ndarray | None, Phase98GeometryApplicability]:
    if channel in {
        Phase98DisorderChannel.ONSITE,
        Phase98DisorderChannel.HOPPING,
        Phase98DisorderChannel.PAIRING,
    }:
        if not isinstance(disorder.state, np.ndarray):
            raise TypeError("matrix disorder channel returned a non-matrix state")
        return (
            clean_geometry,
            disorder.state,
            Phase98GeometryApplicability.CLEAN_PRIMARY,
        )
    if not isinstance(disorder.state, Geometry):
        raise TypeError("geometry disorder channel returned a non-geometry state")
    applicability = (
        Phase98GeometryApplicability.COORDINATE_DISORDER
        if channel is Phase98DisorderChannel.COORDINATE
        else Phase98GeometryApplicability.REMOVAL_DISORDER
    )
    return disorder.state, None, applicability


def _passing_candidate_keys(
    summaries: tuple[Phase98DisorderChannelSummary, ...],
) -> set[str]:
    by_role: dict[str, dict[Phase98DisorderChannel, bool]] = {}
    for summary in summaries:
        if summary.role_key.startswith("reference_"):
            continue
        by_role.setdefault(summary.role_key, {})[summary.channel] = summary.passes
    return {
        role
        for role, channels in by_role.items()
        if set(channels) == set(Phase98DisorderChannel) and all(channels.values())
    }


def _reference_channel_comparisons(
    summaries: tuple[Phase98DisorderChannelSummary, ...],
) -> list[dict[str, _JSONValue]]:
    strongest_reference = {
        channel: max(
            (
                item.observed_fraction
                for item in summaries
                if item.channel is channel and item.role_key.startswith("reference_")
            ),
            default=float("nan"),
        )
        for channel in Phase98DisorderChannel
    }
    return [
        {
            "role_key": item.role_key,
            "channel": item.channel.value,
            "candidate_fraction": item.observed_fraction,
            "strongest_eligible_primary_reference_fraction": strongest_reference[item.channel],
            "interpretation": "descriptive_finite_size_difference_only",
        }
        for item in summaries
        if not item.role_key.startswith("reference_")
    ]


def _encode_reference(record: Phase98ReferenceRecord) -> dict[str, _JSONValue]:
    return {
        "key": record.key,
        "generation_seed": record.generation_seed,
        "geometry_id": exact_geometry_id(record.geometry),
        "graph_fingerprint": canonical_graph_hash(record.geometry),
        "generator_provenance": _jsonify(record.geometry.metadata.get("generation")),
        "scientific": _encode_scientific(record.scientific),
    }


def _encode_descriptive_reference(
    key: str,
    geometry: Geometry,
    run: Any,
    grid: Any,
) -> dict[str, _JSONValue]:
    return {
        "key": key,
        "matched_primary_stratum": False,
        "excluded_from_R_ranking_selection_and_disorder": True,
        "geometry_id": exact_geometry_id(geometry),
        "graph_fingerprint": canonical_graph_hash(geometry),
        "site_count": geometry.n_sites,
        "edge_count": geometry.n_edges,
        "boundary_site_count": len(geometry.boundary_sites),
        "run_valid": bool(run.is_valid),
        "pipeline": _encode_pipeline_run(run),
        "topology_grid": None if grid is None else _encode_topology_grid(grid),
    }


def _encode_trial_candidate(
    candidate_index: int,
    scientific: Phase98ScientificEvaluation | None,
    ranking: CandidateRankingEntry,
) -> dict[str, _JSONValue]:
    return {
        "candidate_index": candidate_index,
        "scientific": None if scientific is None else _encode_scientific(scientific),
        "ranking": {
            "rank": ranking.rank,
            "unranked_reason": (
                None if ranking.unranked_reason is None else ranking.unranked_reason.value
            ),
            "values": _jsonify(ranking.values),
        },
    }


def _encode_scientific(result: Phase98ScientificEvaluation) -> dict[str, _JSONValue]:
    return {
        "clean_eligible": result.clean_eligible,
        "gate_reasons": list(result.gate_reasons),
        "geometry_constraints": _encode_constraint_report(result.geometry_constraints),
        "pipeline": _encode_pipeline_run(result.run),
        "localizer_protection_proxy": (
            None if result.topology_grid is None else result.localizer_protection_proxy
        ),
        "minimum_boundary_weight_first_four": (
            None
            if result.boundary_signature is None
            else result.minimum_boundary_weight_first_four
        ),
        "topology_grid": (
            None if result.topology_grid is None else _encode_topology_grid(result.topology_grid)
        ),
        "boundary_signature": (
            None
            if result.boundary_signature is None
            else {
                "particle_hole_pairs_by_state_index": [
                    list(pair) for pair in result.boundary_signature.particle_hole_pairs
                ],
                "pairing_cost": result.boundary_signature.pairing_cost,
                "maximum_pair_residual": result.boundary_signature.maximum_pair_residual,
                "boundary_localized_count": result.boundary_signature.boundary_localized_count,
                "minimum_boundary_weight_first_four": (
                    result.boundary_signature.minimum_boundary_weight_first_four
                ),
                "reasons": list(result.boundary_signature.reasons),
                "states": [
                    {
                        "state_index": state.state_index,
                        "energy": state.energy,
                        "ipr": state.ipr,
                        "boundary_weight": state.boundary_weight,
                        "site_probability": state.localization.probability.tolist(),
                        "component_probabilities": (
                            state.localization.component_probabilities.tolist()
                        ),
                        "majorana": {
                            "site_probability": state.majorana.site_probability.tolist(),
                            "particle_probability": state.majorana.particle_probability.tolist(),
                            "hole_probability": state.majorana.hole_probability.tolist(),
                            "polarization": _complex_array(state.majorana.polarization),
                            "polarization_magnitude": (
                                state.majorana.polarization_magnitude.tolist()
                            ),
                            "total_polarization": {
                                "real": state.majorana.total_polarization.real,
                                "imag": state.majorana.total_polarization.imag,
                            },
                            "self_conjugacy": state.majorana.self_conjugacy,
                            "polarization_norm": state.majorana.polarization_norm,
                            "particle_weight": state.majorana.particle_weight,
                            "hole_weight": state.majorana.hole_weight,
                        },
                    }
                    for state in result.boundary_signature.states
                ],
            }
        ),
    }


def _encode_constraint_report(report: Any) -> dict[str, _JSONValue]:
    return {
        "applicability": report.applicability.value,
        "is_applicable": report.is_applicable,
        "base_validation_valid": report.base_validation.is_valid,
        "base_validation_issues": [
            {"code": item.code, "message": item.message, "severity": item.severity.value}
            for item in report.base_validation.issues
        ],
        "issues": [
            {"code": item.code, "message": item.message} for item in report.issues
        ],
        "measurements": _jsonify(report.measurements),
    }


def _encode_pipeline_run(run: Any) -> dict[str, _JSONValue]:
    reproducibility = run.reproducibility
    return {
        "is_valid": run.is_valid,
        "validity_issues": [
            {
                "code": issue.code,
                "message": issue.message,
                "severity": issue.severity.value,
                "category": issue.category.value,
            }
            for issue in run.validity.issues
        ],
        "stage_failure": (
            None
            if run.failure is None
            else {
                "stage": run.failure.stage.value,
                "error_type": run.failure.error_type,
                "message": run.failure.message,
            }
        ),
        "reproducibility": (
            None
            if reproducibility is None
            else {
                "seed": reproducibility.seed,
                "model_name": reproducibility.model_name,
                "model_parameters": _jsonify(reproducibility.model_parameters),
                "geometry_id": reproducibility.geometry_id,
                "geometry_id_scheme": reproducibility.geometry_id_scheme,
                "solver_name": reproducibility.solver_name,
                "solver_settings": _jsonify(reproducibility.solver_settings),
                "evaluation_settings": _jsonify(reproducibility.evaluation_settings),
                "code_version": reproducibility.code_version,
                "code_version_source": reproducibility.code_version_source,
                "warnings": list(reproducibility.warnings),
            }
        ),
        "evaluation_warnings": (
            [] if run.evaluation is None else list(run.evaluation.warnings)
        ),
        "finite_spectrum_gap": (
            None if run.evaluation is None else run.evaluation.gap
        ),
    }


def _encode_topology_grid(grid: Any) -> dict[str, _JSONValue]:
    return {
        "bott": [
            {
                "periods": item.coordinate_periods.tolist(),
                "estimate": item.bott_estimate,
                "index": item.bott_index,
                "quantization_error": item.quantization_error,
                "is_quantized": item.is_quantized,
                "minimum_fermi_distance": item.minimum_fermi_distance,
                "minimum_projected_position_singular_value": (
                    item.minimum_projected_position_singular_value
                ),
                "minimum_branch_cut_distance": item.minimum_branch_cut_distance,
                "maximum_hermiticity_residual": item.maximum_hermiticity_residual,
                "maximum_unitarity_residual": item.maximum_unitarity_residual,
                "tolerance": item.tolerance,
                "quantization_tolerance": item.quantization_tolerance,
            }
            for item in grid.bott
        ],
        "local_chern": [
            {
                "bulk_depth_slot": index,
                "bulk_site_count": int(np.count_nonzero(item.bulk_mask)),
                "bulk_chern_estimate": item.bulk_chern_estimate,
                "chern_number": item.chern_number,
                "quantization_error": item.quantization_error,
                "is_quantized": item.is_quantized,
                "minimum_fermi_distance": item.minimum_fermi_distance,
                "finite_sample_trace_residual": item.finite_sample_trace_residual,
                "maximum_hermiticity_residual": item.maximum_hermiticity_residual,
                "maximum_projector_residual": item.maximum_projector_residual,
                "tolerance": item.tolerance,
                "quantization_tolerance": item.quantization_tolerance,
            }
            for index, item in enumerate(grid.local_chern)
        ],
        "localizer": [
            {
                "probe_position": item.probe_position.tolist(),
                "energy": item.energy,
                "kappa": item.kappa,
                "signature": item.signature,
                "local_chern_number": item.local_chern_number,
                "is_invertible": item.is_invertible,
                "localizer_gap": item.localizer_gap,
                "positive_eigenvalue_count": item.positive_eigenvalue_count,
                "negative_eigenvalue_count": item.negative_eigenvalue_count,
                "zero_eigenvalue_count": item.zero_eigenvalue_count,
                "minimum_energy_distance": item.minimum_energy_distance,
                "maximum_hamiltonian_hermiticity_residual": (
                    item.maximum_hamiltonian_hermiticity_residual
                ),
                "maximum_localizer_hermiticity_residual": (
                    item.maximum_localizer_hermiticity_residual
                ),
                "tolerance": item.tolerance,
            }
            for item in grid.localizer
        ],
        "representatives": [
            {
                "method": item.method.value,
                "invariant_value": item.invariant_value,
                "is_topological": item.is_topological,
                "confidence": {
                    "is_resolved": item.confidence.is_resolved,
                    "is_quantized": item.confidence.is_quantized,
                    "minimum_gap": item.confidence.minimum_gap,
                    "gap_kind": item.confidence.gap_kind,
                    "quantization_error": item.confidence.quantization_error,
                    "maximum_residual": item.confidence.maximum_residual,
                    "convergence_checked": item.confidence.convergence_checked,
                    "diagnostics": _jsonify(item.confidence.diagnostics),
                },
                "warnings": list(item.warnings),
            }
            for item in grid.representatives
        ],
        "localizer_protection_proxy": grid.localizer_protection_proxy,
    }


def _encode_disorder_provenance(disorder: DisorderOutcome) -> dict[str, _JSONValue]:
    provenance = disorder.provenance
    return {
        "disorder_key": provenance.disorder_key,
        "disorder_version": provenance.disorder_version,
        "parameters": _jsonify(provenance.parameters),
        "seed": provenance.seed,
        "rng_algorithm": provenance.rng_algorithm,
        "source": {
            "target": provenance.source.target.value,
            "identifier": provenance.source.identifier,
            "scheme": provenance.source.scheme,
        },
        "result": {
            "target": provenance.result.target.value,
            "identifier": provenance.result.identifier,
            "scheme": provenance.result.scheme,
        },
    }


def _encode_baseline_statistics(
    statistics: SearchBaselineStatistics,
) -> dict[str, _JSONValue]:
    return {
        "criterion_key": statistics.criterion.key,
        "criterion_description": statistics.criterion.description,
        "trial_count": statistics.trial_count,
        "successful_trial_count": statistics.successful_trial_count,
        "trial_successes": list(statistics.trial_successes),
        "trial_success_fraction": statistics.trial_success_fraction,
        "trial_success_standard_error": statistics.trial_success_standard_error,
        "confidence_level": statistics.confidence_level,
        "uncertainty_method": statistics.uncertainty_method.value,
        "trial_success_confidence_interval": list(
            statistics.trial_success_confidence_interval
        ),
        "candidates_per_trial": statistics.candidates_per_trial,
        "total_candidate_count": statistics.total_candidate_count,
        "successful_candidate_count": statistics.successful_candidate_count,
        "candidate_success_fraction": statistics.candidate_success_fraction,
        "invalid_candidate_count": statistics.invalid_candidate_count,
        "callback_failure_count": statistics.callback_failure_count,
        "warnings": list(statistics.warnings),
    }


def _encode_selection(item: Phase98SelectedCandidate) -> dict[str, _JSONValue]:
    return {
        "trial_index": item.trial_index,
        "candidate_index": item.candidate_index,
        "geometry_id": item.geometry_id,
        "graph_fingerprint": canonical_graph_hash(item.geometry),
        "ranking_values": _jsonify(item.ranking_entry.values),
        "rank": item.ranking_entry.rank,
    }


def _encode_channel_summary(
    item: Phase98DisorderChannelSummary,
    destination: Path,
) -> dict[str, _JSONValue]:
    return {
        "role_key": item.role_key,
        "channel": item.channel.value,
        "seed_role": item.seed_role,
        "successful_count": item.successful_count,
        "total_count": item.total_count,
        "observed_fraction": item.observed_fraction,
        "wilson_interval_95": list(item.wilson_interval),
        "passes": item.passes,
        "operational_failure_count": item.operational_failure_count,
        "invalid_geometry_count": item.invalid_geometry_count,
        "invalid_evaluation_count": item.invalid_evaluation_count,
        "manifest": _artifact_link(destination, item.manifest),
    }


def _encode_dry_run(item: Phase98DryRunRecord) -> dict[str, _JSONValue]:
    return {
        "role": "dry_run_only",
        "seed": item.seed,
        "geometry_id": item.exact_geometry_id,
        "serialization_sha256": item.exact_serialization_sha256,
        "complete_attempt_count": item.complete_attempt_count,
        "proposal_count": item.proposal_count,
    }


def _channel_stress_definition(channel: Phase98DisorderChannel) -> dict[str, _JSONValue]:
    if channel is Phase98DisorderChannel.ONSITE:
        return {"transform": "uniform_onsite_disorder", "width": 1.0}
    if channel is Phase98DisorderChannel.HOPPING:
        return {"transform": "uniform_hopping_disorder", "width": 0.5}
    if channel is Phase98DisorderChannel.PAIRING:
        return {
            "transform": "uniform_pairing_disorder",
            "channel": "chiral_p_wave",
            "width": 0.5,
        }
    if channel is Phase98DisorderChannel.COORDINATE:
        return {"transform": "uniform_coordinate_perturbation", "width": 0.20}
    if channel is Phase98DisorderChannel.EDGE_REMOVAL:
        return {"transform": "random_edge_removal", "removal_probability": 0.05}
    return {"transform": "random_node_removal", "removal_probability": 0.02}


def _artifact_link(root: Path, path: Path) -> dict[str, _JSONValue]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _environment_record() -> dict[str, _JSONValue]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "scipy": version("scipy"),
        "platform": platform.platform(),
        "bytecode_disabled": sys.dont_write_bytecode,
        "rng_algorithm": "numpy.random.PCG64",
        "execution_order": "sequential_stored_request_order",
    }


def _validate_full_run_environment(code_commit: str) -> str:
    if not isinstance(code_commit, str) or _COMMIT_PATTERN.fullmatch(code_commit) is None:
        raise ValueError("code_commit must be a full lowercase 40-hex Git commit")
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError("the accepted Phase-9.8 run requires Python 3.14")
    if not sys.dont_write_bytecode:
        raise RuntimeError("the accepted Phase-9.8 run requires PYTHONDONTWRITEBYTECODE=1")
    if code_commit == PHASE_9_8_PROTOCOL_COMMIT:
        raise ValueError("code_commit must include the later Phase-9.8 implementation")
    _verify_committed_worktree(code_commit)
    return code_commit


def _verify_committed_worktree(code_commit: str) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    safe_directory = repository_root.as_posix()

    def git(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                (
                    "git",
                    "-c",
                    f"safe.directory={safe_directory}",
                    "-C",
                    str(repository_root),
                    *arguments,
                ),
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                "the accepted Phase-9.8 run requires a readable Git worktree"
            ) from error
        return completed.stdout.strip()

    head_commit = git("rev-parse", "HEAD")
    if head_commit != code_commit:
        raise RuntimeError(
            "code_commit does not match the exact Git HEAD used for execution"
        )
    status_lines = tuple(
        line for line in git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
        if line
    )
    unexpected = tuple(
        line
        for line in status_lines
        if line[3:].replace("\\", "/") not in _ALLOWED_WORKTREE_ARTIFACTS
    )
    if unexpected:
        raise RuntimeError(
            "the Phase-9.8 code worktree must be committed before execution; "
            f"unexpected status entries={unexpected!r}"
        )


def _metadata_integer(geometry: Geometry, key: str) -> int:
    value = geometry.metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"generator metadata {key!r} must be an integer")
    return value


def _complex_array(values: np.ndarray) -> dict[str, _JSONValue]:
    array = np.asarray(values, dtype=complex)
    return {"real": array.real.tolist(), "imag": array.imag.tolist()}


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Phase-9.8 artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _jsonify(payload),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"Phase-9.8 artifact already exists: {path}")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _jsonify(value: object) -> _JSONValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("Phase-9.8 JSON artifacts reject non-finite floats")
        return result
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return _complex_array(value)
        return _jsonify(value.tolist())
    raise TypeError(f"unsupported Phase-9.8 JSON value: {type(value).__name__}")
