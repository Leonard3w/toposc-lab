"""User-started, failure-aware campaign for the frozen Phase-10 research protocol."""

from __future__ import annotations

import hashlib
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from toposc_lab.evaluation import GeometryEvaluationRun
from toposc_lab.evaluation.reproducibility import ReproducibilityRecord
from toposc_lab.geometry.generators import BUILTIN_GEOMETRY_GENERATORS
from toposc_lab.search._research_report import render_report, research_summary
from toposc_lab.search._research_runtime import ResearchMonitor, research_environment
from toposc_lab.search._research_storage import (
    AttemptLedger,
    ResearchAbort,
    encode_record,
    json_bytes,
    load_record,
    load_sealed,
    publish_derived,
    save_record,
)
from toposc_lab.search.checkpoint import create_search_checkpoint, save_search_checkpoint
from toposc_lab.search.generation_loop import GenerationLoopResult, GenerationReproductionRequest
from toposc_lab.search.geometry_genome import GeometryGenome
from toposc_lab.search.phase_10_research import (
    CONFIRMATION_SEEDS,
    DERIVATION_ID,
    PREFLIGHT_SEEDS,
    REFERENCE_SEEDS,
    RESEARCH_PROTOCOL_COMMIT,
    RESEARCH_PROTOCOL_ID,
    RESEARCH_PROTOCOL_PATH,
    TRIAL_SEEDS,
    VALIDATION_SEEDS,
    evaluate_research_geometry,
    legal_edge_swaps,
    produce_research_offspring,
    research_protocol,
    sample_research_genome,
    square_reference,
)
from toposc_lab.search.search_benchmark import (
    SearchBenchmarkProtocol,
    SearchBenchmarkResult,
    SearchBenchmarkTrial,
    _audit_evaluations,
    _seed_schedule,
    run_search_benchmark,
)


class SeedRegistry:
    """Audit independently owned streams; named replay and shared roles are explicit."""

    def __init__(self) -> None:
        self.roles: dict[int, str] = {}
        for role, seeds in (
            ("preflight", PREFLIGHT_SEEDS),
            ("trial", TRIAL_SEEDS),
            ("reference", REFERENCE_SEEDS),
            ("validation", VALIDATION_SEEDS),
            ("confirmation", CONFIRMATION_SEEDS),
        ):
            for seed in seeds:
                self.claim(seed, f"{role}/{seed}")

    def claim(self, seed: int, role: str) -> None:
        if seed in self.roles and self.roles[seed] != role:
            raise ResearchAbort(f"seed collision: {role} and {self.roles[seed]}")
        self.roles[seed] = role

    def trial(self, root: int, protocol: SearchBenchmarkProtocol) -> dict[str, Any]:
        evolution, samples, evaluations = _seed_schedule(root, protocol)
        self.claim(evolution, f"{root}/evolution")
        for index, seed in enumerate(samples):
            self.claim(seed, f"{root}/sample/{index}")
        transitions = np.random.PCG64(evolution).random_raw(6)
        for index, seed in enumerate(transitions):
            self.claim(int(seed), f"{root}/transition/{index}")
        return {
            "root": root,
            "evolution": evolution,
            "sampling": samples,
            "evaluation": evaluations,
            "transitions": tuple(int(s) for s in transitions),
        }


@contextmanager
def _campaign_lock(root: Path) -> Iterator[None]:
    """OS-owned lock releases on crashes; a stale filename does not block resume."""
    with (root / "campaign.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError(
                "Dieses Ergebnisverzeichnis wird bereits von einem Lauf verwendet"
            ) from error
        try:
            yield
        finally:
            stream.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _output_path(output: Path) -> Path:
    root = output.resolve()
    results = Path(__file__).resolve().parents[3] / "results"
    if not root.is_relative_to(results.resolve()) or not root.name.startswith("phase_10_research"):
        raise ValueError("Ein neues results/phase_10_research...-Verzeichnis verwenden")
    return root


def run_research_campaign(
    output: Path,
    *,
    mode: Literal["preflight", "full", "resume"],
) -> Path:
    """Run exactly the accepted campaign; main physics is never started implicitly."""
    if mode not in ("preflight", "full", "resume"):
        raise ValueError("unknown research mode")
    environment = research_environment()
    root = _output_path(output)
    if mode == "preflight":
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        raise ValueError("Zuerst den Vorlauf in einem neuen Ergebnisverzeichnis starten")
    try:
        with _campaign_lock(root):
            if mode == "resume":
                mode = "full" if (root / "full").is_dir() else "preflight"
                resume = True
            else:
                resume = False
            stage = root / mode
            if mode == "full":
                dry_stage = root / "preflight"
                dry_manifest = load_record(dry_stage / "manifest.json")
                dry_complete = load_record(dry_stage / "complete.json")
                if (
                    dry_manifest["environment"] != environment
                    or not dry_complete["preflight_passed"]
                ):
                    raise ResearchAbort("Passender bestandener Vorlauf erforderlich")
                _verify_completion(dry_stage)
            if not resume:
                stage.mkdir(exist_ok=False)
            with ResearchMonitor(root / "events.jsonl") as monitor:
                try:
                    return _execute_stage(root, stage, mode, environment, monitor, resume=resume)
                except BaseException as error:
                    monitor.emit("interrupted", error_type=type(error).__name__, error=str(error))
                    raise
    except ResearchAbort as error:
        raise RuntimeError(str(error)) from error


def _execute_stage(
    root: Path,
    stage: Path,
    mode: str,
    environment: dict[str, Any],
    monitor: ResearchMonitor,
    *,
    resume: bool,
) -> Path:
    preflight = mode == "preflight"
    total = 129 if preflight else 2083
    protocol_file = Path(__file__).resolve().parents[3] / RESEARCH_PROTOCOL_PATH
    metadata = {
        "protocol_id": RESEARCH_PROTOCOL_ID,
        "protocol_commit": RESEARCH_PROTOCOL_COMMIT,
        "storage_amendment": "TOPOSC-P10-EVO-RS-001-A1",
        "checkpoint_layout": "immutable-generation-files.v1",
        "environment": environment,
        "mode": mode,
        "total_evaluation_attempts": total,
        "protocol_text": protocol_file.read_text(encoding="utf-8"),
        "seed_roles": {
            "preflight": PREFLIGHT_SEEDS,
            "trials": TRIAL_SEEDS,
            "references": REFERENCE_SEEDS,
            "validation_reserved": VALIDATION_SEEDS,
            "confirmation_reserved": CONFIRMATION_SEEDS,
        },
    }
    manifest = stage / "manifest.json"
    if manifest.exists():
        if load_record(manifest) != metadata:
            raise ResearchAbort(
                "Protokoll, Code oder numerische Umgebung stimmen beim Resume nicht überein"
            )
    else:
        save_record(manifest, metadata)
    estimate = None
    if not preflight:
        dry_stage = root / "preflight"
        dry_manifest = load_record(dry_stage / "manifest.json")
        dry_complete = load_record(dry_stage / "complete.json")
        if dry_manifest["environment"] != environment or not dry_complete["preflight_passed"]:
            raise ResearchAbort("Passender bestandener Vorlauf erforderlich")
        _verify_completion(dry_stage)
        estimate = dry_complete["seconds_per_evaluation"]
    monitor.emit(
        "starting",
        stage=mode,
        total=total,
        completed=0,
        resume=resume,
        estimated_seconds_per_evaluation=estimate,
    )
    if (stage / "complete.json").exists():
        _verify_completion(stage)
        monitor.emit("complete_already_sealed", completed=total, last_sealed=f"{mode}/complete")
        return stage / "report.md"
    seeds = PREFLIGHT_SEEDS[:2] if preflight else TRIAL_SEEDS
    registry = SeedRegistry()
    started = time.perf_counter()
    panel = _reference_panel(stage, preflight, environment["code_commit"], monitor)
    reference_proxy = panel["reference_proxy"]
    reference_count = 1 if preflight else 35
    if reference_proxy is None:
        raise ResearchAbort("Keine zulässige primäre Referenz; Suche wird nicht gestartet")
    reference_record = panel["provenance"]
    protocol = research_protocol(environment["code_commit"], reference_proxy, seeds)
    schedules = {seed: registry.trial(seed, protocol) for seed in seeds}
    if preflight:
        _geometry_checks(stage, monitor)
    trials = []
    for index, seed in enumerate(seeds):
        monitor.emit(
            "trial_start",
            trial=f"{index + 1}/{len(seeds)}",
            arm=None,
            generation=None,
            slot=None,
            completed=reference_count + index * 64,
        )
        directory = stage / f"trial_{index:02d}"
        single = replace(protocol, trial_seeds=(seed,))
        if (directory / "sealed.json").exists():
            trial = load_sealed(directory)
            if not isinstance(trial, SearchBenchmarkTrial):
                raise ResearchAbort("invalid sealed trial")
            for transition in trial.evolution.transitions:
                for child in transition.offspring:
                    operator_seed = child.proposal.operator_seed
                    if operator_seed is not None:
                        registry.claim(
                            operator_seed,
                            f"{seed}/operator/{transition.population.generation_index}/"
                            f"{child.proposal.validation_source_selection_index}",
                        )
        else:
            trial = _trial(
                directory,
                single,
                schedules[seed],
                registry,
                monitor,
                reference_proxy=reference_proxy,
                completed_base=reference_count + index * 64,
            )
        # Reconstructed objects have fresh identities; verify frozen values before
        # binding them to the corresponding reconstructed protocol for the audit.
        if (
            trial.evolution.definition != single.definition
            or trial.evolution.initial_population.validity_policy != single.validity_policy
        ):
            raise ResearchAbort("stored trial strategy differs from frozen protocol")
        audited = replace(
            single,
            definition=trial.evolution.definition,
            validity_policy=trial.evolution.initial_population.validity_policy,
        )
        SearchBenchmarkResult(audited, (trial,))
        reference_record = _audit_evaluations(trial, audited, (None,) * 32, reference_record)
        trials.append(trial)
        monitor.emit(
            "trial_sealed",
            completed=reference_count + (index + 1) * 64,
            last_sealed=f"{mode}/trial_{index:02d}",
        )
    immutable_trials = tuple(trials)
    summary_path = stage / "analysis.json"
    if summary_path.exists():
        summary = load_record(summary_path)
    else:
        summary = research_summary(immutable_trials, reference_proxy, preflight=preflight)
        summary["environment"] = environment
        summary["protocol_commit"] = RESEARCH_PROTOCOL_COMMIT
        summary["total_evaluation_attempts"] = total
        operator_rows = []
        for index, trial in enumerate(immutable_trials):
            directory = stage / f"trial_{index:02d}"
            sealed = load_record(directory / "sealed.json")
            operations = [
                load_record(path)
                for path in sorted(
                    (directory / sealed["execution"]).glob("mutation_*_operation.json")
                )
            ]
            expected = sum(len(t.offspring) for t in trial.evolution.transitions)
            if len(operations) != expected:
                raise ResearchAbort("sealed mutation ledger does not cover every offspring")
            noops = sum(op["reason"] == "no_legal_swap" for op in operations)
            operator_rows.append(
                {
                    "trial_seed": trial.seed,
                    "offspring_count": expected,
                    "no_legal_swap_count": noops,
                    "no_legal_swap_fraction": None if expected == 0 else noops / expected,
                }
            )
        summary["mutation_statistics"] = operator_rows
        started_count = len(list(stage.rglob("evaluation_*_input.json"))) + len(
            list((stage / "references").rglob("reference_*_input.json"))
        )
        summary["execution_accounting"] = {
            "logical_evaluation_budget": total,
            "recorded_evaluation_starts": started_count,
            "additional_starts_from_replay": max(0, started_count - total),
            "starts_may_include_interruption_immediately_before_solver": True,
        }
        summary["current_execution_measurements"] = getattr(monitor, "measurements", {})
        summary["timing_scope"] = (
            "Observed generation, mutation, physics and ledger/checkpoint publication times; "
            "shared-start generation has its own category. Monitor/reporting and replay loading "
            "remain in total elapsed time; category times are not a full wall-time partition."
        )
        summary["all_execution_timing_events"] = "../events.jsonl"
        save_record(summary_path, summary)
    # Derived report files are idempotent on resume, without changing any outcome.
    render_report(stage, summary, immutable_trials)
    publish_derived(stage / "summary.json", json_bytes(summary))
    elapsed = time.perf_counter() - started
    artifacts = _artifact_inventory(stage)
    save_record(
        stage / "complete.json",
        {
            "preflight_passed": preflight,
            "artifacts": artifacts,
            "seconds_per_evaluation": elapsed
            / max(1, getattr(monitor, "evaluations_observed", total)),
            "estimate_excludes_prior_execution_time": resume,
        },
    )
    monitor.emit("complete", completed=total, last_sealed=f"{mode}/complete")
    return stage / "report.md"


def _reference_panel(
    stage: Path, preflight: bool, code: str, monitor: ResearchMonitor
) -> dict[str, Any]:
    directory = stage / "references"
    if (directory / "sealed.json").exists():
        result: dict[str, Any] = load_sealed(directory)
        return result
    ledger = _campaign_ledger(directory, monitor)
    recipes: list[tuple[str, int | None, str | None]] = [("square", None, None)]
    if not preflight:
        recipes += [("hard_core_planar_reference", s, None) for s in REFERENCE_SEEDS]
        recipes += [
            ("ammann_beenker_patch", None, "ammann_beenker"),
            ("sierpinski_carpet", None, "sierpinski"),
        ]
    records = []
    provenance: ReproducibilityRecord | None = None
    eligible = []
    for index, (recipe, seed, descriptive) in enumerate(recipes):
        monitor.emit("reference_start", arm="reference", slot=index + 1)
        ledger.record(f"reference_{index:02d}_request.json", {"recipe": recipe, "seed": seed})
        generation_started = time.perf_counter()
        if recipe == "square":
            geometry = square_reference()
        else:
            parameters = (
                {"radius": 4.0, "spacing": 1.0}
                if descriptive == "ammann_beenker"
                else ({"order": 2, "spacing": 1.0} if descriptive == "sierpinski" else {})
            )
            geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
                recipe, parameters=parameters, seed=seed
            )
        monitor.emit(
            "reference_generated",
            category="generation_reference",
            operation_seconds=time.perf_counter() - generation_started,
        )
        ledger.record(f"reference_{index:02d}_input.json", GeometryGenome.from_geometry(geometry))
        started = time.perf_counter()
        try:
            evaluated = evaluate_research_geometry(geometry, code, descriptive=descriptive)
        except Exception as error:
            ledger.record(
                f"reference_{index:02d}_outcome.json",
                {"error": type(error).__name__, "message": str(error)},
            )
            raise ResearchAbort(f"reference evaluation failed: {error}") from error
        physics_seconds = time.perf_counter() - started
        record = {
            "recipe": recipe,
            "seed": seed,
            "genome": GeometryGenome.from_geometry(geometry),
            "run": evaluated.run,
            "scientific": evaluated.scientific,
        }
        ledger.record(f"reference_{index:02d}_outcome.json", record)
        records.append(record)
        if descriptive is None:
            if not evaluated.run.is_valid:
                raise ResearchAbort(
                    "Operative primäre Referenzinvalidität verhindert den Suchstart"
                )
            provenance = evaluated.run.reproducibility if provenance is None else provenance
            if evaluated.scientific["clean_eligible"]:
                eligible.append(evaluated.scientific["localizer_protection_proxy"])
        monitor.emit(
            "reference_done",
            completed=index + 1,
            category="physics",
            operation_seconds=physics_seconds,
        )
    panel = {
        "records": records,
        "reference_proxy": max(eligible, default=None),
        "provenance": provenance,
    }
    ledger.seal(panel)
    monitor.emit("panel_sealed", last_sealed=f"{stage.name}/references")
    return panel


def _geometry_checks(stage: Path, monitor: ResearchMonitor) -> None:
    directory = stage / "geometry_checks"
    if (directory / "sealed.json").exists():
        load_sealed(directory)
        return
    ledger = _campaign_ledger(directory, monitor)
    records = []
    for seed in PREFLIGHT_SEEDS[2:]:
        monitor.emit("geometry_check", arm="geometry_only", seed=seed)
        ledger.record(f"{seed}_request.json", {"seed": seed})
        genome = sample_research_genome(seed)
        record = {"seed": seed, "genome": genome, "legal_swap_count": len(legal_edge_swaps(genome))}
        ledger.record(f"{seed}_result.json", record)
        records.append(record)
    ledger.seal(records)


def _trial(
    directory: Path,
    protocol: SearchBenchmarkProtocol,
    schedule: dict[str, Any],
    registry: SeedRegistry,
    monitor: ResearchMonitor,
    *,
    reference_proxy: float,
    completed_base: int,
) -> SearchBenchmarkTrial:
    ledger = _campaign_ledger(directory, monitor)
    ledger.record("schedule.json", schedule)
    sample_index = 0
    attempt = 0
    hits = 0
    shared_outcomes: dict[int, bytes] = {}

    def outcome(index: int, value: Any) -> None:
        try:
            data = encode_record(value)
        except Exception as error:
            raise ResearchAbort(f"cannot encode scientific outcome: {error}") from error
        if index < 8:
            shared_outcomes[index] = data
        if 32 <= index < 40 and shared_outcomes[index - 32] != data:
            raise ResearchAbort("shared deterministic start evaluations differ between arms")
        ledger.record(f"evaluation_{index:02d}_outcome.json", value)

    def sampler(seed: int) -> GeometryGenome:
        nonlocal sample_index
        index = sample_index
        sample_index += 1
        ledger.record(f"sample_{index:02d}_request.json", {"seed": seed})
        monitor.emit(
            "generating", arm="shared_initial" if index < 8 else "random", sampling_seed=seed
        )
        started = time.perf_counter()
        try:
            genome = sample_research_genome(seed)
        except Exception as error:
            ledger.record(
                f"sample_{index:02d}_result.json",
                {
                    "error": type(error).__name__,
                    "message": str(error),
                },
            )
            raise
        generation_seconds = time.perf_counter() - started
        ledger.record(f"sample_{index:02d}_result.json", genome)
        monitor.emit(
            "generated",
            category="generation_shared_initial" if index < 8 else "generation_random",
            operation_seconds=generation_seconds,
        )
        return genome

    def evaluator(genome: GeometryGenome, seed: None) -> GeometryEvaluationRun:
        nonlocal attempt, hits
        index = attempt
        attempt += 1
        if seed is not None:
            raise ResearchAbort("clean physics must not receive an evaluation seed")
        arm = "evolution" if index < 32 else "random"
        ledger.record(f"evaluation_{index:02d}_input.json", genome)
        monitor.emit(
            "evaluating",
            arm=arm,
            generation=(index % 32) // 8,
            slot=index % 32 + 1,
            completed=completed_base + index,
            hits=hits,
        )
        started = time.perf_counter()
        try:
            evaluated = evaluate_research_geometry(genome.to_geometry(), protocol.code_version)
        except Exception as error:
            physics_seconds = time.perf_counter() - started
            outcome(
                index,
                {
                    "error": type(error).__name__,
                    "message": str(error),
                },
            )
            monitor.emit(
                "evaluation_failed",
                completed=completed_base + index + 1,
                category="physics",
                operation_seconds=physics_seconds,
            )
            raise
        physics_seconds = time.perf_counter() - started
        outcome(
            index,
            {
                "run": evaluated.run,
                "scientific": evaluated.scientific,
            },
        )
        # This display is observational; the benchmark criterion makes the final decision.
        values = evaluated.scientific
        hits += int(
            values["clean_eligible"]
            and values["localizer_protection_proxy"] >= 1.10 * reference_proxy
        )
        monitor.emit(
            "evaluated",
            completed=completed_base + index + 1,
            hits=hits,
            category="physics",
            operation_seconds=physics_seconds,
        )
        return evaluated.run

    def producer(request: GenerationReproductionRequest) -> Any:
        generation = request.target_generation_index
        ledger.record(
            f"reproduction_{generation}_input.json",
            {
                "seed": request.seed,
                "selected": request.selection,
                "required_offspring_count": request.required_offspring_count,
            },
        )
        monitor.emit("mutating", arm="evolution", generation=generation)
        started = time.perf_counter()
        storage_before = ledger.storage_seconds

        def record(data: dict[str, Any]) -> None:
            slot = data["parent_slot"]
            registry.claim(
                data["operator_seed"], f"{schedule['root']}/operator/{generation}/{slot}"
            )
            ledger.record(f"mutation_{generation}_{slot}_operation.json", data)

        proposals = produce_research_offspring(request, record=record)
        mutation_seconds = max(
            0.0, time.perf_counter() - started - (ledger.storage_seconds - storage_before)
        )
        ledger.record(f"reproduction_{generation}_output.json", proposals)
        monitor.emit("mutated", category="mutation", operation_seconds=mutation_seconds)
        return proposals

    def checkpoint(prefix: GenerationLoopResult) -> None:
        started = time.perf_counter()
        try:
            save_search_checkpoint(
                ledger.execution / f"checkpoint_generation_{prefix.config.generation_count:04d}.zip",
                create_search_checkpoint(
                    prefix,
                    requested_generation_count=3,
                    evaluator_identifier=DERIVATION_ID,
                    code_version=protocol.code_version,
                ),
                overwrite=False,
            )
        except Exception as error:
            raise ResearchAbort(f"checkpoint failed: {error}") from error
        monitor.emit(
            "checkpoint_saved",
            category="storage",
            operation_seconds=time.perf_counter() - started,
        )

    result = run_search_benchmark(
        protocol,
        sampler=sampler,
        evaluator=evaluator,
        offspring_producer=producer,
        checkpoint_callback=checkpoint,
    )
    trial = result.trials[0]
    if attempt != 64 or sample_index != 32:
        raise ResearchAbort("trial did not consume the frozen evaluation/sampling budget")
    ledger.seal(trial)
    return trial


def _campaign_ledger(directory: Path, monitor: ResearchMonitor) -> AttemptLedger:
    return AttemptLedger(
        directory,
        on_storage=lambda seconds: monitor.emit(
            "artifact_saved", category="storage", operation_seconds=seconds
        ),
    )


def _artifact_inventory(stage: Path) -> dict[str, str]:
    return {
        p.relative_to(stage).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(stage.rglob("*"))
        if p.is_file() and p.name != "complete.json" and not p.name.startswith(".writing-")
    }


def _verify_completion(stage: Path) -> None:
    completion = load_record(stage / "complete.json")
    if completion["artifacts"] != _artifact_inventory(stage):
        raise ResearchAbort("completed campaign artifact inventory/hash differs")
