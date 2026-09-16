"""Durable experiment loop with stage-level cooperative controls and recovery."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from toposc_lab.discovery.storage import atomic_json, read_json, writer_lease
from toposc_lab.research.config import ExperimentConfig
from toposc_lab.research.provenance import capture, verify
from toposc_lab.research.reporting import diagnostics, final_report
from toposc_lab.research.storage import ResearchStore, atomic_text, dumps, utc_now

TERMINAL = {"COMPLETED", "STOPPED", "ARCHIVED"}


class ResearchEngine:
    """The database commits plans, RNG state and stage results before advancing.

    A pause exits after the current exact stage; reopening the same engine is a
    resume. Started-but-uncommitted attempts remain charged and are retried with
    the original stage seed up to retry_limit. Hooks are durable fault boundaries.
    """

    def __init__(self, directory: str | Path, *, hook: Callable[[str, dict[str, Any]], None] | None = None,
                 evaluator: Any = None) -> None:
        from toposc_lab.research.physics import create_evaluator, provenance_from_manifest
        from toposc_lab.research.space import SPACE_REGISTRY
        from toposc_lab.research.strategies import create_strategy
        from toposc_lab.research.surrogate import SURROGATE_REGISTRY

        self.directory = Path(directory)
        self.store = ResearchStore(directory)
        self.config = ExperimentConfig(**self.store.get("config"))
        self.manifest = self.store.get("manifest")
        self.space = SPACE_REGISTRY[self.config.geometry_space](**self.config.space)
        self.strategy = create_strategy(self.config.algorithm, self.space, self.config.seed,
                                        experiment_id=self.manifest["experiment_id"],
                                        code_version=self.manifest["git_commit"], **self.config.search)
        saved = self.store.get("strategy")
        if saved is not None:
            self.strategy.resume(saved)
        else:
            self.strategy.initialize()
        self.surrogate = SURROGATE_REGISTRY[self.config.surrogate_model](
            seed=self.config.seed, **self.config.surrogate)
        saved_model = self.store.get("model")
        if saved_model:
            self.surrogate.resume(saved_model)
        protocol = self.config.physics_protocol()
        self.evaluator = evaluator or create_evaluator(
            protocol, provenance=provenance_from_manifest(self.manifest, protocol))
        self.hook = hook
        self.started = 0.0
        self.previous_elapsed = 0.0

    @classmethod
    def create(cls, directory: str | Path, config: ExperimentConfig | dict[str, Any]) -> Path:
        config = config if isinstance(config, ExperimentConfig) else ExperimentConfig(**config)
        config.validate_plugins()
        directory = Path(directory).resolve()
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("Experiment directory must be empty; use resume or clone")
        directory.mkdir(parents=True, exist_ok=True)
        for folder in ("candidates", "checkpoints", "models", "plots", "reports", "logs"):
            (directory / folder).mkdir()
        manifest = capture(directory)
        experiment_id = "EXP-" + uuid4().hex[:16]
        manifest.update(experiment_id=experiment_id, config_sha256=config.fingerprint,
                        blas_threads=config.blas_threads)
        store = ResearchStore(directory, create=True)
        state = {"experiment_id": experiment_id, "status": "CREATED", "cycle": 0,
                 "generated": 0, "exact_evaluations": 0, "elapsed_seconds": 0.0,
                 "exact_seconds": 0.0, "baselines_done": False, "pending": [],
                 "created": utc_now(), "last_checkpoint_count": 0,
                 "remaining_exact_budget": config.exact_budget,
                 "warnings": ["Finite-system evidence only; no Majorana/phase certificate"]}
        with store.connect() as db:
            store.put(db, "config", "current", config.to_dict())
            store.put(db, "manifest", "current", manifest)
            store.put(db, "state", "current", state)
            store.event(db, "experiment_created", {"config": config.to_dict(), "manifest": manifest})
        atomic_text(directory / "config.json", json.dumps(config.to_dict(), indent=2))
        atomic_json(directory / "state.json", state)
        atomic_json(directory / "manifest.json", manifest)
        return directory

    def _hook(self, name: str, **payload: Any) -> None:
        if self.hook:
            self.hook(name, payload)

    def _state(self, **changes: Any) -> dict[str, Any]:
        state: dict[str, Any] = self.store.get("state")
        attempts = self.store.attempts()
        state.update(exact_evaluations=len(attempts),
                     exact_seconds=sum(a["seconds"] or 0 for a in attempts),
                     remaining_exact_budget=max(0, self.config.exact_budget - len(attempts)))
        if self.started:
            state["elapsed_seconds"] = self.previous_elapsed + time.monotonic() - self.started
        state.update(changes)
        return state

    def _save_state(self, **changes: Any) -> dict[str, Any]:
        state = self._state(**changes)
        self.store.save("state", state)
        return state

    def _control(self) -> bool:
        with self.store.connect(readonly=True) as db:
            requests = db.execute("SELECT * FROM controls WHERE consumed=0 ORDER BY number").fetchall()
        actions = [r["action"] for r in requests]
        if "checkpoint" in actions:
            self.checkpoint("manual")
        if requests:
            self._hook("control_pending")
            state = self._state()
            if "stop" in actions:
                state.update(status="FINALIZING", final_status="STOPPED", finish_reason="manual_stop")
            elif "pause" in actions:
                state.update(status="PAUSED")
            with self.store.connect() as db:
                self.store.put(db, "state", "current", state)
                for row in requests:
                    db.execute("UPDATE controls SET consumed=1 WHERE number=?", (row["number"],))
                self.store.event(db, "controls_applied", {"actions": actions})
            self._hook("control_committed")
        if "stop" in actions or "pause" in actions:
            if "stop" in actions:
                self._finish("STOPPED", "manual_stop")
            else:
                self.checkpoint("pause")
            return True
        if self._state()["elapsed_seconds"] >= self.config.wall_seconds:
            self._save_state(status="PAUSED", pause_reason="wall_clock_budget")
            self.checkpoint("wall_clock_budget")
            return True
        return False

    def _stage(self, candidate: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any] | None:
        from toposc_lab.research.space import geometry_from_payload

        identity = candidate["id"] + ":" + stage["key"]
        saved: dict[str, Any] | None = self.store.get("exact_result", identity)
        if saved is not None:
            return saved
        attempts = self.store.attempts()
        previous = [a for a in attempts if a["candidate"] == candidate["id"]
                    and a["stage"] == stage["key"]]
        if len(previous) > self.config.retry_limit:
            return {"status": "failed", "error": "Stage retry limit exhausted", "stage": stage}
        if len(attempts) >= self.config.exact_budget:
            return None
        with self.store.connect() as db:
            cursor = db.execute("INSERT INTO attempts(candidate,stage,status,started) VALUES(?,?,?,?)",
                                (candidate["id"], stage["key"], "started", utc_now()))
            number = cursor.lastrowid
            self.store.event(db, "exact_started", {"attempt": number, "candidate": candidate["id"],
                                                   "stage": stage})
        self._save_state(current_candidate=candidate["id"], current_stage=stage["key"])
        self._hook("attempt_started", attempt=number)
        before = time.monotonic()
        try:
            result: dict[str, Any] = self.evaluator.evaluate(geometry_from_payload(candidate["geometry"]), stage)
            dumps(result)  # Reject NaN/inf and nonserializable exact output before any score.
        except Exception as error:  # noqa: BLE001 -- persist plugin/solver failures as evidence
            result = {"status": "failed", "error": f"{type(error).__name__}: {error}", "stage": stage}
        self._hook("after_simulation", attempt=number)
        seconds = time.monotonic() - before
        failed = result.get("status") == "failed"
        with self.store.connect() as db:
            self.store.put(db, "exact_result", identity, result)
            db.execute("UPDATE attempts SET status=?,finished=?,seconds=?,error=? WHERE number=?",
                       ("failed" if failed else "complete", utc_now(), seconds, result.get("error"), number))
            if failed:
                self.store.put(db, "error", str(number), result)
            self.store.event(db, "exact_finished", {"attempt": number, "failed": failed})
        self._hook("stage_saved", attempt=number)
        self._save_state()
        state = self.store.get("state")
        if state["exact_evaluations"] - state["last_checkpoint_count"] >= self.config.checkpoint_every:
            self.checkpoint("interval")
        return result

    def _evaluate_candidate(self, identity: str) -> bool:
        candidate = self.store.get("candidate", identity)
        if candidate.get("observed"):
            return True
        results: dict[str, Any] = {}
        for stage in self.evaluator.plan():
            if self._control():
                return False
            result = self._stage(candidate, stage)
            if result is None:
                partial = self.evaluator.summarize(results)
                candidate.update(partial, exact_results=results, origin="partial_exact", score=None)
                candidate["warnings"] = [*candidate.get("warnings", []),
                                         "Exact budget exhausted before full validation"]
                self.store.save("candidate", candidate, identity)
                self._finish("COMPLETED", "exact_budget_exhausted")
                return False
            results[stage["key"]] = result
            if result.get("status") == "failed":
                break
        try:
            summary = self.evaluator.summarize(results)
            if any(r.get("status") == "failed" for r in results.values()):
                summary.update(validation_state="FAILED", score=None)
            score = summary.get("score")
            if score is not None and (isinstance(score, bool) or not math.isfinite(score)):
                raise ValueError("Invalid nonfinite scientific score")
            dumps(summary)
        except Exception as error:  # noqa: BLE001 -- invalid plugin summaries cannot become elites
            summary = {"validation_state": "FAILED", "score": None,
                       "warnings": [str(error)], "origin": "exact"}
        candidate.update(summary, exact_results=results, origin="exact", observed=True,
                         exact_evaluation_count=len(self.store.attempts()), completed=utc_now())
        if candidate.get("prediction") and candidate.get("score") is not None:
            self.surrogate.observe(candidate["prediction"], candidate["score"])
        if candidate.get("score") is not None and candidate.get("validation_state") != "FAILED":
            self.strategy.observe(candidate, {**summary, "origin": "exact"})
        state = self._state()
        with self.store.connect() as db:
            self.store.put(db, "candidate", identity, candidate)
            self.store.put(db, "strategy", "current", self.strategy.checkpoint())
            self.store.put(db, "model", "current", self.surrogate.checkpoint())
            self.store.put(db, "state", "current", state)
            for key, value in candidate.get("validation_results", {}).items():
                self.store.put(db, "validation_result", identity + ":" + key, value)
            self.store.event(db, "candidate_completed", {"candidate": identity,
                                                       "validation_state": candidate["validation_state"]})
        atomic_json(self.directory / "candidates" / (identity + ".json"), candidate)
        self._hook("candidate_saved", candidate=identity)
        return True

    def _baseline_candidates(self) -> list[dict[str, Any]]:
        from toposc_lab.research.descriptors import compute_descriptors
        from toposc_lab.research.space import geometry_to_payload

        baseline_candidates = []
        rng = np.random.default_rng(self.config.seed + 991)
        for family in self.config.baselines:
            geometry = self.space.reference()
            if family == "random_rewired":
                accepted = 0
                for _ in range(100):
                    edited, _ = self.space.mutate(geometry, rng, "local_rewiring")
                    if not self.space.validate(edited):
                        geometry = edited
                        accepted += 1
                    if accepted >= 8:
                        break
            if self.space.validate(geometry):
                raise ValueError(f"Baseline {family} is incompatible with configured constraints")
            if family == "random_rewired" and self.space.hash(geometry) == self.space.hash(self.space.reference()):
                with self.store.connect() as db:
                    self.store.event(db, "baseline_unavailable", {
                        "family": family, "reason": "No distinct admissible rewiring found; regular duplicate omitted"})
                continue
            identity = "baseline-" + family
            baseline_candidates.append({"id": identity, "candidate_id": identity,
                "geometry": geometry_to_payload(geometry), "descriptors": compute_descriptors(geometry),
                "baseline": True, "family": family, "generation": 0, "parent": None,
                "lineage": [], "mutation": {"operator": "baseline"}, "seed": self.config.seed + 991,
                "created": utc_now(), "code_version": self.manifest["git_commit"],
                "experiment_id": self.manifest["experiment_id"], "validation_state": "PROPOSED"})
        return baseline_candidates

    def _plan(self) -> bool:
        state = self._state()
        stage_cost = len(self.evaluator.plan())
        slots = (self.config.exact_budget - state["exact_evaluations"]) // stage_cost
        if slots < 1:
            self._finish("COMPLETED", "insufficient_budget_for_complete_candidate")
            return False
        if not state["baselines_done"]:
            candidates = self._baseline_candidates()[:slots]
            selected = candidates
            state.update(pending_kind="baselines")
        else:
            remaining = self.config.candidate_budget - state["generated"]
            if remaining <= 0 or state["cycle"] >= self.config.cycles:
                self._finish("COMPLETED", "candidate_or_cycle_budget")
                return False
            complete = [c for c in self.store.all("candidate")
                        if c.get("origin") == "exact" and c.get("score") is not None
                        and c.get("validation_state") != "FAILED"]
            if self.strategy.uses_surrogate and len(complete) >= 4 and (
                    not self.surrogate.fitted or state["cycle"] % self.config.retrain_every == 0
                    ) and state.get("last_trained_cycle") != state["cycle"]:
                self._save_state(surrogate_status="training")
                self.surrogate.fit(complete)
                with self.store.connect() as db:
                    state["last_trained_cycle"] = state["cycle"]
                    self.store.put(db, "state", "current", state)
                    self.store.put(db, "model", "current", self.surrogate.checkpoint())
                    self.store.put(db, "model_version", str(state["cycle"]), self.surrogate.summarize())
                    self.store.event(db, "surrogate_retrained", self.surrogate.summarize())
                atomic_json(self.directory / "models" / f"cycle-{state['cycle']:06d}.json",
                            self.surrogate.checkpoint())
            self.strategy.proposal_limit = remaining
            proposals = self.strategy.propose(min(self.config.pool_size, remaining),
                max_attempts=min(remaining, max(30, self.config.pool_size * 30)))
            rejections = self.strategy.drain_rejections()
            state["generated"] += len(proposals) + len(rejections)
            selected = self.strategy.select(proposals, min(self.config.batch_size, slots),
                                             surrogate=self.surrogate if self.surrogate.fitted else None)
            candidates = proposals + rejections
            state.update(pending_kind="search", surrogate_status="ready" if self.surrogate.fitted
                         else "awaiting_at_least_four_exact_labels")
        selected_ids = {c["id"] for c in selected}
        state["pending"] = [c["id"] for c in selected]
        with self.store.connect() as db:
            for candidate in candidates:
                candidate.setdefault("candidate_id", candidate["id"])
                candidate.setdefault("baseline", False)
                candidate["selected"] = candidate["id"] in selected_ids
                self.store.put(db, "candidate", candidate["id"], candidate)
                if candidate.get("prediction"):
                    self.store.put(db, "surrogate_prediction", candidate["id"], candidate["prediction"])
                self.store.put(db, "geometry", candidate["id"], candidate["geometry"])
                self.store.put(db, "descriptors", candidate["id"], candidate.get("descriptors", {}))
                self.store.put(db, "lineage", candidate["id"], {k: candidate.get(k) for k in
                               ("parent", "lineage", "mutation", "generation", "seed")})
            self.store.put(db, "state", "current", state)
            self.store.put(db, "strategy", "current", self.strategy.checkpoint())
            self.store.event(db, "plan_saved", {"cycle": state["cycle"], "selected": state["pending"],
                                               "generated": state["generated"]})
        self._hook("plan_saved", cycle=state["cycle"])
        return True

    def checkpoint(self, reason: str = "manual") -> dict[str, Any]:
        from toposc_lab.research.service import ResearchService

        self._save_state()
        snapshot = ResearchService.snapshot(self.directory, detect_interrupted=False)
        report = diagnostics(snapshot)
        state = self._state(**{k: report[k] for k in ("best_score", "best_validated_candidate",
                                                    "archive_coverage", "duplicate_rate", "invalid_rate",
                                                    "surrogate_error", "diversity")})
        state["last_checkpoint_count"] = state["exact_evaluations"]
        report.update(reason=reason, created=utc_now(), cycle=state["cycle"])
        identity = f"{len(self.store.all('checkpoint')):06d}"
        checkpoint = {"state": state, "strategy": self.strategy.checkpoint(),
                      "config_sha256": self.config.fingerprint, "report": report}
        with self.store.connect() as db:
            self.store.put(db, "state", "current", state)
            self.store.put(db, "checkpoint", identity, checkpoint)
            self.store.put(db, "history", identity, {**report, "exact_evaluations": state["exact_evaluations"]})
            self.store.event(db, "checkpoint", {"id": identity, "reason": reason})
        atomic_json(self.directory / "checkpoints" / (identity + ".json"), checkpoint)
        atomic_json(self.directory / "state.json", state)
        atomic_text(self.directory / "reports" / ("checkpoint-" + identity + ".md"),
                    "# Research checkpoint\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n")
        self._hook("checkpoint_saved", checkpoint=identity)
        return report

    def _report(self) -> None:
        from toposc_lab.research.service import ResearchService

        snapshot = ResearchService.snapshot(self.directory, detect_interrupted=False)
        if snapshot["state"]["status"] == "FINALIZING":
            snapshot["state"]["status"] = snapshot["state"]["final_status"]
        report = final_report(snapshot)
        self.store.save("report", {"markdown": report})
        self._hook("report_saved")
        atomic_text(self.directory / "final_report.md", report)

    def repair_exports(self) -> None:
        """Rebuild derived exports from the authoritative checked database; no physics."""
        with self.store.connect(readonly=True) as db:
            rows = db.execute("SELECT * FROM objects WHERE kind='checkpoint'").fetchall()
        for row in rows:
            saved = self.store.decode(row)
            path = self.directory / "checkpoints" / (row["id"] + ".json")
            try:
                if read_json(path) == saved:
                    continue
            except (OSError, ValueError, KeyError, TypeError):
                pass
            atomic_json(path, saved)
            with self.store.connect() as db:
                self.store.event(db, "checkpoint_export_repaired", {"path": path.name})
        report = self.store.get("report")
        state = self.store.get("state")
        if state["status"] in TERMINAL and state["status"] != "ARCHIVED" and not report:
            self._report()
            report = self.store.get("report")
        if report:
            path = self.directory / "final_report.md"
            if not path.exists() or path.read_text(encoding="utf-8") != report["markdown"]:
                atomic_text(path, report["markdown"])
        atomic_json(self.directory / "state.json", state)

    def _finish(self, status: str, reason: str) -> None:
        self._save_state(status="FINALIZING", final_status=status, finish_reason=reason, finished=utc_now())
        self.checkpoint(reason)
        self._report()
        state = self._state(status=status)
        with self.store.connect() as db:
            self.store.put(db, "state", "current", state)
            self.store.event(db, "experiment_finished", {"status": status, "reason": reason})
        atomic_json(self.directory / "state.json", state)

    def run(self, *, max_cycles: int | None = None) -> dict[str, Any]:
        if max_cycles is not None and (type(max_cycles) is not int or max_cycles < 0):
            raise ValueError("max_cycles must be a nonnegative integer")
        with writer_lease(self.directory):
            verify(self.directory, self.manifest)
            self.store.integrity_check()
            if self.config.fingerprint != self.manifest["config_sha256"]:
                raise ValueError("Resume configuration mismatch")
            file_config = ExperimentConfig.from_file(self.directory / "config.json")
            if file_config.fingerprint != self.config.fingerprint:
                raise ValueError("config.json differs from the frozen database configuration; clone a new run")
            if ExperimentConfig(**self.store.get("config")).fingerprint != self.config.fingerprint:
                raise ValueError("Engine configuration changed since construction")
            # Constructing an engine does not acquire its lease. Always reload the
            # current search and model here so an older object cannot lose newer work.
            self.strategy.initialize()
            saved_strategy = self.store.get("strategy")
            if saved_strategy:
                self.strategy.resume(saved_strategy)
            saved_model = self.store.get("model")
            if saved_model:
                self.surrogate.resume(saved_model)
            self.repair_exports()
            state = self.store.get("state")
            if state["status"] == "FINALIZING":
                self._finish(state["final_status"], state["finish_reason"])
                return self._state()
            if state["status"] in TERMINAL:
                return self._state()
            runtime = {"blas_thread_environment": {name: os.environ.get(name) for name in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")}}
            previous_runtime = self.store.get("execution_runtime")
            if previous_runtime is not None and runtime != previous_runtime:
                raise ValueError("Resume BLAS execution environment mismatch")
            if previous_runtime is None:
                self.store.save("execution_runtime", runtime)
            if state.get("pause_reason") == "wall_clock_budget" and state["elapsed_seconds"] >= self.config.wall_seconds:
                raise ValueError("Wall-clock budget exhausted; clone with new compute limits")
            self.previous_elapsed = state["elapsed_seconds"]
            self.started = time.monotonic()
            with self.store.connect() as db:
                db.execute("UPDATE attempts SET status='interrupted',finished=? WHERE status='started'", (utc_now(),))
                db.execute("UPDATE controls SET consumed=1 WHERE action='resume' AND consumed=0")
                self.store.event(db, "run_started" if state["status"] == "CREATED" else "resumed",
                                 {"pid": os.getpid(), "cycle": state["cycle"]})
            self._save_state(status="RUNNING", pid=os.getpid(), pause_reason=None)
            initial_cycle = state["cycle"]
            try:
                while True:
                    if self._control():
                        break
                    state = self.store.get("state")
                    if not state["pending"]:
                        if max_cycles is not None and state["cycle"] - initial_cycle >= max_cycles:
                            self._save_state(status="PAUSED")
                            self.checkpoint("bounded_run_pause")
                            break
                        if not self._plan():
                            break
                        state = self.store.get("state")
                    for identity in state["pending"]:
                        if not self._evaluate_candidate(identity):
                            return self._state()
                    state = self._state(pending=[])
                    if state.get("pending_kind") == "baselines":
                        state["baselines_done"] = True
                    else:
                        state["cycle"] += 1
                    self.store.save("state", state)
                    self.checkpoint("cycle_complete")
                    self._hook("cycle_committed", cycle=state["cycle"])
            except BaseException:
                # Includes graceful keyboard interruption; a hard process crash leaves RUNNING,
                # detected by the OS-released lease when the monitor/restart reads it.
                if self.store.get("state")["status"] not in TERMINAL | {"PAUSED", "FINALIZING"}:
                    self._save_state(status="INTERRUPTED")
                raise
            finally:
                self.started = 0.0
            return self._state()
