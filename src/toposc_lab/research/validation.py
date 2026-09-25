"""Bounded fixed-cohort study using the existing ResearchEngine stage runtime.

CLI: python -m toposc_lab.research.validation --help
No search strategy, surrogate, or historical experiment is executed here.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from toposc_lab.discovery.storage import atomic_json, writer_lease
from toposc_lab.research.config import ExperimentConfig
from toposc_lab.research.engine import TERMINAL, ResearchEngine
from toposc_lab.research.physics import provenance_from_manifest
from toposc_lab.research.provenance import capture, verify
from toposc_lab.research.storage import ResearchStore, atomic_text, dumps, utc_now
from toposc_lab.research.validation_cohort import SPACE, digest, freeze_cohort, validate_cohort
from toposc_lab.research.validation_diagnostics import ValidationEvaluator

VERSION = "phase18.fixed-cohort-validation.v1"
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
PILOT_SEEDS = [180101, 180102]
CONFIRMATION_SEEDS = list(range(181001, 181051))


def pilot_settings(cohort: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": VERSION,
        "mode": "pilot",
        "cohort_sha256": cohort["sha256"],
        "widths": [1.2, 3.0, 6.0, 9.0, 12.0],
        "seeds": PILOT_SEEDS,
        "exact_budget": 120,
        "wall_seconds": 3600.0,
        "clean_seed": 180100,
        "retry_limit": 1,
        "checkpoint_every": 10,
        "protocol_document": "docs/decisions/phase18_validation_protocol.md",
    }


def make_config(settings: dict[str, Any], cohort: dict[str, Any]) -> ExperimentConfig:
    validate_cohort(cohort)
    if settings.get("version") != VERSION or settings.get("mode") not in {
        "pilot",
        "confirmation",
        "engineering",
    }:
        raise ValueError("Unknown validation study version/mode")
    if settings.get("cohort_sha256") != cohort["sha256"]:
        raise ValueError("Study cohort mismatch")
    if settings.get("mode") == "pilot" and settings["seeds"] != PILOT_SEEDS:
        raise ValueError("Pilot seeds are frozen")
    if settings.get("mode") == "confirmation" and settings["seeds"] != CONFIRMATION_SEEDS:
        raise ValueError("Confirmation requires the reserved 50 fresh seeds")
    if settings.get("mode") != "engineering" and (
        cohort["space"] != SPACE or len(cohort["candidates"]) != 10
    ):
        raise ValueError("Scientific cohort requires ten candidates in the frozen 100-site space")
    if any(w <= 0 for w in settings["widths"]):
        raise ValueError("Use one clean stage, not repeated W=0 disorder samples")
    if set(settings["seeds"]) & {17000, 17001, 17002, 17003, 17004}:
        raise ValueError("Historical seeds cannot be reused")
    if settings["clean_seed"] in settings["seeds"]:
        raise ValueError("Clean seed must be separate")
    if settings.get("exact_budget") is None or settings.get("wall_seconds") is None:
        raise ValueError("Explicit exact-attempt and wall-clock budgets required before creation")
    return ExperimentConfig(
        name="Phase 18 " + settings["mode"],
        exact_budget=settings["exact_budget"],
        wall_seconds=settings["wall_seconds"],
        retry_limit=settings["retry_limit"],
        checkpoint_every=settings["checkpoint_every"],
        space=cohort["space"],
        objective="robustness_quality_mean",
        physics={
            "disorder_widths": settings["widths"],
            "disorder_seeds": settings["seeds"],
            "clean_seed": settings["clean_seed"],
            "confirmation": False,
        },
    )


class ValidationStudy(ResearchEngine):
    """Fixed schedule; inherits charged exact stages, clocks and durable controls.

    Overrides search lifecycle/checkpoints only. No production search policy or
    old adapter changes. SQLite results are the resume cursor, so crash recovery
    never depends on a possibly stale exported file.
    """

    def __init__(self, directory: str | Path, *, hook: Any = None, evaluator: Any = None) -> None:
        self.directory = Path(directory)
        self.store = ResearchStore(directory)
        self.settings = self.store.get("validation_settings")
        self.cohort = self.store.get("validation_cohort")
        if self.settings is None:
            raise ValueError("Not a fixed-cohort validation study")
        self.config = make_config(self.settings, self.cohort)
        self.manifest = self.store.get("manifest")
        protocol = self.config.physics_protocol()
        self.evaluator = evaluator or ValidationEvaluator(
            protocol, provenance_from_manifest(self.manifest, protocol)
        )
        self.hook = hook
        self.started = 0.0
        self.previous_elapsed = 0.0

    @classmethod
    def create_study(
        cls, directory: str | Path, settings: dict[str, Any], cohort: dict[str, Any]
    ) -> Path:
        config = make_config(settings, cohort)
        directory = Path(directory).resolve()
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("Study directory must be new/empty")
        directory.mkdir(parents=True, exist_ok=True)
        for folder in ("checkpoints", "reports", "plots", "logs"):
            (directory / folder).mkdir()
        manifest = capture(directory)
        manifest.update(
            experiment_id="VAL-" + digest(settings)[:16],
            study_type=VERSION,
            config_sha256=config.fingerprint,
            settings_sha256=digest(settings),
            cohort_sha256=cohort["sha256"],
            blas_threads=1,
        )
        protocol_path = Path(__file__).resolve().parents[3] / settings["protocol_document"]
        protocol_text = protocol_path.read_text(encoding="utf-8")
        manifest["protocol_document_sha256"] = digest(protocol_text)
        atomic_text(directory / "protocol.md", protocol_text)
        store = ResearchStore(directory, create=True)
        state = {
            "experiment_id": manifest["experiment_id"],
            "status": "CREATED",
            "cycle": 0,
            "elapsed_seconds": 0.0,
            "exact_seconds": 0.0,
            "last_checkpoint_count": 0,
            "exact_evaluations": 0,
            "remaining_exact_budget": config.exact_budget,
            "planned_realizations": len(cohort["candidates"])
            * (1 + len(settings["widths"]) * len(settings["seeds"])),
        }
        with store.connect() as db:
            for kind, value in (
                ("validation_settings", settings),
                ("validation_cohort", cohort),
                ("config", config.to_dict()),
                ("manifest", manifest),
                ("state", state),
            ):
                store.put(db, kind, "current", value)
        atomic_text(directory / "study.json", json.dumps(settings, indent=2))
        atomic_text(directory / "cohort.json", json.dumps(cohort, indent=2))
        atomic_json(directory / "manifest.json", manifest)
        atomic_json(directory / "state.json", state)
        return directory

    def checkpoint(self, reason: str = "manual") -> dict[str, Any]:
        state = self._state()
        state["last_checkpoint_count"] = state["exact_evaluations"]
        saved = {
            "reason": reason,
            "state": state,
            "created": utc_now(),
            "settings_sha256": digest(self.settings),
            "cohort_sha256": self.cohort["sha256"],
            "complete_stage_records": self.store.count("exact_result"),
        }
        identity = f"{self.store.count('checkpoint'):06d}"
        with self.store.connect() as db:
            self.store.put(db, "state", "current", state)
            self.store.put(db, "checkpoint", identity, saved)
            self.store.event(db, "checkpoint", {"reason": reason, "id": identity})
        atomic_json(self.directory / "checkpoints" / (identity + ".json"), saved)
        atomic_json(self.directory / "state.json", state)
        self._hook("checkpoint_saved", checkpoint=identity)
        return saved

    def _report(self) -> None:
        from toposc_lab.research.validation_reporting import export_results

        report = export_results(self.directory)
        self.store.save("report", {"markdown": report})
        self._hook("report_saved")

    def run(
        self, *, max_stages: int | None = None, max_cycles: int | None = None
    ) -> dict[str, Any]:
        if max_cycles is not None:
            raise ValueError("Fixed validation uses max_stages, not search cycles")
        if max_stages is not None and (type(max_stages) is not int or max_stages < 0):
            raise ValueError("max_stages must be a nonnegative integer")
        with writer_lease(self.directory):
            verify(self.directory, self.manifest)
            self.store.integrity_check()
            for key, value in (
                ("validation_settings", self.settings),
                ("validation_cohort", self.cohort),
            ):
                if self.store.get(key) != value:
                    raise ValueError("Study changed since construction")
            if digest(self.settings) != self.manifest["settings_sha256"]:
                raise ValueError("Frozen settings mismatch")
            if self.cohort["sha256"] != self.manifest["cohort_sha256"]:
                raise ValueError("Frozen cohort mismatch")
            if self.config.fingerprint != self.manifest["config_sha256"]:
                raise ValueError("Frozen engine config mismatch")
            for name, expected in (("study.json", self.settings), ("cohort.json", self.cohort)):
                if json.loads((self.directory / name).read_text(encoding="utf-8")) != expected:
                    raise ValueError(f"Frozen {name} modified")
            if (
                digest((self.directory / "protocol.md").read_text(encoding="utf-8"))
                != self.manifest["protocol_document_sha256"]
            ):
                raise ValueError("Frozen protocol document modified")
            state = self.store.get("state")
            if state["status"] == "FINALIZING":
                self._finish(state["final_status"], state["finish_reason"])
                return self._state()
            if state["status"] in TERMINAL:
                # Rebuild exports, never recalculate completed evidence.
                self._report()
                return self._state()
            runtime = {k: os.environ.get(k) for k in THREAD_VARIABLES}
            if self.settings["mode"] != "engineering" and any(v != "1" for v in runtime.values()):
                raise ValueError("Set all four BLAS thread variables to 1 before Python starts")
            previous = self.store.get("execution_runtime")
            if previous is not None and previous != runtime:
                raise ValueError("Resume BLAS execution environment mismatch")
            self.store.save("execution_runtime", runtime)
            self.previous_elapsed = state["elapsed_seconds"]
            self.started = time.monotonic()
            with self.store.connect() as db:
                db.execute(
                    "UPDATE attempts SET status='interrupted',finished=? WHERE status='started'",
                    (utc_now(),),
                )
                db.execute("UPDATE controls SET consumed=1 WHERE action='resume' AND consumed=0")
            self._save_state(status="RUNNING", pid=os.getpid())
            initial_attempts = len(self.store.attempts())
            try:
                for stage in self.evaluator.plan():
                    # Same realization across the whole cohort before moving on.
                    for candidate in self.cohort["candidates"]:
                        if self._control():
                            return self._state()
                        identity = candidate["id"] + ":" + stage["key"]
                        if self.store.get("exact_result", identity) is not None:
                            continue
                        if (
                            max_stages is not None
                            and len(self.store.attempts()) - initial_attempts >= max_stages
                        ):
                            self._save_state(status="PAUSED")
                            self.checkpoint("bounded_stage_pause")
                            return self._state()
                        result = self._stage(candidate, stage)
                        if result is None:
                            self._finish("STOPPED", "exact_budget_exhausted_incomplete")
                            return self._state()
                        if self.store.get("exact_result", identity) is None:
                            # Preserve explicit retry-exhaustion failure, without claiming an exact call.
                            self.store.save("exact_result", result, identity)
                        self._hook(
                            "validation_stage_completed", candidate=candidate["id"], stage=stage
                        )
                    self._save_state(cycle=self.store.count("exact_result"))
                self._finish("COMPLETED", "all_planned_realizations_recorded")
                return self._state()
            except BaseException:
                if self.store.get("state")["status"] not in TERMINAL | {"PAUSED", "FINALIZING"}:
                    self._save_state(status="INTERRUPTED")
                raise
            finally:
                self.started = 0.0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--historical", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("create")
    p.add_argument("directory", type=Path)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--exact-budget", type=int)
    p.add_argument("--wall-seconds", type=float)
    p = sub.add_parser("run")
    p.add_argument("directory", type=Path)
    p.add_argument("--max-stages", type=int)
    p = sub.add_parser("report")
    p.add_argument("directory", type=Path)
    for action in ("pause", "stop", "checkpoint"):
        p = sub.add_parser(action)
        p.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        destination = args.output.resolve()
        if destination == args.historical.resolve() or destination.is_relative_to(
            args.historical.resolve()
        ):
            raise ValueError("Preparation must be outside the historical experiment")
        if destination.exists():
            raise FileExistsError("Preparation output must be new")
        cohort = freeze_cohort(args.historical)
        destination.mkdir(parents=True)
        atomic_text(destination / "cohort.json", json.dumps(cohort, indent=2))
        atomic_text(destination / "pilot.json", json.dumps(pilot_settings(cohort), indent=2))
        print(dumps({"cohort_sha256": cohort["sha256"], "members": len(cohort["candidates"])}))
    elif args.action == "create":
        settings = json.loads(args.config.read_text(encoding="utf-8"))
        cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
        if settings["mode"] == "confirmation" and (
            args.exact_budget is None or args.wall_seconds is None
        ):
            raise ValueError(
                "Confirmation is not authorized without both explicit budget arguments"
            )
        if args.exact_budget is not None:
            settings["exact_budget"] = args.exact_budget
        if args.wall_seconds is not None:
            settings["wall_seconds"] = args.wall_seconds
        print(ValidationStudy.create_study(args.directory, settings, cohort))
    elif args.action == "run":
        print(dumps(ValidationStudy(args.directory).run(max_stages=args.max_stages)))
    elif args.action == "report":
        from toposc_lab.research.validation_reporting import export_results

        # A read snapshot is taken for exports; this never invokes an evaluator.
        export_results(args.directory)
        print(args.directory / "final_report.md")
    else:
        ResearchStore(args.directory).request(args.action)
        print("Control queued; applied at next running stage boundary")


if __name__ == "__main__":
    main()
