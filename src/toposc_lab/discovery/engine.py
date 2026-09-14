"""Durable bounded discovery, with independently cached exact validation stages."""

import base64
import os
import platform
import time
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

from toposc_lab.active_learning.benchmark import source_provenance
from toposc_lab.active_learning.exact import Verification
from toposc_lab.active_learning.pool import Candidate
from toposc_lab.active_learning.storage import append_verified
from toposc_lab.active_learning.training import FittedSurrogate
from toposc_lab.data import (
    DatasetRecord,
    ExactPhysicsDataset,
    ReproducibilityMetadata,
    load_dataset,
    record_from_dict,
    record_to_dict,
    save_dataset,
    validate_dataset_record,
)
from toposc_lab.discovery.config import DiscoveryConfig, derived_seed
from toposc_lab.discovery.selection import plan_cycle
from toposc_lab.discovery.storage import (
    atomic_json,
    decode_run,
    encode_run,
    read_json,
    writer_lease,
)
from toposc_lab.discovery.validation import (
    disorder_evidence,
    integrate_validation,
    majorana_evidence,
)
from toposc_lab.generative.generators import Evidence
from toposc_lab.generative.physics import ExactGeometryEvaluator, candidate
from toposc_lab.generative.space import GeometrySearchSpace, structural_distance
from toposc_lab.geometry import Geometry, geometry_from_bytes, geometry_to_bytes

Hook = Callable[[str, dict[str, Any]], None]
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def scientific_record(record: DatasetRecord) -> dict[str, Any]:
    payload = record_to_dict(record)
    payload.pop("record_id")
    payload.pop("provenance")
    return payload


class DiscoveryEngine:
    """Single writer; iterations means additional cycles within the frozen total cap.

    ``seed_dataset`` is a frozen external exclusion archive, not automatically
    imported training evidence. New labels are stored in ``dataset.json``.
    Hooks expose durable fault-injection boundaries for recovery verification.
    """

    def __init__(
        self,
        directory: str | Path,
        config: DiscoveryConfig | None = None,
        *,
        seed_dataset: ExactPhysicsDataset | None = None,
        hook: Hook | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.config = config if config is not None else DiscoveryConfig()
        self.seed_dataset = seed_dataset if seed_dataset is not None else ExactPhysicsDataset(())
        self.hook = hook
        self._fitted_cache: dict[int, FittedSurrogate] = {}
        self.root = Path(__file__).resolve().parents[3]

    def _event(self, event: str, **payload: Any) -> None:
        if self.hook is not None:
            self.hook(event, payload)

    def _initialize(self) -> ReproducibilityMetadata:
        current = source_provenance(self.root)
        environment: dict[str, Any] = {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "platform": platform.platform(),
            "threads": {n: os.environ.get(n) for n in THREADS},
        }
        external_ids = [r.record_id for r in self.seed_dataset.records]
        manifest_path = self.directory / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if (
                manifest["config_sha256"] != self.config.fingerprint
                or manifest["source_sha256"] != current.runtime["source_sha256"]
                or manifest["environment"] != environment
                or manifest["external_record_ids"] != external_ids
            ):
                raise ValueError("resume config/source/runtime/exclusion archive mismatch")
            stored_archive = self.directory / "source.zip"
            if sha256(stored_archive.read_bytes()).hexdigest() != manifest["source_zip_sha256"]:
                raise ValueError("corrupt source snapshot")
            return ReproducibilityMetadata(**manifest["provenance"])
        if any(self.directory.glob("cycle-*")):
            raise ValueError("campaign has cycles but no manifest")
        archive_path = self.directory / "source.zip"
        temporary = self.directory / "source.zip.tmp"
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for folder, pattern in (
                ("src", "*.py"),
                ("tests", "test_discovery*.py"),
                ("scripts", "phase_15*.py"),
            ):
                for path in sorted((self.root / folder).rglob(pattern)):
                    archive.write(path, path.relative_to(self.root).as_posix())
            for name in ("pyproject.toml", "docs/decisions/phase_15_protocol.md"):
                path = self.root / name
                if path.exists():
                    archive.write(path, name)
        os.replace(temporary, archive_path)
        provenance = replace(
            current,
            solver_settings={
                **dict(current.solver_settings),
                "kappas": self.config.kappas,
                "probe": self.config.probe,
            },
            runtime={
                **dict(current.runtime),
                **environment,
                "benchmark": "phase15.autonomy.v1",
                "config_sha256": self.config.fingerprint,
            },
        )
        provenance_dict = {
            name: plain(getattr(provenance, name))
            for name in (
                "seed",
                "git_commit",
                "git_dirty",
                "package_version",
                "solver_name",
                "solver_version",
                "solver_settings",
                "tolerances",
                "timestamp_utc",
                "runtime",
            )
        }
        save_dataset(
            self.directory / "exclusions.json",
            self.seed_dataset,
            overwrite=(self.directory / "exclusions.json").exists(),
        )
        atomic_json(
            manifest_path,
            {
                "config": asdict(self.config),
                "config_sha256": self.config.fingerprint,
                "source_sha256": current.runtime["source_sha256"],
                "source_zip_sha256": sha256(archive_path.read_bytes()).hexdigest(),
                "external_record_ids": external_ids,
                "environment": environment,
                "provenance": provenance_dict,
            },
        )
        return provenance

    def _exact_stage(
        self, path: Path, seed: int, operation: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        if path.exists():
            return cast(dict[str, Any], read_json(path))
        ledger = self.directory / "attempts"
        ledger.mkdir(exist_ok=True)
        number = len(list(ledger.glob("*.json")))
        if number >= self.config.exact_attempt_cap:
            raise RuntimeError("exact attempt cap exhausted; interrupted attempts remain charged")
        entry = ledger / f"{number:06d}.json"
        identity = {
            "number": number,
            "seed": seed,
            "stage": path.relative_to(self.directory).as_posix(),
        }
        atomic_json(entry, {**identity, "status": "started"})
        self._event("attempt_started", **identity)
        started = time.perf_counter()
        result: dict[str, Any]
        try:
            payload = operation()
        except Exception as error:  # noqa: BLE001 -- isolate numerical stage failures
            result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        else:
            self._event("after_simulation", **identity)
            result = {"status": "complete", "payload": payload}
        result["wall_seconds"] = time.perf_counter() - started
        atomic_json(path, result)
        self._event("stage_saved", **identity)
        atomic_json(
            entry, {**identity, "status": result["status"], "wall_seconds": result["wall_seconds"]}
        )
        return result

    def _history(self, count: int) -> tuple[Evidence, ...]:
        history = []
        for cycle in range(count):
            directory = self.directory / f"cycle-{cycle:04d}"
            commit = read_json(directory / "commit.json")
            for index in commit["selected_indices"]:
                outcome = read_json(directory / f"candidate-{index:04d}.json")
                if outcome["status"] == "complete":
                    base = read_json(directory / f"base-{index:04d}.json")["payload"]
                    history.append(
                        Evidence(record_from_dict(outcome["record"]), decode_run(base["run"]))
                    )
        return tuple(history)

    def _archive(self, count: int) -> tuple[Geometry, ...]:
        archive = [r.geometry.to_geometry() for r in self.seed_dataset.records]
        for cycle in range(count):
            plan = read_json(self.directory / f"cycle-{cycle:04d}" / "plan.json")
            archive.extend(geometry_from_bytes(base64.b64decode(g)) for g in plan["geometries"])
        return tuple(archive)

    def _completed(self) -> int:
        count = 0
        while (self.directory / f"cycle-{count:04d}" / "commit.json").exists():
            directory = self.directory / f"cycle-{count:04d}"
            commit = read_json(directory / "commit.json")
            for name, digest in commit["inventory"].items():
                if sha256((directory / name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"committed cycle corruption: {directory / name}")
            count += 1
        checkpoint = self.directory / "checkpoint.json"
        if checkpoint.exists():
            saved = read_json(checkpoint)
            if (
                saved["completed_cycles"] > count
                or saved["config_sha256"] != self.config.fingerprint
            ):
                raise ValueError("checkpoint references missing/incompatible cycles")
        return count

    def _reconcile_dataset(self) -> None:
        expected = []
        for plan_path in sorted(self.directory.glob("cycle-*/plan.json")):
            plan = read_json(plan_path)
            for index in plan["selected_indices"]:
                path = plan_path.parent / f"candidate-{index:04d}.json"
                if path.exists():
                    item = read_json(path)
                    if item["status"] == "complete":
                        expected.append(record_from_dict(item["record"]))
        destination = self.directory / "dataset.json"
        if destination.exists():
            actual = load_dataset(destination)
            lookup = {r.record_id: record_to_dict(r) for r in expected}
            if any(lookup.get(r.record_id) != record_to_dict(r) for r in actual.records):
                raise ValueError("dataset has unjournaled or altered exact labels")
        if len({candidate(r.geometry.to_geometry()).candidate_id for r in expected}) != len(
            expected
        ):
            raise ValueError("duplicate candidate labels in journal")
        save_dataset(
            destination, ExactPhysicsDataset(tuple(expected)), overwrite=destination.exists()
        )

    def _cycle(self, cycle: int, provenance: ReproducibilityMetadata) -> None:
        started = time.perf_counter()
        directory = self.directory / f"cycle-{cycle:04d}"
        directory.mkdir(exist_ok=True)
        plan_path = directory / "plan.json"
        if not plan_path.exists():
            geometries, plan = plan_cycle(
                self.config, cycle, self._history(cycle), self._archive(cycle), self._fitted_cache
            )
            plan["geometries"] = [
                base64.b64encode(geometry_to_bytes(g)).decode() for g in geometries
            ]
            atomic_json(plan_path, plan)
        plan = read_json(plan_path)
        geometries = tuple(geometry_from_bytes(base64.b64decode(g)) for g in plan["geometries"])
        archive = self._archive(cycle)
        for i, geometry in enumerate(geometries):
            if GeometrySearchSpace().reasons(geometry):
                raise ValueError("journal contains incompatible geometry")
            if candidate(geometry).candidate_id != plan["candidate_ids"][i]:
                raise ValueError("journal candidate identity mismatch")
            if any(
                structural_distance(geometry, old) <= self.config.minimum_distance
                for old in archive + geometries[:i]
            ):
                raise ValueError("journal contains duplicate/near-duplicate geometry")
        self._event("plan_saved", cycle=cycle)
        local = replace(provenance, runtime={**dict(provenance.runtime), "discovery_cycle": cycle})
        evaluator = ExactGeometryEvaluator(local)
        for index in plan["selected_indices"]:
            path = directory / f"candidate-{index:04d}.json"
            if path.exists():
                continue
            proposal = candidate(geometries[index])
            seed = derived_seed(self.config.seed, cycle, index, 3)

            def simulate(p: Candidate, s: int) -> dict[str, Any]:
                record, run = evaluator.evaluate(p, s)
                validate_dataset_record(record).raise_for_errors()
                return {"record": record_to_dict(record), "run": encode_run(run)}

            base = self._exact_stage(
                directory / f"base-{index:04d}.json", seed, partial(simulate, proposal, seed)
            )
            if base["status"] != "complete":
                atomic_json(
                    path,
                    {
                        "status": "failed",
                        "stage": "base",
                        "details": base,
                        "candidate_id": proposal.candidate_id,
                    },
                )
                continue
            record = record_from_dict(base["payload"]["record"])
            run = decode_run(base["payload"]["run"])
            confirmation_seed = derived_seed(self.config.seed, cycle, index, 4)

            def confirm(p: Candidate, s: int) -> dict[str, Any]:
                return {"record": record_to_dict(evaluator.evaluate(p, s)[0])}

            confirmation = self._exact_stage(
                directory / f"confirmation-{index:04d}.json",
                confirmation_seed,
                partial(confirm, proposal, confirmation_seed),
            )
            if confirmation["status"] != "complete":
                atomic_json(
                    path,
                    {
                        "status": "failed",
                        "stage": "confirmation",
                        "details": confirmation,
                        "candidate_id": proposal.candidate_id,
                    },
                )
                continue
            repeated = record_from_dict(confirmation["payload"]["record"])
            if scientific_record(record) != scientific_record(repeated):
                raise ValueError("independent exact confirmation mismatch")
            majorana = majorana_evidence(record, run)
            if majorana["operator_phs_residual"] > self.config.tolerance:
                raise ValueError("exact BdG operator violates particle-hole contract")
            members = []
            failure = None
            for member in range(self.config.disorder_samples):
                disorder_seed = derived_seed(self.config.seed, cycle, index, 5, member)
                result = self._exact_stage(
                    directory / f"disorder-{index:04d}-{member:04d}.json",
                    disorder_seed,
                    partial(disorder_evidence, record, disorder_seed, self.config),
                )
                if result["status"] == "complete":
                    members.append(result["payload"])
                else:
                    failure = result
            if failure is not None:
                atomic_json(
                    path,
                    {
                        "status": "failed",
                        "stage": "robustness",
                        "details": failure,
                        "candidate_id": proposal.candidate_id,
                    },
                )
                continue
            integrated = integrate_validation(record, majorana, members, self.config)
            validate_dataset_record(integrated).raise_for_errors()
            atomic_json(
                path,
                {
                    "status": "complete",
                    "candidate_id": proposal.candidate_id,
                    "record": record_to_dict(integrated),
                    "exact_confirmation": True,
                    "common_ood": plan["common_ood"][proposal.candidate_id],
                    "selection_strategy": plan["strategies"][proposal.candidate_id],
                },
            )
            self._event("candidate_saved", cycle=cycle, index=index)
            append_verified(
                self.directory / "dataset.json",
                Verification(proposal.candidate_id, seed, integrated, None),
            )
            self._event("dataset_saved", cycle=cycle, index=index)
        self._reconcile_dataset()
        inventory = {
            p.name: sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.glob("*.json"))
            if p.name != "commit.json"
        }
        atomic_json(
            directory / "commit.json",
            {
                "cycle": cycle,
                "selected_indices": plan["selected_indices"],
                "inventory": inventory,
                "wall_seconds_this_invocation": time.perf_counter() - started,
            },
        )
        self._event("cycle_committed", cycle=cycle)

    def run(self, iterations: int | None = None) -> dict[str, Any]:
        from toposc_lab.discovery.reporting import write_reports

        if iterations is not None and (type(iterations) is not int or iterations < 0):
            raise ValueError("iterations must be nonnegative")
        with writer_lease(self.directory):
            provenance = self._initialize()
            completed = self._completed()
            self._reconcile_dataset()
            target = self.config.cycles if iterations is None else completed + iterations
            if target > self.config.cycles:
                raise ValueError("requested iterations exceed frozen campaign cap")
            for cycle in range(completed, target):
                self._cycle(cycle, provenance)
                atomic_json(
                    self.directory / "checkpoint.json",
                    {
                        "completed_cycles": cycle + 1,
                        "config_sha256": self.config.fingerprint,
                        "record_ids": [
                            r.record_id
                            for r in load_dataset(self.directory / "dataset.json").records
                        ],
                    },
                )
                self._event("checkpoint_saved", cycle=cycle)
            # Also repair the checkpoint when a crash occurred after cycle commit.
            completed = self._completed()
            atomic_json(
                self.directory / "checkpoint.json",
                {
                    "completed_cycles": completed,
                    "config_sha256": self.config.fingerprint,
                    "record_ids": [
                        r.record_id for r in load_dataset(self.directory / "dataset.json").records
                    ],
                },
            )
            return write_reports(self.directory, self.config, completed)
