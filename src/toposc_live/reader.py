"""Version-aware, read-only campaign snapshots from persisted engine evidence."""

import base64
import json
import time
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from toposc_live.models import (
    CampaignSnapshot,
    CandidateSnapshot,
    GeometrySnapshot,
    display,
    majorana_label,
    mapping,
    number,
    sequence,
)
from toposc_live.processes import ProcessService


class ArtifactCache:
    """Stat-based bounded cache; a changed/torn file is retried on the next refresh."""

    def __init__(self) -> None:
        self.entries: OrderedDict[Path, tuple[tuple[int, int], Any, str]] = OrderedDict()
        self.reads = 0

    def read(self, path: Path, warnings: list[str], *, envelope: bool = True) -> Any:
        try:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            cached = self.entries.get(path)
            if cached and cached[0] == signature:
                self.entries.move_to_end(path)
                return cached[1]
            if stat.st_size > 32 * 1024 * 1024:
                raise ValueError("artifact exceeds 32 MiB display limit")
            raw = path.read_bytes()
            self.reads += 1
            after = path.stat()
            if (after.st_mtime_ns, after.st_size) != signature:
                raise ValueError("artifact changed while being read; retry next refresh")
            value = json.loads(raw)
            if envelope:
                if not isinstance(value, dict) or "payload" not in value or "sha256" not in value:
                    raise ValueError("missing checksum envelope; unsupported artifact format")
                serialized = json.dumps(
                    value["payload"], sort_keys=True, allow_nan=False, indent=2
                ).encode()
                if sha256(serialized).hexdigest() != value["sha256"]:
                    raise ValueError("checksum mismatch")
                value = value["payload"]
            self.entries[path] = (signature, value, sha256(raw).hexdigest())
            while len(self.entries) > 1024:
                self.entries.popitem(last=False)
            return value
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError, RecursionError) as error:
            self.entries.pop(path, None)
            warnings.append(f"{path.name}: {error}")
            return None


def discover_campaigns(
    roots: tuple[Path, ...], recent: tuple[Path, ...] = (), depth: int = 3
) -> tuple[Path, ...]:
    """Bounded scan, no symlink traversal, no results writes or fixed campaign path."""
    found = {p.resolve() for p in recent if p.is_dir()}

    def visit(path: Path, remaining: int) -> None:
        if path.is_symlink() or not path.is_dir():
            return
        if (path / "manifest.json").is_file() or (path / "leaderboard.json").is_file():
            found.add(path.resolve())
            return
        if remaining <= 0:
            return
        try:
            for child in path.iterdir():
                if not child.name.startswith("."):
                    visit(child, remaining - 1)
        except OSError:
            return

    for root in roots:
        visit(root.expanduser(), depth)
    return tuple(sorted(found, key=lambda p: str(p).casefold()))


def geometry_snapshot(
    record: dict[str, Any], report: dict[str, Any] | None = None
) -> GeometrySnapshot:
    geometry = mapping(record.get("geometry"))
    report = report or {}
    try:
        if geometry.get("archive"):
            if geometry.get("archive_schema_version", 1) != 1:
                raise ValueError("unsupported geometry archive version")
            from toposc_lab.geometry import geometry_from_bytes

            archive = base64.b64decode(geometry["archive"], validate=True)
            identifier = geometry.get("exact_id")
            if (
                identifier
                and identifier != "geometry-archive-v1-sha256:" + sha256(archive).hexdigest()
            ):
                raise ValueError("geometry identity checksum mismatch")
            decoded = geometry_from_bytes(archive)
            coordinates = decoded.coordinates.tolist() if decoded.coordinates is not None else []
            edges = [(e.source, e.target) for e in decoded.edges]
            boundary = tuple(decoded.boundary_sites)
        else:
            coordinates, edges = report.get("coordinates", []), report.get("edges", [])
            boundary = ()
        points = tuple(tuple(float(v) for v in point) for point in coordinates)
        links = tuple(tuple(edge) for edge in edges)
        if not points:
            return GeometrySnapshot()
        dimension = len(points[0])
        if dimension < 1 or any(
            len(p) != dimension or any(number(v) is None for v in p) for p in points
        ):
            raise ValueError("invalid coordinates")
        if any(
            len(e) != 2 or any(type(i) is not int or not 0 <= i < len(points) for i in e)
            for e in links
        ):
            raise ValueError("invalid edge endpoints")
        return GeometrySnapshot(
            points,
            tuple((edge[0], edge[1]) for edge in links),
            boundary,
            f"Stored {dimension}D embedding"
            + ("; first two coordinates projected" if dimension > 2 else ""),
        )
    except (ValueError, TypeError, KeyError, OSError, IndexError) as error:
        return GeometrySnapshot(message=f"unavailable: {error}")


def parse_candidate(
    record: dict[str, Any],
    outcome: dict[str, Any],
    *,
    generator: str,
    cycle: Any,
    source: str,
    report: dict[str, Any] | None = None,
) -> CandidateSnapshot | None:
    if record.get("schema_version", 1) != 1 or record.get("result_kind") != "exact":
        return None
    observables = {
        o.get("kind"): mapping(o.get("values"))
        for o in sequence(record.get("observables"))
        if isinstance(o, dict) and isinstance(o.get("kind"), str)
    }
    metrics = observables.get("finite_geometry_quality", {})
    topology: Any = next(iter(sequence(record.get("topology"))), {})
    topology = mapping(topology)
    robustness = [mapping(r) for r in sequence(record.get("robustness"))]
    robust_display = [
        {
            "protocol": r.get("protocol"),
            "statistics": r.get("statistics"),
            "uncertainty": r.get("uncertainty"),
            "scope": mapping(r.get("parameters")).get("ensemble_scope"),
        }
        for r in robustness
    ]
    return CandidateSnapshot(
        record_id=display(record.get("record_id")),
        geometry_id=display(mapping(record.get("geometry")).get("exact_id")),
        generator=generator,
        seed=mapping(record.get("provenance")).get("seed"),
        cycle=cycle,
        quality=number(metrics.get("quality")),
        topology=display(topology.get("validity")),
        indices=mapping(topology.get("parameters")).get("indices"),
        minimum_abs_energy=number(metrics.get("minimum_abs_energy")),
        boundary_weight=metrics.get("boundary_weights", metrics.get("boundary_weight")),
        ood=outcome.get("common_ood"),
        confirmation=outcome.get("exact_confirmation"),
        robustness=robust_display or None,
        majorana=majorana_label(record),
        geometry=geometry_snapshot(record, report),
        source=source,
    )


class CampaignReader:
    """One instance per refresh worker. Widgets only receive domain snapshots."""

    def group_member(self, directory: Path) -> tuple[CampaignSnapshot, dict[str, Any]]:
        """Read-only group preflight; reject incomplete identity or corrupt evidence.

        Unlike ordinary inspection, aggregation requires a full current config,
        validated exact records and committed-cycle inventories. No recovery,
        numerical evaluation or source/runtime resume check is performed.
        """
        from dataclasses import fields

        from toposc_lab.data import record_from_dict, validate_dataset_record
        from toposc_lab.discovery.config import DiscoveryConfig

        directory = directory.resolve()
        warnings: list[str] = []

        def required(path: Path) -> dict[str, Any]:
            value = self.cache.read(path, warnings)
            if not isinstance(value, dict) or warnings:
                raise ValueError(f"Group evidence unavailable: {path}: {warnings}")
            return value

        manifest = required(directory / "manifest.json")
        config = mapping(manifest.get("config"))
        if set(config) != {f.name for f in fields(DiscoveryConfig)}:
            raise ValueError(f"Group requires a complete supported configuration: {directory}")
        official = DiscoveryConfig(**config)
        if official.fingerprint != manifest.get("config_sha256"):
            raise ValueError(f"Group configuration fingerprint mismatch: {directory}")
        checkpoint = required(directory / "checkpoint.json")
        if (directory / "summary.json").exists():
            summary = required(directory / "summary.json")
            if summary.get("schema_version", 1) != 1:
                raise ValueError("Unsupported group summary schema")
        if checkpoint.get("config_sha256") != official.fingerprint:
            raise ValueError(f"Group checkpoint fingerprint mismatch: {directory}")
        completed = checkpoint.get("completed_cycles")
        if type(completed) is not int or not 0 <= completed <= official.cycles:
            raise ValueError(f"Invalid completed-cycle count: {directory}")
        # Check every artifact the aggregate consumes, including optional journals
        # if present. A torn refresh fails visibly instead of becoming a zero.
        for path in sorted((directory / "attempts").glob("*.json")):
            required(path)
        for cycle_dir in sorted(directory.glob("cycle-*")):
            if not cycle_dir.is_dir():
                continue
            plan = required(cycle_dir / "plan.json")
            audit = mapping(plan.get("audit"))
            if any(
                type(audit.get(k)) is not int or audit[k] < 0
                for k in ("attempted", "invalid", "duplicate")
            ) or not isinstance(plan.get("geometries"), list):
                raise ValueError(f"Proposal counts unavailable: {cycle_dir}")
            expected_method = (
                "random" if cycle_dir.name == "cycle-0000" else official.resolved_generator
            )
            if plan.get("method") != expected_method:
                raise ValueError(f"Stored generator/configuration mismatch: {cycle_dir}")
            for pattern in ("plan.json", "candidate-*.json", "base-*.json"):
                for path in cycle_dir.glob(pattern):
                    required(path)
        for cycle in range(completed):
            cycle_dir = directory / f"cycle-{cycle:04d}"
            commit = required(cycle_dir / "commit.json")
            inventory = commit.get("inventory")
            if commit.get("cycle") != cycle or not isinstance(inventory, dict) or not inventory:
                raise ValueError(f"Invalid committed inventory: {cycle_dir}")
            for name, digest in inventory.items():
                if not isinstance(name, str) or Path(name).name != name:
                    raise ValueError(f"Unsafe inventory path: {cycle_dir}")
                path = cycle_dir / name
                required(path)
                if self.cache.entries[path][2] != digest:
                    raise ValueError(f"Committed inventory mismatch: {path}")
            if "plan.json" not in inventory:
                raise ValueError(f"Committed plan missing: {cycle_dir}")
            actual = {p.name for p in cycle_dir.glob("*.json") if p.name != "commit.json"}
            if set(inventory) != actual:
                raise ValueError(f"Committed inventory file set mismatch: {cycle_dir}")
            if sum(name.startswith("candidate-") for name in inventory) != official.batch_size:
                raise ValueError(f"Committed candidate count mismatch: {cycle_dir}")
        snapshot = self.load(directory)
        for candidate in snapshot.candidates:
            path = directory / candidate.source
            outcome = required(path)
            record = mapping(outcome.get("record", mapping(outcome.get("payload")).get("record")))
            validate_dataset_record(record_from_dict(record)).raise_for_errors()
            model = mapping(record.get("model"))
            parameters = mapping(model.get("parameters"))
            if model.get("model_name") != official.model or any(
                parameters.get(key) != config[key]
                for key in ("hopping", "pairing", "chemical_potential", "chirality")
            ):
                raise ValueError(f"Stored record/model configuration mismatch: {path}")
            if candidate.quality is None:
                raise ValueError(f"Exact quality unavailable: {path}")
        if (
            required(directory / "checkpoint.json") != checkpoint
            or required(directory / "manifest.json") != manifest
        ):
            raise ValueError("Campaign changed during group preflight; retry next refresh")
        if (
            snapshot.status == "Completed"
            and len(snapshot.candidates) != official.cycles * official.batch_size
        ):
            raise ValueError(f"Completed campaign candidate inventory is incomplete: {directory}")
        return snapshot, manifest

    def __init__(self, processes: ProcessService | None = None) -> None:
        self.cache = ArtifactCache()
        self.processes = processes or ProcessService()
        self._candidate_cache: dict[Path, tuple[Any, CandidateSnapshot | None]] = {}

    def load(self, directory: Path) -> CampaignSnapshot:
        directory = directory.resolve()
        warnings: list[str] = []

        def read(relative: str | Path) -> Any:
            return self.cache.read(directory / relative, warnings)

        manifest = mapping(read("manifest.json"))
        config = mapping(manifest.get("config"))
        summary = mapping(read("summary.json"))
        checkpoint = mapping(read("checkpoint.json"))
        job = self.processes.inspect(directory)
        state = mapping(job.get("state"))
        name = str(job.get("name", directory.name))
        if config.get("schema_version", 1) != 1 or summary.get("schema_version", 1) != 1:
            return CampaignSnapshot(
                directory,
                name,
                config=config,
                warnings=("Unsupported campaign schema; scientific fields unavailable.",),
            )
        if not manifest:
            warnings.append("Manifest unavailable; compatibility and scientific scope unavailable.")
        if manifest.get("config_sha256") and checkpoint.get("config_sha256") not in (
            None,
            manifest["config_sha256"],
        ):
            warnings.append("Checkpoint/configuration fingerprint mismatch.")
        # Inspect stored fingerprints without requiring historical configs to validate today.
        if manifest.get("config_sha256"):
            fingerprint = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            if fingerprint != manifest["config_sha256"]:
                warnings.append("Manifest configuration fingerprint mismatch.")
        generator = display(summary.get("generator", config.get("generator")))
        if generator == "auto":
            generator = "auto (resolved generator unavailable until plan/report)"
        attempts = []
        for path in sorted((directory / "attempts").glob("*.json")):
            item = read(path)
            if isinstance(item, dict):
                attempts.append(item)
        completed_attempts = sum(a.get("status") == "complete" for a in attempts)
        candidates: list[CandidateSnapshot] = []
        events: list[str] = []
        totals = {
            "Generated": 0,
            "Valid pool": 0,
            "Invalid": 0,
            "Duplicates / near-duplicates": 0,
            "Selected OOD": 0,
        }
        plans_seen = 0
        current_cycle: Any = None
        stage_numbers: dict[str, int] = {}
        successful_count = 0
        for attempt in attempts:
            if attempt.get("status") == "complete":
                successful_count += 1
                stage_numbers[str(attempt.get("stage"))] = successful_count
            events.append(
                f"Exact attempt {display(attempt.get('number'))}: {display(attempt.get('status'))} — {display(attempt.get('stage'))}"
            )
        for cycle_dir in sorted(directory.glob("cycle-*")):
            if not cycle_dir.is_dir():
                continue
            try:
                cycle = int(cycle_dir.name.removeprefix("cycle-"))
            except ValueError:
                continue
            current_cycle = cycle
            plan = mapping(read(cycle_dir / "plan.json"))
            cycle_generator = display(plan.get("method", generator))
            if plan:
                plans_seen += 1
                audit = mapping(plan.get("audit"))
                for label, key in (
                    ("Generated", "attempted"),
                    ("Invalid", "invalid"),
                    ("Duplicates / near-duplicates", "duplicate"),
                ):
                    value = number(audit.get(key))
                    if value is not None:
                        totals[label] += int(value)
                totals["Valid pool"] += len(sequence(plan.get("geometries")))
                ids, ood = sequence(plan.get("candidate_ids")), mapping(plan.get("common_ood"))
                for index in sequence(plan.get("selected_indices")):
                    if type(index) is int and 0 <= index < len(ids):
                        totals["Selected OOD"] += mapping(ood.get(ids[index])).get("is_ood") is True
                for event in sequence(audit.get("events"))[-30:]:
                    item = mapping(event)
                    events.append(
                        f"Cycle {cycle}, proposal {display(item.get('attempt'))}: "
                        f"{display(item.get('status'))}; {display(item.get('reasons'))}"
                    )
            commit = mapping(read(cycle_dir / "commit.json"))
            if commit:
                events.append(f"Cycle {cycle}: checkpoint commit stored")
            # Prefer integrated journals, but expose exact base results during validation.
            paths = {
                p.name.replace("base-", "candidate-"): p for p in cycle_dir.glob("base-*.json")
            }
            paths.update({p.name: p for p in cycle_dir.glob("candidate-*.json")})
            for path in paths.values():
                outcome = mapping(read(path))
                if outcome.get("status") != "complete":
                    continue
                record = mapping(
                    outcome.get("record", mapping(outcome.get("payload")).get("record"))
                )
                signature = self.cache.entries.get(path, (None,))[0]
                previous = self._candidate_cache.get(path)
                if previous and previous[0] == signature:
                    candidate = previous[1]
                else:
                    candidate = parse_candidate(
                        record,
                        outcome,
                        generator=cycle_generator,
                        cycle=cycle,
                        source=str(path.relative_to(directory)),
                    )
                    self._candidate_cache[path] = (signature, candidate)
                if candidate is None:
                    warnings.append(f"{path.name}: unsupported or non-exact record; omitted.")
                    continue
                if candidate.geometry.message.startswith("unavailable:"):
                    warnings.append(f"{path.name}: {candidate.geometry.message}")
                base_stage = f"{cycle_dir.name}/{path.name.replace('candidate-', 'base-')}"
                candidate = replace(candidate, exact_evaluation=stage_numbers.get(base_stage))
                candidates.append(candidate)
                events.append(
                    f"Cycle {cycle}: exact candidate {candidate.geometry_id}; quality {display(candidate.quality)}"
                )
        if not list(directory.glob("cycle-*/candidate-*.json")) and not list(
            directory.glob("cycle-*/base-*.json")
        ):
            # Older report-only campaigns are inspectable without synthesizing missing fields.
            for row in sequence(read("leaderboard.json")):
                row = mapping(row)
                if row.get("kind") != "exact":
                    continue
                identifier = display(row.get("geometry_id"))
                name_part = identifier.rsplit(":", 1)[-1]
                report = (
                    mapping(read(Path("reports") / f"{name_part}.json"))
                    if name_part.isalnum()
                    else {}
                )
                candidate = parse_candidate(
                    mapping(report.get("record")),
                    row,
                    generator=display(row.get("generator", generator)),
                    cycle=row.get("cycle"),
                    source="leaderboard.json",
                    report=report,
                )
                candidates.append(
                    candidate
                    or CandidateSnapshot(
                        display(row.get("record_id")),
                        identifier,
                        display(row.get("generator", generator)),
                        row.get("seed"),
                        row.get("cycle"),
                        number(row.get("quality")),
                        topology=display(row.get("topology")),
                        majorana=majorana_label(row),
                        robustness=row.get("robustness_mean"),
                        source="leaderboard.json",
                    )
                )
        candidates = [replace(c, campaign_seed=config.get("seed")) for c in candidates]
        curves: dict[str, list[tuple[int, float]]] = {}
        bests: dict[str, float] = {}
        for candidate in sorted(candidates, key=lambda c: c.exact_evaluation or 0):
            if candidate.exact_evaluation is None or candidate.quality is None:
                continue
            bests[candidate.generator] = max(
                bests.get(candidate.generator, candidate.quality), candidate.quality
            )
            curves.setdefault(candidate.generator, []).append(
                (candidate.exact_evaluation, bests[candidate.generator])
            )
        candidates.sort(key=lambda c: (c.quality is None, -(c.quality or 0), c.geometry_id))
        completed = number(checkpoint.get("completed_cycles", summary.get("completed_cycles")))
        planned = number(config.get("cycles"))
        status = str(job.get("status", "Unknown"))
        if status not in ("Running", "Starting", "Failed"):
            if completed is not None and planned is not None and completed >= planned:
                status = "Completed" if not any("mismatch" in w for w in warnings) else "Unknown"
            elif status == "Exited":
                status = "Interrupted"  # Deliberate --iterations boundary; resumable.
        cap = summary.get("exact_attempt_cap")
        if cap is None and config:
            try:
                from toposc_lab.discovery.config import DiscoveryConfig

                cap = DiscoveryConfig(**config).exact_attempt_cap
            except (ValueError, TypeError):
                pass  # Historical config is displayable even when not launchable today.
        has_ledger = (directory / "attempts").is_dir()
        charged = len(attempts) if has_ledger else summary.get("exact_attempts")
        cap = number(cap)
        charged = number(charged)
        progress = {
            "Campaign seed": config.get("seed"),
            "Latest exact attempt seed": attempts[-1].get("seed") if attempts else None,
            "Current cycle (zero-based)": current_cycle,
            "Completed cycles": completed,
            "Planned cycles": planned,
            "Exact completed": completed_attempts if has_ledger else None,
            "Exact attempts charged": charged,
            "Exact attempt cap": cap,
            "Remaining attempt budget": max(0, cap - charged)
            if cap is not None and charged is not None
            else None,
            **(totals if plans_seen else {k: None for k in totals}),
        }
        stage_counts: dict[str, int] = {}
        for a in attempts:
            stage = str(a.get("stage"))
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        health = {
            "Checkpoint": "checkpoint.json" if checkpoint else None,
            "Latest checkpoint cycles": completed,
            "Process": job.get("status", "unavailable — external process not tracked"),
            "Process identity verified": job.get("alive"),
            "Retries (repeated stage attempts)": sum(n - 1 for n in stage_counts.values())
            if has_ledger
            else None,
            "Failed calculations": sum(a.get("status") == "failed" for a in attempts)
            if has_ledger
            else summary.get("failed_attempts"),
            "Started / unfinalized attempts": sum(a.get("status") == "started" for a in attempts)
            if has_ledger
            else None,
            "Latest successful exact": next(
                (a.get("stage") for a in reversed(attempts) if a.get("status") == "complete"), None
            ),
            "Recovery": "resume invocation" if job.get("resume") else None,
            "Exit code": state.get("exit_code"),
            "Process error": state.get("error"),
            "Integrity scope": "Read artifact envelopes verified; engine performs full committed inventory audit on resume.",
        }
        started = state.get("started_at", mapping(manifest.get("provenance")).get("timestamp_utc"))
        elapsed = None
        if isinstance(started, (float, int)):
            elapsed = max(0, state.get("ended_at", time.time()) - started)
            started = datetime.fromtimestamp(started).astimezone().isoformat(timespec="seconds")
        if not job:
            warnings.append(
                "External process liveness and elapsed wall time unavailable; no running/interrupted state inferred from file age."
            )
        if status == "Completed":
            events.append("Campaign completed: stored cycle count reached configured total")
        if not curves:
            warnings.append(
                "Exact-evaluation quality curve unavailable without a successful attempt ledger."
            )
        return CampaignSnapshot(
            directory,
            str(job.get("name", directory.name)),
            status,
            config,
            progress,
            health,
            tuple(candidates),
            {key: tuple(value) for key, value in curves.items()},
            tuple(events[-100:]),
            tuple(dict.fromkeys(warnings)),
            self.processes.log_tail(job),
            started,
            elapsed,
        )
