"""Descriptive multi-seed analysis of validated snapshots; no scientific execution."""

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from toposc_live.models import CampaignSnapshot, CandidateSnapshot, number
from toposc_live.reader import CampaignReader


@dataclass(frozen=True)
class GroupMember:
    directory: Path
    role: str = "included"
    reason: str = ""
    expected_seed: int | None = None


@dataclass(frozen=True)
class GroupDefinition:
    name: str
    members: tuple[GroupMember, ...]

    @classmethod
    def read(cls, path: Path) -> "GroupDefinition":
        data = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or data.get("schema_version") != 1
            or not isinstance(data.get("name"), str)
            or not data["name"].strip()
            or not isinstance(data.get("members"), list)
        ):
            raise ValueError("Unsupported campaign group definition")
        for item in data["members"]:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("directory"), str)
                or not item["directory"].strip()
                or item.get("role", "included") not in ("included", "excluded", "technical_dropout")
                or not isinstance(item.get("reason", ""), str)
                or (
                    item.get("expected_seed") is not None and type(item["expected_seed"]) is not int
                )
            ):
                raise ValueError("Malformed campaign group member")
        return cls(
            data["name"],
            tuple(
                GroupMember(
                    (path.parent / item["directory"]).resolve(),
                    item.get("role", "included"),
                    item.get("reason", ""),
                    item.get("expected_seed"),
                )
                for item in data["members"]
            ),
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "members": [
                {**asdict(m), "directory": str(m.directory.resolve())} for m in self.members
            ],
        }


@dataclass(frozen=True)
class GroupSnapshot:
    definition: GroupDefinition
    members: tuple[dict[str, Any], ...]
    config: dict[str, Any]
    statistics: dict[str, Any]
    candidates: tuple[CandidateSnapshot, ...]
    curves: dict[str, tuple[tuple[int, float], ...]]
    warnings: tuple[str, ...]

    def payload(self, top_n: int = 20) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.definition.name,
            "definition": self.definition.payload(),
            "compatible_configuration": self.config,
            "statistics": self.statistics,
            "members": self.members,
            "top_n": top_n,
            "top_candidates": [candidate_row(c) for c in self.candidates[:top_n]],
            "curves": self.curves,
            "progress_axis": "completed cycles",
            "aggregate_curve": "Equal-seed mean at the common completed-cycle horizon; no extrapolation",
            "warnings": self.warnings,
            "scope": "Stored finite-model exact quality; descriptive only. No Majorana, thermodynamic or saturation claim.",
        }


def candidate_row(candidate: CandidateSnapshot) -> dict[str, Any]:
    return {
        "Campaign Seed": candidate.campaign_seed,
        "Exact Eval Seed": candidate.seed,
        "record_id": candidate.record_id,
        "geometry_id": candidate.geometry_id,
        "quality": candidate.quality,
        "generator": candidate.generator,
        "cycle_zero_based": candidate.cycle,
        "topology": candidate.topology,
        "localizer_indices": candidate.indices,
        "minimum_abs_energy": candidate.minimum_abs_energy,
        "robustness": candidate.robustness,
        "OOD": candidate.ood,
        "confirmation": candidate.confirmation,
        "Majorana": candidate.majorana,
        "source": candidate.source,
    }


def compatibility(config: dict[str, Any]) -> dict[str, Any]:
    # Every official field participates, including budget/selection/disorder controls.
    # Only replicate identity and the non-scientific figure-output option differ.
    return {k: v for k, v in config.items() if k not in ("seed", "plots")}


def summarize(
    definition: GroupDefinition,
    loaded: tuple[tuple[CampaignSnapshot, dict[str, Any]], ...],
) -> GroupSnapshot:
    """Pure descriptive aggregation; loaded evidence has passed reader preflight."""
    if not definition.members or len(definition.members) != len(loaded):
        raise ValueError("A group requires explicitly selected campaign directories")
    seeds: set[int] = set()
    paths: set[Path] = set()
    included: list[CampaignSnapshot] = []
    members: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None
    environments: set[str] = set()
    sources: set[str] = set()
    exclusions: set[str] = set()
    protocols: set[str] = set()
    warnings: list[str] = []
    curves: dict[str, tuple[tuple[int, float], ...]] = {}
    for member, (snapshot, manifest) in zip(definition.members, loaded, strict=True):
        seed = snapshot.config.get("seed")
        if type(seed) is not int or seed in seeds or member.directory.resolve() in paths:
            raise ValueError("Campaign seed IDs and directories must be unique and available")
        if member.expected_seed is not None and seed != member.expected_seed:
            raise ValueError(f"Expected campaign seed {member.expected_seed}, found {seed}")
        seeds.add(seed)
        paths.add(member.directory.resolve())
        if member.role not in ("included", "technical_dropout", "excluded"):
            raise ValueError(f"Unknown group member role: {member.role}")
        if member.role != "included" and not member.reason.strip():
            raise ValueError("Exclusions require an explicit reason")
        row: dict[str, Any] = {
            "Campaign Seed": seed,
            "directory": str(member.directory.resolve()),
            "role": member.role,
            "reason": member.reason,
            "status": snapshot.status,
            "best_exact_quality": snapshot.best.quality if snapshot.best else None,
            "source_sha256": manifest.get("source_sha256"),
            **snapshot.progress,
        }
        if member.role == "included":
            signature = compatibility(snapshot.config)
            if reference is not None and signature != reference:
                changed = sorted(
                    k
                    for k in set(signature) | set(reference)
                    if signature.get(k) != reference.get(k)
                )
                raise ValueError(f"Incompatible campaign seed {seed}: {', '.join(changed)}")
            reference = signature
            environments.add(json.dumps(manifest.get("environment"), sort_keys=True))
            exclusions.add(json.dumps(manifest.get("external_record_ids"), sort_keys=True))
            sources.add(str(manifest.get("source_sha256", "unavailable")))
            protocol = manifest.get("provenance", {}).get("runtime", {}).get("benchmark")
            protocols.add(str(protocol))
            included.append(snapshot)
            completed = int(snapshot.progress.get("Completed cycles") or 0)
            points: list[tuple[int, float]] = []
            best: float | None = None
            last_improvement = None
            for cycle in range(completed):
                qualities = [
                    c.quality
                    for c in snapshot.candidates
                    if c.cycle == cycle and c.quality is not None
                ]
                if not qualities:
                    break  # Never fabricate missing-cycle evidence or extrapolate.
                value = max(qualities)
                if best is None or value > best:
                    best, last_improvement = value, cycle + 1
                points.append((cycle + 1, best))
            curves[str(seed)] = tuple(points)
            row["last_improvement_completed_cycle"] = last_improvement
            row["last_cycle_gain"] = points[-1][1] - points[-2][1] if len(points) >= 2 else None
        members.append(row)
        warnings.extend(f"Seed {seed}: {w}" for w in snapshot.warnings)
    if not included or reference is None:
        raise ValueError("A group needs at least one included campaign")
    if len(environments) != 1 or len(exclusions) != 1 or len(protocols) != 1:
        raise ValueError(
            "Incompatible stored protocol, runtime environment or external exclusion cohort"
        )
    if len(sources) > 1 or "unavailable" in sources:
        warnings.append(
            "Source fingerprints differ or are unavailable; this is a descriptive configuration-matched group, not identical-source or causal evidence."
        )
    candidates = tuple(
        sorted(
            (c for s in included for c in s.candidates),
            key=lambda c: (c.quality is None, -(c.quality or 0), c.campaign_seed, c.record_id),
        )
    )
    bests = [s.best.quality for s in included if s.best is not None and s.best.quality is not None]
    threshold = number(reference.get("success_threshold"))
    if threshold is None:
        raise ValueError("Frozen threshold unavailable")
    observed = [c for c in candidates if c.quality is not None]
    candidate_successes = sum(c.quality is not None and c.quality >= threshold for c in candidates)
    seed_successes = sum(q >= threshold for q in bests)
    totals: dict[str, Any] = {}
    for key in (
        "Generated",
        "Valid pool",
        "Invalid",
        "Duplicates / near-duplicates",
        "Exact completed",
        "Exact attempts charged",
    ):
        values = [number(s.progress.get(key)) for s in included]
        totals[key] = (
            sum(v for v in values if v is not None) if all(v is not None for v in values) else None
        )
    stats = {
        "included_seeds": len(included),
        "complete_seeds": sum(s.status == "Completed" for s in included),
        "failed_seeds": sum(s.status == "Failed" for s in included),
        "incomplete_seeds": sum(s.status not in ("Completed", "Failed") for s in included),
        "technical_dropouts": sum(m.role == "technical_dropout" for m in definition.members),
        "excluded_seeds": sum(m.role != "included" for m in definition.members),
        "all_member_failed_seeds": sum(s.status == "Failed" for s, _ in loaded),
        "per_seed_best_observed_count": len(bests),
        "global_best_exact_quality": max(bests) if bests else None,
        "per_seed_best_mean": statistics.mean(bests) if bests else None,
        "per_seed_best_median": statistics.median(bests) if bests else None,
        "per_seed_best_sample_sd": statistics.stdev(bests) if len(bests) > 1 else None,
        "per_seed_best_min": min(bests) if bests else None,
        "per_seed_best_max": max(bests) if bests else None,
        "frozen_threshold": threshold,
        "exact_candidates": len(observed),
        "candidate_success_count": candidate_successes,
        "candidate_success_fraction": candidate_successes / len(observed) if observed else None,
        "seed_success_count": seed_successes,
        "seed_success_fraction_among_observed": seed_successes / len(bests) if bests else None,
        **totals,
    }
    horizon = min((len(points) for points in curves.values()), default=0)
    curves["Group mean"] = tuple(
        (i + 1, statistics.mean(points[i][1] for points in curves.values())) for i in range(horizon)
    )
    if stats["complete_seeds"] != len(included):
        warnings.append(
            "Some included seeds are unfinished/failed: observed-best statistics and success fractions are provisional, not completed-run rates."
        )
    return GroupSnapshot(
        definition, tuple(members), reference, stats, candidates, curves, tuple(warnings)
    )


def load_group(reader: CampaignReader, definition: GroupDefinition) -> GroupSnapshot:
    loaded = []
    for member in definition.members:
        if member.role == "included":
            loaded.append(reader.group_member(member.directory))
        else:
            # Excluded corrupt/dropout data remain inspectable without entering arithmetic.
            loaded.append((reader.load(member.directory.resolve()), {}))
    return summarize(definition, tuple(loaded))


def export_group(snapshot: GroupSnapshot, path: Path, top_n: int = 20) -> None:
    """Exclusive-create export outside campaign directories; never rewrite evidence.

    CSV is a long-form summary, member and candidate table with explicit row_type.
    Nested diagnostics are JSON strings so no stored fields are flattened away.
    """
    path = path.resolve()
    if any(path.is_relative_to(m.directory.resolve()) for m in snapshot.definition.members):
        raise ValueError("Exports must be outside every campaign directory")
    if top_n < 1:
        raise ValueError("Top N must be positive")
    payload = snapshot.payload(top_n)
    if path.suffix.lower() == ".json":
        content = json.dumps(payload, indent=2, allow_nan=False)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content + "\n")
    elif path.suffix.lower() == ".csv":
        rows = [{"row_type": "summary", **snapshot.statistics}]
        rows.extend({"row_type": "member", **m} for m in snapshot.members)
        rows.extend({"row_type": "candidate", **c} for c in payload["top_candidates"])
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with path.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: json.dumps(v) if isinstance(v, (dict, list, tuple)) else v
                        for k, v in row.items()
                    }
                )
    else:
        raise ValueError("Choose a .json or .csv export")
