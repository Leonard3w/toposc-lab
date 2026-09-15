"""Qt-independent display contracts; missing scientific evidence stays missing."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(value) else None
    return None


def display(value: Any) -> str:
    if value is None or value == "":
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.9g}" if math.isfinite(value) else "unavailable"
    if isinstance(value, (list, tuple, dict)):
        import json

        return json.dumps(value, ensure_ascii=False)
    return str(value)


def majorana_label(record: dict[str, Any]) -> str:
    """Copy explicit disclaimers; absence is never converted into a claim."""
    nodes = [record]
    while nodes:
        node = nodes.pop()
        if node.get("majorana_claim") is False or node.get("status") == "diagnostics_only":
            return "diagnostics_only; majorana_claim=false — no Majorana claim"
        for value in node.values():
            if isinstance(value, dict):
                nodes.append(value)
            elif isinstance(value, list):
                nodes.extend(v for v in value if isinstance(v, dict))
    return "unavailable — no Majorana claim inferred"


@dataclass(frozen=True)
class GeometrySnapshot:
    coordinates: tuple[tuple[float, ...], ...] = ()
    edges: tuple[tuple[int, int], ...] = ()
    boundary: tuple[int, ...] = ()
    message: str = "unavailable"


@dataclass(frozen=True)
class CandidateSnapshot:
    record_id: str
    geometry_id: str = "unavailable"
    generator: str = "unavailable"
    seed: Any = None
    cycle: Any = None
    quality: float | None = None
    topology: str = "unavailable"
    indices: Any = None
    minimum_abs_energy: float | None = None
    boundary_weight: Any = None
    ood: Any = None
    confirmation: Any = None
    robustness: Any = None
    majorana: str = "unavailable — no Majorana claim inferred"
    geometry: GeometrySnapshot = field(default_factory=GeometrySnapshot)
    exact_evaluation: int | None = None
    source: str = ""
    campaign_seed: Any = None

    @property
    def selection_key(self) -> str:
        return f"{self.campaign_seed}:{self.record_id}"

    def details(self) -> str:
        fields = {
            "Geometry ID": self.geometry_id,
            "Record ID": self.record_id,
            "Generator": self.generator,
            "Campaign Seed": self.campaign_seed,
            "Exact evaluation seed": self.seed,
            "Cycle (zero-based)": self.cycle,
            "Exact quality": self.quality,
            "Majorana": self.majorana,
            "Topology status (stored)": self.topology,
            "Localizer indices": self.indices,
            "Minimum |E|": self.minimum_abs_energy,
            "Boundary diagnostics": self.boundary_weight,
            "OOD (stored)": self.ood,
            "Exact confirmation": self.confirmation,
            "Robustness (stored)": self.robustness,
            "Geometry view": self.geometry.message,
            "Source": self.source,
        }
        return "\n".join(f"{key}: {display(value)}" for key, value in fields.items())


@dataclass(frozen=True)
class CampaignSnapshot:
    directory: Path
    name: str
    status: str = "Unknown"
    config: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, Any] = field(default_factory=dict)
    health: dict[str, Any] = field(default_factory=dict)
    candidates: tuple[CandidateSnapshot, ...] = ()
    curves: dict[str, tuple[tuple[int, float], ...]] = field(default_factory=dict)
    events: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    log: str = ""
    started: Any = None
    elapsed: float | None = None

    @property
    def threshold(self) -> float | None:
        return number(self.config.get("success_threshold"))

    @property
    def best(self) -> CandidateSnapshot | None:
        return next((c for c in self.candidates if c.quality is not None), None)

    @property
    def threshold_label(self) -> str:
        if self.best is None or self.threshold is None:
            return "Threshold status unavailable"
        assert self.best.quality is not None
        return (
            "Frozen threshold reached"
            if self.best.quality >= self.threshold
            else "Below frozen threshold"
        )
