"""Configurable presentation of the existing exact chiral-p-wave evaluation.

Historical adapters remain frozen. This versioned adapter changes settings and
evidence selection, never the Hamiltonian, pairing convention, or Q definition.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from hashlib import sha256
from typing import Any

import numpy as np

from toposc_lab.geometry import Geometry
from toposc_lab.research.embedded import EmbeddedDomain
from toposc_lab.research.physics import (
    EVIDENCE_SCOPE as LEGACY_SCOPE,
)
from toposc_lab.research.physics import (
    OBJECTIVE_REGISTRY,
    FiniteSystemEvaluator,
    PhysicsAdapterDefinition,
    PhysicsProtocol,
    register_physics_adapter,
)
from toposc_lab.research.storage import dumps
from toposc_lab.research.validation_diagnostics import ValidationEvaluator

ADAPTER_ID = "phase20.configurable-chiral-p-wave.v1"
STUDIO_ADAPTER_ID = ADAPTER_ID
EVIDENCE_SCOPE = "Finite embedded-domain localizer and localization diagnostics; no phase or Majorana claim."
# These are declared algorithm definitions, not editable controls. The exact
# adapter and diagnostic implementations retain their validated constants.
FROZEN_NUMERICAL_SETTINGS = {
    "robustness_uncertainty_method": "wilson_score",
    "pairing_plane_axes": [0, 1],
    "boundary_shell_bin_width": 1.0,
    "success_threshold": 0.20,
}


@dataclass(frozen=True)
class StudioPhysicsProtocol(PhysicsProtocol):
    adapter_id: str = ADAPTER_ID
    probe_rule: str = "declared_embedded_domain_center"
    domain: EmbeddedDomain = field(default_factory=EmbeddedDomain)
    clean_reference: bool = True
    paired_seeds: bool = True
    energy_cutoff: float = 0.5
    group_tolerance: float = 1e-8
    boundary_widths: tuple[float, ...] = (1, 2, 3)
    near_zero_count: int = 4
    majorana_zero_tolerance: float = 1e-10
    majorana_splitting_tolerance: float = 1e-3
    majorana_splitting_phs_tolerance: float = 1e-8
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if isinstance(self.domain, dict):
            object.__setattr__(self, "domain", EmbeddedDomain(**self.domain))
        if not isinstance(self.domain, EmbeddedDomain):
            raise TypeError("domain must be an EmbeddedDomain")
        if self.adapter_id != ADAPTER_ID or self.probe_rule != "declared_embedded_domain_center":
            raise ValueError("unsupported studio adapter or probe convention")
        for name in ("clean_reference", "paired_seeds"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("hopping", "chemical_potential", "pairing"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if type(self.chirality) is not int or self.chirality not in (-1, 1):
            raise ValueError("chirality must be -1 or +1")
        for name in ("tolerance", "energy_cutoff", "group_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 1e-14 <= self.tolerance <= 1e-4:
            raise ValueError("numerical tolerance must be between 1e-14 and 1e-4")
        if type(self.near_zero_count) is not int or not 1 <= self.near_zero_count <= 8192:
            raise ValueError("near_zero_count must be an integer between 1 and 8192")
        for name in ("majorana_zero_tolerance", "majorana_splitting_tolerance",
                     "majorana_splitting_phs_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.majorana_splitting_tolerance <= self.majorana_zero_tolerance:
            raise ValueError("majorana_splitting_tolerance must exceed majorana_zero_tolerance")
        if (isinstance(self.confidence_level, bool) or not np.isfinite(self.confidence_level)
                or not 0 < self.confidence_level < 1):
            raise ValueError("confidence_level must be strictly between zero and one")
        for name in ("kappas", "boundary_widths"):
            values = tuple(getattr(self, name))
            if (not values or len(values) > 32 or len(set(values)) != len(values)
                    or any(isinstance(v, bool) or not np.isfinite(v) or v <= 0 for v in values)):
                raise ValueError(f"{name} requires 1..32 distinct positive finite values")
            object.__setattr__(self, name, values)
        if not self.clean_reference and (self.confirmation or self.objective == "clean_quality"):
            raise ValueError("clean confirmation and clean_quality require clean_reference")
        if not self.clean_reference and "robustness" not in self.validators:
            raise ValueError("at least a clean reference or disorder ensemble is required")
        # Reuse all established seed, ensemble, objective and validator checks.
        # Only the explicitly configurable fields are replaced for that check.
        legacy = {f.name: getattr(self, f.name) for f in fields(PhysicsProtocol)}
        frozen = PhysicsProtocol()
        for name in ("adapter_id", "hopping", "chemical_potential", "pairing", "chirality",
                     "tolerance", "kappas", "probe_rule"):
            legacy[name] = getattr(frozen, name)
        checked = PhysicsProtocol(**legacy)
        for name in ("disorder_widths", "disorder_seeds", "validators"):
            object.__setattr__(self, name, getattr(checked, name))


class StudioFiniteEvaluator(FiniteSystemEvaluator):
    adapter_id = ADAPTER_ID
    evidence_scope = EVIDENCE_SCOPE
    geometry_family = "embedded_2d"

    def probe(self, geometry: Geometry) -> tuple[float, float]:
        self.protocol.domain.validate(geometry)
        return self.protocol.domain.center

    def plan(self) -> list[dict[str, Any]]:
        stages = super().plan()
        return stages if self.protocol.clean_reference else [s for s in stages if s["kind"] == "disorder"]


class _StageEvaluator(StudioFiniteEvaluator):
    """Pass the realized seed through the unchanged exact evaluator's plan check."""

    def __init__(self, protocol: Any, provenance: Any, stage: dict[str, Any]) -> None:
        super().__init__(protocol, provenance)
        object.__setattr__(self, "stage", stage)

    def plan(self) -> list[dict[str, Any]]:
        return [dict(self.stage)]


class StudioEvaluator:
    """Shared stage protocol with optional spatial diagnostics and exact summaries."""

    def __init__(self, protocol: StudioPhysicsProtocol, provenance: Any = None) -> None:
        self.protocol, self.provenance = protocol, provenance
        self.base = StudioFiniteEvaluator(protocol, provenance)

    def plan(self) -> list[dict[str, Any]]:
        return self.base.plan()

    def evaluate(self, geometry: Geometry, stage: Mapping[str, Any]) -> dict[str, Any]:
        if dict(stage) not in self.plan():
            raise ValueError("stage must match the serialized protocol plan")
        actual = dict(stage)
        if not self.protocol.paired_seeds and stage["kind"] == "disorder":
            assert geometry.coordinates is not None
            identity = dumps({
                "coordinates": geometry.coordinates.tolist(),
                "edges": sorted(sorted((e.source, e.target)) for e in geometry.edges),
                "boundary": sorted(geometry.boundary_sites),
            })
            digest = sha256((identity + ":" + str(stage["seed"])).encode()).digest()
            actual["seed"] = int.from_bytes(digest[:8], "big")
        exact = _StageEvaluator(self.protocol, self.provenance, actual)
        evaluator = ValidationEvaluator(
            self.protocol, self.provenance, domain=self.protocol.domain, base=exact,
            energy_cutoff=self.protocol.energy_cutoff,
            group_tolerance=self.protocol.group_tolerance,
            boundary_widths=self.protocol.boundary_widths,
        )
        result = evaluator.evaluate(geometry, actual)
        result["requested_stage"] = dict(stage)
        result["numerical_definitions"] = dict(FROZEN_NUMERICAL_SETTINGS)
        result["seed_policy"] = "paired" if self.protocol.paired_seeds else "independent_by_geometry"
        result["diagnostic_version"] = "phase20.configurable-spatial-boundary.v1"
        return result

    def summarize(self, results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        summary = self.base.summarize(results)
        summary.update(adapter_id=ADAPTER_ID, scope=EVIDENCE_SCOPE)
        summary["warnings"] = [w.replace(LEGACY_SCOPE, EVIDENCE_SCOPE) for w in summary["warnings"]]
        for validation in summary["validation_results"].values():
            if validation.get("scope") == LEGACY_SCOPE:
                validation["scope"] = EVIDENCE_SCOPE
        for group in summary["robustness"]:
            members = [results[s["key"]] for s in self.plan()
                       if s["kind"] == "disorder" and s["width"] == group["width"]
                       and s["key"] in results]
            group["requested_seeds"] = list(self.protocol.disorder_seeds)
            group["seeds"] = [m["stage"]["seed"] for m in members]
            group["scope"] = ("fixed geometry, scalar onsite disorder; " +
                              ("paired seeds across candidates" if self.protocol.paired_seeds
                               else "independent deterministic seeds by physical geometry"))
        if not self.protocol.clean_reference:
            # No fictitious clean result: spectrum evidence comes from actual disorder stages.
            available = [r for r in results.values() if r.get("kind") == "exact"
                         and r.get("status") == "completed" and not r.get("error")]
            summary["score_ready"] = bool(summary["complete"] and available)
            summary["validation_results"]["spectrum"] = {
                "passed": bool(available), "status": "validated" if available else "pending",
                "scope": "finite complete BdG spectra of requested disorder stages",
                "tolerance": self.protocol.tolerance,
            }
            if available and not summary["failed_stages"]:
                summary["validation_state"] = "EXACT_EVALUATED"
                summary["validation_states"] = ["EXACT_EVALUATED"]
            summary["score"] = (OBJECTIVE_REGISTRY[self.protocol.objective].evaluate(summary)
                                if summary["score_ready"] else None)
        summary["clean_reference_requested"] = self.protocol.clean_reference
        return summary


register_physics_adapter(PhysicsAdapterDefinition(
    ADAPTER_ID, EVIDENCE_SCOPE, StudioPhysicsProtocol, StudioEvaluator,
))
