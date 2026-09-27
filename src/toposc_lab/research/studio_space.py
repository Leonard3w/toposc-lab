"""Existing geometry recipes and immutable follow-ups in the research strategy API.

This module owns proposal adapters, not evaluation, storage or lifecycle. The
ResearchEngine still commits each proposal, RNG checkpoint and exact stage.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from toposc_lab.geometry import Geometry, GeometryBoundaryComponent, square
from toposc_lab.geometry.generators.protocol import (
    FunctionGeometryGenerator,
    GeometryGenerationRequest,
)
from toposc_lab.research.descriptors import compute_descriptors
from toposc_lab.research.embedded import EmbeddedDomain
from toposc_lab.research.embedded_cohort import canonical_sites, geometry_id
from toposc_lab.research.space import (
    SPACE_REGISTRY,
    FixedConnectivitySpace,
    geometry_from_payload,
    geometry_to_payload,
)
from toposc_lab.research.storage import utc_now
from toposc_lab.research.strategies import register_strategy

FAMILIES = ("regular", "rewired_square", "amorphous_planar", "constrained_embedded")


def checked_geometry(record: dict[str, Any]) -> Geometry:
    """The archive is authoritative; a conflicting JSON preview is an error."""
    geometry = geometry_from_payload(record["geometry"])
    rebuilt = geometry_to_payload(geometry)
    for name in ("n_sites", "coordinates", "edges", "boundary_sites"):
        if record["geometry"].get(name) != rebuilt[name]:
            raise ValueError(f"Geometry preview differs from archive: {name}")
    recorded = record.get("geometry_sha256")
    if recorded is not None and recorded != geometry_id(geometry):
        raise ValueError("Physical geometry checksum mismatch")
    return geometry


@dataclass
class EmbeddedSamplingSpace:
    families: list[str] = field(default_factory=lambda: list(FAMILIES))
    side: int = 8
    generator: dict[str, Any] = field(default_factory=dict)
    domain: dict[str, Any] = field(default_factory=lambda: asdict(EmbeddedDomain()))
    rewires: list[int] = field(default_factory=lambda: [8, 16, 32, 64])
    min_degree: int = 2
    max_degree: int = 4
    max_bond_length: float = 1.75
    bond_tolerance: float = 0.0
    forbid_crossings: bool = True
    site_crossings: str = "forbid"

    def __post_init__(self) -> None:
        if not self.families or len(set(self.families)) != len(self.families):
            raise ValueError("Select distinct geometry families")
        if set(self.families) - set(FAMILIES):
            raise ValueError("Unsupported embedded geometry family")
        if type(self.side) is not int or not 2 <= self.side <= 64:
            raise ValueError("side must be 2..64 (dense resource limits apply separately)")
        self.physical_domain = EmbeddedDomain(**self.domain)
        self.domain = asdict(self.physical_domain)
        if set(self.families) & {"regular", "rewired_square"}:
            expected = (0.0, float(self.side - 1), 0.0, float(self.side - 1))
            if self.physical_domain.bounds != expected:
                raise ValueError("Unit square families require domain [0, side-1]²")
        # Constructor validation does not draw a graph or consume a research seed.
        if "rewired_square" in self.families:
            self._square_space()
        if set(self.families) & {"amorphous_planar", "constrained_embedded"}:
            from toposc_lab.geometry.generators.hard_core_planar import HardCorePlanarConfig

            prepared = HardCorePlanarConfig(**self.generator)
            self.generator = asdict(prepared)
            maximum = prepared.box_maximum
            if self.physical_domain.bounds != (0.0, maximum, 0.0, maximum):
                raise ValueError("Free graph generator requires domain [0, box_maximum]²")
            if self.physical_domain.boundary_shell != prepared.boundary_shell_thickness:
                raise ValueError("Generator and physical boundary widths must agree")

    def _square_space(self) -> FixedConnectivitySpace:
        return FixedConnectivitySpace(
            side=self.side,
            min_degree=self.min_degree,
            max_degree=self.max_degree,
            max_bond_length=self.max_bond_length,
            bond_tolerance=self.bond_tolerance,
            forbid_crossings=self.forbid_crossings,
            site_crossings=self.site_crossings,
            initialization_rewires=tuple(self.rewires),
        )

    def generate(self, family: str, seed: int) -> Geometry:
        if family == "regular":
            g = FunctionGeometryGenerator("regular", square).generate(
                GeometryGenerationRequest(parameters={"n_x": self.side, "n_y": self.side})
            )
        elif family == "rewired_square":
            space = self._square_space()
            g = space.sample(np.random.default_rng(seed))
            reasons = space.validate(g)
            if reasons:
                raise ValueError("Rewired geometry violates constraints: " + ", ".join(reasons))
        else:
            from toposc_lab.geometry.generators.hard_core_planar import (
                constrained_random_embedded_graph,
                hard_core_planar_graph,
            )

            builder = (
                hard_core_planar_graph
                if family == "amorphous_planar"
                else constrained_random_embedded_graph
            )
            g = FunctionGeometryGenerator(family, builder, stochastic=True).generate(
                GeometryGenerationRequest(seed=seed, parameters={"config": self.generator})
            )
        boundary = self.physical_domain.boundary(g.coordinates)
        g = replace(
            g,
            boundary_sites=boundary,
            boundary_components=(GeometryBoundaryComponent("outer", 0, boundary),),
        )
        g = canonical_sites(g)
        self.physical_domain.validate(g)
        return g


@dataclass
class FixedCandidateSpace:
    candidates: list[dict[str, Any]]
    domain: dict[str, Any] = field(default_factory=lambda: asdict(EmbeddedDomain()))

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("Select at least one candidate for the follow-up")
        self.candidates = copy.deepcopy(self.candidates)
        self.physical_domain = EmbeddedDomain(**self.domain)
        seen = set()
        for candidate in self.candidates:
            geometry = checked_geometry(candidate)
            self.physical_domain.validate(geometry)
            identity = geometry_id(geometry)
            if identity in seen:
                raise ValueError("Duplicate physical geometry in fixed candidates")
            seen.add(identity)


class StudioSampling:
    """Outcome-blind sampling of existing recipes, with resumable proposal state."""

    name = "embedded_random"
    uses_surrogate = False

    def __init__(
        self,
        space: EmbeddedSamplingSpace,
        seed: int = 0,
        *,
        experiment_id: str = "",
        code_version: str = "unknown",
    ) -> None:
        self.space, self.seed = space, seed
        self.experiment_id, self.code_version = experiment_id, code_version
        self.proposal_limit: int | None = None
        self.initialize()

    def initialize(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.generation = self.generated = self.cursor = 0
        self.invalid = self.duplicates = self.near_duplicates = 0
        self.seen: set[str] = set()
        self.used_regular = False
        self.archive: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.selection_counts: dict[str, int] = {}
        self._rejections: list[dict[str, Any]] = []

    @property
    def exhausted(self) -> bool:
        return self.space.families == ["regular"] and self.used_regular

    def _next(self) -> tuple[Geometry, dict[str, Any]] | None:
        choices = [f for f in self.space.families if f != "regular" or not self.used_regular]
        if not choices:
            return None
        family = choices[self.cursor % len(choices)]
        self.cursor += 1
        seed = int(self.rng.integers(0, 2**32))
        if family == "regular":
            self.used_regular = True
        self._attempt_metadata = {"family": family, "seed": seed}
        return self.space.generate(family, seed), {"family": family, "seed": seed}

    def propose(self, count: int, *, max_attempts: int | None = None) -> list[dict[str, Any]]:
        if type(count) is not int or count < 0:
            raise ValueError("proposal count must be a nonnegative integer")
        if self.exhausted or count == 0:
            return []
        limit = max_attempts if max_attempts is not None else max(1, count * 30)
        if self.proposal_limit is not None:
            limit = min(limit, self.proposal_limit)
        records = []
        for _ in range(limit):
            if len(records) >= count or self.exhausted:
                break
            self.generated += 1
            self._attempt_metadata = {}
            try:
                entry = self._next()
                if entry is None:
                    self.generated -= 1
                    break
                geometry, source = entry
                identity = geometry_id(geometry)
                if identity in self.seen:
                    self.duplicates += 1
                    raise ValueError("duplicate physical geometry")
                record = {
                    **source,
                    "id": identity,
                    "geometry_hash": identity,
                    "geometry_sha256": identity,
                    "geometry": geometry_to_payload(geometry),
                    "descriptors": compute_descriptors(geometry, include_square_reference=False),
                    "origin": "proposed",
                    "validation_state": "PROPOSED",
                    "score": None,
                    "parent": source.get("source_candidate_id"),
                    "lineage": [],
                    "generation": self.generation,
                    "attempt": self.generated,
                    "mutation": {
                        "operator": "fixed_candidate"
                        if self.name == "fixed_candidates"
                        else "existing_generator",
                        "family": source.get("family"),
                    },
                    "created_at": utc_now(),
                    "code_version": self.code_version,
                    "experiment_id": self.experiment_id,
                    "baseline": source.get("family") == "regular",
                }
                self.seen.add(identity)
                records.append(record)
            except (ValueError, RuntimeError) as error:
                self.invalid += "duplicate" not in str(error)
                self._rejections.append(
                    {
                        **self._attempt_metadata,
                        "id": f"rejected-{self.generated}",
                        "validation_state": "REJECTED",
                        "rejection_reason": str(error),
                        "origin": "proposed",
                        "score": None,
                        "generation": self.generation,
                        "attempt": self.generated,
                    }
                )
        self.generation += 1
        return records

    def select(
        self, proposals: list[dict[str, Any]], count: int, surrogate: Any = None
    ) -> list[dict[str, Any]]:
        result = proposals[:count]
        for record in result:
            record["acquisition"] = {"strategy": self.name, "score": None}
        return result

    def drain_rejections(self) -> list[dict[str, Any]]:
        result, self._rejections = self._rejections, []
        return result

    def observe(self, candidate: dict[str, Any], result: dict[str, Any]) -> None:
        if not any(row["id"] == candidate["id"] for row in self.history):
            # Exact evidence is in the existing store; checkpoint only progress metadata.
            self.history.append(
                {
                    key: value
                    for key, value in {**candidate, **result}.items()
                    if key in {"id", "score", "origin", "family", "validation_state"}
                }
            )

    def checkpoint(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "version": 1,
                "strategy": self.name,
                "rng": self.rng.bit_generator.state,
                "generation": self.generation,
                "generated": self.generated,
                "cursor": self.cursor,
                "invalid": self.invalid,
                "duplicates": self.duplicates,
                "near_duplicates": 0,
                "used_regular": self.used_regular,
                "seen": sorted(self.seen),
                "archive": {},
                "history": self.history,
                "selection_counts": {},
                "rejections": self._rejections,
            }
        )

    def resume(self, checkpoint: dict[str, Any]) -> StudioSampling:
        if checkpoint.get("version") != 1 or checkpoint.get("strategy") != self.name:
            raise ValueError("Incompatible studio strategy checkpoint")
        state = copy.deepcopy(checkpoint)
        self.rng.bit_generator.state = state.pop("rng")
        self.seen = set(state.pop("seen"))
        self._rejections = state.pop("rejections")
        for name in (
            "generation",
            "generated",
            "cursor",
            "invalid",
            "duplicates",
            "near_duplicates",
            "used_regular",
            "archive",
            "history",
            "selection_counts",
        ):
            setattr(self, name, state[name])
        return self

    def summarize(self) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "generated": self.generated,
            "accepted": len(self.seen),
            "invalid": self.invalid,
            "duplicates": self.duplicates,
            "archive_size": 0,
            "archive_coverage": 0,
            "exhausted": self.exhausted,
        }


class FixedCandidateStrategy(StudioSampling):
    """Evaluate a saved selection once, retaining the original site ordering."""

    name = "fixed_candidates"

    @property
    def exhausted(self) -> bool:
        return self.cursor >= len(self.space.candidates)

    def _next(self) -> tuple[Geometry, dict[str, Any]] | None:
        if self.exhausted:
            return None
        original = copy.deepcopy(self.space.candidates[self.cursor])
        self.cursor += 1
        source = {
            key: value
            for key, value in original.items()
            if key
            in {
                "family",
                "seed",
                "source_run",
                "source_experiment_id",
                "source_geometry_sha256",
                "source_candidate_id",
                "selection_note",
            }
        }
        source.setdefault("source_candidate_id", original["id"])
        source.setdefault("family", "selected")
        return checked_geometry(original), source


SPACE_REGISTRY["embedded_sampling"] = EmbeddedSamplingSpace
SPACE_REGISTRY["fixed_candidates"] = FixedCandidateSpace
register_strategy(StudioSampling.name, StudioSampling)
register_strategy(FixedCandidateStrategy.name, FixedCandidateStrategy)
