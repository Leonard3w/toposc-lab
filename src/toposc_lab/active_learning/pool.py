"""Finite, explicitly constrained proposal spaces; no simulated labels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from toposc_lab.data import GeometryRecord, ModelParametersRecord
from toposc_lab.data.dataset_schema import encode_scientific_value
from toposc_lab.geometry import validate_geometry


@dataclass(frozen=True, slots=True)
class Candidate:
    geometry: GeometryRecord
    model: ModelParametersRecord

    @property
    def candidate_id(self) -> str:
        payload = [
            self.geometry.exact_id,
            self.model.model_name,
            self.model.model_version,
            encode_scientific_value(self.model.parameters),
        ]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateSpace:
    """A finite catalog is itself a constraint; bounds also reject invalid inputs.

    Distinct model parameters on the same graph are distinct simulations.
    Held-out geometry families are excluded separately by the training contract.
    """

    candidates: tuple[Candidate, ...]
    minimum_sites: int = 2
    maximum_sites: int = 256
    require_connected: bool = True

    def __post_init__(self) -> None:
        if not 2 <= self.minimum_sites <= self.maximum_sites:
            raise ValueError("invalid site constraints")
        if not self.candidates or not all(isinstance(c, Candidate) for c in self.candidates):
            raise ValueError("a nonempty Candidate catalog is required")
        identifiers = [c.candidate_id for c in self.candidates]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("duplicate candidate in search space")
        for candidate in self.candidates:
            geometry = candidate.geometry.to_geometry()
            if not validate_geometry(geometry).is_valid:
                raise ValueError("invalid candidate geometry")
            if not self.minimum_sites <= geometry.n_sites <= self.maximum_sites:
                raise ValueError("candidate violates site constraints")
            if self.require_connected:
                from toposc_lab.geometry import extract_geometry_descriptors

                if not extract_geometry_descriptors(geometry)["is_connected"]:
                    raise ValueError("candidate must be connected")
            n = candidate.model.parameters.get("n_sites", geometry.n_sites)
            if n != geometry.n_sites:
                raise ValueError("model and geometry site counts disagree")
        object.__setattr__(self, "candidates", tuple(self.candidates))


def generate_candidate_pool(
    space: CandidateSpace,
    *,
    count: int,
    seed: int,
    excluded_ids: frozenset[str] = frozenset(),
    excluded_families: frozenset[str] = frozenset(),
) -> tuple[Candidate, ...]:
    """Sample without replacement; report exhaustion by a shorter/empty pool."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 or seed < 0:
        raise ValueError("positive count and nonnegative seed required")
    available = sorted(
        (
            c
            for c in space.candidates
            if c.candidate_id not in excluded_ids
            and c.geometry.family_fingerprint not in excluded_families
        ),
        key=lambda c: c.candidate_id,
    )
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(available))[:count]
    return tuple(available[int(i)] for i in indices)
