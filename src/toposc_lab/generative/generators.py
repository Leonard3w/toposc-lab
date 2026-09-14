"""Common batch API; reuse Phase-10 evolution and Phase-13 optional acquisition."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from toposc_lab.active_learning.acquisition import AcquisitionConfig, acquire
from toposc_lab.active_learning.training import TrainingConfig, retrain
from toposc_lab.data import DatasetRecord, ExactPhysicsDataset
from toposc_lab.evaluation import ObjectiveDirection
from toposc_lab.generative.physics import PROTOCOL_ID, candidate
from toposc_lab.generative.space import GeometrySearchSpace, NoveltyConfig, structural_distance
from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.ml import RegressionTargetDefinition, RegressionTargetSource
from toposc_lab.search import (
    LexicographicFitnessDefinition,
    TournamentSelectionConfig,
    create_initial_population,
    evaluate_population_fitness,
    geometry_to_genome,
    select_population_members,
)
from toposc_lab.search.geometry_mutations import rewire_edge_mutation
from toposc_lab.search.lexicographic_fitness import DerivedEvaluationRun

Method = Literal["random", "evolution", "patch", "coverage", "active"]
METHODS: tuple[Method, ...] = ("random", "evolution", "patch", "coverage", "active")


@dataclass(frozen=True, slots=True)
class Evidence:
    record: DatasetRecord
    run: DerivedEvaluationRun


@dataclass(slots=True)
class ProposalAudit:
    attempted: int = 0
    invalid: int = 0
    duplicate: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    events: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class GeometryGenerator:
    method: Method
    space: GeometrySearchSpace = field(default_factory=GeometrySearchSpace)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    audit: ProposalAudit = field(default_factory=ProposalAudit)
    maximum_attempts: int = 1000

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError("unknown generator")
        if isinstance(self.maximum_attempts, bool) or self.maximum_attempts < 1:
            raise ValueError("positive attempt cap required")

    def propose(
        self,
        history: tuple[Evidence, ...],
        *,
        seed: int,
        count: int = 4,
        excluded: tuple[Geometry, ...] = (),
    ) -> tuple[Geometry, ...]:
        """No simulations here: history contains only already paid exact records.

        ``excluded`` also includes failed exact attempts. They cannot be retried as
        uncharged proposals. Each call has a deterministic bounded rejection loop.
        """
        if seed < 0 or count < 1:
            raise ValueError("nonnegative seed and positive count required")
        rng = np.random.default_rng(seed)
        archive = [e.record.geometry.to_geometry() for e in history] + list(excluded)
        parent = None
        if self.method == "evolution" and history:
            # Steady-state archive elitism, with existing Phase-10 tournament/fitness.
            elite = sorted(
                history,
                key=lambda e: (
                    -float(cast(float, e.record.observables[0].values["quality"])),
                    e.record.geometry.exact_id,
                ),
            )[:8]
            population = create_initial_population(
                [geometry_to_genome(e.record.geometry.to_geometry()) for e in elite]
            )
            fitness = evaluate_population_fitness(
                population,
                definition=LexicographicFitnessDefinition(
                    ("quality",), (ObjectiveDirection.MAXIMIZE,), PROTOCOL_ID
                ),
                evaluator=lambda member: elite[member.member_index].run,
            )
            selection = select_population_members(
                fitness, config=TournamentSelectionConfig(count, min(3, len(elite))), seed=seed
            )
            parent = [m.population_member.genome for m in selection.selected_members]
        pool_size = max(12, count) if self.method in ("active", "coverage") and history else count
        pool: list[Geometry] = []
        for attempt in range(self.maximum_attempts):
            if len(pool) == pool_size:
                break
            if parent is not None and rng.random() >= 0.2:
                genome = parent[len(pool) % len(parent)]
                for _ in range(2):
                    removable = [
                        i
                        for i, e in enumerate(genome.edges)
                        if (e.source, e.target) not in self.space.perimeter
                    ]
                    current = {(e.source, e.target) for e in genome.edges}
                    available = [e for e in self.space.edge_pool if e not in current]
                    genome = rewire_edge_mutation(
                        genome,
                        int(rng.choice(removable)),
                        GeometryEdge(*available[int(rng.integers(len(available)))]),
                    )
                proposal = genome.to_geometry()
            else:
                proposal = self.space.sample(rng, patch=self.method == "patch")
            self.audit.attempted += 1
            reasons = self.space.reasons(proposal)
            event = {
                "seed": seed,
                "attempt": attempt,
                "edges": [[e.source, e.target] for e in proposal.edges],
                "reasons": list(reasons),
            }
            if reasons:
                self.audit.invalid += 1
                self.audit.reasons.update(reasons)
                event["status"] = "invalid"
            elif any(
                structural_distance(proposal, old) <= self.novelty.minimum_distance
                for old in archive + pool
            ):
                self.audit.duplicate += 1
                event["status"] = "duplicate"
            else:
                pool.append(proposal)
                event["status"] = "pool"
            self.audit.events.append(event)
        if len(pool) != pool_size:
            raise RuntimeError("proposal attempt cap exhausted; never silently relax constraints")
        if self.method == "coverage" and history:
            selected: list[Geometry] = []
            while len(selected) < count:
                index = max(
                    range(len(pool)),
                    key=lambda i: min(
                        structural_distance(pool[i], old) for old in archive + selected
                    ),
                )
                selected.append(pool.pop(index))
            return tuple(selected)
        if self.method == "active" and history:
            fitted = retrain(
                ExactPhysicsDataset(tuple(e.record for e in history)),
                config=TrainingConfig(
                    RegressionTargetDefinition(
                        "finite_geometry_quality",
                        RegressionTargetSource.OBSERVABLE,
                        "finite_geometry_quality",
                        "1",
                        "quality",
                    ),
                    flag_extrapolation=True,
                ),
                seed=seed,
            )
            candidates = tuple(candidate(g) for g in pool)
            predictions = fitted.predict(candidates)
            acquisitions = acquire(
                predictions,
                AcquisitionConfig(
                    batch_size=count, scaling="target_scale", diversity_radius_fraction=0.5
                ),
                feature_coordinates=fitted.feature_coordinates(candidates),
                reference_spacing=fitted.reference_spacing,
            )
            strategies = {s.candidate_id: s.strategy for s in acquisitions}
            for p in predictions:
                self.audit.predictions.append(
                    {
                        "seed": seed,
                        "candidate_id": p.candidate_id,
                        "value": p.value,
                        "uncertainty": p.uncertainty,
                        "is_ood": p.is_ood,
                        "ood_score": p.ood_score,
                        "strategy": strategies.get(p.candidate_id),
                        "training_record_ids": list(p.training_record_ids),
                        "training_fingerprint": p.training_fingerprint,
                        "kind": "predicted",
                    }
                )
            lookup = {c.candidate_id: g for c, g in zip(candidates, pool, strict=True)}
            return tuple(lookup[s.candidate_id] for s in acquisitions)
        return tuple(pool)
