"""Common deterministic search interface and an exact-only quality-diversity archive."""
from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np

from toposc_lab.research.descriptors import (
    DEFAULT_BEHAVIOR_DESCRIPTORS,
    DEFAULT_DESCRIPTOR_BOUNDS,
    DESCRIPTOR_REGISTRY,
    compute_descriptors,
)
from toposc_lab.research.space import (
    MUTATION_REGISTRY,
    FixedConnectivitySpace,
    geometry_from_payload,
    geometry_to_payload,
)

STRATEGY_REGISTRY: dict[str, type[SearchStrategy]] = {}


def register_strategy(name: str, strategy: type[SearchStrategy]) -> None:
    if not name or name in STRATEGY_REGISTRY:
        raise ValueError(f"duplicate/empty strategy: {name}")
    STRATEGY_REGISTRY[name] = strategy


def create_strategy(name: str, space: FixedConnectivitySpace, seed: int = 0,
                    **kwargs: Any) -> SearchStrategy:
    return STRATEGY_REGISTRY[name](space=space, seed=seed, **kwargs)


class SearchStrategy:
    name = "random"
    uses_surrogate = False

    def __init__(self, space: FixedConnectivitySpace, seed: int = 0, *,
                 behavior_descriptors: Sequence[str] = DEFAULT_BEHAVIOR_DESCRIPTORS,
                 descriptor_bounds: Sequence[Sequence[float]] = DEFAULT_DESCRIPTOR_BOUNDS,
                 archive_bins: int | Sequence[int] = (12, 12),
                 mutation_rates: dict[str, float] | None = None,
                 allocation: dict[str, float] | Sequence[float] | None = None,
                 experiment_id: str = "", code_version: str = "unknown",
                 family_fraction: float = 0.5) -> None:
        self.space, self.seed = space, seed
        self.rng = np.random.default_rng(seed)
        self.behavior_descriptors = tuple(behavior_descriptors)
        if len(set(self.behavior_descriptors)) != len(self.behavior_descriptors):
            raise ValueError("behavior descriptors must be distinct")
        if set(self.behavior_descriptors) - DESCRIPTOR_REGISTRY.keys():
            raise ValueError("unregistered behavior descriptor")
        self.descriptor_bounds = tuple(tuple(map(float, b)) for b in descriptor_bounds)
        self.archive_bins = ((archive_bins,) * len(self.behavior_descriptors)
                             if isinstance(archive_bins, int) else tuple(archive_bins))
        if not (len(self.behavior_descriptors) == len(self.descriptor_bounds) == len(self.archive_bins)):
            raise ValueError("descriptor names, bounds and bin counts must align")
        if not self.behavior_descriptors or any(type(b) is not int or b < 1 for b in self.archive_bins):
            raise ValueError("archive needs positive bin counts")
        if any(len(b) != 2 or not np.all(np.isfinite(b)) or b[0] >= b[1]
               for b in self.descriptor_bounds):
            raise ValueError("descriptor bounds must be finite increasing pairs")
        self.mutation_rates = dict({
            "remove_local_bond": 0.025, "add_local_bond": 0.025, "edge_swap": 0.15,
            "local_rewiring": 0.35, "diagonal_flip": 0.1,
            "local_motif_replacement": 0.2, "boundary_rewiring": 0.15,
        } if mutation_rates is None else mutation_rates)
        if set(self.mutation_rates) - MUTATION_REGISTRY.keys():
            raise ValueError("unregistered mutation")
        rates = np.asarray(list(self.mutation_rates.values()), float)
        if not len(rates) or not np.all(np.isfinite(rates)) or np.any(rates < 0) or rates.sum() <= 0:
            raise ValueError("mutation rates must be finite nonnegative weights")
        self._mutation_probabilities = rates / rates.sum()
        if allocation is None:
            allocation = {"exploitation": 0.6, "uncertainty": 0.2, "novelty": 0.2}
        if isinstance(allocation, dict):
            weights = [allocation.get(k, 0) for k in ("exploitation", "uncertainty", "novelty")]
        else:
            weights = list(allocation)
        allocation_values = np.asarray(weights, float)
        if len(allocation_values) != 3 or not np.all(np.isfinite(allocation_values)) or np.any(allocation_values < 0) or allocation_values.sum() <= 0:
            raise ValueError("allocation must contain three nonnegative weights")
        self.allocation = dict(zip(("exploitation", "uncertainty", "novelty"),
                                   (allocation_values / allocation_values.sum()).tolist(), strict=True))
        if not 0 < family_fraction <= 1:
            raise ValueError("family_fraction must be in (0,1]")
        self.family_fraction = family_fraction
        self.experiment_id, self.code_version = experiment_id, code_version
        self.proposal_limit: int | None = None
        self.initialize()

    def initialize(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.generation = 0
        self.generated = 0
        self.invalid = 0
        self.duplicates = 0
        self.near_duplicates = 0
        self.archive: dict[str, dict[str, Any]] = {}
        self.history: list[dict[str, Any]] = []
        self.seen: dict[str, dict[str, Any]] = {}
        self._orbits: dict[str, tuple[frozenset[tuple[int, int]], ...]] = {}
        self._rejections: list[dict[str, Any]] = []
        self.selection_counts: Counter[str] = Counter()

    def cell(self, descriptors: dict[str, float]) -> tuple[int, ...] | None:
        indices = []
        for name, (low, high), bins in zip(self.behavior_descriptors, self.descriptor_bounds,
                                          self.archive_bins, strict=True):
            value = descriptors.get(name)
            if value is None or not np.isfinite(value) or not low <= value <= high:
                return None
            indices.append(min(bins - 1, int((value - low) / (high - low) * bins)))
        return tuple(indices)

    def _parent(self) -> dict[str, Any] | None:
        if self.name == "random" or not self.history:
            return None
        if self.name in ("map_elites", "surrogate_map_elites") and self.archive:
            # JSON stores object keys canonically. Parent choice must use the same
            # cell ordering before and after a persisted resume.
            elites = [self.archive[key] for key in sorted(self.archive)]
            # Equal cell probability prevents the most populated region dominating breeding.
            return elites[int(self.rng.integers(len(elites)))]
        population = [r for r in self.history if _exact_eligible(r)]
        if not population:
            return None
        indices = self.rng.choice(len(population), min(3, len(population)), replace=False)
        return max((population[int(i)] for i in indices), key=lambda r: (r["score"], r["id"]))

    def propose(self, count: int, *, max_attempts: int | None = None) -> list[dict[str, Any]]:
        if count < 0:
            raise ValueError("proposal count must be nonnegative")
        limit = max_attempts if max_attempts is not None else max(1, count * 30)
        if self.proposal_limit is not None:
            limit = min(limit, self.proposal_limit)
        proposed: list[dict[str, Any]] = []
        for _ in range(max(0, limit)):
            if len(proposed) >= count:
                break
            self.generated += 1
            parent = self._parent()
            proposal_seed = int(self.rng.integers(0, 2**32))
            rng = np.random.default_rng(proposal_seed)
            if parent is None:
                geometry = self.space.sample(rng)
                mutation = {"operator": "random_wiring", "added": [], "removed": []}
            else:
                operator = str(rng.choice(tuple(self.mutation_rates), p=self._mutation_probabilities))
                geometry, mutation = self.space.mutate(geometry_from_payload(parent["geometry"]),
                                                       rng, operator)
            geometry_hash = self.space.hash(geometry)
            record: dict[str, Any] = {
                "id": geometry_hash, "geometry_hash": geometry_hash,
                "geometry": geometry_to_payload(geometry), "descriptors": {},
                "origin": "proposed", "validation_state": "PROPOSED", "score": None,
                "generation": self.generation, "parent": None if parent is None else parent["id"],
                "lineage": [] if parent is None else [*parent.get("lineage", []), parent["id"]],
                "mutation": mutation, "seed": proposal_seed,
                "created_at": datetime.now(UTC).isoformat(),
                "code_version": self.code_version, "experiment_id": self.experiment_id,
                "family": "fixed_connectivity", "attempt": self.generated,
                "lineage_family": geometry_hash if parent is None else
                    parent.get("lineage_family", parent["id"]),
            }
            reasons = self.space.validate(geometry)
            if reasons:
                self.invalid += 1
                reason = ",".join(reasons)
            elif geometry_hash in self.seen:
                self.duplicates += 1
                reason = "duplicate"
            else:
                edge_set = frozenset(self.space.edges(geometry))
                near = any(min(len(edge_set ^ variant) / max(1, len(edge_set | variant))
                               for variant in orbit) < self.space.minimum_distance
                           for orbit in self._orbits.values())
                if near:
                    self.near_duplicates += 1
                    reason = "near_duplicate"
                else:
                    reason = ""
            if reason:
                record.update(id=f"{geometry_hash}-rejected-{self.generated}",
                              validation_state="REJECTED", rejection_reason=reason)
                self._rejections.append(record)
                continue
            record["descriptors"] = compute_descriptors(geometry)
            record["archive_cell"] = self.cell(record["descriptors"])
            self.seen[geometry_hash] = copy.deepcopy(record)
            self._orbits[geometry_hash] = self.space.orbit(geometry)
            proposed.append(record)
        self.generation += 1
        return proposed

    def drain_rejections(self) -> list[dict[str, Any]]:
        result, self._rejections = self._rejections, []
        return result

    def observe(self, candidate: dict[str, Any], result: dict[str, Any]) -> None:
        merged = {**candidate, **result}
        # Eigensystems and detailed stage records belong in the dataset, not in
        # the repeatedly checkpointed breeding population and elite archive.
        record = copy.deepcopy({key: value for key, value in merged.items() if key in {
            "id", "candidate_id", "geometry_hash", "geometry", "descriptors", "origin",
            "validation_state", "score", "generation", "parent", "lineage", "mutation",
            "seed", "created_at", "created", "completed", "code_version", "experiment_id",
            "family", "lineage_family", "attempt", "archive_cell", "prediction",
            "surrogate_prediction", "acquisition", "raw_metrics", "metrics", "baseline",
            "validation_results", "warnings", "exact_evaluation_count",
        }})
        if record.get("score") is not None and (
            not isinstance(record["score"], (int, float)) or isinstance(record["score"], bool)
            or not np.isfinite(record["score"])
        ):
            record.update(score=None, validation_state="FAILED")
            record["warnings"] = [*record.get("warnings", []), "Nonfinite/invalid exact score rejected"]
        if any(r["id"] == record["id"] for r in self.history):
            return
        self.history.append(record)
        if not _exact_eligible(record):
            return
        geometry = geometry_from_payload(record["geometry"])
        key = self.space.hash(geometry)
        if self.space.validate(geometry):
            return
        self.seen[key] = copy.deepcopy(record)
        self._orbits[key] = self.space.orbit(geometry)
        cell = self.cell(record["descriptors"])
        if cell is None:
            return
        cell_key = ",".join(map(str, cell))
        record["archive_cell"] = list(cell)
        old = self.archive.get(cell_key)
        if old is None or record["score"] > old["score"]:
            self.archive[cell_key] = record

    def _novelty(self, candidate: dict[str, Any]) -> float:
        cell = self.cell(candidate["descriptors"])
        occupied = Counter(tuple(r.get("archive_cell") or ()) for r in self.history)
        coverage = 1 / (1 + occupied.get(cell or (), 0))
        if not self.history:
            return coverage
        names = self.behavior_descriptors
        scale = np.array([b - a for a, b in self.descriptor_bounds])
        vector = np.array([candidate["descriptors"][k] for k in names])
        distances = [float(np.linalg.norm((vector - np.array([r["descriptors"][k] for k in names]))
                                          / scale)) for r in self.history if r.get("descriptors")]
        return coverage + (min(distances) if distances else 0)

    def select(self, proposals: list[dict[str, Any]], count: int,
               surrogate: Any = None) -> list[dict[str, Any]]:
        if count < 0:
            raise ValueError("selection count must be nonnegative")
        count = min(count, len(proposals))
        if not self.uses_surrogate or surrogate is None or not surrogate.fitted:
            result = proposals[:count]
            for candidate in result:
                candidate["acquisition"] = {"strategy": "warm_start" if
                    self.uses_surrogate else self.name, "score": None}
            return result
        predictions = surrogate.predict(proposals)
        for candidate, prediction in zip(proposals, predictions, strict=True):
            candidate["surrogate_prediction"] = prediction
            candidate["prediction"] = prediction
            candidate["validation_state"] = "SURROGATE_SCREENED"
        remaining = list(proposals)
        selected: list[dict[str, Any]] = []
        cells: Counter[tuple[int, ...] | None] = Counter()
        families: Counter[str] = Counter()
        lineages: Counter[str] = Counter()
        cap = max(1, int(np.ceil(count * self.family_fraction)))
        for _ in range(count):
            total = sum(self.selection_counts.values()) + 1
            mode = max(self.allocation, key=lambda k:
                       self.allocation[k] * total - self.selection_counts[k])
            family_eligible = [r for r in remaining if families[r.get("family", "unknown")] < cap]
            eligible = family_eligible or remaining
            lineage_eligible = [r for r in eligible if lineages[r.get("lineage_family", r["id"])] < cap]
            eligible = lineage_eligible or eligible
            cell_eligible = [r for r in eligible if cells[self.cell(r["descriptors"])] < cap]
            eligible = cell_eligible or eligible

            def acquisition(record: dict[str, Any], mode: str = mode) -> tuple[float, str]:
                prediction = record["prediction"]
                if mode == "exploitation":
                    # Penalize unsupported extrapolation for exploitation; such candidates
                    # remain available to uncertainty and novelty selections.
                    value = prediction["score"] - (prediction["uncertainty"] if prediction["ood"] else 0)
                elif mode == "uncertainty":
                    value = prediction["uncertainty"]
                else:
                    value = self._novelty(record)
                return float(value), record["id"]

            winner = max(eligible, key=acquisition)
            value = acquisition(winner)[0]
            winner["acquisition"] = {"strategy": mode, "score": value,
                                      "novelty": self._novelty(winner)}
            selected.append(winner)
            remaining.remove(winner)
            cells[self.cell(winner["descriptors"])] += 1
            families[winner.get("family", "unknown")] += 1
            lineages[winner.get("lineage_family", winner["id"])] += 1
            self.selection_counts[mode] += 1
        return selected

    def checkpoint(self) -> dict[str, Any]:
        return copy.deepcopy({"version": 1, "strategy": self.name, "rng": self.rng.bit_generator.state,
            "generation": self.generation, "generated": self.generated, "invalid": self.invalid,
            "duplicates": self.duplicates, "near_duplicates": self.near_duplicates,
            "archive": self.archive, "history": self.history, "seen": self.seen,
            "rejections": self._rejections, "selection_counts": dict(self.selection_counts)})

    def resume(self, checkpoint: dict[str, Any]) -> SearchStrategy:
        if checkpoint.get("version") != 1 or checkpoint.get("strategy") != self.name:
            raise ValueError("incompatible strategy checkpoint")
        state = copy.deepcopy(checkpoint)
        self.rng.bit_generator.state = state["rng"]
        for key in ("generation", "generated", "invalid", "duplicates", "near_duplicates",
                    "archive", "history", "seen"):
            setattr(self, key, state[key])
        self._rejections = state["rejections"]
        self.selection_counts = Counter(state["selection_counts"])
        self._orbits = {key: self.space.orbit(geometry_from_payload(record["geometry"]))
                        for key, record in self.seen.items()}
        if any(not _exact_eligible(record) for record in self.archive.values()):
            raise ValueError("checkpoint archive contains unvalidated surrogate or failed scores")
        return self

    def summarize(self) -> dict[str, Any]:
        cells = Counter(tuple(r.get("archive_cell") or ()) for r in self.history)
        families = Counter(r.get("family", "unknown") for r in self.history)
        lineages = Counter(r.get("lineage_family", r["id"]) for r in self.history)
        total = sum(cells.values())
        proportions = np.array(list(cells.values()), float) / max(1, total)
        diversity = float(-np.sum(proportions * np.log(proportions))) if total else 0.0
        exact = [r for r in self.history if _exact_eligible(r)]
        return {"strategy": self.name, "generation": self.generation,
                "generated": self.generated, "accepted": len(self.seen), "invalid": self.invalid,
                "duplicates": self.duplicates, "near_duplicates": self.near_duplicates,
                "duplicate_rate": self.duplicates / max(1, self.generated),
                "near_duplicate_rate": self.near_duplicates / max(1, self.generated),
                "invalid_rate": self.invalid / max(1, self.generated),
                "archive_size": len(self.archive),
                "archive_coverage": len(self.archive) / int(np.prod(self.archive_bins)),
                "archive_total_cells": int(np.prod(self.archive_bins)), "diversity": diversity,
                "family_concentration": max(families.values(), default=0) / max(1, total),
                "family_distribution": dict(families),
                "lineage_concentration": max(lineages.values(), default=0) / max(1, total),
                "lineage_distribution": dict(lineages),
                "cell_concentration": float(proportions.max()) if total else 0.0,
                "cell_distribution": {str(k): v for k, v in cells.items()},
                "allocation": self.allocation, "selection_counts": dict(self.selection_counts),
                "exact_evaluations": len(exact),
                "best_score": max((r["score"] for r in exact), default=None)}


def _exact_eligible(record: dict[str, Any]) -> bool:
    score = record.get("score")
    return (record.get("origin") == "exact" and isinstance(score, (float, int))
            and not isinstance(score, bool) and np.isfinite(score)
            and record.get("validation_state") in {"EXACT_EVALUATED", "TOPOLOGY_VALIDATED",
                "MAJORANA_VALIDATED", "ROBUSTNESS_VALIDATED", "FINITE_SIZE_VALIDATED"})


class RandomSearch(SearchStrategy):
    name = "random"


class EvolutionarySearch(SearchStrategy):
    name = "evolution"


class MAPElites(SearchStrategy):
    name = "map_elites"


class SurrogateMAPElites(SearchStrategy):
    name = "surrogate_map_elites"
    uses_surrogate = True


for _strategy in (RandomSearch, EvolutionarySearch, MAPElites, SurrogateMAPElites):
    register_strategy(_strategy.name, _strategy)
