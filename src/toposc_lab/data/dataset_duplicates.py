"""Conservative exact and relation-aware geometry duplicate detection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Literal, TypeAlias

import numpy as np

from toposc_lab.data.dataset_schema import DatasetRecord, GeometryRecord
from toposc_lab.geometry import Geometry, canonical_graph_hash

CoordinateRelation: TypeAlias = Literal["ignore", "pairwise_distance"]


class DuplicateRelation(str, Enum):
    """Strength of evidence returned by duplicate assessment."""

    EXACT_SNAPSHOT = "exact_snapshot"
    GRAPH_ISOMORPHIC = "graph_isomorphic"
    COORDINATE_NEAR_DUPLICATE = "coordinate_near_duplicate"
    POSSIBLE_FINGERPRINT_COLLISION = "possible_fingerprint_collision"
    DISTINCT = "distinct"


@dataclass(frozen=True, slots=True)
class DuplicatePolicy:
    """Fields that define a scientifically intended geometry relation.

    Coordinates may be ignored, or compared through all pairwise distances so
    translations, rotations, reflections, and allowed site relabelings do not
    create artificial families. Faces and arbitrary metadata are intentionally
    outside this graph-relation policy and remain protected by exact snapshot IDs.
    """

    include_site_types: bool = True
    include_edge_types: bool = True
    include_boundary: bool = True
    coordinate_relation: CoordinateRelation = "ignore"
    coordinate_tolerance: float = 0.0
    max_backtracking_states: int = 100_000

    def __post_init__(self) -> None:
        for name in ("include_site_types", "include_edge_types", "include_boundary"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.coordinate_relation not in ("ignore", "pairwise_distance"):
            raise ValueError("coordinate_relation is unsupported")
        if isinstance(self.coordinate_tolerance, bool) or not isinstance(
            self.coordinate_tolerance, Real
        ):
            raise TypeError("coordinate_tolerance must be a real number")
        tolerance = float(self.coordinate_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("coordinate_tolerance must be finite and nonnegative")
        if isinstance(self.max_backtracking_states, bool) or not isinstance(
            self.max_backtracking_states, Integral
        ):
            raise TypeError("max_backtracking_states must be an integer")
        if int(self.max_backtracking_states) < 1:
            raise ValueError("max_backtracking_states must be positive")
        object.__setattr__(self, "coordinate_tolerance", tolerance)
        object.__setattr__(self, "max_backtracking_states", int(self.max_backtracking_states))


@dataclass(frozen=True, slots=True)
class DuplicateAssessment:
    """A duplicate decision that preserves uncertainty instead of guessing."""

    relation: DuplicateRelation
    equivalent: bool | None
    reason: str
    site_mapping: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class DuplicateGroups:
    """Deterministic connected duplicate groups and uncertain comparisons."""

    groups: tuple[tuple[str, ...], ...]
    uncertain_pairs: tuple[tuple[str, str], ...]


_DEFAULT_DUPLICATE_POLICY = DuplicatePolicy()


def assess_geometry_duplicate(
    first: GeometryRecord | Geometry,
    second: GeometryRecord | Geometry,
    *,
    policy: DuplicatePolicy = _DEFAULT_DUPLICATE_POLICY,
) -> DuplicateAssessment:
    """Assess exact equality, configured isomorphism, and coordinate proximity."""
    if not isinstance(policy, DuplicatePolicy):
        raise TypeError("policy must be DuplicatePolicy")
    first_record = first if isinstance(first, GeometryRecord) else GeometryRecord.from_geometry(first)
    second_record = second if isinstance(second, GeometryRecord) else GeometryRecord.from_geometry(second)
    if first_record.exact_id == second_record.exact_id:
        exact_mapping = tuple(range(first_record.to_geometry().n_sites))
        return DuplicateAssessment(
            relation=DuplicateRelation.EXACT_SNAPSHOT,
            equivalent=True,
            reason="exact serialized geometry identifiers match",
            site_mapping=exact_mapping,
        )

    first_geometry = first_record.to_geometry()
    second_geometry = second_record.to_geometry()
    first_fingerprint = _policy_fingerprint(first_geometry, policy)
    second_fingerprint = _policy_fingerprint(second_geometry, policy)
    if first_fingerprint != second_fingerprint:
        return DuplicateAssessment(
            relation=DuplicateRelation.DISTINCT,
            equivalent=False,
            reason="configured canonical graph fingerprints differ",
        )

    mapping, exhausted = _isomorphism_mapping(first_geometry, second_geometry, policy)
    if exhausted:
        return DuplicateAssessment(
            relation=DuplicateRelation.POSSIBLE_FINGERPRINT_COLLISION,
            equivalent=None,
            reason="exact isomorphism search exceeded max_backtracking_states",
        )
    if mapping is None:
        return DuplicateAssessment(
            relation=DuplicateRelation.DISTINCT,
            equivalent=False,
            reason="fingerprints match but exact graph isomorphism was disproved",
        )
    if policy.coordinate_relation == "ignore":
        return DuplicateAssessment(
            relation=DuplicateRelation.GRAPH_ISOMORPHIC,
            equivalent=True,
            reason="exact configured graph isomorphism was established",
            site_mapping=mapping,
        )
    if not _coordinates_match(first_geometry, second_geometry, mapping, policy.coordinate_tolerance):
        return DuplicateAssessment(
            relation=DuplicateRelation.DISTINCT,
            equivalent=False,
            reason="graphs are isomorphic but pairwise coordinate distances differ",
            site_mapping=mapping,
        )
    return DuplicateAssessment(
        relation=DuplicateRelation.COORDINATE_NEAR_DUPLICATE,
        equivalent=True,
        reason="graph isomorphism and pairwise coordinate tolerance were satisfied",
        site_mapping=mapping,
    )


def find_duplicate_groups(
    records: tuple[DatasetRecord, ...],
    *,
    policy: DuplicatePolicy = _DEFAULT_DUPLICATE_POLICY,
) -> DuplicateGroups:
    """Group proven geometry duplicates; uncertain fingerprint collisions stay separate."""
    items = tuple(records)
    if any(not isinstance(record, DatasetRecord) for record in items):
        raise TypeError("records must contain only DatasetRecord values")
    ordered = tuple(sorted(items, key=lambda record: record.record_id))
    parents = list(range(len(ordered)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    uncertain: list[tuple[str, str]] = []
    for first_index, first in enumerate(ordered):
        for second_index in range(first_index + 1, len(ordered)):
            second = ordered[second_index]
            assessment = assess_geometry_duplicate(first.geometry, second.geometry, policy=policy)
            if assessment.equivalent is True:
                first_root = root(first_index)
                second_root = root(second_index)
                if first_root != second_root:
                    parents[second_root] = first_root
            elif assessment.equivalent is None:
                uncertain.append((first.record_id, second.record_id))
    grouped: dict[int, list[str]] = {}
    for index, record in enumerate(ordered):
        grouped.setdefault(root(index), []).append(record.record_id)
    groups = tuple(sorted(tuple(values) for values in grouped.values()))
    return DuplicateGroups(groups=groups, uncertain_pairs=tuple(uncertain))


def _policy_fingerprint(geometry: Geometry, policy: DuplicatePolicy) -> str:
    return str(
        canonical_graph_hash(
            geometry,
            include_site_types=policy.include_site_types,
            include_edge_types=policy.include_edge_types,
            include_boundary=policy.include_boundary,
        )
    )


def _isomorphism_mapping(
    first: Geometry,
    second: Geometry,
    policy: DuplicatePolicy,
) -> tuple[tuple[int, ...] | None, bool]:
    if first.n_sites != second.n_sites or first.n_edges != second.n_edges:
        return None, False
    first_adjacency = _adjacency(first, policy)
    second_adjacency = _adjacency(second, policy)
    first_nodes = tuple(_node_signature(first, site, policy, first_adjacency) for site in first.site_indices)
    second_nodes = tuple(_node_signature(second, site, policy, second_adjacency) for site in second.site_indices)
    if Counter(first_nodes) != Counter(second_nodes):
        return None, False
    candidates = {
        site: tuple(target for target in second.site_indices if second_nodes[target] == first_nodes[site])
        for site in first.site_indices
    }
    mapping: dict[int, int] = {}
    used: set[int] = set()
    state_count = 0

    def search() -> tuple[int, ...] | None:
        nonlocal state_count
        state_count += 1
        if state_count > policy.max_backtracking_states:
            raise _SearchLimit
        if len(mapping) == first.n_sites:
            completed = tuple(mapping[site] for site in first.site_indices)
            if policy.coordinate_relation == "pairwise_distance" and not _coordinates_match(
                first,
                second,
                completed,
                policy.coordinate_tolerance,
            ):
                return None
            return completed
        site = min(
            (item for item in first.site_indices if item not in mapping),
            key=lambda item: (sum(target not in used for target in candidates[item]), item),
        )
        for target in candidates[site]:
            if target in used or not _mapping_consistent(
                site, target, mapping, first_adjacency, second_adjacency
            ):
                continue
            mapping[site] = target
            used.add(target)
            result = search()
            if result is not None:
                return result
            used.remove(target)
            del mapping[site]
        return None

    try:
        return search(), False
    except _SearchLimit:
        return None, True


class _SearchLimit(Exception):
    pass


def _adjacency(
    geometry: Geometry,
    policy: DuplicatePolicy,
) -> tuple[dict[int, tuple[str | None, bool]], ...]:
    adjacency: list[dict[int, tuple[str | None, bool]]] = [
        {} for _ in geometry.site_indices
    ]
    for edge in geometry.edges:
        label = (
            edge.edge_type if policy.include_edge_types else None,
            edge.boundary_crossing if policy.include_boundary else False,
        )
        adjacency[edge.source][edge.target] = label
        adjacency[edge.target][edge.source] = label
    return tuple(adjacency)


def _node_signature(
    geometry: Geometry,
    site: int,
    policy: DuplicatePolicy,
    adjacency: tuple[dict[int, tuple[str | None, bool]], ...],
) -> tuple[object, ...]:
    site_type = None
    if policy.include_site_types and geometry.site_types is not None:
        site_type = geometry.site_types[site]
    boundary = policy.include_boundary and site in geometry.boundary_sites
    return site_type, boundary, len(adjacency[site]), tuple(sorted(Counter(adjacency[site].values()).items(), key=repr))


def _mapping_consistent(
    site: int,
    target: int,
    mapping: dict[int, int],
    first_adjacency: tuple[dict[int, tuple[str | None, bool]], ...],
    second_adjacency: tuple[dict[int, tuple[str | None, bool]], ...],
) -> bool:
    for mapped_site, mapped_target in mapping.items():
        first_edge = first_adjacency[site].get(mapped_site)
        second_edge = second_adjacency[target].get(mapped_target)
        if first_edge != second_edge:
            return False
    return True


def _coordinates_match(
    first: Geometry,
    second: Geometry,
    mapping: tuple[int, ...],
    tolerance: float,
) -> bool:
    if first.coordinates is None or second.coordinates is None:
        return first.coordinates is None and second.coordinates is None
    if first.coordinates.shape[1] != second.coordinates.shape[1]:
        return False
    reordered = second.coordinates[np.asarray(mapping, dtype=int)]
    first_distances = np.linalg.norm(first.coordinates[:, None, :] - first.coordinates[None, :, :], axis=2)
    second_distances = np.linalg.norm(reordered[:, None, :] - reordered[None, :, :], axis=2)
    return bool(np.allclose(first_distances, second_distances, rtol=0.0, atol=tolerance))
