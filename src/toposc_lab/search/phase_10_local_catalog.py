"""Physics-free local rewire catalog derived from the accepted measurement panel."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from functools import cache, lru_cache
from itertools import combinations
from math import isclose
from typing import Any, cast

import numpy as np

from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.geometry.validation import validate_geometry
from toposc_lab.search.geometry_genome import GeometryGenome
from toposc_lab.search.phase_9_8_evaluation import _segments_intersect
from toposc_lab.search.phase_10_measurement_validation import (
    ARMS,
    FULL_SIZES,
    OFFSETS,
    Arm,
    intervention_definition,
    measurement_reference,
    validate_measurement_geometry,
)

Edge = tuple[int, int]
EdgePair = tuple[Edge, Edge]
Operation = tuple[EdgePair, EdgePair]


@dataclass(frozen=True, slots=True)
class LocalCatalogOrigin:
    """One fixed measurement patch that generated a candidate operation."""

    offset: int
    arm: Arm
    measurement_sites: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LocalCatalogCandidate:
    """One unique, fully validated geometry and all of its patch origins."""

    n: int
    geometry_id: str
    genome: GeometryGenome
    removed_edges: tuple[Edge, Edge]
    added_edges: tuple[Edge, Edge]
    affected_sites: tuple[int, int, int, int]
    origins: tuple[LocalCatalogOrigin, ...]
    known_measurement_variant: bool
    rotation_signature: str
    dihedral_signature: str


@dataclass(frozen=True, slots=True)
class LocalCatalogSizeCounts:
    """Auditable enumeration counts for one square size."""

    n: int
    patch_origins: int
    alternative_pairing_proposals: int
    proposal_rejections: tuple[tuple[str, int], ...]
    prefiltered_origin_operations: int
    unique_prefiltered_operations: int
    duplicate_prefiltered_origins: int
    validation_rejections: tuple[tuple[str, int], ...]
    valid_origin_memberships: int
    valid_candidates: int
    known_measurement_variants: int
    rotation_classes: int
    dihedral_classes: int


@dataclass(frozen=True, slots=True)
class LocalCatalog:
    """Complete in-memory catalog; it contains no model or physics result."""

    candidates: tuple[LocalCatalogCandidate, ...]
    sizes: tuple[LocalCatalogSizeCounts, ...]


@lru_cache(maxsize=1)
def build_local_rewire_catalog() -> LocalCatalog:
    """Enumerate the bounded local rewire space without evaluating physics."""
    all_candidates: list[LocalCatalogCandidate] = []
    all_counts: list[LocalCatalogSizeCounts] = []
    for n in FULL_SIZES:
        candidates, counts = _build_size_catalog(n)
        all_candidates.extend(candidates)
        all_counts.append(counts)
    return LocalCatalog(tuple(all_candidates), tuple(all_counts))


def local_catalog_summary(
    catalog: LocalCatalog | None = None, *, include_candidates: bool = False
) -> dict[str, Any]:
    """Return a JSON-compatible audit summary, optionally with all exact IDs."""
    catalog = build_local_rewire_catalog() if catalog is None else catalog
    size_rows = [
        {
            "n": counts.n,
            "patch_origins": counts.patch_origins,
            "alternative_pairing_proposals": counts.alternative_pairing_proposals,
            "proposal_rejections": dict(counts.proposal_rejections),
            "prefiltered_origin_operations": counts.prefiltered_origin_operations,
            "unique_prefiltered_operations": counts.unique_prefiltered_operations,
            "duplicate_prefiltered_origins": counts.duplicate_prefiltered_origins,
            "validation_rejections": dict(counts.validation_rejections),
            "valid_origin_memberships": counts.valid_origin_memberships,
            "valid_candidates": counts.valid_candidates,
            "known_measurement_variants": counts.known_measurement_variants,
            "rotation_classes": counts.rotation_classes,
            "dihedral_classes": counts.dihedral_classes,
        }
        for counts in catalog.sizes
    ]
    summary: dict[str, Any] = {
        "kind": "physics_free_local_rewire_catalog",
        "physics_evaluations": 0,
        "sizes": size_rows,
        "total_valid_candidates": len(catalog.candidates),
        "total_known_measurement_variants": sum(
            candidate.known_measurement_variant for candidate in catalog.candidates
        ),
        "candidate_ids_sha256": hashlib.sha256(
            "\n".join(candidate.geometry_id for candidate in catalog.candidates).encode("ascii")
        ).hexdigest(),
    }
    if include_candidates:
        summary["candidates"] = [_candidate_record(candidate) for candidate in catalog.candidates]
    return summary


def render_local_catalog_report(catalog: LocalCatalog | None = None) -> str:
    """Render concise human-readable counts for the CLI."""
    summary = local_catalog_summary(catalog)
    lines = [
        "Phase-10 localer Geometriekatalog (keine Physikberechnung)",
        "",
    ]
    for row in summary["sizes"]:
        lines.append(
            f"n={row['n']}: {row['valid_candidates']} gültige eindeutige Kandidaten, "
            f"{row['known_measurement_variants']} bekannte Messvarianten, "
            f"{row['rotation_classes']} Rotationsklassen"
        )
        lines.append(
            f"  Vorschläge={row['alternative_pairing_proposals']}, "
            f"Bestandsablehnung={row['proposal_rejections']['added_edge_already_exists']}, "
            f"Längen-Ablehnung={row['proposal_rejections']['added_edge_length_not_sqrt2']}, "
            f"Kreuzungs-Ablehnung={row['validation_rejections']['straight_edge_crossing']}"
        )
    lines.extend(
        (
            "",
            f"Gesamt: {summary['total_valid_candidates']} gültige Kandidaten",
            f"ID-Listen-SHA-256: {summary['candidate_ids_sha256']}",
            "Hamiltonians/Solveraufrufe: 0",
            "Mit --details werden alle exakten Geometrie-IDs als JSON ausgegeben.",
        )
    )
    return "\n".join(lines)


def validate_local_catalog_candidate(
    reference: Geometry,
    geometry: Geometry,
    removed_edges: tuple[Edge, Edge],
    added_edges: tuple[Edge, Edge],
    n: int,
) -> tuple[str, ...]:
    """Validate the complete fixed-reference delta contract for one candidate."""
    issues: list[str] = []
    if exact_geometry_id(reference) != _validated_reference_id(n):
        raise ValueError("local catalog reference must be the exact validated square")
    generic = validate_geometry(geometry, require_connected=True)
    if not generic.is_valid:
        issues.extend(issue.code for issue in generic.errors)
    reference_pairs = _edge_pairs(reference)
    candidate_pairs = _edge_pairs(geometry)
    removed = set(removed_edges)
    added = set(added_edges)
    if len(removed) != 2 or len(added) != 2:
        issues.append("two_edge_budget")
    removed_sites = {site for edge in removed_edges for site in edge}
    added_sites = {site for edge in added_edges for site in edge}
    if len(removed_sites) != 4:
        issues.append("four_distinct_endpoints")
    if added_sites != removed_sites or len(added_sites) != 4:
        issues.append("endpoint_set_changed")
    if not removed.issubset(reference_pairs):
        issues.append("removed_edge_missing")
    if added.intersection(reference_pairs):
        issues.append("added_edge_already_exists")
    if candidate_pairs != (reference_pairs - removed) | added:
        issues.append("edge_delta")
    if geometry.n_sites != reference.n_sites or geometry.n_edges != reference.n_edges:
        issues.append("size_budget")
    if not _same_nonedge_fields(reference, geometry):
        issues.append("nonedge_fields_changed")
    if not _edge_semantics_are_fixed(geometry):
        issues.append("edge_semantics_changed")
    if _degrees(geometry) != _degrees(reference):
        issues.append("degree_sequence_changed")
    coordinates = geometry.coordinates
    if coordinates is None or any(
        not isclose(
            float(np.dot(coordinates[target] - coordinates[source],
                         coordinates[target] - coordinates[source])),
            2.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for source, target in added_edges
    ):
        issues.append("added_edge_length_not_sqrt2")
    if coordinates is not None and _added_edges_cross(coordinates, candidate_pairs, added_edges):
        issues.append("straight_edge_crossing")
    return tuple(dict.fromkeys(issues))


def _build_size_catalog(
    n: int,
) -> tuple[tuple[LocalCatalogCandidate, ...], LocalCatalogSizeCounts]:
    reference = measurement_reference(n)
    reference_pairs = _edge_pairs(reference)
    operations: dict[Operation, list[LocalCatalogOrigin]] = {}
    rejections: Counter[str] = Counter()
    proposals = 0
    prefiltered_origins = 0
    known = {
        _operation(
            intervention_definition(n, offset, arm)["removed_edges"],
            intervention_definition(n, offset, arm)["added_edges"],
        )
        for offset in OFFSETS
        for arm in ARMS
    }
    for offset in OFFSETS:
        for arm in ARMS:
            definition = intervention_definition(n, offset, arm)
            origin = LocalCatalogOrigin(offset, arm, definition["measurement_sites"])
            patch = set(origin.measurement_sites)
            local_edges = sorted(edge for edge in reference_pairs if set(edge) <= patch)
            for first, second in combinations(local_edges, 2):
                sites = {*first, *second}
                if len(sites) != 4:
                    continue
                ordered_sites = cast(tuple[int, int, int, int], tuple(sorted(sites)))
                removed = _edge_pair(first, second)
                for added in _alternative_matchings(ordered_sites, removed):
                    proposals += 1
                    if any(edge in reference_pairs for edge in added):
                        rejections["added_edge_already_exists"] += 1
                        continue
                    if not _edges_have_squared_length(reference, added, 2.0):
                        rejections["added_edge_length_not_sqrt2"] += 1
                        continue
                    prefiltered_origins += 1
                    operations.setdefault((removed, added), []).append(origin)

    candidates: list[LocalCatalogCandidate] = []
    validation_rejections: Counter[str] = Counter()
    for (removed, added), origins in sorted(operations.items()):
        geometry = _rewired_geometry(reference, removed, added)
        issues = validate_local_catalog_candidate(reference, geometry, removed, added, n)
        if issues:
            validation_rejections.update(issues)
            continue
        genome = GeometryGenome.from_geometry(geometry)
        geometry_id = exact_geometry_id(geometry)
        if exact_geometry_id(genome.to_geometry()) != geometry_id:
            raise AssertionError("catalog genome round trip changed exact geometry identity")
        affected_sites = cast(
            tuple[int, int, int, int], tuple(sorted(site for edge in removed for site in edge))
        )
        candidates.append(
            LocalCatalogCandidate(
                n=n,
                geometry_id=geometry_id,
                genome=genome,
                removed_edges=removed,
                added_edges=added,
                affected_sites=affected_sites,
                origins=tuple(origins),
                known_measurement_variant=(removed, added) in known,
                rotation_signature=_symmetry_signature(n, removed, added, reflect=False),
                dihedral_signature=_symmetry_signature(n, removed, added, reflect=True),
            )
        )
    if len({candidate.geometry_id for candidate in candidates}) != len(candidates):
        raise AssertionError("exact geometry IDs did not deduplicate the local catalog")
    counts = LocalCatalogSizeCounts(
        n=n,
        patch_origins=len(OFFSETS) * len(ARMS),
        alternative_pairing_proposals=proposals,
        proposal_rejections=tuple(sorted(rejections.items())),
        prefiltered_origin_operations=prefiltered_origins,
        unique_prefiltered_operations=len(operations),
        duplicate_prefiltered_origins=prefiltered_origins - len(operations),
        validation_rejections=tuple(sorted(validation_rejections.items())),
        valid_origin_memberships=sum(len(candidate.origins) for candidate in candidates),
        valid_candidates=len(candidates),
        known_measurement_variants=sum(candidate.known_measurement_variant for candidate in candidates),
        rotation_classes=len({candidate.rotation_signature for candidate in candidates}),
        dihedral_classes=len({candidate.dihedral_signature for candidate in candidates}),
    )
    return tuple(candidates), counts


def _alternative_matchings(
    sites: tuple[int, int, int, int], removed: EdgePair
) -> tuple[EdgePair, ...]:
    first, second, third, fourth = sites
    matchings = (
        ((first, second), (third, fourth)),
        ((first, third), (second, fourth)),
        ((first, fourth), (second, third)),
    )
    alternatives = tuple(_edge_pair(*matching) for matching in matchings)
    return tuple(matching for matching in alternatives if matching != removed)


def _operation(removed: Any, added: Any) -> Operation:
    removed_edges = tuple(_normalize_edge(edge) for edge in removed)
    added_edges = tuple(_normalize_edge(edge) for edge in added)
    if len(removed_edges) != 2 or len(added_edges) != 2:
        raise ValueError("local catalog operations require two removed and two added edges")
    return _edge_pair(*removed_edges), _edge_pair(*added_edges)


def _normalize_edge(edge: Any) -> Edge:
    values = tuple(int(site) for site in edge)
    if len(values) != 2 or values[0] == values[1]:
        raise ValueError("catalog edges require two different endpoints")
    return min(values), max(values)


def _edge_pair(first: Edge, second: Edge) -> EdgePair:
    return (first, second) if first < second else (second, first)


def _edges_have_squared_length(geometry: Geometry, edges: tuple[Edge, Edge], value: float) -> bool:
    coordinates = geometry.coordinates
    if coordinates is None:
        return False
    return all(
        isclose(
            float(np.dot(coordinates[target] - coordinates[source],
                         coordinates[target] - coordinates[source])),
            value,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for source, target in edges
    )


def _rewired_geometry(
    reference: Geometry, removed: tuple[Edge, Edge], added: tuple[Edge, Edge]
) -> Geometry:
    coordinates = reference.coordinates
    assert coordinates is not None
    pairs = (_edge_pairs(reference) - set(removed)) | set(added)
    return Geometry(
        n_sites=reference.n_sites,
        edges=tuple(
            GeometryEdge(
                source,
                target,
                edge_type="size_calibration_coupling",
                displacement=tuple(
                    float(value) for value in coordinates[target] - coordinates[source]
                ),
            )
            for source, target in sorted(pairs)
        ),
        coordinates=coordinates,
        embedding_dimension=reference.embedding_dimension,
        boundary_sites=reference.boundary_sites,
        boundary_components=reference.boundary_components,
        site_types=reference.site_types,
        dimension_records=reference.dimension_records,
        rooted_tree=reference.rooted_tree,
        metadata=reference.metadata,
        faces=reference.faces,
    )


def _same_nonedge_fields(reference: Geometry, candidate: Geometry) -> bool:
    reference_coordinates = reference.coordinates
    candidate_coordinates = candidate.coordinates
    return (
        reference_coordinates is not None
        and candidate_coordinates is not None
        and np.array_equal(reference_coordinates, candidate_coordinates)
        and candidate.embedding_dimension == reference.embedding_dimension
        and candidate.boundary_sites == reference.boundary_sites
        and candidate.boundary_components == reference.boundary_components
        and candidate.site_types == reference.site_types
        and candidate.dimension_records == reference.dimension_records
        and candidate.rooted_tree == reference.rooted_tree
        and dict(candidate.metadata) == dict(reference.metadata)
        and candidate.faces == reference.faces
    )


def _edge_semantics_are_fixed(geometry: Geometry) -> bool:
    coordinates = geometry.coordinates
    if coordinates is None:
        return False
    return all(
        edge.source < edge.target
        and edge.edge_type == "size_calibration_coupling"
        and not edge.boundary_crossing
        and not edge.metadata
        and edge.displacement is not None
        and np.allclose(
            edge.displacement,
            coordinates[edge.target] - coordinates[edge.source],
            rtol=0.0,
            atol=1.0e-12,
        )
        for edge in geometry.edges
    )


def _added_edges_cross(
    coordinates: np.ndarray, candidate_pairs: set[Edge], added_edges: tuple[Edge, Edge]
) -> bool:
    for first in added_edges:
        for second in candidate_pairs:
            if first == second or set(first).intersection(second):
                continue
            if _segments_intersect(
                coordinates[first[0]], coordinates[first[1]],
                coordinates[second[0]], coordinates[second[1]],
            ):
                return True
    return False


def _symmetry_signature(
    n: int,
    removed: tuple[Edge, Edge],
    added: tuple[Edge, Edge],
    *,
    reflect: bool,
) -> str:
    variants = []
    reflection_values = (False, True) if reflect else (False,)
    for reflected in reflection_values:
        for rotations in range(4):
            variants.append(
                (
                    _transform_edges(n, removed, rotations, reflected),
                    _transform_edges(n, added, rotations, reflected),
                )
            )
    canonical = min(variants)
    payload = json.dumps(canonical, separators=(",", ":")).encode("ascii")
    prefix = "d4" if reflect else "c4"
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()}"


def _transform_edges(
    n: int, edges: EdgePair, rotations: int, reflected: bool
) -> EdgePair:
    transformed: list[Edge] = []
    for source, target in edges:
        endpoints = []
        for site in (source, target):
            x_value, y_value = divmod(site, n)
            if reflected:
                x_value = n - 1 - x_value
            for _ in range(rotations):
                x_value, y_value = y_value, n - 1 - x_value
            endpoints.append(x_value * n + y_value)
        transformed.append((min(endpoints), max(endpoints)))
    return _edge_pair(transformed[0], transformed[1])


def _candidate_record(candidate: LocalCatalogCandidate) -> dict[str, Any]:
    return {
        "n": candidate.n,
        "geometry_id": candidate.geometry_id,
        "removed_edges": candidate.removed_edges,
        "added_edges": candidate.added_edges,
        "affected_sites": candidate.affected_sites,
        "origins": [
            {
                "offset": origin.offset,
                "arm": origin.arm,
                "measurement_sites": origin.measurement_sites,
            }
            for origin in candidate.origins
        ],
        "known_measurement_variant": candidate.known_measurement_variant,
        "rotation_signature": candidate.rotation_signature,
        "dihedral_signature": candidate.dihedral_signature,
    }


def _edge_pairs(geometry: Geometry) -> set[Edge]:
    return {(edge.source, edge.target) for edge in geometry.edges}


def _degrees(geometry: Geometry) -> tuple[int, ...]:
    return tuple(len(geometry.neighbors(site)) for site in range(geometry.n_sites))


@cache
def _validated_reference_id(n: int) -> str:
    reference = measurement_reference(n)
    report = validate_measurement_geometry(reference, n)
    if not report["is_valid"]:
        raise AssertionError("measurement reference no longer satisfies its frozen contract")
    return exact_geometry_id(reference)
