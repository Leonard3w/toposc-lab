"""Outcome-blind family recipes composed from existing graph infrastructure."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import numpy as np

from toposc_lab.geometry import Geometry, GeometryEdge, square
from toposc_lab.geometry.generators.hard_core_planar import (
    constrained_random_embedded_graph,
    hard_core_planar_graph,
)
from toposc_lab.geometry.generators.protocol import (
    FunctionGeometryGenerator,
    GeometryGenerationRequest,
)
from toposc_lab.research.descriptors import compute_descriptors
from toposc_lab.research.embedded import EmbeddedDomain
from toposc_lab.research.space import (
    FixedConnectivitySpace,
    geometry_from_payload,
    geometry_to_payload,
)
from toposc_lab.research.validation_cohort import digest
from toposc_lab.search.mutation_validity import (
    MutationValidityPolicy,
    validate_geometry_constraints,
)

VERSION = "phase19.embedded-cohort.v1"
FAMILIES = ("rewired_square", "amorphous_planar", "constrained_embedded")


def comparison_policy() -> MutationValidityPolicy:
    return MutationValidityPolicy(
        require_connected=True,
        require_coordinates=True,
        required_embedding_dimension=2,
        minimum_site_count=64,
        maximum_site_count=64,
        minimum_edge_count=112,
        maximum_edge_count=112,
        minimum_degree=2,
        maximum_degree=4,
        coordinate_lower_bounds=(0.0, 0.0),
        coordinate_upper_bounds=(7.0, 7.0),
        minimum_site_separation=0.55,
        maximum_edge_length=1.75,
        minimum_boundary_site_count=24,
        maximum_boundary_site_count=32,
        forbid_straight_edge_crossings=True,
    )


def canonical_sites(geometry: Geometry) -> Geometry:
    """Stable lexicographic physical ordering for matching independent onsite fields."""
    assert geometry.coordinates is not None
    order = np.lexsort((geometry.coordinates[:, 1], geometry.coordinates[:, 0]))
    inverse = np.argsort(order)
    # Study inputs are open one-skeletons; do not silently discard richer cell complexes.
    if geometry.faces or geometry.rooted_tree or any(e.boundary_crossing for e in geometry.edges):
        raise ValueError("Study recipes must return open one-skeleton graphs")
    for edge in geometry.edges:
        if edge.displacement is not None and not np.array_equal(
            edge.displacement,
            geometry.coordinates[edge.target] - geometry.coordinates[edge.source],
        ):
            raise ValueError("Cannot discard a nonphysical edge displacement")
    from toposc_lab.geometry import GeometryBoundaryComponent

    edges = tuple(
        sorted(
            (
                GeometryEdge(
                    int(inverse[e.source]),
                    int(inverse[e.target]),
                    edge_type=e.edge_type,
                    metadata=e.metadata,
                )
                for e in geometry.edges
            ),
            key=lambda e: (min(e.source, e.target), max(e.source, e.target)),
        )
    )
    return replace(
        geometry,
        coordinates=geometry.coordinates[order],
        edges=edges,
        site_types=None
        if geometry.site_types is None
        else tuple(geometry.site_types[i] for i in order),
        boundary_sites=frozenset(int(inverse[i]) for i in geometry.boundary_sites),
        boundary_components=tuple(
            GeometryBoundaryComponent(
                c.kind, c.component_index, frozenset(int(inverse[i]) for i in c.sites)
            )
            for c in geometry.boundary_components
        ),
    )


def geometry_id(geometry: Geometry) -> str:
    """Exact physical identity, excluding incidental generation metadata."""
    assert geometry.coordinates is not None
    return digest(
        {
            "coordinates": geometry.coordinates.tolist(),
            "edges": sorted(sorted((e.source, e.target)) for e in geometry.edges),
            "boundary": sorted(geometry.boundary_sites),
        }
    )


def rewired_square(*, seed: int) -> Geometry:
    space = FixedConnectivitySpace(
        side=8,
        min_degree=2,
        max_degree=4,
        max_bond_length=1.75,
        bond_tolerance=0,
        forbid_crossings=True,
        initialization_rewires=(8, 16, 32, 64),
    )
    return space.sample(np.random.Generator(np.random.PCG64(seed)))


RECIPES = {
    "rewired_square": FunctionGeometryGenerator("rewired_square", rewired_square, stochastic=True),
    "amorphous_planar": FunctionGeometryGenerator(
        "amorphous_planar", hard_core_planar_graph, stochastic=True
    ),
    "constrained_embedded": FunctionGeometryGenerator(
        "constrained_embedded", constrained_random_embedded_graph, stochastic=True
    ),
}


def prepare_cohort(
    *, per_family: int = 50, seed: int = 190000, max_proposals: int = 1500
) -> dict[str, Any]:
    if type(per_family) is not int or not 1 <= per_family <= 100:
        raise ValueError("per_family must be 1..100")
    if type(max_proposals) is not int or max_proposals < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Positive proposal budget and nonnegative seed required")
    domain, policy = EmbeddedDomain(), comparison_policy()
    candidates: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    seen: set[str] = set()
    rng = np.random.Generator(np.random.PCG64(seed))

    def accept(g: Geometry, family: str, generation_seed: int | None) -> list[str]:
        g = canonical_sites(g)
        report = validate_geometry_constraints(g, policy=policy)
        reasons = [i.code for i in report.issues]
        if reasons:
            return reasons
        domain.validate(g)
        identity = geometry_id(g)
        if identity in seen:
            return ["duplicate_physical_geometry"]
        seen.add(identity)
        features = compute_descriptors(g, include_square_reference=False)
        features["boundary_site_fraction"] = len(g.boundary_sites) / g.n_sites
        candidates.append(
            {
                "id": "regular" if family == "regular" else family + "_" + identity[:16],
                "family": family,
                "geometry_sha256": identity,
                "seed": generation_seed,
                "geometry": geometry_to_payload(g),
                "descriptors": features,
                "descriptor_unavailable": {
                    "regular_edge_distance": "square-reference distance is not a cross-family feature"
                },
            }
        )
        return []

    assert not accept(square(8, 8), "regular", None)
    for family in FAMILIES:
        count = 0
        for _ in range(max_proposals):
            generation_seed = int(rng.integers(0, 2**32))
            try:
                g = RECIPES[family].generate(GeometryGenerationRequest(seed=generation_seed))
                reasons = accept(g, family, generation_seed)
                internal = dict(g.metadata.get("rejected_attempt_reason_counts", {}))
            except (ValueError, RuntimeError) as error:
                reasons, internal = [f"{type(error).__name__}: {error}"], {}
            proposals.append(
                {
                    "family": family,
                    "seed": generation_seed,
                    "accepted": not reasons,
                    "reasons": reasons,
                    "internal_rejection_counts": internal,
                }
            )
            if not reasons:
                count += 1
            if count == per_family:
                break
        if count != per_family:
            # Caller may persist the exception report; never silently shrink/relax a family.
            raise CohortPreparationError(f"{family}: only {count}/{per_family} accepted", proposals)
    result = {
        "version": VERSION,
        "domain": asdict(domain),
        "policy": asdict(policy),
        "seed": seed,
        "per_family": per_family,
        "max_proposals_per_family": max_proposals,
        "candidates": candidates,
        "proposals": proposals,
        "new_exact_calls": 0,
        "site_order": "lexicographic x/y; common iid onsite draws by index, not same spatial field",
        "ensemble": "constrained algorithmic random samples; not uniform over valid graphs",
        "selection": "all valid unique proposals retained until fixed family count, no physics selection",
    }
    return {**result, "sha256": digest(result)}


class CohortPreparationError(RuntimeError):
    def __init__(self, message: str, proposals: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.proposals = proposals


def validate_cohort(cohort: dict[str, Any]) -> None:
    if cohort.get("version") != VERSION or cohort.get("sha256") != digest(
        {k: v for k, v in cohort.items() if k != "sha256"}
    ):
        raise ValueError("Embedded cohort version/checksum mismatch")
    domain = EmbeddedDomain(**cohort["domain"])
    policy = MutationValidityPolicy(**cohort["policy"])
    ids, geometries = set(), set()
    for candidate in cohort["candidates"]:
        g = geometry_from_payload(candidate["geometry"])
        validate_geometry_constraints(g, policy=policy).raise_for_errors()
        domain.validate(g)
        key = geometry_id(g)
        if candidate["id"] in ids or key in geometries or key != candidate["geometry_sha256"]:
            raise ValueError("Duplicate/mismatched candidate identity")
        ids.add(candidate["id"])
        geometries.add(key)
        rebuilt = geometry_to_payload(g)
        if any(
            rebuilt[k] != candidate["geometry"][k]
            for k in ("coordinates", "edges", "boundary_sites", "n_sites")
        ):
            raise ValueError("Geometry preview mismatch")
    if "regular" not in ids:
        raise ValueError("Regular reference required")
