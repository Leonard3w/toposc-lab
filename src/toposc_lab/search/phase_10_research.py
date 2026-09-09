"""Frozen physical adapter and edge-swap strategy for TOPOSC-P10-EVO-RS-001."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from toposc_lab.evaluation import GeometryEvaluationRun, ObjectiveDirection
from toposc_lab.geometry import Geometry, GeometryBoundaryComponent, GeometryEdge
from toposc_lab.geometry.generators import BUILTIN_GEOMETRY_GENERATORS
from toposc_lab.geometry.generators.hard_core_planar import hard_core_planar_edge_pool
from toposc_lab.search.generation_loop import (
    GenerationLoopConfig,
    GenerationReproductionRequest,
    OffspringProposal,
)
from toposc_lab.search.geometry_genome import GeometryGenome
from toposc_lab.search.geometry_mutations import rewire_edge_mutation
from toposc_lab.search.lexicographic_fitness import (
    DerivedEvaluationRun,
    LexicographicFitness,
    LexicographicFitnessDefinition,
)
from toposc_lab.search.mutation_validity import MutationValidityPolicy
from toposc_lab.search.phase_9_8_evaluation import (
    Phase98GeometryApplicability,
    build_phase_9_8_ammann_beenker_topology_inputs,
    build_phase_9_8_sierpinski_topology_inputs,
    evaluate_phase_9_8_descriptive_geometry,
    evaluate_phase_9_8_primary_geometry,
    validate_phase_9_8_geometry,
)
from toposc_lab.search.population_elitism import ElitismConfig
from toposc_lab.search.population_fitness import PopulationFitnessMember
from toposc_lab.search.population_selection import TournamentSelectionConfig

# Reuse the frozen experiment's complete scientific summary, not its runner.
from toposc_lab.search.random_search_experiment import (
    _encode_pipeline_run,
    _encode_scientific,
    _encode_topology_grid,
)
from toposc_lab.search.search_benchmark import BenchmarkSuccessCriterion, SearchBenchmarkProtocol

RESEARCH_PROTOCOL_ID = "TOPOSC-P10-EVO-RS-001"
RESEARCH_PROTOCOL_COMMIT = "71e159f9eeb67de17b551cb820c4f3599830ca3b"
RESEARCH_PROTOCOL_PATH = "docs/decisions/pre_phase_10_research_protocol_v1.md"
DERIVATION_ID = "phase98.clean-screen.raw-quantities.v1"
OPERATOR_ID = "phase10_research_delaunay_edge_swap_v1"
PREFLIGHT_SEEDS = tuple(range(10_799_900, 10_799_910))
TRIAL_SEEDS = tuple(range(10_800_000, 10_800_032))
REFERENCE_SEEDS = tuple(range(10_801_000, 10_801_032))
VALIDATION_SEEDS = tuple(range(10_810_000, 10_810_064))
CONFIRMATION_SEEDS = tuple(range(10_820_000, 10_820_128))
QUANTITIES = (
    "clean_eligible",
    "localizer_protection_proxy",
    "minimum_boundary_weight_first_four",
)
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class ResearchEvaluation:
    """Lossless pipeline object and frozen scientific gates/grids kept separately."""

    run: GeometryEvaluationRun
    scientific: dict[str, Any]


def evaluate_research_geometry(
    geometry: Geometry, code_version: str, *, descriptive: str | None = None
) -> ResearchEvaluation:
    """Evaluate the inherited clean physics without decorative evaluation seeds."""
    if descriptive is not None:
        if descriptive == "ammann_beenker":
            inputs = build_phase_9_8_ammann_beenker_topology_inputs(geometry)
        elif descriptive == "sierpinski":
            inputs = build_phase_9_8_sierpinski_topology_inputs(geometry)
        else:
            raise ValueError("unknown descriptive reference")
        run, grid = evaluate_phase_9_8_descriptive_geometry(
            geometry,
            inputs=inputs,
            code_version=code_version,
        )
        return ResearchEvaluation(
            run,
            {
                "descriptive": descriptive,
                "matched_primary_stratum": False,
                "pipeline": _encode_pipeline_run(run),
                "topology_grid": None if grid is None else _encode_topology_grid(grid),
            },
        )
    scientific = evaluate_phase_9_8_primary_geometry(geometry, code_version=code_version)
    quantities = {}
    if scientific.run.is_valid:
        quantities = dict(
            zip(
                QUANTITIES,
                (
                    float(scientific.clean_eligible),
                    scientific.localizer_protection_proxy,
                    scientific.minimum_boundary_weight_first_four,
                ),
                strict=True,
            )
        )
    return ResearchEvaluation(
        DerivedEvaluationRun.from_run(
            scientific.run, identifier=DERIVATION_ID, quantities=quantities
        ),
        _encode_scientific(scientific),
    )


def research_validity_policy() -> MutationValidityPolicy:
    """Common neutral resource constraints; the physical gate is checked additionally."""
    return MutationValidityPolicy(
        require_connected=True,
        require_coordinates=True,
        required_embedding_dimension=2,
        minimum_site_count=64,
        maximum_site_count=64,
        minimum_edge_count=112,
        maximum_edge_count=112,
        minimum_boundary_site_count=24,
        maximum_boundary_site_count=32,
        minimum_degree=2,
        maximum_degree=4,
        coordinate_lower_bounds=(0.0, 0.0),
        coordinate_upper_bounds=(7.0, 7.0),
        minimum_site_separation=0.55,
        maximum_edge_length=1.75,
        forbid_straight_edge_crossings=True,
        numerical_tolerance=1e-12,
    )


def research_protocol(
    code_version: str, reference_proxy: float, trial_seeds: tuple[int, ...]
) -> SearchBenchmarkProtocol:
    """Bind the frozen strategy to a sealed reference threshold and explicit trials."""
    if not np.isfinite(reference_proxy) or reference_proxy < 0.20:
        raise ValueError("reference proxy requires an eligible primary reference")

    def hit(member: PopulationFitnessMember) -> bool:
        fitness = member.fitness
        return (
            isinstance(fitness, LexicographicFitness)
            and fitness.values[0] == 1.0
            and fitness.values[1] >= 1.10 * reference_proxy
        )

    return SearchBenchmarkProtocol(
        identifier=RESEARCH_PROTOCOL_ID,
        candidate_space_identifier="phase98.primary.delaunay.v1",
        sampler_identifier="hard_core_planar_graph.v1",
        evaluator_identifier=DERIVATION_ID,
        producer_identifier=OPERATOR_ID,
        code_version=code_version,
        population_size=8,
        generation_config=GenerationLoopConfig(
            3, TournamentSelectionConfig(8, 2), ElitismConfig(1)
        ),
        trial_seeds=trial_seeds,
        evaluation_mode="deterministic",
        definition=LexicographicFitnessDefinition(
            QUANTITIES,
            (ObjectiveDirection.MAXIMIZE,) * 3,
            DERIVATION_ID,
        ),
        validity_policy=research_validity_policy(),
        criterion=BenchmarkSuccessCriterion(
            "phase10.screening-strong.v1", "clean and L >= 1.10 R", hit
        ),
    )


def sample_research_genome(seed: int) -> GeometryGenome:
    """Use the registry seed directly, with no outer retry or parameter sampler."""
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate("hard_core_planar_graph", seed=seed)
    require_research_geometry(geometry)
    return GeometryGenome.from_geometry(geometry)


def require_research_geometry(geometry: Geometry) -> None:
    """Enforce both the Phase-9.8 physical gate and this search's fixed Delaunay pool."""
    report = validate_phase_9_8_geometry(
        geometry, applicability=Phase98GeometryApplicability.CLEAN_PRIMARY
    )
    if not report.is_applicable:
        raise ValueError(f"research geometry contract failed: {report.issues}")
    assert geometry.coordinates is not None
    pool = set(hard_core_planar_edge_pool(geometry.coordinates))
    edges = tuple((e.source, e.target) for e in geometry.edges)
    if edges != tuple(sorted(edges)) or not set(edges).issubset(pool):
        raise ValueError("search edges must be sorted oriented members of the Delaunay pool")


def square_reference() -> Geometry:
    """Add the already frozen physical outer-boundary annotation to the square."""
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "square",
        parameters={
            "n_x": 8,
            "n_y": 8,
            "spacing": 1.0,
            "boundary_x": "open",
            "boundary_y": "open",
        },
    )
    return replace(
        geometry,
        boundary_components=(GeometryBoundaryComponent("outer", 0, geometry.boundary_sites),),
    )


def legal_edge_swaps(genome: GeometryGenome) -> tuple[tuple[int, int, int], ...]:
    """Enumerate the complete frozen neighborhood, without any physics evaluations.

    The validated fixed embedding and planar Delaunay pool preserve all spatial
    constraints. Only degree and connectedness can change, so those are checked
    for each proposal. The selected finished graph receives full validation too.
    Each entry is (removed edge index, added source, added target).
    """
    require_research_geometry(genome.to_geometry())
    assert genome.coordinates is not None
    present = tuple((e.source, e.target) for e in genome.edges)
    absent = sorted(set(hard_core_planar_edge_pool(genome.coordinates)) - set(present))
    adjacency: list[set[int]] = [set() for _ in range(genome.n_sites)]
    for a, b in present:
        adjacency[a].add(b)
        adjacency[b].add(a)
    degrees = [len(neighbors) for neighbors in adjacency]
    legal: list[tuple[int, int, int]] = []
    for index, (a, b) in enumerate(present):
        for c, d in absent:
            changes: dict[int, int] = {}
            for site, delta in ((a, -1), (b, -1), (c, 1), (d, 1)):
                changes[site] = changes.get(site, 0) + delta
            if any(not 2 <= degrees[site] + delta <= 4 for site, delta in changes.items()):
                continue
            visited = {0}
            stack = [0]
            while stack:
                site = stack.pop()
                neighbors = adjacency[site].copy()
                if site == a:
                    neighbors.discard(b)
                if site == b:
                    neighbors.discard(a)
                if site == c:
                    neighbors.add(d)
                if site == d:
                    neighbors.add(c)
                for neighbor in sorted(neighbors):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            if len(visited) == genome.n_sites:
                legal.append((index, c, d))
    return tuple(legal)


def produce_research_offspring(
    request: GenerationReproductionRequest,
    *,
    record: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[OffspringProposal, ...]:
    """One uniformly sampled legal edge swap per explicitly selected parent slot."""
    rng = np.random.PCG64(request.seed)
    proposals = []
    for slot in range(request.required_offspring_count):
        parent = request.selected_members[slot].population_member.genome
        operator_seed = int(rng.random_raw())
        legal = legal_edge_swaps(parent)
        chosen: tuple[int, int, int] | None = None
        child = parent
        if legal:
            chosen = legal[
                int(np.random.Generator(np.random.PCG64(operator_seed)).integers(len(legal)))
            ]
            index, a, b = chosen
            assert parent.coordinates is not None
            edge = GeometryEdge(
                a, b, displacement=tuple(parent.coordinates[b] - parent.coordinates[a])
            )
            child = rewire_edge_mutation(parent, index, edge)
            child = replace(
                child, edges=tuple(sorted(child.edges, key=lambda e: (e.source, e.target)))
            )
            require_research_geometry(child.to_geometry())
        if record is not None:
            record(
                {
                    "generation": request.target_generation_index,
                    "parent_slot": slot,
                    "operator_seed": operator_seed,
                    "legal_swap_count": len(legal),
                    "chosen_swap": None if chosen is None else list(chosen),
                    "reason": "edge_swap" if chosen is not None else "no_legal_swap",
                }
            )
        proposals.append(OffspringProposal(child, (slot,), slot, OPERATOR_ID, operator_seed))
    return tuple(proposals)
