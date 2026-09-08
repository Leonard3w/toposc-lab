"""Small engineering demo: graph search, not a superconductivity experiment.

Run from the repository with PYTHONPATH=src and PYTHONDONTWRITEBYTECODE=1.
No files are written. All choices below are explicit demonstration policies,
not defaults of the search library or a Phase-9.8 scientific protocol.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import BasisLayout
from toposc_lab.evaluation import (
    BasicScalarScore,
    BasicScoreComponent,
    GeometryEvaluationRun,
    GeometryModelAdapter,
    ObjectiveDirection,
    evaluate_geometry,
)
from toposc_lab.geometry import Geometry, GeometryEdge
from toposc_lab.search import (
    BenchmarkSuccessCriterion,
    ElitismConfig,
    GenerationLoopConfig,
    GenerationReproductionRequest,
    GeometryGenome,
    MutationValidityPolicy,
    OffspringProposal,
    PopulationFitnessMember,
    ScalarFitnessDefinition,
    SearchBenchmarkProtocol,
    SearchBenchmarkResult,
    TournamentSelectionConfig,
    add_edge_mutation,
    geometry_to_genome,
    remove_edge_mutation,
    run_search_benchmark,
)

CODE_VERSION = "phase-10-engineering-demo.v1"
RING = tuple((site, site + 1) for site in range(5)) + ((0, 5),)
CHORDS = tuple(pair for pair in combinations(range(6), 2) if pair not in RING)


class DemoGraphModel(BaseModel):
    """Six-site real hopping Hamiltonian; deliberately NOT a BdG/class-D model."""

    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry

    @property
    def parameters(self) -> dict[str, Any]:
        return {"onsite": (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0), "hopping": 0.4}

    @property
    def basis_layout(self) -> BasisLayout:
        return BasisLayout((6,))

    def hamiltonian(self) -> NDArray[np.complex128]:
        matrix = np.diag([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).astype(np.complex128)
        for edge in self.geometry.edges:
            matrix[edge.source, edge.target] = matrix[edge.target, edge.source] = 0.4
        return matrix


def sample_genome(seed: int) -> GeometryGenome:
    """Uniform draw among the 36 unordered pairs of nine optional chords."""
    rng = np.random.Generator(np.random.PCG64(seed))
    selected = sorted(int(index) for index in rng.choice(len(CHORDS), 2, replace=False))
    return geometry_to_genome(
        Geometry(
            n_sites=6,
            edges=tuple(GeometryEdge(*edge) for edge in (*RING, *(CHORDS[i] for i in selected))),
        )
    )


def evaluate_genome(genome: GeometryGenome, seed: int) -> GeometryEvaluationRun:
    return evaluate_geometry(
        genome.to_geometry(),
        adapter=GeometryModelAdapter(DemoGraphModel),
        seed=seed,
        code_version=CODE_VERSION,
    )


def produce_offspring(request: GenerationReproductionRequest) -> tuple[OffspringProposal, ...]:
    """Replace one chord; preserve six sites, eight edges, and the connected ring."""
    rng = np.random.Generator(np.random.PCG64(request.seed))
    proposals: list[OffspringProposal] = []
    for _ in range(request.required_offspring_count):
        slot = int(rng.integers(len(request.selected_members)))
        parent = request.selected_members[slot].population_member.genome
        present = {(edge.source, edge.target) for edge in parent.edges}
        removable = [
            i for i, edge in enumerate(parent.edges) if (edge.source, edge.target) in CHORDS
        ]
        absent = [pair for pair in CHORDS if pair not in present]
        genome = remove_edge_mutation(parent, removable[int(rng.integers(len(removable)))])
        genome = add_edge_mutation(genome, GeometryEdge(*absent[int(rng.integers(len(absent)))]))
        proposals.append(
            OffspringProposal(
                genome,
                (slot,),
                slot,
                "demo.replace-one-chord.v1",
                request.seed,
            )
        )
    return tuple(proposals)


def is_demo_hit(member: PopulationFitnessMember) -> bool:
    assert isinstance(member.fitness, BasicScalarScore)
    return member.fitness.value >= 0.65


def demo_protocol() -> SearchBenchmarkProtocol:
    return SearchBenchmarkProtocol(
        identifier="phase-10-engineering-demo.v1",
        candidate_space_identifier="six-site-ring-plus-two-chords.v1",
        sampler_identifier="uniform-two-chords.v1",
        evaluator_identifier="demo-real-hopping-gap.v1",
        producer_identifier="demo.replace-one-chord.v1",
        code_version=CODE_VERSION,
        population_size=6,
        generation_config=GenerationLoopConfig(
            4, TournamentSelectionConfig(6, 2), ElitismConfig(1)
        ),
        trial_seeds=(10201, 10202, 10203, 10204, 10205),
        definition=ScalarFitnessDefinition(
            {BasicScoreComponent.NORMALIZED_GAP: 1.0},
            ObjectiveDirection.MAXIMIZE,
        ),
        validity_policy=MutationValidityPolicy(
            require_connected=True,
            minimum_site_count=6,
            maximum_site_count=6,
            minimum_edge_count=8,
            maximum_edge_count=8,
        ),
        criterion=BenchmarkSuccessCriterion(
            "demo-normalized-gap-at-least-0.65.v1",
            "Engineering score >= 0.65; no topology or physical success claim.",
            is_demo_hit,
        ),
    )


def run_demo() -> SearchBenchmarkResult:
    return run_search_benchmark(
        demo_protocol(),
        sampler=sample_genome,
        evaluator=evaluate_genome,
        offspring_producer=produce_offspring,
    )


def main() -> None:
    result = run_demo()
    print("Phase 10 engineering demo -- NOT a superconductivity/discovery benchmark")
    print(f"Attempts per arm and trial: {result.protocol.attempts_per_arm}")
    print("seed   evolution_best  random_best  delta(+ favors evolution)  first_hit(evo/random)")
    for trial in result.trials:
        left = trial.evolution_arm.final_progress
        right = trial.random_arm.final_progress
        print(
            f"{trial.seed}  {left.best_scalar_score!s:16} {right.best_scalar_score!s:16} "
            f"{trial.scalar_quality_difference!s:24} "
            f"{left.first_success_attempt}/{right.first_success_attempt}",
        )
    print("None = no hit/no available scalar, not zero. No files written.")
    print("Five small trials are illustrative; no significance or physical advantage is claimed.")


if __name__ == "__main__":
    main()
