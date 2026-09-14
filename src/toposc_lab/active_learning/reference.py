"""Exact finite Kitaev-family benchmark; bulk and finite evidence stay separate."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from toposc_lab.active_learning.pool import Candidate
from toposc_lab.data import (
    DatasetRecord,
    GeometryRecord,
    ModelParametersRecord,
    ObservableResultRecord,
    ReproducibilityMetadata,
    SpectrumRecord,
    TopologyResultRecord,
    TopologyValidity,
    create_dataset_record,
)
from toposc_lab.evaluation import GeometryModelAdapter, evaluate_geometry
from toposc_lab.models.geometry_kitaev_chain import GeometryKitaevChain
from toposc_lab.models.kitaev_chain import KitaevChainParameters
from toposc_lab.observables.majorana import majorana_diagnostics
from toposc_lab.search.lexicographic_fitness import DerivedEvaluationRun
from toposc_lab.topology.pfaffian import one_dimensional_pfaffian_invariant
from toposc_lab.topology.symmetry import SymmetryClassification

REFERENCE_ID = "phase13.finite-kitaev-quality.v1"


def reference_candidate(n_sites: int, mu: float = 1.8) -> Candidate:
    model = GeometryKitaevChain(
        KitaevChainParameters(
            n_sites=n_sites, hopping=1.0, chemical_potential=mu, pairing=0.5, boundary="open"
        )
    )
    return Candidate(
        GeometryRecord.from_geometry(model.geometry),
        ModelParametersRecord(model.model_name, "1", model.parameters),
    )


def quality(metrics: dict, *, splitting_scale: float = 0.01) -> float:
    """Explicit finite-family engineering score, never a Majorana certificate."""
    return float(
        metrics["bulk_topological"]
        * metrics["isolation"]
        / (1 + metrics["isolation"])
        * metrics["boundary_weight"]
        / (1 + metrics["splitting"] / splitting_scale)
    )


@dataclass(frozen=True, slots=True)
class KitaevReferenceEvaluator:
    provenance: ReproducibilityMetadata

    def __call__(self, candidate: Candidate, seed: int) -> DatasetRecord:
        return self.evaluate(candidate, seed)[0]

    def evaluate(
        self, candidate: Candidate, seed: int
    ) -> tuple[DatasetRecord, DerivedEvaluationRun]:
        params = KitaevChainParameters(**dict(candidate.model.parameters))
        model = GeometryKitaevChain(params)
        expected = reference_candidate(params.n_sites, params.chemical_potential)
        if candidate.candidate_id != expected.candidate_id:
            raise ValueError("reference evaluator only supports declared clean open Kitaev chains")
        run = evaluate_geometry(
            model.geometry,
            adapter=GeometryModelAdapter(
                lambda g: model, nambu_basis_resolver=lambda m: m.nambu_basis
            ),
            seed=seed,
            code_version=self.provenance.git_commit,
        )
        if not run.is_valid or run.simulation_result is None:
            raise ValueError(f"exact evaluation failed: {run.failure}")
        simulation = run.simulation_result
        e = simulation.eigenvalues
        indices = np.argsort(np.abs(e))[:2]
        splitting = float(np.mean(np.abs(e[indices])))
        next_energy = float(np.sort(np.abs(e))[2])
        diagnostics = [
            majorana_diagnostics(simulation.eigenvectors, int(i), model.nambu_basis)
            for i in indices
        ]
        boundary_weight = float(
            np.mean(
                [sum(d.site_probability[:2]) + sum(d.site_probability[-2:]) for d in diagnostics]
            )
        )
        mu = params.chemical_potential
        invariant = one_dimensional_pfaffian_invariant(
            np.diag([-mu - 2.0, mu + 2.0]),
            np.diag([-mu + 2.0, mu - 2.0]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            SymmetryClassification.from_signature(
                time_reversal_square=1, particle_hole_square=1, chiral_symmetry=True
            ),
            tolerance=1e-10,
        )
        metrics = {
            "splitting": splitting,
            "next_excitation": next_energy,
            "isolation": max(0.0, next_energy - splitting),
            "boundary_weight": boundary_weight,
            "polarization_norm": float(np.mean([d.polarization_norm for d in diagnostics])),
            "phs_mismatch": float(np.max(np.abs(e + e[::-1]))),
            "bulk_topological": float(invariant.is_topological),
        }
        if metrics["phs_mismatch"] > 1e-10:
            raise ValueError("particle-hole spectrum mismatch")
        metrics["quality"] = quality(metrics)
        record = create_dataset_record(
            geometry=candidate.geometry,
            model=candidate.model,
            spectrum=SpectrumRecord(tuple(float(x) for x in e), "hopping", "all", len(e), True),
            observables=(
                ObservableResultRecord(
                    "finite_chain_quality",
                    "1",
                    metrics,
                    conventions={
                        "splitting_scale": 0.01,
                        "boundary_sites_per_end": 2,
                        "finite_system": True,
                        "majorana_claim": False,
                    },
                ),
            ),
            topology=(
                TopologyResultRecord(
                    "pfaffian_1d",
                    "1",
                    TopologyValidity.VALID,
                    invariant.invariant,
                    invariant.is_topological,
                    {"bulk_reference": "translation_invariant_kitaev_endpoints", "mu": mu},
                    {"particle_hole": 1e-10, "antisymmetry": 1e-10},
                    warnings=(
                        "Bulk parent invariant, not an invariant inferred from the finite graph.",
                    ),
                ),
            ),
            robustness=(),
            provenance=replace(self.provenance, seed=seed),
        )
        return record, DerivedEvaluationRun.from_run(
            run, identifier=REFERENCE_ID, quantities={"quality": metrics["quality"]}
        )
