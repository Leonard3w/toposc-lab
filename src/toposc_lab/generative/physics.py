"""Existing chiral-p-wave solver and finite localizer evidence, without phase claims."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

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
    validate_dataset_record,
)
from toposc_lab.evaluation import GeometryModelAdapter, evaluate_geometry
from toposc_lab.generative.space import GeometrySearchSpace
from toposc_lab.geometry import Geometry
from toposc_lab.models.chiral_p_wave import ChiralPWaveModel, ChiralPWaveParameters
from toposc_lab.observables.majorana import majorana_diagnostics
from toposc_lab.search.lexicographic_fitness import DerivedEvaluationRun
from toposc_lab.topology import SymmetryClassification, spectral_localizer

PROTOCOL_ID = "phase14.planar-wiring.v1"
KAPPAS = (0.1, 0.2, 0.3)


def candidate(geometry: Geometry, mu: float = 2.0) -> Candidate:
    params = ChiralPWaveParameters(hopping=1.0, chemical_potential=mu, pairing=1.0, chirality=1)
    return Candidate(
        GeometryRecord.from_geometry(geometry, family_label="planar_wiring_6x6"),
        ModelParametersRecord("chiral_p_wave", "1", params.model_dump()),
    )


@dataclass(frozen=True, slots=True)
class ExactGeometryEvaluator:
    provenance: ReproducibilityMetadata

    def evaluate(
        self, proposal: Candidate, seed: int
    ) -> tuple[DatasetRecord, DerivedEvaluationRun]:
        geometry = proposal.geometry.to_geometry()
        reasons = GeometrySearchSpace().reasons(geometry)
        if reasons:
            raise ValueError(f"inadmissible geometry: {reasons}")
        params = ChiralPWaveParameters.model_validate(dict(proposal.model.parameters))
        if proposal.candidate_id != candidate(geometry, params.chemical_potential).candidate_id:
            raise ValueError("unsupported physical parameters/model identity")
        if params.chemical_potential not in (1.9, 2.0, 2.1, 8.0):
            raise ValueError("chemical potential outside search/reference/confirmation contract")
        model = ChiralPWaveModel(geometry, params)
        run = evaluate_geometry(
            geometry,
            adapter=GeometryModelAdapter(
                lambda g: model, nambu_basis_resolver=lambda m: model.nambu_basis
            ),
            seed=seed,
            code_version=self.provenance.git_commit,
        )
        if not run.is_valid or run.simulation_result is None:
            raise ValueError(f"exact evaluation failed: {run.failure}")
        simulation = run.simulation_result
        energies = simulation.eigenvalues
        hamiltonian = model.hamiltonian()
        phs = float(np.max(np.abs(energies + energies[::-1])))
        # Component-major Nambu basis: both particle and hole coordinates repeat.
        assert geometry.coordinates is not None  # checked by the space contract
        coordinates = np.tile(geometry.coordinates, (2, 1))
        classification = SymmetryClassification.from_signature(
            time_reversal_square=None, particle_hole_square=1, chiral_symmetry=False
        )
        localizers = tuple(
            spectral_localizer(
                hamiltonian,
                coordinates,
                (2.5, 2.5),
                classification,
                kappa=k,
                tolerance=1e-10,
            )
            for k in KAPPAS
        )
        indices = [r.local_chern_number for r in localizers]
        eligible = (
            all(r.is_invertible for r in localizers)
            and indices[0] not in (None, 0)
            and len(set(indices)) == 1
            and phs <= 1e-10
        )
        diagnostics = [
            majorana_diagnostics(simulation.eigenvectors, int(i), model.nambu_basis)
            for i in np.argsort(np.abs(energies))[:4]
        ]
        weights = [
            float(sum(d.site_probability[i] for i in geometry.boundary_sites)) for d in diagnostics
        ]
        gap = min(r.localizer_gap for r in localizers)
        metrics: dict[str, Any] = {
            "quality": float(eligible) * gap,
            "eligible": float(eligible),
            "success": float(eligible and gap >= 0.2),
            "localizer_gap": gap,
            "boundary_weight": float(np.mean(weights)),
            "boundary_weight_minimum": min(weights),
            "boundary_weights": weights,
            "polarization_norms": [float(d.polarization_norm) for d in diagnostics],
            "minimum_abs_energy": float(np.min(np.abs(energies))),
            "phs_mismatch": phs,
            "alternate_quality": float(eligible) * gap * float(np.mean(weights)),
        }
        record = create_dataset_record(
            geometry=proposal.geometry,
            model=proposal.model,
            spectrum=SpectrumRecord(tuple(float(x) for x in energies), "hopping", "all", 72, True),
            observables=(
                ObservableResultRecord(
                    "finite_geometry_quality",
                    "1",
                    metrics,
                    conventions={
                        "finite_system": True,
                        "majorana_claim": False,
                        "phase_claim": False,
                        "novelty_in_fitness": False,
                    },
                ),
            ),
            topology=(
                TopologyResultRecord(
                    "spectral_localizer_2d_grid",
                    "1",
                    TopologyValidity.VALID
                    if all(r.is_invertible for r in localizers) and len(set(indices)) == 1
                    else TopologyValidity.UNRESOLVED,
                    indices[0] if len(set(indices)) == 1 else None,
                    bool(indices[0]) if len(set(indices)) == 1 and indices[0] is not None else None,
                    {
                        "kappas": KAPPAS,
                        "probe": (2.5, 2.5),
                        "localizer_gaps": tuple(r.localizer_gap for r in localizers),
                        "indices": tuple(indices),
                        "signatures": tuple(r.signature for r in localizers),
                    },
                    {"numerical": 1e-10},
                    reason="Index disagreement or unresolved localizer"
                    if not (all(r.is_invertible for r in localizers) and len(set(indices)) == 1)
                    else None,
                    warnings=("Finite center-probe evidence; no thermodynamic or Majorana claim.",),
                ),
            ),
            robustness=(),
            provenance=replace(self.provenance, seed=seed),
        )
        validate_dataset_record(record).raise_for_errors()
        return record, DerivedEvaluationRun.from_run(
            run,
            identifier=PROTOCOL_ID,
            quantities={"quality": metrics["quality"]},
        )
