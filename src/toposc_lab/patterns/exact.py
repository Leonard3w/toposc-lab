"""Narrow 59-edge simplification adapter; unchanged BdG/localizer definitions."""

from dataclasses import replace
from typing import Any

import numpy as np

from toposc_lab.data import (
    DatasetRecord,
    ObservableResultRecord,
    ReproducibilityMetadata,
    SpectrumRecord,
    TopologyResultRecord,
    TopologyValidity,
    create_dataset_record,
    validate_dataset_record,
)
from toposc_lab.evaluation import GeometryModelAdapter, evaluate_geometry
from toposc_lab.generative.physics import KAPPAS, candidate
from toposc_lab.geometry import Geometry
from toposc_lab.models.chiral_p_wave import ChiralPWaveModel, ChiralPWaveParameters
from toposc_lab.observables.majorana import majorana_diagnostics
from toposc_lab.patterns.interventions import deletion_reasons
from toposc_lab.search.lexicographic_fitness import DerivedEvaluationRun
from toposc_lab.topology import SymmetryClassification, spectral_localizer


def evaluate_deletion(
    geometry: Geometry, seed: int, provenance: ReproducibilityMetadata
) -> tuple[DatasetRecord, DerivedEvaluationRun]:
    reasons = deletion_reasons(geometry)
    if reasons:
        raise ValueError(f"inadmissible single-edge simplification: {reasons}")
    proposal = candidate(geometry)
    model = ChiralPWaveModel(
        geometry, ChiralPWaveParameters.model_validate(dict(proposal.model.parameters))
    )
    run = evaluate_geometry(
        geometry,
        adapter=GeometryModelAdapter(
            lambda g: model,
            nambu_basis_resolver=lambda m: model.nambu_basis,
        ),
        seed=seed,
        code_version=provenance.git_commit,
    )
    if not run.is_valid or run.simulation_result is None:
        raise ValueError(f"exact simplification failed: {run.failure}")
    simulation = run.simulation_result
    energies = simulation.eigenvalues
    hamiltonian = model.hamiltonian()
    phs = float(np.max(np.abs(energies + energies[::-1])))
    exchange = model.nambu_basis.particle_hole_operator
    operator_phs = float(np.max(np.abs(exchange @ hamiltonian.conj() @ exchange + hamiltonian)))
    assert geometry.coordinates is not None
    symmetry = SymmetryClassification.from_signature(
        time_reversal_square=None,
        particle_hole_square=1,
        chiral_symmetry=False,
    )
    localizers = [
        spectral_localizer(
            hamiltonian,
            np.tile(geometry.coordinates, (2, 1)),
            (2.5, 2.5),
            symmetry,
            kappa=k,
            tolerance=1e-10,
        )
        for k in KAPPAS
    ]
    indices = [r.local_chern_number for r in localizers]
    resolved = all(r.is_invertible for r in localizers) and len(set(indices)) == 1
    eligible = resolved and indices[0] not in (None, 0) and max(phs, operator_phs) <= 1e-10
    gap = min(r.localizer_gap for r in localizers)
    diagnostics = [
        majorana_diagnostics(simulation.eigenvectors, int(i), model.nambu_basis)
        for i in np.argsort(np.abs(energies))[:4]
    ]
    weights = [
        float(sum(d.site_probability[i] for i in geometry.boundary_sites)) for d in diagnostics
    ]
    quality = float(eligible) * gap
    metrics: dict[str, Any] = {
        "quality": quality,
        "eligible": float(eligible),
        "success": float(quality >= 0.2),
        "localizer_gap": gap,
        "boundary_weight": float(np.mean(weights)),
        "boundary_weight_minimum": min(weights),
        "boundary_weights": weights,
        "polarization_norms": [float(d.polarization_norm) for d in diagnostics],
        "minimum_abs_energy": float(np.min(np.abs(energies))),
        "phs_mismatch": phs,
        "operator_phs_residual": operator_phs,
        "alternate_quality": quality * float(np.mean(weights)),
    }
    record = create_dataset_record(
        geometry=proposal.geometry,
        model=proposal.model,
        spectrum=SpectrumRecord(tuple(float(e) for e in energies), "hopping", "all", 72, True),
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
                    "scope": "phase16a.single-edge-deletion.v1",
                },
            ),
        ),
        topology=(
            TopologyResultRecord(
                "spectral_localizer_2d_grid",
                "1",
                TopologyValidity.VALID if resolved else TopologyValidity.UNRESOLVED,
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
                reason=None if resolved else "Index disagreement or unresolved localizer",
                warnings=("Single-edge deletion diagnostic; no phase or Majorana claim.",),
            ),
        ),
        robustness=(),
        provenance=replace(provenance, seed=seed),
    )
    validate_dataset_record(record).raise_for_errors()
    return record, DerivedEvaluationRun.from_run(
        run, identifier="phase16a.single-edge-deletion.v1", quantities={"quality": quality}
    )
