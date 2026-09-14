"""Exact evidence contracts, with no autonomous Majorana or phase certificate."""

from dataclasses import asdict
from typing import Any

import numpy as np

from toposc_lab.data import (
    DatasetRecord,
    ObservableResultRecord,
    RobustnessResultRecord,
    create_dataset_record,
)
from toposc_lab.discovery.config import DiscoveryConfig
from toposc_lab.generative.physics import KAPPAS
from toposc_lab.models.chiral_p_wave import ChiralPWaveModel, ChiralPWaveParameters
from toposc_lab.observables.majorana import finite_size_splitting_diagnostics, majorana_diagnostics
from toposc_lab.robustness.disorder import exact_hamiltonian_id
from toposc_lab.robustness.ensemble import DisorderEnsembleRequest
from toposc_lab.robustness.metrics import RobustnessFractionMetric
from toposc_lab.robustness.onsite import apply_uniform_onsite_disorder
from toposc_lab.robustness.uncertainty import estimate_robustness_uncertainty
from toposc_lab.search.lexicographic_fitness import DerivedEvaluationRun
from toposc_lab.topology import SymmetryClassification, spectral_localizer


def model_for(record: DatasetRecord) -> ChiralPWaveModel:
    return ChiralPWaveModel(
        record.geometry.to_geometry(),
        ChiralPWaveParameters.model_validate(dict(record.model.parameters)),
    )


def majorana_evidence(record: DatasetRecord, run: DerivedEvaluationRun) -> dict[str, Any]:
    model = model_for(record)
    simulation = run.simulation_result
    assert simulation is not None
    energies = simulation.eigenvalues
    hamiltonian = model.hamiltonian()
    exchange = model.nambu_basis.particle_hole_operator
    residual = float(np.max(np.abs(exchange @ hamiltonian.conj() @ exchange + hamiltonian)))
    states = []
    for index in np.argsort(np.abs(energies))[:4]:
        d = majorana_diagnostics(simulation.eigenvectors, int(index), model.nambu_basis)
        states.append(
            {
                "index": int(index),
                "energy": float(energies[index]),
                "site_probability": d.site_probability.tolist(),
                "ipr": float(np.sum(d.site_probability**2)),
                "boundary_weight": float(
                    sum(d.site_probability[i] for i in model.geometry.boundary_sites)
                ),
                "self_conjugacy": d.self_conjugacy,
                "polarization_norm": d.polarization_norm,
                "polarization_real": d.polarization.real.tolist(),
                "polarization_imag": d.polarization.imag.tolist(),
                "particle_weight": d.particle_weight,
                "hole_weight": d.hole_weight,
            }
        )
    splitting = finite_size_splitting_diagnostics(energies)
    return {
        "status": "diagnostics_only",
        "majorana_claim": False,
        "states": states,
        "operator_phs_residual": residual,
        "splitting": asdict(splitting),
        "zero_tolerance": 1e-10,
        "splitting_tolerance": 1e-3,
        "splitting_phs_tolerance": 1e-8,
        "reason": "Finite chiral-p-wave boundary eigenstates do not establish separated "
        "Majorana zero modes; no vortex/defect or size-convergence evidence.",
        "basis_caution": "Individual eigenstates can rotate within degenerate subspaces.",
    }


def disorder_evidence(record: DatasetRecord, seed: int, config: DiscoveryConfig) -> dict[str, Any]:
    model = model_for(record)
    realization = apply_uniform_onsite_disorder(
        model.geometry,
        model.hamiltonian(),
        width=config.onsite_width,
        seed=seed,
        nambu_basis=model.nambu_basis,
    )
    matrix = np.asarray(realization.state)
    energies, vectors = np.linalg.eigh(matrix)
    exchange = model.nambu_basis.particle_hole_operator
    phs = float(np.max(np.abs(exchange @ matrix.conj() @ exchange + matrix)))
    spectral_phs = float(np.max(np.abs(energies + energies[::-1])))
    symmetry = SymmetryClassification.from_signature(
        time_reversal_square=None, particle_hole_square=1, chiral_symmetry=False
    )
    assert model.geometry.coordinates is not None
    coordinates = np.tile(model.geometry.coordinates, (2, 1))
    localizers = [
        spectral_localizer(
            matrix, coordinates, config.probe, symmetry, kappa=k, tolerance=config.tolerance
        )
        for k in KAPPAS
    ]
    indices = [r.local_chern_number for r in localizers]
    eligible = (
        all(r.is_invertible for r in localizers)
        and indices[0] not in (None, 0)
        and len(set(indices)) == 1
        and max(phs, spectral_phs) <= config.tolerance
    )
    quality = min(r.localizer_gap for r in localizers) if eligible else 0.0
    low = [
        majorana_diagnostics(vectors, int(i), model.nambu_basis)
        for i in np.argsort(np.abs(energies))[:4]
    ]
    return {
        "kind": "exact",
        "seed": seed,
        "hamiltonian_id": exact_hamiltonian_id(matrix),
        "onsite_width": config.onsite_width,
        "distribution": "uniform [-width/2,width/2]",
        "spectrum": energies.tolist(),
        "indices": indices,
        "localizer_gaps": [r.localizer_gap for r in localizers],
        "operator_phs_residual": phs,
        "spectral_phs_residual": spectral_phs,
        "eligible": bool(eligible),
        "quality": quality,
        "success": bool(eligible and quality >= config.success_threshold),
        "minimum_abs_energy": float(np.min(np.abs(energies))),
        "boundary_weights": [
            float(sum(d.site_probability[i] for i in model.geometry.boundary_sites)) for d in low
        ],
        "polarization_norms": [d.polarization_norm for d in low],
        "majorana_claim": False,
    }


def integrate_validation(
    record: DatasetRecord,
    majorana: dict[str, Any],
    members: list[dict[str, Any]],
    config: DiscoveryConfig,
) -> DatasetRecord:
    seeds = tuple(m["seed"] for m in members)
    values = np.array([m["quality"] for m in members])
    metric = RobustnessFractionMetric(
        "frozen_quality_success",
        "Exact eligible quality >=0.20",
        DisorderEnsembleRequest(seeds),
        tuple(m["success"] for m in members),
    )
    interval = estimate_robustness_uncertainty(metric)
    robustness = RobustnessResultRecord(
        "uniform_onsite_finite_localizer",
        "1",
        seeds,
        {
            "width": config.onsite_width,
            "success_threshold": config.success_threshold,
            "members": tuple(members),
            "ensemble_scope": "fixed geometry, scalar onsite disorder only",
        },
        {
            "quality_mean": float(np.mean(values)),
            "quality_min": float(np.min(values)),
            "quality_sd": float(np.std(values, ddof=1)),
            "success_fraction": metric.value,
            "eligible_fraction": float(np.mean([m["eligible"] for m in members])),
            "samples": len(members),
        },
        {
            "quality_standard_error": float(np.std(values, ddof=1) / np.sqrt(len(values))),
            "success_wilson_lower": interval.lower_bound,
            "success_wilson_upper": interval.upper_bound,
        },
    )
    return create_dataset_record(
        geometry=record.geometry,
        model=record.model,
        spectrum=record.spectrum,
        topology=record.topology,
        provenance=record.provenance,
        observables=(
            *record.observables,
            ObservableResultRecord("discovery_majorana_diagnostics", "1", majorana),
            ObservableResultRecord(
                "finite_size_family_validation",
                "1",
                {
                    "status": "unavailable",
                    "phase_claim": False,
                    "reason": "Fixed 36-site wiring stratum has no declared size-extension map; "
                    "unrelated sizes are not a scaling sequence of this candidate.",
                },
            ),
        ),
        robustness=(robustness,),
    )
