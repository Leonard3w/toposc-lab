"""Versioned exact finite-system physics for connectivity research.

This adapter extends the Phase-14/15 center-probe localizer algorithm to a
configurable square set of sites. It does not extend its scientific claim to a
thermodynamic phase, a bulk gap, or validated Majorana modes. Every call to
``evaluate`` performs one independently budgeted exact stage; the engine owns
attempt accounting, failures, and persistence.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal, TypeGuard, cast

import numpy as np

from toposc_lab.data import (
    GeometryRecord,
    ModelParametersRecord,
    ObservableResultRecord,
    ReproducibilityMetadata,
    SpectrumRecord,
    TopologyResultRecord,
    TopologyValidity,
    create_dataset_record,
    record_to_dict,
    validate_dataset_record,
)
from toposc_lab.generative.physics import KAPPAS
from toposc_lab.geometry import Geometry
from toposc_lab.models.chiral_p_wave import ChiralPWaveModel, ChiralPWaveParameters
from toposc_lab.observables.majorana import finite_size_splitting_diagnostics, majorana_diagnostics
from toposc_lab.robustness.disorder import exact_hamiltonian_id
from toposc_lab.robustness.ensemble import DisorderEnsembleRequest
from toposc_lab.robustness.metrics import RobustnessFractionMetric
from toposc_lab.robustness.onsite import apply_uniform_onsite_disorder
from toposc_lab.robustness.uncertainty import estimate_robustness_uncertainty
from toposc_lab.topology import SymmetryClassification, spectral_localizer

ADAPTER_ID = "phase17.fixed-sites-chiral-p-wave.v1"
EVIDENCE_SCOPE = "Finite square-site center-probe evidence; no phase or Majorana claim."


@dataclass(frozen=True, slots=True)
class ObjectiveDefinition:
    key: str
    description: str
    evaluate: Callable[[dict[str, Any]], float | None]


@dataclass(frozen=True, slots=True)
class ValidatorDefinition:
    key: str
    description: str
    state: str
    evaluate: Callable[[dict[str, Any], PhysicsProtocol], dict[str, Any]]


OBJECTIVE_REGISTRY: dict[str, ObjectiveDefinition] = {}
VALIDATOR_REGISTRY: dict[str, ValidatorDefinition] = {}


def register_objective(definition: ObjectiveDefinition) -> None:
    """Register a named objective without silently replacing an existing rule."""
    if not definition.key.isidentifier() or definition.key in OBJECTIVE_REGISTRY:
        raise ValueError("objective key must be a new identifier")
    if not callable(definition.evaluate):
        raise TypeError("objective evaluator must be callable")
    OBJECTIVE_REGISTRY[definition.key] = definition


def register_validator(definition: ValidatorDefinition) -> None:
    """Add an explicit evidence gate; existing gates cannot be overwritten."""
    if not definition.key.isidentifier() or definition.key in VALIDATOR_REGISTRY:
        raise ValueError("validator key must be a new identifier")
    if not callable(definition.evaluate):
        raise TypeError("validator evaluator must be callable")
    VALIDATOR_REGISTRY[definition.key] = definition


@dataclass(frozen=True, slots=True)
class PhysicsProtocol:
    adapter_id: str = ADAPTER_ID
    disorder_widths: tuple[float, ...] = (0.0, 0.2, 0.4)
    disorder_seeds: tuple[int, ...] = (17001, 17002, 17003, 17004)
    clean_seed: int = 17000
    confirmation: bool = True
    objective: str = "robustness_success_fraction"
    validators: tuple[str, ...] = (
        "spectrum",
        "topology",
        "majorana",
        "robustness",
        "reproducibility",
        "finite_size",
    )
    finite_size: bool = False
    model: str = "chiral_p_wave"
    hopping: float = 1.0
    chemical_potential: float = 2.0
    pairing: float = 1.0
    chirality: int = 1
    basis: str = "component_major"
    solver: str = "numpy.linalg.eigh"
    tolerance: float = 1e-10
    success_threshold: float = 0.20
    kappas: tuple[float, ...] = KAPPAS
    probe_rule: str = "square_bounding_box_center"

    def __post_init__(self) -> None:
        for name in ("disorder_widths", "disorder_seeds", "kappas", "validators"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.adapter_id != ADAPTER_ID:
            raise ValueError("a new physical contract requires a registered versioned adapter")
        for name in ("confirmation", "finite_size"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if type(self.clean_seed) is not int or self.clean_seed < 0:
            raise ValueError("clean_seed must be a nonnegative integer")
        if any(type(seed) is not int or seed < 0 for seed in self.disorder_seeds):
            raise ValueError("disorder seeds must be nonnegative integers")
        if len(set(self.disorder_seeds)) != len(self.disorder_seeds):
            raise ValueError("disorder seeds must be unique")
        if any(
            isinstance(width, bool) or not math.isfinite(width) or width < 0
            for width in self.disorder_widths
        ):
            raise ValueError("disorder widths must be finite nonnegative numbers")
        if tuple(sorted(set(self.disorder_widths))) != self.disorder_widths:
            raise ValueError("disorder widths must be unique and increasing")
        if self.objective not in OBJECTIVE_REGISTRY:
            raise ValueError(f"unknown objective: {self.objective}")
        if len(set(self.validators)) != len(self.validators) or any(
            key not in VALIDATOR_REGISTRY for key in self.validators
        ):
            raise ValueError("validators must be unique registered keys")
        if "spectrum" not in self.validators:
            raise ValueError("exact spectrum validation is mandatory")
        if "robustness" in self.validators and (
            not self.disorder_widths or len(self.disorder_seeds) < 2
        ):
            raise ValueError("robustness requires widths and at least two distinct seeds")
        if self.objective.startswith("robustness_") and "robustness" not in self.validators:
            raise ValueError("the robustness objective requires its exact ensemble validator")
        scientific = (
            self.model,
            self.hopping,
            self.chemical_potential,
            self.pairing,
            self.chirality,
            self.basis,
            self.solver,
            self.tolerance,
            self.success_threshold,
            self.kappas,
            self.probe_rule,
        )
        if (
            scientific
            != (
                "chiral_p_wave",
                1.0,
                2.0,
                1.0,
                1,
                "component_major",
                "numpy.linalg.eigh",
                1e-10,
                0.20,
                KAPPAS,
                "square_bounding_box_center",
            )
            or type(self.chirality) is not int
            or any(
                isinstance(value, bool)
                for value in (
                    self.hopping,
                    self.chemical_potential,
                    self.pairing,
                    self.tolerance,
                    self.success_threshold,
                    *self.kappas,
                )
            )
        ):
            raise ValueError("scientific definitions are frozen in this versioned adapter")


def _valid_result(result: Mapping[str, Any] | None) -> TypeGuard[Mapping[str, Any]]:
    return bool(
        result
        and result.get("kind") == "exact"
        and not result.get("error")
        and result.get("status") != "failed"
    )


@dataclass(frozen=True, slots=True)
class FiniteSystemEvaluator:
    protocol: PhysicsProtocol = field(default_factory=PhysicsProtocol)
    provenance: ReproducibilityMetadata | None = None

    def plan(self) -> list[dict[str, Any]]:
        """Same seed protocol for all candidates and matched references."""
        stages = [{"key": "clean", "kind": "clean", "seed": self.protocol.clean_seed, "width": 0.0}]
        if self.protocol.confirmation:
            stages.append(
                {
                    "key": "confirmation",
                    "kind": "confirmation",
                    "seed": self.protocol.clean_seed,
                    "width": 0.0,
                }
            )
        if "robustness" in self.protocol.validators:
            for width_index, width in enumerate(self.protocol.disorder_widths):
                for seed_index, seed in enumerate(self.protocol.disorder_seeds):
                    stages.append(
                        {
                            "key": f"disorder_{width_index}_{seed_index}",
                            "kind": "disorder",
                            "seed": seed,
                            "width": width,
                        }
                    )
        return stages

    def evaluate(self, geometry: Geometry, stage: Mapping[str, Any]) -> dict[str, Any]:
        """Compute one clean, confirmation, or single-realization exact stage."""
        if dict(stage) not in self.plan():
            raise ValueError("stage must match the serialized protocol plan")
        probe = _square_probe(geometry)
        params = ChiralPWaveParameters(
            hopping=self.protocol.hopping,
            chemical_potential=self.protocol.chemical_potential,
            pairing=self.protocol.pairing,
            chirality=cast(Literal[-1, 1], self.protocol.chirality),
        )
        model = ChiralPWaveModel(geometry, params)
        clean_matrix = model.hamiltonian()
        matrix = clean_matrix
        seed = int(stage["seed"])
        width = float(stage["width"])
        if stage["kind"] == "disorder":
            matrix = np.asarray(
                apply_uniform_onsite_disorder(
                    geometry, clean_matrix, width=width, seed=seed, nambu_basis=model.nambu_basis
                ).state
            )
        hermiticity = float(np.max(np.abs(matrix - matrix.conj().T)))
        if hermiticity > self.protocol.tolerance or not np.isfinite(matrix).all():
            raise ValueError("Hamiltonian failed finite Hermitian matrix validation")
        energies, vectors = np.linalg.eigh(matrix)
        if not np.isfinite(energies).all() or not np.isfinite(vectors).all():
            raise ValueError("eigensolver returned non-finite results")
        residual = float(np.max(np.abs(matrix @ vectors - vectors * energies)))
        if residual > self.protocol.tolerance:
            raise ValueError("eigensystem residual exceeds the numerical tolerance")
        exchange = model.nambu_basis.particle_hole_operator
        operator_phs = float(np.max(np.abs(exchange @ matrix.conj() @ exchange + matrix)))
        spectral_phs = float(np.max(np.abs(energies + energies[::-1])))
        symmetry = SymmetryClassification.from_signature(
            time_reversal_square=None, particle_hole_square=1, chiral_symmetry=False
        )
        assert geometry.coordinates is not None
        coordinates = np.tile(geometry.coordinates, (2, 1))
        localizers = [
            spectral_localizer(
                matrix, coordinates, probe, symmetry, kappa=kappa, tolerance=self.protocol.tolerance
            )
            for kappa in self.protocol.kappas
        ]
        indices = [localizer.local_chern_number for localizer in localizers]
        consistent = all(item.is_invertible for item in localizers) and len(set(indices)) == 1
        eligible = bool(
            consistent
            and indices[0] not in (None, 0)
            and max(operator_phs, spectral_phs) <= self.protocol.tolerance
        )
        gap = float(min(item.localizer_gap for item in localizers))
        quality = gap if eligible else 0.0
        states = []
        for index in np.argsort(np.abs(energies))[:4]:
            diagnostic = majorana_diagnostics(vectors, int(index), model.nambu_basis)
            states.append(
                {
                    "index": int(index),
                    "energy": float(energies[index]),
                    "site_probability": diagnostic.site_probability.tolist(),
                    "ipr": float(np.sum(diagnostic.site_probability**2)),
                    "boundary_weight": float(
                        sum(diagnostic.site_probability[i] for i in geometry.boundary_sites)
                    ),
                    "self_conjugacy": float(diagnostic.self_conjugacy),
                    "polarization_norm": float(diagnostic.polarization_norm),
                    "polarization_real": diagnostic.polarization.real.tolist(),
                    "polarization_imag": diagnostic.polarization.imag.tolist(),
                    "particle_weight": float(diagnostic.particle_weight),
                    "hole_weight": float(diagnostic.hole_weight),
                }
            )
        majorana: dict[str, Any] = {
            "status": "diagnostics_only",
            "majorana_claim": False,
            "states": states,
            "splitting": asdict(finite_size_splitting_diagnostics(energies)),
            "zero_tolerance": 1e-10,
            "splitting_tolerance": 1e-3,
            "splitting_phs_tolerance": 1e-8,
            "reason": "Boundary eigenstates and polarization do not establish separated "
            "Majorana modes; no defect or converged size-extension evidence.",
            "basis_caution": "Eigenstates can rotate within degenerate subspaces.",
        }
        boundary_weights = [state["boundary_weight"] for state in states]
        metrics = {
            "quality": quality,
            "eligible": eligible,
            "success": bool(eligible and quality >= self.protocol.success_threshold),
            "localizer_gap": gap,
            "minimum_abs_energy": float(np.min(np.abs(energies))),
            "boundary_weight": float(np.mean(boundary_weights)),
            "boundary_weight_minimum": min(boundary_weights),
            "boundary_weights": boundary_weights,
            "polarization_norms": [state["polarization_norm"] for state in states],
            "operator_phs_residual": operator_phs,
            "spectral_phs_residual": spectral_phs,
            "hermiticity_residual": hermiticity,
            "eigensystem_residual": residual,
            "bulk_gap": None,
            "bulk_gap_status": "unavailable_without_bulk_edge_separation",
        }
        result: dict[str, Any] = {
            "kind": "exact",
            "status": "completed",
            "stage": dict(stage),
            "adapter_id": ADAPTER_ID,
            "seed": seed,
            "onsite_width": width,
            "hamiltonian_id": exact_hamiltonian_id(matrix),
            "n_sites": geometry.n_sites,
            "basis": self.protocol.basis,
            "spectrum": energies.tolist(),
            "metrics": metrics,
            "indices": indices,
            "localizer_gaps": [float(x.localizer_gap) for x in localizers],
            "localizer_signatures": [int(x.signature) for x in localizers],
            "probe": list(probe),
            "kappas": list(self.protocol.kappas),
            "topology": {
                "status": "validated"
                if eligible
                else "unresolved"
                if not consistent
                else "trivial",
                "finite_system": True,
                "phase_claim": False,
                "eligible": eligible,
                "indices": indices,
            },
            "majorana": majorana,
            "phase_claim": False,
            "majorana_claim": False,
            "scope": EVIDENCE_SCOPE,
            "distribution": "uniform [-width/2,width/2]",
            "onsite_offsets": np.real(np.diag(matrix - clean_matrix)[: geometry.n_sites]).tolist(),
        }
        provenance = self.provenance or ReproducibilityMetadata(
            seed=seed,
            git_commit="unrecorded",
            git_dirty=True,
            package_version="0.1.0",
            solver_name="exact_diagonalization",
            solver_version=np.__version__,
            solver_settings={"solver": self.protocol.solver, "basis": self.protocol.basis},
            tolerances={"numerical": self.protocol.tolerance},
            timestamp_utc=datetime.now(UTC).isoformat(),
        )
        provenance = replace(
            provenance,
            seed=seed,
            solver_settings={
                **dict(provenance.solver_settings),
                "adapter_id": ADAPTER_ID,
                "protocol": asdict(self.protocol),
                "stage": dict(stage),
            },
        )
        record = create_dataset_record(
            geometry=GeometryRecord.from_geometry(geometry, family_label="fixed_square_sites"),
            model=ModelParametersRecord(self.protocol.model, "1", params.model_dump()),
            spectrum=SpectrumRecord(
                tuple(float(x) for x in energies), "hopping", "all", 2 * geometry.n_sites, True
            ),
            observables=(
                ObservableResultRecord(
                    "finite_geometry_quality",
                    "2",
                    metrics,
                    conventions={
                        "finite_system": True,
                        "phase_claim": False,
                        "majorana_claim": False,
                        "adapter_id": ADAPTER_ID,
                    },
                ),
                ObservableResultRecord("research_majorana_diagnostics", "1", majorana),
                ObservableResultRecord(
                    "research_exact_stage",
                    "1",
                    {
                        "stage": dict(stage),
                        "onsite_offsets": result["onsite_offsets"],
                        "hamiltonian_id": result["hamiltonian_id"],
                    },
                ),
            ),
            topology=(
                TopologyResultRecord(
                    "spectral_localizer_2d_grid",
                    "2",
                    TopologyValidity.VALID if consistent else TopologyValidity.UNRESOLVED,
                    indices[0] if consistent else None,
                    bool(indices[0]) if consistent else None,
                    {
                        "kappas": self.protocol.kappas,
                        "probe": probe,
                        "localizer_gaps": result["localizer_gaps"],
                        "indices": tuple(indices),
                    },
                    {"numerical": self.protocol.tolerance},
                    reason=None if consistent else "Localizer unresolved or indices disagree",
                    warnings=(EVIDENCE_SCOPE,),
                ),
            ),
            robustness=(),
            provenance=provenance,
        )
        validate_dataset_record(record).raise_for_errors()
        result["dataset_record"] = record_to_dict(record)
        return result

    def summarize(self, results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """Preserve partial evidence, and never turn an absent stage into a good score."""
        plan = self.plan()
        known = {stage["key"] for stage in plan}
        if set(results) - known:
            raise ValueError("results contain stages outside this protocol")
        missing = sorted(known - set(results))
        failed = sorted(key for key, result in results.items() if not _valid_result(result))
        clean = dict(results["clean"]) if _valid_result(results.get("clean")) else None
        confirmation = results.get("confirmation")
        confirmed = False
        comparison: dict[str, Any] = {"status": "not_requested"}
        if self.protocol.confirmation:
            comparison = {"status": "pending"}
            if clean and _valid_result(confirmation):
                assert confirmation is not None
                left = np.asarray(clean["spectrum"])
                right = np.asarray(confirmation["spectrum"])
                difference = (
                    float(np.max(np.abs(left - right))) if left.shape == right.shape else None
                )
                quality_difference = abs(
                    clean["metrics"]["quality"] - confirmation["metrics"]["quality"]
                )
                confirmed = bool(
                    difference is not None
                    and difference <= self.protocol.tolerance
                    and quality_difference <= self.protocol.tolerance
                    and clean["hamiltonian_id"] == confirmation["hamiltonian_id"]
                    and clean["indices"] == confirmation["indices"]
                )
                comparison = {
                    "status": "validated" if confirmed else "failed",
                    "max_spectrum_difference": difference,
                    "quality_difference": quality_difference,
                    "tolerance": self.protocol.tolerance,
                }
        robustness = _ensemble_summaries(self.protocol, plan, results)
        complete = not missing and not failed
        score_ready = bool(complete and clean and (confirmed or not self.protocol.confirmation))
        raw = dict(clean["metrics"]) if clean else {}
        raw["robustness_success_fraction"] = (
            float(np.mean([group["success_fraction"] for group in robustness]))
            if robustness and all(group["complete"] for group in robustness)
            else None
        )
        raw["robustness_quality_mean"] = (
            float(np.mean([group["quality_mean"] for group in robustness]))
            if robustness
            and all(group["complete"] and group["quality_mean"] is not None for group in robustness)
            else None
        )
        summary: dict[str, Any] = {
            "adapter_id": ADAPTER_ID,
            "origin": "exact",
            "scope": EVIDENCE_SCOPE,
            "raw_metrics": raw,
            "robustness": robustness,
            "reproducibility": comparison,
            "complete": complete,
            "score_ready": score_ready,
            "missing_stages": missing,
            "failed_stages": failed,
            "exact_stages_completed": len(results) - len(failed),
            "clean": clean,
            "majorana": clean["majorana"] if clean else {"status": "pending"},
            "phase_claim": False,
            "majorana_claim": False,
            "critical_disorder_strength": {
                "status": "unavailable",
                "value": None,
                "reason": "Sampled finite-localizer success fractions do not define a "
                "thermodynamic critical disorder strength W_c.",
            },
            "finite_size": {
                "status": "unavailable",
                "requested": self.protocol.finite_size,
                "phase_claim": False,
                "reason": "No declared candidate-preserving size-extension "
                "map; unrelated site counts are not a finite-size scaling sequence.",
            },
        }
        validations = {
            key: VALIDATOR_REGISTRY[key].evaluate(summary, self.protocol)
            for key in self.protocol.validators
        }
        states = [
            VALIDATOR_REGISTRY[key].state
            for key, result in validations.items()
            if result.get("passed")
        ]
        order = [
            "EXACT_EVALUATED",
            "TOPOLOGY_VALIDATED",
            "MAJORANA_VALIDATED",
            "ROBUSTNESS_VALIDATED",
            "FINITE_SIZE_VALIDATED",
        ]
        state = next((item for item in reversed(order) if item in states), "PROPOSED")
        if failed or comparison["status"] == "failed":
            state = "FAILED"
        summary.update(
            validation_state=state,
            validation_states=states,
            validation_results=validations,
            objective=self.protocol.objective,
            score=OBJECTIVE_REGISTRY[self.protocol.objective].evaluate(summary)
            if score_ready
            else None,
            warnings=[
                EVIDENCE_SCOPE,
                summary["finite_size"]["reason"],
                "minimum_abs_energy is a finite spectral gap, not a bulk gap.",
            ],
        )
        return summary


def _square_probe(geometry: Geometry) -> tuple[float, float]:
    side = math.isqrt(geometry.n_sites)
    if side < 2 or side * side != geometry.n_sites or geometry.coordinates is None:
        raise ValueError("adapter requires a square set of at least four fixed sites")
    expected = {(float(x), float(y)) for y in range(side) for x in range(side)}
    if (
        geometry.coordinates.shape != (geometry.n_sites, 2)
        or set(map(tuple, geometry.coordinates.tolist())) != expected
    ):
        raise ValueError("adapter requires unit-spaced fixed square sites anchored at (0,0)")
    boundary = frozenset(
        index
        for index, (x, y) in enumerate(geometry.coordinates)
        if x in (0, side - 1) or y in (0, side - 1)
    )
    if geometry.boundary_sites != boundary:
        raise ValueError("outer geometric boundary must remain fixed")
    for edge in geometry.edges:
        if edge.boundary_crossing or (
            edge.displacement is not None
            and not np.array_equal(
                edge.displacement,
                geometry.coordinates[edge.target] - geometry.coordinates[edge.source],
            )
        ):
            raise ValueError("periodic or modified displacement conventions are unsupported")
    return ((side - 1) / 2, (side - 1) / 2)


def _ensemble_summaries(
    protocol: PhysicsProtocol, plan: list[dict[str, Any]], results: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    if "robustness" not in protocol.validators:
        return groups
    for width in protocol.disorder_widths:
        stages = [
            stage for stage in plan if stage["kind"] == "disorder" and stage["width"] == width
        ]
        members = [results.get(stage["key"]) for stage in stages]
        missing = [
            stage["key"] for stage, member in zip(stages, members, strict=True) if member is None
        ]
        failures = tuple(
            index
            for index, member in enumerate(members)
            if member is not None and not _valid_result(member)
        )
        successes = tuple(
            bool(_valid_result(member) and member["metrics"]["success"]) for member in members
        )
        metric = RobustnessFractionMetric(
            "frozen_quality_success",
            "Exact eligible quality >=0.20",
            DisorderEnsembleRequest(protocol.disorder_seeds),
            successes,
            failures,
        )
        interval = estimate_robustness_uncertainty(metric)
        qualities = [member["metrics"]["quality"] for member in members if _valid_result(member)]
        complete = not missing and not failures
        groups.append(
            {
                "width": width,
                "seeds": list(protocol.disorder_seeds),
                "complete": complete,
                "requested_samples": len(stages),
                "completed_samples": len(qualities),
                "successful_samples": metric.successful_count,
                "missing_stages": missing,
                "execution_failure_indices": list(failures),
                "successes": list(successes),
                "success_fraction": metric.value if not missing else None,
                "success_wilson_lower": interval.lower_bound if not missing else None,
                "success_wilson_upper": interval.upper_bound if not missing else None,
                "confidence_level": 0.95,
                "uncertainty_method": "wilson_score",
                "quality_mean": float(np.mean(qualities)) if qualities else None,
                "quality_min": float(np.min(qualities)) if qualities else None,
                "quality_sd": float(np.std(qualities, ddof=1)) if len(qualities) > 1 else None,
                "scope": "fixed geometry, scalar onsite disorder, same seeds across candidates",
                "success_threshold": protocol.success_threshold,
                "failure_denominator": "Execution failures count as unsuccessful requested samples.",
                "phase_claim": False,
                "majorana_claim": False,
            }
        )
    return groups


def _spectrum_validation(summary: dict[str, Any], protocol: PhysicsProtocol) -> dict[str, Any]:
    passed = summary["clean"] is not None
    return {
        "passed": passed,
        "status": "validated" if passed else "pending",
        "scope": "finite complete BdG spectrum",
        "tolerance": protocol.tolerance,
    }


def _topology_validation(summary: dict[str, Any], protocol: PhysicsProtocol) -> dict[str, Any]:
    clean = summary["clean"]
    passed = bool(clean and clean["metrics"]["eligible"])
    return {
        "passed": passed,
        "status": clean["topology"]["status"] if clean else "pending",
        "scope": EVIDENCE_SCOPE,
        "phase_claim": False,
    }


def _majorana_validation(summary: dict[str, Any], protocol: PhysicsProtocol) -> dict[str, Any]:
    return {"passed": False, **summary["majorana"], "majorana_claim": False}


def _robustness_validation(summary: dict[str, Any], protocol: PhysicsProtocol) -> dict[str, Any]:
    groups = summary["robustness"]
    passed = bool(
        summary["score_ready"]
        and summary["raw_metrics"].get("success")
        and groups
        and all(group["complete"] and all(group["successes"]) for group in groups)
    )
    return {
        "passed": passed,
        "status": "validated"
        if passed
        else "evaluated"
        if all(group["complete"] for group in groups)
        else "pending",
        "scope": "All requested finite-localizer samples satisfy the frozen success criterion; "
        "this is no claim about unobserved seeds, widths, or Majorana robustness.",
        "phase_claim": False,
        "majorana_claim": False,
    }


def _reproducibility_validation(
    summary: dict[str, Any], protocol: PhysicsProtocol
) -> dict[str, Any]:
    return {
        "passed": summary["reproducibility"]["status"] == "validated",
        **summary["reproducibility"],
    }


def _finite_size_validation(summary: dict[str, Any], protocol: PhysicsProtocol) -> dict[str, Any]:
    return {"passed": False, **summary["finite_size"]}


for _objective in (
    ObjectiveDefinition(
        "robustness_success_fraction",
        "Mean empirical success fraction over the declared sampled widths (not W_c).",
        lambda summary: summary["raw_metrics"].get("robustness_success_fraction"),
    ),
    ObjectiveDefinition(
        "robustness_quality_mean",
        "Mean exact eligible finite-localizer quality over requested disorder samples.",
        lambda summary: summary["raw_metrics"].get("robustness_quality_mean"),
    ),
    ObjectiveDefinition(
        "clean_quality",
        "Exact clean eligible finite-localizer quality.",
        lambda summary: summary["raw_metrics"].get("quality"),
    ),
):
    register_objective(_objective)

for _validator in (
    ValidatorDefinition(
        "spectrum", "Exact finite spectrum", "EXACT_EVALUATED", _spectrum_validation
    ),
    ValidatorDefinition(
        "topology", "Finite center-probe localizers", "TOPOLOGY_VALIDATED", _topology_validation
    ),
    ValidatorDefinition("majorana", "Diagnostics only", "MAJORANA_VALIDATED", _majorana_validation),
    ValidatorDefinition(
        "robustness",
        "All sampled finite-evidence successes",
        "ROBUSTNESS_VALIDATED",
        _robustness_validation,
    ),
    ValidatorDefinition(
        "reproducibility",
        "Independent clean confirmation",
        "EXACT_EVALUATED",
        _reproducibility_validation,
    ),
    ValidatorDefinition(
        "finite_size", "No declared extension", "FINITE_SIZE_VALIDATED", _finite_size_validation
    ),
):
    register_validator(_validator)

EVALUATOR_REGISTRY: dict[str, type[FiniteSystemEvaluator]] = {ADAPTER_ID: FiniteSystemEvaluator}


@dataclass(frozen=True, slots=True)
class PhysicsAdapterDefinition:
    """Factories keep future physical protocols outside the workbench engine."""

    key: str
    description: str
    protocol_factory: Callable[..., Any]
    evaluator_factory: Callable[..., Any]


PHYSICS_REGISTRY: dict[str, PhysicsAdapterDefinition] = {}


def register_physics_adapter(definition: PhysicsAdapterDefinition) -> None:
    if not definition.key or definition.key in PHYSICS_REGISTRY:
        raise ValueError("physics adapter key must be new and nonempty")
    if not callable(definition.protocol_factory) or not callable(definition.evaluator_factory):
        raise TypeError("physics adapter factories must be callable")
    PHYSICS_REGISTRY[definition.key] = definition


def create_physics_protocol(settings: Mapping[str, Any], *, objective: str) -> Any:
    """Resolve a versioned protocol with one canonical objective choice."""
    key = settings.get("adapter_id", ADAPTER_ID)
    if key not in PHYSICS_REGISTRY:
        raise ValueError(f"unknown physics adapter: {key}")
    if "objective" in settings and settings["objective"] != objective:
        raise ValueError("physics.objective conflicts with the top-level experiment objective")
    return PHYSICS_REGISTRY[key].protocol_factory(**{**settings, "objective": objective})


def create_evaluator(protocol: Any, *, provenance: ReproducibilityMetadata) -> Any:
    return PHYSICS_REGISTRY[protocol.adapter_id].evaluator_factory(
        protocol=protocol, provenance=provenance
    )


def provenance_from_manifest(manifest: Mapping[str, Any], protocol: Any) -> ReproducibilityMetadata:
    """Use the archived launch provenance, never a new capture during resume."""
    environment = dict(manifest["environment"])
    packages = environment.get("packages", {})
    return ReproducibilityMetadata(
        seed=0,
        git_commit=manifest["git_commit"],
        git_dirty=manifest["git_dirty"],
        package_version=packages.get("toposc-lab", "source-checkout"),
        solver_name="exact_diagonalization",
        solver_version=packages.get("numpy", "unrecorded"),
        solver_settings={
            "adapter_id": protocol.adapter_id,
            "blas_threads": manifest.get("blas_threads", 1),
        },
        tolerances={"numerical": getattr(protocol, "tolerance", 1e-10)},
        timestamp_utc=manifest["timestamp"],
        runtime={
            "environment": environment,
            "source_sha256": manifest["source_sha256"],
            "source_zip_sha256": manifest["source_zip_sha256"],
            "experiment_id": manifest["experiment_id"],
        },
    )


register_physics_adapter(
    PhysicsAdapterDefinition(ADAPTER_ID, EVIDENCE_SCOPE, PhysicsProtocol, FiniteSystemEvaluator)
)
