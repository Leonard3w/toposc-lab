"""Unified, failure-aware orchestration of one geometry evaluation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
import math
from numbers import Integral, Real
from typing import Any, TypeAlias

import numpy as np

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import BasisLayout, SimulationResult
from toposc_lab.evaluation.descriptors import evaluate_geometry_descriptors
from toposc_lab.evaluation.eigenstates import evaluate_eigenstates
from toposc_lab.evaluation.majorana import evaluate_majorana_diagnostics
from toposc_lab.evaluation.reproducibility import (
    ReproducibilityRecord,
    create_reproducibility_record,
)
from toposc_lab.evaluation.results import GeometryEvaluation
from toposc_lab.evaluation.spectral import evaluate_spectrum
from toposc_lab.evaluation.topology import (
    TopologyIntegrationInput,
    integrate_topology_results,
)
from toposc_lab.evaluation.validity import (
    CandidateFailureStage,
    CandidateStageFailure,
    CandidateValidityPolicy,
    CandidateValidityReport,
    ModelGeometryRequirements,
    validate_candidate,
)
from toposc_lab.geometry.base import Geometry
from toposc_lab.hamiltonians.nambu import NambuBasis
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver
from toposc_lab.topology.dispatch import TopologyDispatchDecision

ModelFactory: TypeAlias = Callable[[Geometry], BaseModel]
NambuBasisResolver: TypeAlias = Callable[[BaseModel], NambuBasis]

_WARNING_PREFIX = "[evaluate_geometry] "


@dataclass(frozen=True, slots=True)
class GeometryModelAdapter:
    """Explicitly bind arbitrary geometries to one physical-model factory."""

    model_factory: ModelFactory
    requirements: ModelGeometryRequirements = ModelGeometryRequirements()
    nambu_basis_resolver: NambuBasisResolver | None = None

    def __post_init__(self) -> None:
        if not callable(self.model_factory):
            raise TypeError("model_factory must be callable")
        if not isinstance(self.requirements, ModelGeometryRequirements):
            raise TypeError("requirements must be ModelGeometryRequirements")
        if self.nambu_basis_resolver is not None and not callable(
            self.nambu_basis_resolver
        ):
            raise TypeError("nambu_basis_resolver must be callable or None")


@dataclass(frozen=True, slots=True)
class GeometryEvaluationConfig:
    """Numerical conventions and explicit final-validity requirements."""

    reference_energy: float = 0.0
    zero_mode_tolerance: float = 1.0e-10
    low_energy_count: int = 8
    boundary_localization_threshold: float = 0.5
    numerical_tolerance: float = 1.0e-10
    require_resolved_topology: bool = False
    require_topology_convergence: bool = False
    topology_convergence_checked: bool = False

    def __post_init__(self) -> None:
        reference = _finite_real(self.reference_energy, name="reference_energy")
        zero_tolerance = _positive_finite_real(
            self.zero_mode_tolerance,
            name="zero_mode_tolerance",
        )
        low_energy_count = _positive_integer(
            self.low_energy_count,
            name="low_energy_count",
        )
        boundary_threshold = _unit_interval_real(
            self.boundary_localization_threshold,
            name="boundary_localization_threshold",
        )
        numerical_tolerance = _positive_finite_real(
            self.numerical_tolerance,
            name="numerical_tolerance",
        )
        for name, value in (
            ("require_resolved_topology", self.require_resolved_topology),
            ("require_topology_convergence", self.require_topology_convergence),
            ("topology_convergence_checked", self.topology_convergence_checked),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")

        object.__setattr__(self, "reference_energy", reference)
        object.__setattr__(self, "zero_mode_tolerance", zero_tolerance)
        object.__setattr__(self, "low_energy_count", low_energy_count)
        object.__setattr__(
            self,
            "boundary_localization_threshold",
            boundary_threshold,
        )
        object.__setattr__(self, "numerical_tolerance", numerical_tolerance)


@dataclass(frozen=True, slots=True)
class GeometryEvaluationContext:
    """Read-only stage context supplied to a method-specific topology hook."""

    geometry: Geometry
    model: BaseModel
    hamiltonian: np.ndarray
    simulation_result: SimulationResult
    evaluation: GeometryEvaluation

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, Geometry):
            raise TypeError("geometry must be Geometry")
        if not isinstance(self.model, BaseModel):
            raise TypeError("model must be BaseModel")
        if not isinstance(self.simulation_result, SimulationResult):
            raise TypeError("simulation_result must be SimulationResult")
        if not isinstance(self.evaluation, GeometryEvaluation):
            raise TypeError("evaluation must be GeometryEvaluation")
        matrix = np.asarray(self.hamiltonian, dtype=complex)
        if matrix.shape != (
            self.simulation_result.dimension,
            self.simulation_result.dimension,
        ):
            raise ValueError("hamiltonian shape must match the simulation result")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("hamiltonian must contain only finite values")
        if matrix.flags.writeable:
            matrix = matrix.copy()
            matrix.setflags(write=False)
        object.__setattr__(self, "hamiltonian", matrix)


TopologyEvaluationHook: TypeAlias = Callable[
    [GeometryEvaluationContext],
    Iterable[TopologyIntegrationInput],
]


@dataclass(frozen=True, slots=True)
class GeometryEvaluationRun:
    """Separated numerical, scientific, and validity outputs of one pipeline run."""

    simulation_result: SimulationResult | None
    evaluation: GeometryEvaluation | None
    validity: CandidateValidityReport
    reproducibility: ReproducibilityRecord | None = None
    failure: CandidateStageFailure | None = None

    def __post_init__(self) -> None:
        if self.simulation_result is not None and not isinstance(
            self.simulation_result,
            SimulationResult,
        ):
            raise TypeError("simulation_result must be SimulationResult or None")
        if self.evaluation is not None and not isinstance(
            self.evaluation,
            GeometryEvaluation,
        ):
            raise TypeError("evaluation must be GeometryEvaluation or None")
        if not isinstance(self.validity, CandidateValidityReport):
            raise TypeError("validity must be CandidateValidityReport")
        if self.reproducibility is not None and not isinstance(
            self.reproducibility,
            ReproducibilityRecord,
        ):
            raise TypeError(
                "reproducibility must be ReproducibilityRecord or None"
            )
        if self.failure is not None and not isinstance(
            self.failure,
            CandidateStageFailure,
        ):
            raise TypeError("failure must be CandidateStageFailure or None")
        if self.evaluation is not None and (
            self.simulation_result is None or self.reproducibility is None
        ):
            raise ValueError(
                "an evaluation requires simulation and reproducibility results"
            )
        if self.validity.is_valid and (
            self.simulation_result is None
            or self.evaluation is None
            or self.reproducibility is None
            or self.failure is not None
        ):
            raise ValueError(
                "a valid run requires simulation and evaluation results without failure"
            )
        if self.failure is not None:
            failure_code = f"{self.failure.stage.value}_failure"
            if failure_code not in {issue.code for issue in self.validity.errors}:
                raise ValueError("validity report must contain the recorded stage failure")

    @property
    def is_valid(self) -> bool:
        """Whether all requested stages completed and final validity passed."""
        return self.validity.is_valid


def evaluate_geometry(
    geometry: Geometry,
    *,
    adapter: GeometryModelAdapter,
    config: GeometryEvaluationConfig | None = None,
    solver: ExactDiagonalizationSolver | None = None,
    topology_hook: TopologyEvaluationHook | None = None,
    topology_dispatch: TopologyDispatchDecision | None = None,
    seed: int | None = None,
    code_version: str | None = None,
) -> GeometryEvaluationRun:
    """Run the Phase 7 evaluation stages for one explicitly adapted geometry.

    Candidate-level exceptions raised by model, Hamiltonian, solver,
    evaluation, or topology work are converted into a stage failure and an
    invalid run. Invalid preflight or numerical checks stop later work without
    inventing an exception. API contract errors are raised before execution.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be Geometry")
    if not isinstance(adapter, GeometryModelAdapter):
        raise TypeError("adapter must be GeometryModelAdapter")
    if config is None:
        config = GeometryEvaluationConfig()
    elif not isinstance(config, GeometryEvaluationConfig):
        raise TypeError("config must be GeometryEvaluationConfig or None")
    if solver is None:
        solver = ExactDiagonalizationSolver()
    elif not isinstance(solver, ExactDiagonalizationSolver):
        raise TypeError("solver must be ExactDiagonalizationSolver or None")
    if topology_hook is not None and not callable(topology_hook):
        raise TypeError("topology_hook must be callable or None")
    if topology_dispatch is not None and not isinstance(
        topology_dispatch,
        TopologyDispatchDecision,
    ):
        raise TypeError("topology_dispatch must be TopologyDispatchDecision or None")
    if (topology_hook is None) != (topology_dispatch is None):
        raise ValueError("topology_hook and topology_dispatch must be supplied together")
    _validate_reproducibility_inputs(seed=seed, code_version=code_version)

    policy = _validity_policy(adapter=adapter, config=config)
    execution_policy = replace(
        policy,
        require_resolved_topology=False,
        require_topology_convergence=False,
    )
    preflight = validate_candidate(geometry, policy=execution_policy)
    if not preflight.is_valid:
        return GeometryEvaluationRun(
            simulation_result=None,
            evaluation=None,
            validity=preflight,
        )

    model: BaseModel | None = None
    basis_layout: BasisLayout | None = None
    matrix: np.ndarray | None = None
    simulation_result: SimulationResult | None = None
    evaluation: GeometryEvaluation | None = None
    reproducibility: ReproducibilityRecord | None = None
    model_parameters: dict[str, Any] | None = None
    try:
        model = adapter.model_factory(geometry)
        if not isinstance(model, BaseModel):
            raise TypeError("model_factory must return BaseModel")
        basis_layout = model.basis_layout
        if not isinstance(basis_layout, BasisLayout):
            raise TypeError("model.basis_layout must return BasisLayout")
        model_parameters = model.parameters
        reproducibility = create_reproducibility_record(
            geometry,
            seed=seed,
            model_name=model.model_name,
            model_parameters=model_parameters,
            solver_name=f"{type(solver).__module__}.{type(solver).__qualname__}",
            solver_settings=_solver_settings(solver),
            evaluation_settings=_evaluation_settings(config),
            code_version=code_version,
        )
    except Exception as error:
        return _failed_run(
            geometry,
            policy=execution_policy,
            stage=CandidateFailureStage.MODEL_CONSTRUCTION,
            error=error,
        )
    assert model is not None
    assert basis_layout is not None
    assert model_parameters is not None
    assert reproducibility is not None

    try:
        matrix = np.asarray(model.hamiltonian(), dtype=complex).copy()
        matrix.setflags(write=False)
    except Exception as error:
        return _failed_run(
            geometry,
            policy=execution_policy,
            stage=CandidateFailureStage.HAMILTONIAN_CONSTRUCTION,
            error=error,
            basis_layout=basis_layout,
            reproducibility=reproducibility,
        )
    assert matrix is not None

    presolve = validate_candidate(
        geometry,
        policy=execution_policy,
        basis_layout=basis_layout,
        hamiltonian=matrix,
    )
    if not presolve.is_valid:
        return GeometryEvaluationRun(
            simulation_result=None,
            evaluation=None,
            validity=presolve,
            reproducibility=reproducibility,
        )

    try:
        eigensystem = solver.solve(matrix)
        simulation_result = SimulationResult(
            model_name=model.model_name,
            eigenvalues=eigensystem.eigenvalues,
            eigenvectors=eigensystem.eigenvectors,
            basis_layout=basis_layout,
            parameters=model_parameters,
        )
    except Exception as error:
        return _failed_run(
            geometry,
            policy=execution_policy,
            stage=CandidateFailureStage.SOLVER,
            error=error,
            basis_layout=basis_layout,
            hamiltonian=matrix,
            reproducibility=reproducibility,
        )
    assert simulation_result is not None

    postsolve = validate_candidate(
        geometry,
        policy=execution_policy,
        basis_layout=basis_layout,
        hamiltonian=matrix,
        simulation_result=simulation_result,
    )
    if not postsolve.is_valid:
        return GeometryEvaluationRun(
            simulation_result=simulation_result,
            evaluation=None,
            validity=postsolve,
            reproducibility=reproducibility,
        )

    try:
        evaluation = evaluate_spectrum(
            simulation_result.eigenvalues,
            reference_energy=config.reference_energy,
            zero_mode_tolerance=config.zero_mode_tolerance,
            low_energy_count=config.low_energy_count,
        )
        evaluation = evaluate_eigenstates(
            evaluation,
            simulation_result.eigenvectors,
            basis_layout=basis_layout,
            geometry=geometry,
            boundary_localization_threshold=(
                config.boundary_localization_threshold
            ),
        )
        if adapter.nambu_basis_resolver is None:
            evaluation = _append_pipeline_warning(
                evaluation,
                "Majorana diagnostics were not requested because the model adapter "
                "has no Nambu-basis resolver.",
            )
        else:
            nambu_basis = adapter.nambu_basis_resolver(model)
            if not isinstance(nambu_basis, NambuBasis):
                raise TypeError("nambu_basis_resolver must return NambuBasis")
            evaluation = evaluate_majorana_diagnostics(
                evaluation,
                simulation_result.eigenvectors,
                nambu_basis=nambu_basis,
            )
        evaluation = evaluate_geometry_descriptors(evaluation, geometry)
    except Exception as error:
        return _failed_run(
            geometry,
            policy=execution_policy,
            stage=CandidateFailureStage.EVALUATION,
            error=error,
            basis_layout=basis_layout,
            hamiltonian=matrix,
            simulation_result=simulation_result,
            evaluation=evaluation,
            reproducibility=reproducibility,
        )
    assert evaluation is not None

    if topology_hook is None:
        evaluation = _append_pipeline_warning(
            evaluation,
            "Topology diagnostics were not requested because no topology hook and "
            "dispatch decision were supplied.",
        )
    else:
        assert topology_dispatch is not None
        try:
            context = GeometryEvaluationContext(
                geometry=geometry,
                model=model,
                hamiltonian=matrix,
                simulation_result=simulation_result,
                evaluation=evaluation,
            )
            topology_results = tuple(topology_hook(context))
            evaluation = integrate_topology_results(
                evaluation,
                topology_results,
                dispatch_decision=topology_dispatch,
                convergence_checked=config.topology_convergence_checked,
            )
        except Exception as error:
            return _failed_run(
                geometry,
                policy=execution_policy,
                stage=CandidateFailureStage.TOPOLOGY,
                error=error,
                basis_layout=basis_layout,
                hamiltonian=matrix,
                simulation_result=simulation_result,
                evaluation=evaluation,
                reproducibility=reproducibility,
            )

    final_validity = validate_candidate(
        geometry,
        policy=policy,
        basis_layout=basis_layout,
        hamiltonian=matrix,
        simulation_result=simulation_result,
        evaluation=evaluation,
    )
    return GeometryEvaluationRun(
        simulation_result=simulation_result,
        evaluation=evaluation,
        validity=final_validity,
        reproducibility=reproducibility,
    )


def _failed_run(
    geometry: Geometry,
    *,
    policy: CandidateValidityPolicy,
    stage: CandidateFailureStage,
    error: Exception,
    basis_layout: BasisLayout | None = None,
    hamiltonian: np.ndarray | None = None,
    simulation_result: SimulationResult | None = None,
    evaluation: GeometryEvaluation | None = None,
    reproducibility: ReproducibilityRecord | None = None,
) -> GeometryEvaluationRun:
    failure = CandidateStageFailure.from_exception(stage, error)
    validity = validate_candidate(
        geometry,
        policy=policy,
        basis_layout=basis_layout,
        hamiltonian=hamiltonian,
        simulation_result=simulation_result,
        evaluation=evaluation,
        failures=(failure,),
    )
    return GeometryEvaluationRun(
        simulation_result=simulation_result,
        evaluation=evaluation,
        validity=validity,
        reproducibility=reproducibility,
        failure=failure,
    )


def _validity_policy(
    *,
    adapter: GeometryModelAdapter,
    config: GeometryEvaluationConfig,
) -> CandidateValidityPolicy:
    return CandidateValidityPolicy(
        model_geometry=adapter.requirements,
        numerical_tolerance=config.numerical_tolerance,
        require_resolved_topology=config.require_resolved_topology,
        require_topology_convergence=config.require_topology_convergence,
    )


def _append_pipeline_warning(
    evaluation: GeometryEvaluation,
    message: str,
) -> GeometryEvaluation:
    warning = f"{_WARNING_PREFIX}{message}"
    if warning in evaluation.warnings:
        return evaluation
    return replace(evaluation, warnings=evaluation.warnings + (warning,))


def _solver_settings(
    solver: ExactDiagonalizationSolver,
) -> dict[str, object]:
    if type(solver) is ExactDiagonalizationSolver:
        return {
            "backend": "numpy.linalg.eigh",
            "spectrum": "full",
        }
    return {}


def _evaluation_settings(
    config: GeometryEvaluationConfig,
) -> dict[str, object]:
    return {
        "reference_energy": config.reference_energy,
        "zero_mode_tolerance": config.zero_mode_tolerance,
        "low_energy_count": config.low_energy_count,
        "boundary_localization_threshold": (
            config.boundary_localization_threshold
        ),
        "numerical_tolerance": config.numerical_tolerance,
        "require_resolved_topology": config.require_resolved_topology,
        "require_topology_convergence": config.require_topology_convergence,
        "topology_convergence_checked": config.topology_convergence_checked,
    }


def _validate_reproducibility_inputs(
    *,
    seed: int | None,
    code_version: str | None,
) -> None:
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer or None")
        if int(seed) < 0:
            raise ValueError("seed must be non-negative")
    if code_version is not None:
        if not isinstance(code_version, str):
            raise TypeError("code_version must be a string or None")
        if not code_version.strip():
            raise ValueError("code_version must be a non-empty string or None")


def _finite_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite_real(value: float, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _unit_interval_real(value: float, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return result


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result
