from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import cast

import numpy as np
import pytest

from toposc_lab.core.model import BaseModel
from toposc_lab.core.results import BasisLayout
from toposc_lab.evaluation import (
    CandidateFailureStage,
    CandidateValidityReport,
    GeometryEvaluationConfig,
    GeometryEvaluationContext,
    GeometryEvaluationRun,
    GeometryModelAdapter,
    ModelGeometryRequirements,
    evaluate_geometry,
)
from toposc_lab.geometry import Geometry, chain, square
from toposc_lab.hamiltonians import NambuBasis
from toposc_lab.models.chiral_p_wave import ChiralPWaveModel, ChiralPWaveParameters
from toposc_lab.solvers.exact_diagonalization import (
    EigenSystem,
    ExactDiagonalizationSolver,
)
from toposc_lab.topology import (
    NumericalConfidence,
    TopologyCapability,
    TopologyDispatchDecision,
    TopologyMethod,
    TopologyResult,
    dispatch_topology_methods,
)
from toposc_lab.topology.dispatch import TopologyDispatchContext
from toposc_lab.topology.symmetry import SymmetryClassification


class _GeometryModel(BaseModel):
    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry

    @property
    def basis_layout(self) -> BasisLayout:
        return BasisLayout(
            spatial_shape=(self.geometry.n_sites,),
            components_per_site=1,
            component_labels=("orbital",),
        )

    def hamiltonian(self) -> np.ndarray:
        return np.diag(np.linspace(-1.0, 1.0, self.geometry.n_sites))


class _BdGGeometryModel(BaseModel):
    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry

    @property
    def nambu_basis(self) -> NambuBasis:
        return NambuBasis(n_sites=self.geometry.n_sites, ordering="component_major")

    @property
    def basis_layout(self) -> BasisLayout:
        return self.nambu_basis.basis_layout

    def hamiltonian(self) -> np.ndarray:
        dimension = self.basis_layout.dimension
        return np.diag(np.linspace(-2.0, 2.0, dimension))


class _HamiltonianFailureModel(_GeometryModel):
    def hamiltonian(self) -> np.ndarray:
        raise ValueError("physical terms cannot be assembled")


class _WrongBasisModel(_GeometryModel):
    @property
    def basis_layout(self) -> BasisLayout:
        return BasisLayout(spatial_shape=(self.geometry.n_sites + 1,))


class _NonHermitianModel(_GeometryModel):
    def hamiltonian(self) -> np.ndarray:
        return np.array([[0.0, 1.0], [0.0, 0.0]])


class _FailingSolver(ExactDiagonalizationSolver):
    def solve(self, hamiltonian: np.ndarray) -> EigenSystem:
        raise np.linalg.LinAlgError("eigensolver did not converge")


def _model_factory(geometry: Geometry) -> BaseModel:
    return _GeometryModel(geometry)


def _bdg_factory(geometry: Geometry) -> BaseModel:
    return _BdGGeometryModel(geometry)


def _nambu_resolver(model: BaseModel) -> NambuBasis:
    return cast(_BdGGeometryModel, model).nambu_basis


def _chiral_factory(geometry: Geometry) -> BaseModel:
    return ChiralPWaveModel(
        geometry,
        ChiralPWaveParameters(
            hopping=1.0,
            chemical_potential=0.5,
            pairing=0.3,
            chirality=1,
            plane_axes=(0, 1),
        ),
    )


def _chiral_nambu_resolver(model: BaseModel) -> NambuBasis:
    return cast(ChiralPWaveModel, model).nambu_basis


def _adapter() -> GeometryModelAdapter:
    return GeometryModelAdapter(model_factory=_model_factory)


def _class_d_dispatch() -> TopologyDispatchDecision:
    return dispatch_topology_methods(
        TopologyDispatchContext(
            physical_dimension=1,
            embedding_dimension=1,
            classification=SymmetryClassification.from_signature(
                time_reversal_square=None,
                particle_hole_square=1,
                chiral_symmetry=False,
            ),
            capabilities=frozenset(
                {
                    TopologyCapability.TRANSLATION_INVARIANT_BULK,
                    TopologyCapability.BULK_GAP_EVIDENCE,
                    TopologyCapability.BLOCH_PARTICLE_HOLE_ENDPOINTS,
                }
            ),
        )
    )


def _pfaffian_topology_result() -> TopologyResult:
    return TopologyResult(
        invariant_value=-1,
        is_topological=True,
        invariant_group="Z2",
        method=TopologyMethod.PFAFFIAN_1D,
        applicability_assumptions=("The class-D Pfaffian assumptions hold.",),
        confidence=NumericalConfidence(
            is_resolved=True,
            is_quantized=True,
            minimum_gap=0.4,
            gap_kind="endpoint_energy_gap",
            quantization_error=0.0,
            maximum_residual=1.0e-14,
            convergence_checked=True,
        ),
        warnings=(),
    )


def test_pipeline_produces_separated_complete_core_results() -> None:
    run = evaluate_geometry(
        chain(2),
        adapter=_adapter(),
        config=GeometryEvaluationConfig(low_energy_count=2),
    )

    assert run.is_valid
    assert run.failure is None
    assert run.simulation_result is not None
    assert run.evaluation is not None
    assert run.simulation_result.eigenvalues == pytest.approx([-1.0, 1.0])
    assert run.evaluation.gap == pytest.approx(2.0)
    assert run.evaluation.low_energy_states == {0: -1.0, 1: 1.0}
    assert set(run.evaluation.ipr) == {0, 1}
    assert set(run.evaluation.localization) == {0, 1}
    assert run.evaluation.majorana_metrics == {}
    assert run.evaluation.geometry_descriptors["site_count"] == 2
    assert run.evaluation.topology == ()
    assert any("no Nambu-basis resolver" in item for item in run.evaluation.warnings)
    assert any("no topology hook" in item for item in run.evaluation.warnings)
    assert not hasattr(run, "score")
    assert run.reproducibility is not None
    assert run.reproducibility.model_name == "_GeometryModel"
    assert run.reproducibility.geometry_id.startswith(
        "toposc-geometry-archive-v1-sha256:"
    )
    assert run.reproducibility.solver_settings == {
        "backend": "numpy.linalg.eigh",
        "spectrum": "full",
    }
    assert run.reproducibility.evaluation_settings["low_energy_count"] == 2


def test_nambu_adapter_enables_majorana_stage_explicitly() -> None:
    run = evaluate_geometry(
        chain(2),
        adapter=GeometryModelAdapter(
            model_factory=_bdg_factory,
            nambu_basis_resolver=_nambu_resolver,
        ),
        config=GeometryEvaluationConfig(low_energy_count=2),
    )

    assert run.is_valid
    assert run.evaluation is not None
    assert len(run.evaluation.majorana_metrics) == 2
    assert not any("no Nambu-basis resolver" in item for item in run.evaluation.warnings)


def test_production_chiral_model_runs_through_explicit_2d_adapter() -> None:
    geometry = square(2, 2)
    run = evaluate_geometry(
        geometry,
        adapter=GeometryModelAdapter(
            model_factory=_chiral_factory,
            requirements=ModelGeometryRequirements(
                require_connected=True,
                require_edges=True,
                required_spatial_axes=(0, 1),
            ),
            nambu_basis_resolver=_chiral_nambu_resolver,
        ),
        config=GeometryEvaluationConfig(low_energy_count=4),
    )

    assert run.is_valid
    assert run.simulation_result is not None
    assert run.evaluation is not None
    assert run.simulation_result.model_name == "ChiralPWaveModel"
    assert run.simulation_result.basis_layout.n_sites == geometry.n_sites
    assert run.evaluation.geometry_descriptors["site_count"] == geometry.n_sites
    assert set(run.evaluation.majorana_metrics) == set(
        run.evaluation.low_energy_states
    )


def test_topology_hook_receives_separated_context_and_is_integrated() -> None:
    observed: list[GeometryEvaluationContext] = []

    def topology_hook(
        context: GeometryEvaluationContext,
    ) -> tuple[TopologyResult, ...]:
        observed.append(context)
        assert context.evaluation.geometry_descriptors["site_count"] == 2
        assert context.hamiltonian.flags.writeable is False
        return (_pfaffian_topology_result(),)

    run = evaluate_geometry(
        chain(2),
        adapter=_adapter(),
        config=GeometryEvaluationConfig(
            low_energy_count=2,
            require_resolved_topology=True,
            require_topology_convergence=True,
        ),
        topology_hook=topology_hook,
        topology_dispatch=_class_d_dispatch(),
    )

    assert run.is_valid
    assert len(observed) == 1
    assert run.evaluation is not None
    assert run.evaluation.topology[0].method is TopologyMethod.PFAFFIAN_1D


def test_invalid_preflight_stops_before_model_factory() -> None:
    calls = 0

    def model_factory(geometry: Geometry) -> BaseModel:
        nonlocal calls
        calls += 1
        return _GeometryModel(geometry)

    run = evaluate_geometry(
        chain(2),
        adapter=GeometryModelAdapter(
            model_factory=model_factory,
            requirements=ModelGeometryRequirements(required_spatial_axes=(0, 1)),
        ),
    )

    assert not run.is_valid
    assert calls == 0
    assert run.failure is None
    assert run.simulation_result is None
    assert run.evaluation is None
    assert "missing_required_spatial_axes" in {
        issue.code for issue in run.validity.errors
    }


def test_basis_geometry_mismatch_stops_before_solver() -> None:
    run = evaluate_geometry(
        chain(2),
        adapter=GeometryModelAdapter(
            model_factory=lambda geometry: _WrongBasisModel(geometry)
        ),
    )

    assert not run.is_valid
    assert run.failure is None
    assert "basis_geometry_site_mismatch" in {
        issue.code for issue in run.validity.errors
    }


def test_nonhermitian_model_hamiltonian_stops_before_solver() -> None:
    run = evaluate_geometry(
        chain(2),
        adapter=GeometryModelAdapter(
            model_factory=lambda geometry: _NonHermitianModel(geometry)
        ),
        solver=_FailingSolver(),
    )

    assert not run.is_valid
    assert run.failure is None
    assert run.simulation_result is None
    assert "nonhermitian_hamiltonian" in {
        issue.code for issue in run.validity.errors
    }


@pytest.mark.parametrize(
    ("adapter", "solver", "stage", "message"),
    [
        (
            GeometryModelAdapter(
                model_factory=lambda geometry: (_ for _ in ()).throw(
                    ValueError("unsupported geometry")
                )
            ),
            None,
            CandidateFailureStage.MODEL_CONSTRUCTION,
            "unsupported geometry",
        ),
        (
            GeometryModelAdapter(
                model_factory=lambda geometry: _HamiltonianFailureModel(geometry)
            ),
            None,
            CandidateFailureStage.HAMILTONIAN_CONSTRUCTION,
            "physical terms cannot be assembled",
        ),
        (
            _adapter(),
            _FailingSolver(),
            CandidateFailureStage.SOLVER,
            "eigensolver did not converge",
        ),
    ],
)
def test_candidate_execution_exceptions_become_stage_failures(
    adapter: GeometryModelAdapter,
    solver: ExactDiagonalizationSolver | None,
    stage: CandidateFailureStage,
    message: str,
) -> None:
    run = evaluate_geometry(chain(2), adapter=adapter, solver=solver)

    assert not run.is_valid
    assert run.failure is not None
    assert run.failure.stage is stage
    assert message in run.failure.message
    assert f"{stage.value}_failure" in {
        issue.code for issue in run.validity.errors
    }


def test_invalid_nambu_resolver_output_is_an_evaluation_failure() -> None:
    def bad_resolver(model: BaseModel) -> NambuBasis:
        return cast(NambuBasis, object())

    run = evaluate_geometry(
        chain(2),
        adapter=GeometryModelAdapter(
            model_factory=_model_factory,
            nambu_basis_resolver=bad_resolver,
        ),
    )

    assert not run.is_valid
    assert run.failure is not None
    assert run.failure.stage is CandidateFailureStage.EVALUATION
    assert run.simulation_result is not None
    assert run.evaluation is not None


def test_topology_exception_retains_completed_non_topological_evaluation() -> None:
    def bad_topology_hook(
        context: GeometryEvaluationContext,
    ) -> tuple[TopologyResult, ...]:
        raise RuntimeError("topology backend failed")

    run = evaluate_geometry(
        chain(2),
        adapter=_adapter(),
        topology_hook=bad_topology_hook,
        topology_dispatch=_class_d_dispatch(),
    )

    assert not run.is_valid
    assert run.failure is not None
    assert run.failure.stage is CandidateFailureStage.TOPOLOGY
    assert run.simulation_result is not None
    assert run.evaluation is not None
    assert run.evaluation.geometry_descriptors["site_count"] == 2


def test_required_topology_without_hook_finishes_as_invalid_not_exception() -> None:
    run = evaluate_geometry(
        chain(2),
        adapter=_adapter(),
        config=GeometryEvaluationConfig(require_resolved_topology=True),
    )

    assert not run.is_valid
    assert run.failure is None
    assert run.evaluation is not None
    assert "missing_required_topology" in {
        issue.code for issue in run.validity.errors
    }


def test_pipeline_result_is_immutable() -> None:
    run = evaluate_geometry(chain(2), adapter=_adapter())

    with pytest.raises(FrozenInstanceError):
        run.failure = None  # type: ignore[misc]


def test_run_rejects_valid_report_without_completed_results() -> None:
    with pytest.raises(ValueError, match="valid run requires"):
        GeometryEvaluationRun(
            simulation_result=None,
            evaluation=None,
            validity=CandidateValidityReport(()),
        )


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"zero_mode_tolerance": 0.0}, ValueError, "must be positive"),
        ({"low_energy_count": 0}, ValueError, "must be positive"),
        (
            {"boundary_localization_threshold": 1.1},
            ValueError,
            "between zero and one",
        ),
        ({"numerical_tolerance": np.inf}, ValueError, "must be finite"),
        ({"require_resolved_topology": 1}, TypeError, "must be a boolean"),
    ],
)
def test_config_rejects_invalid_numerical_conventions(
    kwargs: dict[str, object],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        GeometryEvaluationConfig(**kwargs)  # type: ignore[arg-type]


def test_topology_hook_and_dispatch_must_be_supplied_together() -> None:
    with pytest.raises(ValueError, match="must be supplied together"):
        evaluate_geometry(
            chain(2),
            adapter=_adapter(),
            topology_hook=lambda context: (),
        )
