from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from toposc_lab.core import BasisLayout
from toposc_lab.evaluation import (
    GeometryEvaluation,
    evaluate_eigenstates,
    evaluate_majorana_diagnostics,
    evaluate_spectrum,
)
from toposc_lab.geometry import Geometry
from toposc_lab.hamiltonians import NambuBasis
from toposc_lab.models.geometry_kitaev_chain import GeometryKitaevChain
from toposc_lab.models.kitaev_chain import KitaevChainParameters
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver


def _site_major_test_vectors() -> np.ndarray:
    inverse_sqrt_two = 1.0 / np.sqrt(2.0)
    return np.column_stack(
        (
            np.asarray([inverse_sqrt_two, -inverse_sqrt_two, 0.0, 0.0]),
            np.asarray([inverse_sqrt_two, inverse_sqrt_two, 0.0, 0.0]),
            np.asarray([0.0, 0.0, 1.0, 0.0]),
            np.asarray([0.0, 0.0, 0.0, 1.0]),
        )
    )


def _prepared_evaluation(
    ordering: str = "component_major",
) -> tuple[GeometryEvaluation, np.ndarray, NambuBasis]:
    basis = NambuBasis(n_sites=2, ordering=ordering)  # type: ignore[arg-type]
    vectors = NambuBasis(n_sites=2, ordering="site_major").reorder_states(
        _site_major_test_vectors(),
        ordering=ordering,  # type: ignore[arg-type]
    )
    spectral = evaluate_spectrum(
        np.asarray([-2.0, -0.1, 0.1, 2.0]),
        low_energy_count=2,
    )
    evaluation = evaluate_eigenstates(
        spectral,
        vectors,
        basis_layout=basis.basis_layout,
        geometry=Geometry(
            n_sites=2,
            coordinates=np.asarray([[0.0], [1.0]]),
            boundary_sites=frozenset({0, 1}),
        ),
    )
    return evaluation, vectors, basis


@pytest.mark.parametrize("ordering", ["component_major", "site_major"])
def test_majorana_evaluation_is_explicitly_nambu_order_aware(ordering: str) -> None:
    evaluation, vectors, basis = _prepared_evaluation(ordering)

    result = evaluate_majorana_diagnostics(
        evaluation,
        vectors,
        nambu_basis=basis,
    )

    assert tuple(result.majorana_metrics) == (1, 2)
    assert result.majorana_metrics[1].self_conjugacy == pytest.approx(1.0)
    assert result.majorana_metrics[2].self_conjugacy == pytest.approx(0.0)
    assert result.majorana_metrics[1].particle_weight == pytest.approx(0.5)
    assert result.majorana_metrics[1].hole_weight == pytest.approx(0.5)
    assert result.majorana_metrics[2].particle_weight == pytest.approx(1.0)
    assert result.majorana_metrics[2].hole_weight == pytest.approx(0.0)


def test_majorana_evaluation_preserves_independent_evidence_fields() -> None:
    evaluation, vectors, basis = _prepared_evaluation()

    result = evaluate_majorana_diagnostics(
        evaluation,
        vectors,
        nambu_basis=basis,
    )

    assert result.gap == evaluation.gap
    assert result.low_energy_states == evaluation.low_energy_states
    assert result.zero_mode_count == 0
    assert result.ipr == evaluation.ipr
    assert result.localization == evaluation.localization
    assert result.topology == ()
    assert any("does not by itself establish zero energy" in item for item in result.warnings)
    assert any("degenerate subspace" in item for item in result.warnings)
    assert any("every selected low-energy state" in item for item in result.warnings)


def test_majorana_and_localization_site_probabilities_are_cross_checked() -> None:
    evaluation, vectors, basis = _prepared_evaluation()

    result = evaluate_majorana_diagnostics(
        evaluation,
        vectors,
        nambu_basis=basis,
    )

    for state_index in result.low_energy_states:
        assert np.allclose(
            result.majorana_metrics[state_index].site_probability,
            result.localization[state_index].probability,
        )


def test_majorana_evaluation_is_idempotent_for_warnings() -> None:
    evaluation, vectors, basis = _prepared_evaluation()

    first = evaluate_majorana_diagnostics(
        evaluation,
        vectors,
        nambu_basis=basis,
    )
    second = evaluate_majorana_diagnostics(
        first,
        vectors,
        nambu_basis=basis,
    )

    assert second.warnings == first.warnings
    assert second.majorana_metrics.keys() == first.majorana_metrics.keys()


def test_majorana_evaluation_requires_complete_phase_7_3_results() -> None:
    with pytest.raises(ValueError, match="complete Phase 7.3"):
        evaluate_majorana_diagnostics(
            GeometryEvaluation(
                low_energy_states={0: 0.0},
                zero_mode_count=1,
            ),
            np.asarray([[1.0], [1.0]]),
            nambu_basis=NambuBasis(n_sites=1),
        )


def test_majorana_evaluation_rejects_inconsistent_phase_7_3_ipr() -> None:
    evaluation, vectors, basis = _prepared_evaluation()
    inconsistent = replace(
        evaluation,
        ipr={1: 0.25, 2: evaluation.ipr[2]},
    )

    with pytest.raises(ValueError, match="IPR values"):
        evaluate_majorana_diagnostics(
            inconsistent,
            vectors,
            nambu_basis=basis,
        )


def test_majorana_evaluation_rejects_component_label_mismatch() -> None:
    basis = NambuBasis(n_sites=2)
    spectral = GeometryEvaluation(
        low_energy_states={0: 0.0},
        zero_mode_count=1,
    )
    vectors = np.asarray([[1.0], [0.0], [1.0], [0.0]])
    evaluation = evaluate_eigenstates(
        spectral,
        vectors,
        basis_layout=BasisLayout(
            spatial_shape=(2,),
            components_per_site=2,
            ordering="component_major",
            component_labels=("particle", "hole"),
        ),
        geometry=Geometry(n_sites=2),
    )

    with pytest.raises(ValueError, match="component labels"):
        evaluate_majorana_diagnostics(
            evaluation,
            vectors,
            nambu_basis=basis,
        )


def test_majorana_evaluation_rejects_ordering_mismatch_via_probability_check() -> None:
    basis = NambuBasis(n_sites=2, ordering="component_major")
    vectors = NambuBasis(n_sites=2, ordering="site_major").reorder_states(
        _site_major_test_vectors(),
        ordering="component_major",
    )
    spectral = evaluate_spectrum(
        np.asarray([-2.0, -0.1, 0.1, 2.0]),
        low_energy_count=2,
    )
    evaluation = evaluate_eigenstates(
        spectral,
        vectors,
        basis_layout=BasisLayout(
            spatial_shape=(2,),
            components_per_site=2,
            ordering="site_major",
            component_labels=basis.component_labels,
        ),
        geometry=Geometry(n_sites=2),
    )

    with pytest.raises(ValueError, match="does not match"):
        evaluate_majorana_diagnostics(
            evaluation,
            vectors,
            nambu_basis=basis,
        )


def test_majorana_evaluation_rejects_missing_eigenvector_column() -> None:
    evaluation, vectors, basis = _prepared_evaluation()

    with pytest.raises(ValueError, match="outside the available eigenvector range"):
        evaluate_majorana_diagnostics(
            GeometryEvaluation(
                gap=evaluation.gap,
                low_energy_states={4: 0.1},
                zero_mode_count=0,
                ipr={4: evaluation.ipr[1]},
                localization={4: evaluation.localization[1]},
            ),
            vectors,
            nambu_basis=basis,
        )


def test_geometry_kitaev_result_crosses_all_phase_7_2_to_7_4_layers() -> None:
    model = GeometryKitaevChain(
        KitaevChainParameters(
            n_sites=8,
            hopping=1.0,
            chemical_potential=0.5,
            pairing=0.8,
            boundary="open",
        )
    )
    solver_result = ExactDiagonalizationSolver().solve_model(model)
    spectral = evaluate_spectrum(solver_result.eigenvalues, low_energy_count=2)
    eigenstates = evaluate_eigenstates(
        spectral,
        solver_result.eigenvectors,
        basis_layout=solver_result.basis_layout,
        geometry=model.geometry,
    )

    result = evaluate_majorana_diagnostics(
        eigenstates,
        solver_result.eigenvectors,
        nambu_basis=model.nambu_basis,
    )

    assert result.majorana_metrics.keys() == result.low_energy_states.keys()
    for state_index, diagnostics in result.majorana_metrics.items():
        assert np.allclose(
            diagnostics.site_probability,
            result.localization[state_index].probability,
        )
        assert 0.0 <= diagnostics.self_conjugacy <= 1.0
    assert result.topology == ()
