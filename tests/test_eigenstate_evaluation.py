from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.core import BasisLayout
from toposc_lab.evaluation import (
    GeometryEvaluation,
    evaluate_eigenstates,
    evaluate_spectrum,
)
from toposc_lab.geometry import Geometry
from toposc_lab.observables import localization_profile_on_geometry
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver


def _embedded_geometry() -> Geometry:
    return Geometry(
        n_sites=4,
        coordinates=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        ),
        boundary_sites=frozenset({0, 3}),
    )


def test_geometry_localization_uses_explicit_boundary_and_coordinates() -> None:
    geometry = _embedded_geometry()
    eigenvectors = np.eye(4)[:, [1, 0, 2, 3]]

    profile = localization_profile_on_geometry(
        eigenvectors,
        state_index=1,
        basis_layout=BasisLayout(spatial_shape=(4,)),
        geometry=geometry,
    )

    assert np.array_equal(profile.probability, [1.0, 0.0, 0.0, 0.0])
    assert profile.component_probabilities.shape == (4, 1)
    assert np.array_equal(profile.center_of_mass, [0.0, 0.0])
    assert profile.inverse_participation_ratio == pytest.approx(1.0)
    assert profile.participation_ratio == pytest.approx(1.0)
    assert profile.edge_weight == pytest.approx(1.0)
    assert profile.bulk_weight == pytest.approx(0.0)
    assert profile.is_edge_localized


def test_geometry_localization_supports_arbitrary_embedding_dimension() -> None:
    geometry = Geometry(
        n_sites=2,
        coordinates=np.asarray([[0.0, 0.0, 0.0, 0.0], [2.0, 4.0, 6.0, 8.0]]),
        boundary_sites=frozenset({1}),
    )
    eigenvectors = np.asarray([[0.5], [np.sqrt(0.75)]])

    profile = localization_profile_on_geometry(
        eigenvectors,
        state_index=0,
        basis_layout=BasisLayout(spatial_shape=(2,)),
        geometry=geometry,
        boundary_threshold=0.7,
    )

    assert np.allclose(profile.probability, [0.25, 0.75])
    assert np.allclose(profile.center_of_mass, [1.5, 3.0, 4.5, 6.0])
    assert profile.inverse_participation_ratio == pytest.approx(0.625)
    assert profile.edge_weight == pytest.approx(0.75)
    assert profile.is_edge_localized


def test_geometry_localization_respects_component_major_basis() -> None:
    geometry = Geometry(
        n_sites=2,
        coordinates=np.asarray([[0.0], [1.0]]),
        boundary_sites=frozenset({1}),
    )
    layout = BasisLayout(
        spatial_shape=(2,),
        components_per_site=2,
        ordering="component_major",
        component_labels=("particle", "hole"),
    )
    eigenvectors = np.asarray([[0.0], [1.0], [0.0], [1.0]])

    profile = localization_profile_on_geometry(
        eigenvectors,
        state_index=0,
        basis_layout=layout,
        geometry=geometry,
    )

    assert np.array_equal(profile.probability, [0.0, 1.0])
    assert np.array_equal(profile.component_probabilities, [[0.0, 0.0], [0.5, 0.5]])
    assert profile.component_labels == ("particle", "hole")
    assert profile.edge_weight == pytest.approx(1.0)


def test_abstract_geometry_has_explicitly_unavailable_center_of_mass() -> None:
    profile = localization_profile_on_geometry(
        np.asarray([[1.0], [0.0]]),
        state_index=0,
        basis_layout=BasisLayout(spatial_shape=(2,)),
        geometry=Geometry(n_sites=2),
    )

    assert profile.center_of_mass.shape == (0,)
    assert profile.edge_weight == 0.0
    assert profile.bulk_weight == 1.0
    assert not profile.is_edge_localized


def test_eigenstate_evaluation_populates_all_selected_states_only() -> None:
    spectral = evaluate_spectrum(
        np.asarray([-2.0, -0.2, 0.2, 2.0]),
        low_energy_count=2,
    )
    eigenvectors = np.eye(4)[:, [1, 0, 2, 3]]

    result = evaluate_eigenstates(
        spectral,
        eigenvectors,
        basis_layout=BasisLayout(spatial_shape=(4,)),
        geometry=_embedded_geometry(),
    )

    assert result.gap == spectral.gap
    assert result.low_energy_states == spectral.low_energy_states
    assert result.zero_mode_count == spectral.zero_mode_count
    assert result.ipr == {1: 1.0, 2: 1.0}
    assert tuple(result.localization) == (1, 2)
    assert result.localization[1].edge_weight == pytest.approx(1.0)
    assert result.localization[2].edge_weight == pytest.approx(0.0)
    assert result.majorana_metrics == {}
    assert result.topology == ()
    assert not any("have not been evaluated" in warning for warning in result.warnings)
    assert any("do not establish Majorana character" in warning for warning in result.warnings)
    assert any("only geometry.boundary_sites" in warning for warning in result.warnings)


def test_matrix_level_solver_output_enters_geometry_eigenstate_evaluation() -> None:
    geometry = _embedded_geometry()
    eigensystem = ExactDiagonalizationSolver().solve(
        np.diag([-2.0, -0.2, 0.2, 2.0])
    )
    spectral = evaluate_spectrum(eigensystem.eigenvalues, low_energy_count=2)

    result = evaluate_eigenstates(
        spectral,
        eigensystem.eigenvectors,
        basis_layout=BasisLayout(spatial_shape=(geometry.n_sites,)),
        geometry=geometry,
    )

    assert result.low_energy_states == {1: -0.2, 2: 0.2}
    assert result.ipr == {1: 1.0, 2: 1.0}
    assert result.localization[1].center_of_mass.shape == (2,)


def test_eigenstate_evaluation_warns_for_abstract_boundaryless_geometry() -> None:
    evaluation = GeometryEvaluation(
        low_energy_states={0: 0.1},
        zero_mode_count=0,
        warnings=("Existing warning.",),
    )

    result = evaluate_eigenstates(
        evaluation,
        np.asarray([[1.0], [0.0]]),
        basis_layout=BasisLayout(spatial_shape=(2,)),
        geometry=Geometry(n_sites=2),
    )

    assert result.warnings[0] == "Existing warning."
    assert any("no explicit boundary sites" in warning for warning in result.warnings)
    assert any("no embedding coordinates" in warning for warning in result.warnings)
    assert result.localization[0].center_of_mass.size == 0


def test_eigenstate_evaluation_is_idempotent_for_warnings() -> None:
    evaluation = GeometryEvaluation(
        low_energy_states={0: 0.1},
        zero_mode_count=0,
    )
    geometry = Geometry(n_sites=2)
    vectors = np.asarray([[1.0], [0.0]])
    layout = BasisLayout(spatial_shape=(2,))

    first = evaluate_eigenstates(
        evaluation,
        vectors,
        basis_layout=layout,
        geometry=geometry,
    )
    second = evaluate_eigenstates(
        first,
        vectors,
        basis_layout=layout,
        geometry=geometry,
    )

    assert second.warnings == first.warnings


def test_eigenstate_evaluation_requires_selected_low_energy_states() -> None:
    with pytest.raises(ValueError, match="must contain low-energy states"):
        evaluate_eigenstates(
            GeometryEvaluation(),
            np.eye(2),
            basis_layout=BasisLayout(spatial_shape=(2,)),
            geometry=Geometry(n_sites=2),
        )


def test_eigenstate_evaluation_rejects_site_count_mismatch() -> None:
    with pytest.raises(ValueError, match="same site count"):
        evaluate_eigenstates(
            GeometryEvaluation(low_energy_states={0: 0.0}, zero_mode_count=1),
            np.eye(2),
            basis_layout=BasisLayout(spatial_shape=(2,)),
            geometry=Geometry(n_sites=3),
        )


def test_eigenstate_evaluation_rejects_missing_eigenvector_column() -> None:
    with pytest.raises(ValueError, match="outside the available eigenvector range"):
        evaluate_eigenstates(
            GeometryEvaluation(low_energy_states={2: 0.2}, zero_mode_count=0),
            np.eye(2),
            basis_layout=BasisLayout(spatial_shape=(2,)),
            geometry=Geometry(n_sites=2),
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf])
def test_eigenstate_evaluation_rejects_invalid_boundary_threshold(
    threshold: float,
) -> None:
    with pytest.raises(ValueError, match="threshold"):
        evaluate_eigenstates(
            GeometryEvaluation(low_energy_states={0: 0.0}, zero_mode_count=1),
            np.asarray([[1.0], [0.0]]),
            basis_layout=BasisLayout(spatial_shape=(2,)),
            geometry=Geometry(n_sites=2),
            boundary_localization_threshold=threshold,
        )


def test_eigenstate_evaluation_rejects_nonfinite_eigenvectors() -> None:
    with pytest.raises(ValueError, match="finite"):
        evaluate_eigenstates(
            GeometryEvaluation(low_energy_states={0: 0.0}, zero_mode_count=1),
            np.asarray([[np.nan], [0.0]]),
            basis_layout=BasisLayout(spatial_shape=(2,)),
            geometry=Geometry(n_sites=2),
        )
