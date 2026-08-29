import numpy as np
import pytest

from toposc_lab.core.results import SimulationResult
from toposc_lab.hamiltonians import NambuBasis
from toposc_lab.observables.majorana import (
    majorana_diagnostics,
    majorana_diagnostics_from_result,
)


@pytest.mark.parametrize("ordering", ["component_major", "site_major"])
def test_canonical_majorana_has_unit_self_conjugacy(ordering: str) -> None:
    basis = NambuBasis(n_sites=2, ordering=ordering)  # type: ignore[arg-type]
    site_major_state = np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0)
    state = NambuBasis(n_sites=2, ordering="site_major").reorder_states(
        site_major_state,
        ordering=ordering,  # type: ignore[arg-type]
    )

    diagnostics = majorana_diagnostics(state[:, np.newaxis], 0, basis)

    assert np.allclose(diagnostics.site_probability, [1.0, 0.0])
    assert np.allclose(diagnostics.polarization, [1.0, 0.0])
    assert diagnostics.self_conjugacy == pytest.approx(1.0)
    assert diagnostics.particle_weight == pytest.approx(0.5)
    assert diagnostics.hole_weight == pytest.approx(0.5)


def test_polarization_matches_antiunitary_particle_hole_overlap() -> None:
    basis = NambuBasis(n_sites=2)
    state = np.array([1.0, 2.0j, 0.5j, -1.0], dtype=complex)
    normalized_state = state / np.linalg.norm(state)

    diagnostics = majorana_diagnostics(state[:, np.newaxis], 0, basis)
    expected = np.vdot(
        normalized_state,
        basis.particle_hole_operator @ normalized_state.conj(),
    )

    assert diagnostics.total_polarization == pytest.approx(expected)
    assert diagnostics.self_conjugacy == pytest.approx(abs(expected))


def test_pure_particle_state_has_zero_majorana_polarization() -> None:
    basis = NambuBasis(n_sites=2)
    state = np.array([1.0, 0.0, 0.0, 0.0])

    diagnostics = majorana_diagnostics(state[:, np.newaxis], 0, basis)

    assert np.allclose(diagnostics.polarization, 0.0)
    assert diagnostics.self_conjugacy == 0.0
    assert diagnostics.particle_weight == pytest.approx(1.0)
    assert diagnostics.hole_weight == 0.0


def test_opposite_local_polarizations_cancel_globally() -> None:
    basis = NambuBasis(n_sites=2, ordering="site_major")
    state = np.array([1.0, 1.0, 1.0, -1.0]) / 2.0

    diagnostics = majorana_diagnostics(state[:, np.newaxis], 0, basis)

    assert np.allclose(diagnostics.polarization, [0.5, -0.5])
    assert np.allclose(diagnostics.polarization_magnitude, [0.5, 0.5])
    assert diagnostics.self_conjugacy == 0.0
    assert diagnostics.polarization_norm == pytest.approx(1.0)


def test_spinful_polarization_pairs_only_matching_components() -> None:
    basis = NambuBasis(
        n_sites=1,
        normal_components_per_site=2,
        ordering="site_major",
        normal_component_labels=("up", "down"),
    )
    state = np.array([1.0, 0.0, 1.0, 0.0]) / np.sqrt(2.0)

    diagnostics = majorana_diagnostics(state[:, np.newaxis], 0, basis)

    assert diagnostics.polarization[0] == pytest.approx(1.0)
    assert diagnostics.self_conjugacy == pytest.approx(1.0)


def test_result_wrapper_rejects_mismatched_nambu_layout() -> None:
    result = SimulationResult(
        model_name="Test",
        eigenvalues=np.array([0.0]),
        eigenvectors=np.ones((2, 1)),
        basis_layout=NambuBasis(n_sites=1).basis_layout,
    )

    with pytest.raises(ValueError, match="does not match"):
        majorana_diagnostics_from_result(
            result,
            0,
            NambuBasis(n_sites=1, ordering="site_major"),
        )


def test_majorana_diagnostics_rejects_zero_norm_state() -> None:
    with pytest.raises(ValueError, match="positive norm"):
        majorana_diagnostics(
            np.zeros((2, 1)),
            0,
            NambuBasis(n_sites=1),
        )
