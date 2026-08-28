from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.geometry import Geometry, GeometryEdge, chain, irregular_cluster
from toposc_lab.hamiltonians import build_tight_binding_hamiltonian
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver


def test_spinless_chain_produces_expected_matrix() -> None:
    geometry = chain(4)

    hamiltonian = build_tight_binding_hamiltonian(
        geometry,
        onsite=0.5,
        hopping=-1.0,
    )

    expected = np.asarray(
        [
            [0.5, -1.0, 0.0, 0.0],
            [-1.0, 0.5, -1.0, 0.0],
            [0.0, -1.0, 0.5, -1.0],
            [0.0, 0.0, -1.0, 0.5],
        ],
        dtype=complex,
    )
    assert np.array_equal(hamiltonian, expected)


def test_complex_hopping_uses_edge_orientation_and_hermitian_conjugate() -> None:
    geometry = Geometry(n_sites=2, edges=(GeometryEdge(1, 0),))

    hamiltonian = build_tight_binding_hamiltonian(
        geometry,
        hopping=1.0 + 2.0j,
    )

    assert hamiltonian[1, 0] == 1.0 + 2.0j
    assert hamiltonian[0, 1] == 1.0 - 2.0j
    assert np.array_equal(hamiltonian, hamiltonian.conj().T)


def test_site_and_edge_callables_create_inhomogeneous_terms() -> None:
    geometry = irregular_cluster()

    hamiltonian = build_tight_binding_hamiltonian(
        geometry,
        onsite=lambda site: 0.1 * site,
        hopping=lambda edge: 2.0 if edge.edge_type == "long_range" else -1.0,
    )

    assert np.allclose(np.diag(hamiltonian), 0.1 * np.arange(geometry.n_sites))
    assert hamiltonian[0, 5] == 2.0
    assert hamiltonian[0, 1] == -1.0
    assert hamiltonian[0, 3] == 0.0


def test_site_and_edge_mappings_create_inhomogeneous_terms() -> None:
    geometry = chain(3)
    onsite = {0: -1.0, 1: 0.0, 2: 2.0}
    hopping = {
        geometry.edges[0]: 1.0 + 2.0j,
        geometry.edges[1]: -3.0,
    }

    hamiltonian = build_tight_binding_hamiltonian(
        geometry,
        onsite=onsite,
        hopping=hopping,
    )

    expected = np.asarray(
        [
            [-1.0, 1.0 + 2.0j, 0.0],
            [1.0 - 2.0j, 0.0, -3.0],
            [0.0, -3.0, 2.0],
        ],
        dtype=complex,
    )
    assert np.array_equal(hamiltonian, expected)


def test_mapping_values_may_be_component_matrices() -> None:
    geometry = chain(2)
    sigma_z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    hopping_matrix = np.asarray([[0.0, 1.0j], [2.0, 0.0]], dtype=complex)

    hamiltonian = build_tight_binding_hamiltonian(
        geometry,
        onsite={0: sigma_z, 1: -sigma_z},
        hopping={geometry.edges[0]: hopping_matrix},
        components_per_site=2,
    )

    assert np.array_equal(hamiltonian[0:2, 0:2], sigma_z)
    assert np.array_equal(hamiltonian[2:4, 2:4], -sigma_z)
    assert np.array_equal(hamiltonian[0:2, 2:4], hopping_matrix)
    assert np.array_equal(hamiltonian[2:4, 0:2], hopping_matrix.conj().T)


def test_incomplete_onsite_mapping_is_rejected() -> None:
    with pytest.raises(ValueError, match="onsite mapping has no value for site 1"):
        build_tight_binding_hamiltonian(
            chain(2),
            onsite={0: 1.0},
            hopping=-1.0,
        )


def test_incomplete_hopping_mapping_is_rejected() -> None:
    geometry = chain(3)

    with pytest.raises(ValueError, match=r"hopping mapping has no value for edge \(1, 2\)"):
        build_tight_binding_hamiltonian(
            geometry,
            hopping={geometry.edges[0]: -1.0},
        )


def test_multicomponent_terms_use_site_major_blocks() -> None:
    geometry = Geometry(n_sites=2, edges=(GeometryEdge(0, 1),))
    sigma_z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    hopping = np.asarray([[1.0, 2.0j], [0.5, -1.0]], dtype=complex)

    hamiltonian = build_tight_binding_hamiltonian(
        geometry,
        onsite=lambda site: (site + 1) * sigma_z,
        hopping=hopping,
        components_per_site=2,
    )

    assert hamiltonian.shape == (4, 4)
    assert np.array_equal(hamiltonian[0:2, 0:2], sigma_z)
    assert np.array_equal(hamiltonian[2:4, 2:4], 2.0 * sigma_z)
    assert np.array_equal(hamiltonian[0:2, 2:4], hopping)
    assert np.array_equal(hamiltonian[2:4, 0:2], hopping.conj().T)
    assert np.array_equal(hamiltonian, hamiltonian.conj().T)


def test_scalar_multicomponent_terms_expand_to_identity() -> None:
    hamiltonian = build_tight_binding_hamiltonian(
        chain(2),
        onsite=3.0,
        hopping=-2.0,
        components_per_site=2,
    )

    identity = np.eye(2)
    assert np.array_equal(hamiltonian[0:2, 0:2], 3.0 * identity)
    assert np.array_equal(hamiltonian[0:2, 2:4], -2.0 * identity)


def test_built_hamiltonian_is_compatible_with_exact_diagonalization() -> None:
    hamiltonian = build_tight_binding_hamiltonian(chain(2), hopping=-1.0)

    result = ExactDiagonalizationSolver().solve(hamiltonian)

    assert np.allclose(result.eigenvalues, [-1.0, 1.0])


def test_nonhermitian_onsite_term_is_rejected() -> None:
    onsite = np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=complex)

    with pytest.raises(ValueError, match="must be Hermitian"):
        build_tight_binding_hamiltonian(
            chain(2),
            onsite=onsite,
            hopping=0.0,
            components_per_site=2,
        )


@pytest.mark.parametrize("components_per_site", [0, -1])
def test_invalid_component_count_is_rejected(components_per_site: int) -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_tight_binding_hamiltonian(
            chain(2),
            hopping=-1.0,
            components_per_site=components_per_site,
        )


@pytest.mark.parametrize("components_per_site", [1.5, True])
def test_noninteger_component_count_is_rejected(components_per_site: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        build_tight_binding_hamiltonian(
            chain(2),
            hopping=-1.0,
            components_per_site=components_per_site,  # type: ignore[arg-type]
        )


def test_term_with_wrong_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"shape \(2, 2\)"):
        build_tight_binding_hamiltonian(
            chain(2),
            hopping=np.ones((3, 3)),
            components_per_site=2,
        )


def test_nonfinite_term_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite values"):
        build_tight_binding_hamiltonian(chain(2), onsite=np.nan, hopping=-1.0)
