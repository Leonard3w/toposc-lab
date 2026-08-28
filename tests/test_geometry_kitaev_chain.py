from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.geometry import Geometry
from toposc_lab.models.geometry_kitaev_chain import GeometryKitaevChain
from toposc_lab.models.kitaev_chain import KitaevChainParameters
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver


def _parameters(*, boundary: str = "open", n_sites: int = 6) -> KitaevChainParameters:
    return KitaevChainParameters(
        n_sites=n_sites,
        hopping=1.0,
        chemical_potential=0.3,
        pairing=0.5,
        boundary=boundary,
    )


def test_geometry_kitaev_chain_uses_general_geometry() -> None:
    model = GeometryKitaevChain(_parameters())

    assert isinstance(model.geometry, Geometry)
    assert model.geometry.n_sites == 6
    assert model.geometry.n_edges == 5
    assert model.geometry.metadata["boundary_condition"] == "open"
    assert not model.geometry.has_edge(0, 5)


def test_periodic_geometry_contains_oriented_closing_edge() -> None:
    model = GeometryKitaevChain(_parameters(boundary="periodic"))
    closing_edge = model.geometry.edge_between(5, 0)

    assert model.geometry.n_edges == 6
    assert closing_edge is not None
    assert (closing_edge.source, closing_edge.target) == (5, 0)
    assert closing_edge.boundary_crossing
    assert model.geometry.boundary_sites == frozenset()


def test_geometry_kitaev_hamiltonian_is_hermitian_and_has_expected_shape() -> None:
    model = GeometryKitaevChain(_parameters())
    hamiltonian = model.hamiltonian()

    assert hamiltonian.shape == (12, 12)
    assert np.allclose(hamiltonian, hamiltonian.conj().T)


def test_normal_block_contains_onsite_and_graph_hopping_terms() -> None:
    model = GeometryKitaevChain(_parameters())
    normal_block = model.hamiltonian()[:6, :6]

    assert np.allclose(np.diag(normal_block), -0.3)
    assert normal_block[0, 1] == -1.0
    assert normal_block[0, 5] == 0.0


def test_pairing_block_is_antisymmetric_and_follows_edge_orientation() -> None:
    model = GeometryKitaevChain(_parameters(boundary="periodic"))
    pairing_block = model.hamiltonian()[:6, 6:]

    assert np.array_equal(pairing_block, -pairing_block.T)
    assert pairing_block[0, 1] == 0.5
    assert pairing_block[1, 0] == -0.5
    assert pairing_block[5, 0] == 0.5
    assert pairing_block[0, 5] == -0.5


def test_basis_layout_preserves_legacy_component_major_ordering() -> None:
    model = GeometryKitaevChain(_parameters())

    assert model.basis_layout.spatial_shape == (6,)
    assert model.basis_layout.components_per_site == 2
    assert model.basis_layout.ordering == "component_major"
    assert model.basis_layout.component_labels == ("electron", "hole")


def test_disorder_is_seeded_and_bounded() -> None:
    params = KitaevChainParameters(
        n_sites=8,
        hopping=1.0,
        chemical_potential=0.3,
        pairing=0.5,
        disorder_strength=0.8,
        disorder_seed=17,
    )
    first = GeometryKitaevChain(params)
    second = GeometryKitaevChain(params)

    assert np.array_equal(first.disorder_profile, second.disorder_profile)
    assert np.max(np.abs(first.disorder_profile)) <= 0.4
    assert np.array_equal(first.hamiltonian(), second.hamiltonian())


def test_two_site_periodic_case_remains_a_simple_graph() -> None:
    model = GeometryKitaevChain(_parameters(boundary="periodic", n_sites=2))

    assert model.geometry.n_edges == 1
    assert model.geometry.boundary_sites == frozenset()
    assert model.geometry.metadata["boundary_condition"] == "periodic"
    assert model.hamiltonian().shape == (4, 4)


def test_invalid_boundary_is_rejected() -> None:
    with pytest.raises(ValueError, match="either open or periodic"):
        GeometryKitaevChain(_parameters(boundary="invalid"))


def test_geometry_kitaev_chain_is_solver_compatible() -> None:
    model = GeometryKitaevChain(_parameters())

    result = ExactDiagonalizationSolver().solve_model(model)

    assert result.eigenvalues.shape == (12,)
    assert result.eigenvectors.shape == (12, 12)
    assert result.basis_layout == model.basis_layout
