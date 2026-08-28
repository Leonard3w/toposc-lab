from __future__ import annotations

import numpy as np

from toposc_lab.models.geometry_kitaev_chain import GeometryKitaevChain
from toposc_lab.models.kitaev_chain import KitaevChainParameters


def _model(
    *,
    n_sites: int = 6,
    hopping: float = 1.0,
    chemical_potential: float = 0.3,
    pairing: float = 0.5,
    boundary: str,
) -> GeometryKitaevChain:
    return GeometryKitaevChain(
        KitaevChainParameters(
            n_sites=n_sites,
            hopping=hopping,
            chemical_potential=chemical_potential,
            pairing=pairing,
            boundary=boundary,
        )
    )


def test_open_and_periodic_geometries_have_expected_boundary_structure() -> None:
    open_model = _model(boundary="open")
    periodic_model = _model(boundary="periodic")

    assert open_model.geometry.boundary_sites == frozenset({0, 5})
    assert open_model.geometry.degree(0) == 1
    assert open_model.geometry.degree(5) == 1
    assert not open_model.geometry.has_edge(5, 0)

    assert periodic_model.geometry.boundary_sites == frozenset()
    assert all(
        periodic_model.geometry.degree(site) == 2
        for site in periodic_model.geometry.site_indices
    )
    closing_edge = periodic_model.geometry.edge_between(5, 0)
    assert closing_edge is not None
    assert closing_edge.boundary_crossing
    assert (closing_edge.source, closing_edge.target) == (5, 0)


def test_periodic_hamiltonian_adds_only_oriented_closing_bond_blocks() -> None:
    n_sites = 5
    hopping = 1.2
    pairing = 0.7
    open_hamiltonian = _model(
        n_sites=n_sites,
        hopping=hopping,
        pairing=pairing,
        boundary="open",
    ).hamiltonian()
    periodic_hamiltonian = _model(
        n_sites=n_sites,
        hopping=hopping,
        pairing=pairing,
        boundary="periodic",
    ).hamiltonian()

    expected_difference = np.zeros((2 * n_sites, 2 * n_sites), dtype=complex)

    # Electron and hole normal-state closing hoppings.
    expected_difference[0, n_sites - 1] = -hopping
    expected_difference[n_sites - 1, 0] = -hopping
    expected_difference[n_sites, 2 * n_sites - 1] = hopping
    expected_difference[2 * n_sites - 1, n_sites] = hopping

    # Antisymmetric p-wave closing pair in the stored (N-1) -> 0 orientation.
    expected_difference[n_sites - 1, n_sites] = pairing
    expected_difference[0, 2 * n_sites - 1] = -pairing
    expected_difference[2 * n_sites - 1, 0] = -pairing
    expected_difference[n_sites, n_sites - 1] = pairing

    assert np.array_equal(periodic_hamiltonian - open_hamiltonian, expected_difference)


def test_open_chain_sweet_spot_has_two_exact_zero_modes() -> None:
    n_sites = 6
    hopping = 1.0
    hamiltonian = _model(
        n_sites=n_sites,
        hopping=hopping,
        chemical_potential=0.0,
        pairing=hopping,
        boundary="open",
    ).hamiltonian()

    eigenvalues = np.linalg.eigvalsh(hamiltonian)
    expected = np.concatenate(
        (
            np.full(n_sites - 1, -2.0 * hopping),
            np.zeros(2),
            np.full(n_sites - 1, 2.0 * hopping),
        )
    )

    assert np.allclose(eigenvalues, expected, rtol=0.0, atol=1e-12)
    assert np.count_nonzero(np.abs(eigenvalues) < 1e-12) == 2


def test_periodic_chain_matches_analytical_momentum_space_dispersion() -> None:
    n_sites = 8
    hopping = 1.1
    chemical_potential = 0.4
    pairing = 0.7
    hamiltonian = _model(
        n_sites=n_sites,
        hopping=hopping,
        chemical_potential=chemical_potential,
        pairing=pairing,
        boundary="periodic",
    ).hamiltonian()

    momenta = 2.0 * np.pi * np.arange(n_sites) / n_sites
    positive_energies = np.sqrt(
        (-chemical_potential - 2.0 * hopping * np.cos(momenta)) ** 2
        + (2.0 * pairing * np.sin(momenta)) ** 2
    )
    expected = np.sort(np.concatenate((-positive_energies, positive_energies)))

    assert np.allclose(
        np.linalg.eigvalsh(hamiltonian),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_two_site_periodic_boundary_does_not_duplicate_simple_graph_edge() -> None:
    open_hamiltonian = _model(n_sites=2, boundary="open").hamiltonian()
    periodic_hamiltonian = _model(n_sites=2, boundary="periodic").hamiltonian()

    assert np.array_equal(periodic_hamiltonian, open_hamiltonian)
