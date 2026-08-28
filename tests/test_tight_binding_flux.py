from __future__ import annotations

import numpy as np

from toposc_lab.geometry import ring
from toposc_lab.hamiltonians import build_tight_binding_hamiltonian


def test_ring_with_uniform_peierls_phase_has_expected_flux_spectrum() -> None:
    n_sites = 7
    hopping_amplitude = 1.3
    flux = 0.8
    geometry = ring(n_sites)
    phase_per_edge = flux / n_sites
    hopping = {
        edge: -hopping_amplitude * np.exp(1.0j * phase_per_edge)
        for edge in geometry.edges
    }

    hamiltonian = build_tight_binding_hamiltonian(geometry, hopping=hopping)

    normalized_loop_product = np.prod(
        [hopping[edge] / -hopping_amplitude for edge in geometry.edges]
    )
    momenta = 2.0 * np.pi * np.arange(n_sites) / n_sites
    expected_eigenvalues = np.sort(
        -2.0 * hopping_amplitude * np.cos(momenta + phase_per_edge)
    )

    assert np.allclose(normalized_loop_product, np.exp(1.0j * flux))
    assert np.array_equal(hamiltonian, hamiltonian.conj().T)
    assert np.allclose(
        np.linalg.eigvalsh(hamiltonian),
        expected_eigenvalues,
        rtol=1e-12,
        atol=1e-12,
    )


def test_total_flux_can_be_moved_to_periodic_closing_edge_by_gauge_change() -> None:
    n_sites = 6
    hopping_amplitude = 0.9
    flux = -1.1
    geometry = ring(n_sites)
    phase_per_edge = flux / n_sites

    uniform_hopping = {
        edge: -hopping_amplitude * np.exp(1.0j * phase_per_edge)
        for edge in geometry.edges
    }
    localized_hopping = {
        edge: -hopping_amplitude
        * np.exp(1.0j * flux if edge.boundary_crossing else 0.0j)
        for edge in geometry.edges
    }
    uniform_hamiltonian = build_tight_binding_hamiltonian(
        geometry,
        hopping=uniform_hopping,
    )
    localized_hamiltonian = build_tight_binding_hamiltonian(
        geometry,
        hopping=localized_hopping,
    )

    gauge = np.diag(np.exp(-1.0j * phase_per_edge * np.arange(n_sites)))
    gauge_transformed = gauge.conj().T @ uniform_hamiltonian @ gauge

    assert not np.allclose(uniform_hamiltonian, localized_hamiltonian)
    assert np.allclose(gauge_transformed, localized_hamiltonian, atol=1e-12)
    assert np.allclose(
        np.linalg.eigvalsh(uniform_hamiltonian),
        np.linalg.eigvalsh(localized_hamiltonian),
        rtol=1e-12,
        atol=1e-12,
    )
