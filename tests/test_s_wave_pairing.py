from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.geometry import Geometry, chain
from toposc_lab.hamiltonians import (
    NambuBasis,
    build_bdg_hamiltonian,
    build_onsite_s_wave_pairing,
)


def test_constant_s_wave_pairing_creates_local_spin_singlets() -> None:
    pairing = build_onsite_s_wave_pairing(chain(2), pairing=0.7)
    expected = np.asarray(
        [
            [0.0, 0.7, 0.0, 0.0],
            [-0.7, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.7],
            [0.0, 0.0, -0.7, 0.0],
        ],
        dtype=complex,
    )

    assert np.array_equal(pairing, expected)
    assert np.array_equal(pairing, -pairing.T)


def test_site_mapping_supports_inhomogeneous_complex_pairing() -> None:
    geometry = chain(3)
    coefficients = {0: 0.2 + 0.1j, 1: -0.4j, 2: 0.8}

    pairing = build_onsite_s_wave_pairing(geometry, pairing=coefficients)

    for site, coefficient in coefficients.items():
        spin_up = 2 * site
        spin_down = spin_up + 1
        assert pairing[spin_up, spin_down] == coefficient
        assert pairing[spin_down, spin_up] == -coefficient
    assert np.array_equal(pairing, -pairing.T)


def test_callable_is_evaluated_for_each_site() -> None:
    geometry = chain(3)

    pairing = build_onsite_s_wave_pairing(
        geometry,
        pairing=lambda site: 0.1 * (site + 1),
    )

    assert np.allclose(pairing[[0, 2, 4], [1, 3, 5]], [0.1, 0.2, 0.3])


def test_single_site_s_wave_bdg_spectrum_matches_analytic_result() -> None:
    geometry = Geometry(n_sites=1)
    chemical_potential = 0.6
    gap = 0.7 + 0.2j
    pairing = build_onsite_s_wave_pairing(geometry, pairing=gap)
    basis = NambuBasis(
        n_sites=1,
        normal_components_per_site=2,
        normal_component_labels=("up", "down"),
    )

    hamiltonian = build_bdg_hamiltonian(
        -chemical_potential * np.eye(2),
        pairing,
        basis=basis,
    )

    energy = np.sqrt(chemical_potential**2 + abs(gap) ** 2)
    assert np.allclose(np.linalg.eigvalsh(hamiltonian), [-energy, -energy, energy, energy])


def test_incomplete_onsite_pairing_mapping_is_rejected() -> None:
    with pytest.raises(ValueError, match="no value for site 1"):
        build_onsite_s_wave_pairing(chain(2), pairing={0: 0.5})


def test_nonscalar_onsite_pairing_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be a scalar"):
        build_onsite_s_wave_pairing(chain(2), pairing=np.eye(2))


def test_nonfinite_onsite_pairing_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        build_onsite_s_wave_pairing(chain(2), pairing=np.nan)
