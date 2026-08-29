from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.geometry import chain, irregular_cluster, ring
from toposc_lab.hamiltonians import (
    NambuBasis,
    build_bdg_hamiltonian,
    build_spinless_p_wave_pairing,
)


def test_constant_pairing_on_open_chain_is_antisymmetric() -> None:
    pairing = build_spinless_p_wave_pairing(chain(4), pairing=0.6)
    expected = np.asarray(
        [
            [0.0, 0.6, 0.0, 0.0],
            [-0.6, 0.0, 0.6, 0.0],
            [0.0, -0.6, 0.0, 0.6],
            [0.0, 0.0, -0.6, 0.0],
        ],
        dtype=complex,
    )

    assert np.array_equal(pairing, expected)
    assert np.array_equal(pairing, -pairing.T)


def test_complex_edge_mapping_preserves_orientation_without_conjugation() -> None:
    geometry = ring(4)
    coefficients = {
        edge: (index + 1) * (0.2 + 0.1j)
        for index, edge in enumerate(geometry.edges)
    }

    pairing = build_spinless_p_wave_pairing(geometry, pairing=coefficients)

    for edge, coefficient in coefficients.items():
        assert pairing[edge.source, edge.target] == coefficient
        assert pairing[edge.target, edge.source] == -coefficient
    assert pairing[3, 0] == coefficients[geometry.edges[-1]]
    assert np.array_equal(pairing, -pairing.T)


def test_callable_can_select_pairing_by_edge_metadata() -> None:
    geometry = irregular_cluster()

    pairing = build_spinless_p_wave_pairing(
        geometry,
        pairing=lambda edge: 0.0 if edge.edge_type == "long_range" else 0.4,
    )

    assert pairing[0, 5] == 0.0
    assert pairing[0, 1] == 0.4
    assert pairing[1, 0] == -0.4


def test_pairing_matrix_integrates_with_generic_bdg_builder() -> None:
    geometry = chain(3)
    pairing = build_spinless_p_wave_pairing(geometry, pairing=0.5j)

    hamiltonian = build_bdg_hamiltonian(
        np.zeros((3, 3)),
        pairing,
        basis=NambuBasis(n_sites=3),
    )

    assert np.array_equal(hamiltonian, hamiltonian.conj().T)


def test_incomplete_pairing_mapping_is_rejected() -> None:
    geometry = chain(3)

    with pytest.raises(ValueError, match=r"no value for edge \(1, 2\)"):
        build_spinless_p_wave_pairing(
            geometry,
            pairing={geometry.edges[0]: 0.5},
        )


def test_nonscalar_pairing_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be a scalar"):
        build_spinless_p_wave_pairing(chain(2), pairing=np.eye(2))


@pytest.mark.parametrize("value", [np.nan, np.inf, complex(0.0, np.inf)])
def test_nonfinite_pairing_is_rejected(value: complex) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        build_spinless_p_wave_pairing(chain(2), pairing=value)
