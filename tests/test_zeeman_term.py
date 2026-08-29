from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.geometry import chain
from toposc_lab.hamiltonians import (
    NambuBasis,
    build_bdg_hamiltonian,
    build_onsite_s_wave_pairing,
    build_tight_binding_hamiltonian,
    build_zeeman_term,
    zeeman_matrix,
)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ([1.0, 0.0, 0.0], [[0.0, 1.0], [1.0, 0.0]]),
        ([0.0, 1.0, 0.0], [[0.0, -1.0j], [1.0j, 0.0]]),
        ([0.0, 0.0, 1.0], [[1.0, 0.0], [0.0, -1.0]]),
    ],
)
def test_cartesian_fields_produce_pauli_matrices(
    field: list[float],
    expected: list[list[complex]],
) -> None:
    matrix = zeeman_matrix(field)

    assert np.array_equal(matrix, np.asarray(expected))
    assert np.array_equal(matrix, matrix.conj().T)


def test_site_dependent_fields_create_site_major_onsite_blocks() -> None:
    geometry = chain(2)
    fields = {0: [0.2, 0.0, 0.5], 1: [0.0, -0.3, -0.4]}

    term = build_zeeman_term(geometry, field=fields)

    assert np.array_equal(term[0:2, 0:2], zeeman_matrix(fields[0]))
    assert np.array_equal(term[2:4, 2:4], zeeman_matrix(fields[1]))
    assert np.array_equal(term[0:2, 2:4], np.zeros((2, 2)))
    assert np.array_equal(term, term.conj().T)


def test_callable_field_is_evaluated_for_each_site() -> None:
    geometry = chain(3)

    term = build_zeeman_term(
        geometry,
        field=lambda site: [0.0, 0.0, 0.1 * (site + 1)],
    )

    assert np.allclose(np.diag(term), [0.1, -0.1, 0.2, -0.2, 0.3, -0.3])


def test_zeeman_term_integrates_with_spinful_s_wave_bdg_builder() -> None:
    geometry = chain(2)
    chemical_potential = 0.4
    normal_state = build_tight_binding_hamiltonian(
        geometry,
        onsite=-chemical_potential,
        hopping=-1.0,
        components_per_site=2,
    )
    normal_state += build_zeeman_term(geometry, field=[0.2, -0.1, 0.3])
    pairing = build_onsite_s_wave_pairing(geometry, pairing=0.5)
    basis = NambuBasis(
        n_sites=2,
        normal_components_per_site=2,
        normal_component_labels=("up", "down"),
    )

    hamiltonian = build_bdg_hamiltonian(normal_state, pairing, basis=basis)

    assert np.array_equal(hamiltonian, hamiltonian.conj().T)


def test_incomplete_field_mapping_is_rejected() -> None:
    with pytest.raises(ValueError, match="no value for site 1"):
        build_zeeman_term(chain(2), field={0: [0.0, 0.0, 1.0]})


@pytest.mark.parametrize("field", [[1.0, 2.0], np.eye(3)])
def test_field_with_wrong_shape_is_rejected(field: object) -> None:
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        zeeman_matrix(field)  # type: ignore[arg-type]


def test_complex_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be real"):
        zeeman_matrix([0.0, 1.0j, 0.0])


@pytest.mark.parametrize("field", [[np.nan, 0.0, 0.0], [0.0, np.inf, 0.0]])
def test_nonfinite_field_is_rejected(field: list[float]) -> None:
    with pytest.raises(ValueError, match="finite values"):
        zeeman_matrix(field)
