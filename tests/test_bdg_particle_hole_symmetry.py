from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.hamiltonians import NambuBasis, build_bdg_hamiltonian
from toposc_lab.observables.symmetries import check_bdg_particle_hole_symmetry


@pytest.mark.parametrize("ordering", ["component_major", "site_major"])
def test_generic_bdg_builder_has_basis_aware_particle_hole_symmetry(
    ordering: str,
) -> None:
    basis = NambuBasis(n_sites=2, ordering=ordering)  # type: ignore[arg-type]
    normal_state = np.asarray(
        [[0.3, -1.0 + 0.2j], [-1.0 - 0.2j, -0.4]],
        dtype=complex,
    )
    pairing = np.asarray([[0.0, 0.7j], [-0.7j, 0.0]], dtype=complex)
    hamiltonian = build_bdg_hamiltonian(normal_state, pairing, basis=basis)

    result = check_bdg_particle_hole_symmetry(hamiltonian, basis)

    assert result.satisfied
    assert result.residual == pytest.approx(0.0)
    assert result.tolerance == 1e-10


def test_basis_aware_check_detects_broken_particle_hole_symmetry() -> None:
    basis = NambuBasis(n_sites=2)
    hamiltonian = build_bdg_hamiltonian(
        np.diag([0.3, -0.4]),
        np.zeros((2, 2)),
        basis=basis,
    )
    hamiltonian[basis.hole_index(0), basis.hole_index(0)] += 0.1

    result = check_bdg_particle_hole_symmetry(hamiltonian, basis)

    assert not result.satisfied
    assert result.residual == pytest.approx(0.1)


def test_basis_aware_check_rejects_dimension_mismatch() -> None:
    basis = NambuBasis(n_sites=2)

    with pytest.raises(ValueError, match=r"hamiltonian must have shape \(4, 4\)"):
        check_bdg_particle_hole_symmetry(np.eye(6), basis)
