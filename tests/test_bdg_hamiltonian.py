from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.hamiltonians import NambuBasis, build_bdg_hamiltonian


def test_spinless_component_major_builder_matches_explicit_blocks() -> None:
    normal_state = np.asarray(
        [[-0.3, -1.0 + 0.2j], [-1.0 - 0.2j, 0.4]],
        dtype=complex,
    )
    pairing = np.asarray([[0.0, 0.7], [-0.7, 0.0]], dtype=complex)

    hamiltonian = build_bdg_hamiltonian(
        normal_state,
        pairing,
        basis=NambuBasis(n_sites=2),
    )

    expected = np.block(
        [
            [normal_state, pairing],
            [-pairing.conj(), -normal_state.conj()],
        ]
    )
    assert np.array_equal(hamiltonian, expected)
    assert np.array_equal(hamiltonian, hamiltonian.conj().T)


@pytest.mark.parametrize("ordering", ["component_major", "site_major"])
def test_multicomponent_blocks_follow_explicit_nambu_indices(ordering: str) -> None:
    basis = NambuBasis(
        n_sites=2,
        normal_components_per_site=2,
        ordering=ordering,  # type: ignore[arg-type]
    )
    normal_state = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
    pairing = np.asarray(
        [
            [0.0, 1.0, 2.0j, 0.0],
            [-1.0, 0.0, 0.0, 3.0],
            [-2.0j, 0.0, 0.0, -1.0j],
            [0.0, -3.0, 1.0j, 0.0],
        ],
        dtype=complex,
    )

    hamiltonian = build_bdg_hamiltonian(normal_state, pairing, basis=basis)
    particle_indices = np.asarray(basis.particle_indices)
    hole_indices = np.asarray(basis.hole_indices)

    assert np.array_equal(
        hamiltonian[np.ix_(particle_indices, particle_indices)],
        normal_state,
    )
    assert np.array_equal(
        hamiltonian[np.ix_(particle_indices, hole_indices)],
        pairing,
    )
    assert np.array_equal(
        hamiltonian[np.ix_(hole_indices, particle_indices)],
        -pairing.conj(),
    )
    assert np.array_equal(
        hamiltonian[np.ix_(hole_indices, hole_indices)],
        -normal_state.conj(),
    )
    assert np.array_equal(hamiltonian, hamiltonian.conj().T)


@pytest.mark.parametrize("block_name", ["normal_state", "pairing"])
def test_builder_rejects_block_with_wrong_shape(block_name: str) -> None:
    blocks = {
        "normal_state": np.eye(2),
        "pairing": np.zeros((2, 2)),
    }
    blocks[block_name] = np.zeros((3, 3))

    with pytest.raises(ValueError, match=rf"{block_name} must have shape \(2, 2\)"):
        build_bdg_hamiltonian(
            blocks["normal_state"],
            blocks["pairing"],
            basis=NambuBasis(n_sites=2),
        )


def test_builder_rejects_nonfinite_values() -> None:
    pairing = np.zeros((2, 2), dtype=complex)
    pairing[0, 1] = np.nan

    with pytest.raises(ValueError, match="pairing must contain only finite values"):
        build_bdg_hamiltonian(
            np.eye(2),
            pairing,
            basis=NambuBasis(n_sites=2),
        )


def test_builder_rejects_nonhermitian_normal_state() -> None:
    normal_state = np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=complex)

    with pytest.raises(
        ValueError,
        match=r"normal_state must be Hermitian.*maximum residual",
    ):
        build_bdg_hamiltonian(
            normal_state,
            np.zeros((2, 2)),
            basis=NambuBasis(n_sites=2),
        )


def test_builder_rejects_nonantisymmetric_pairing() -> None:
    pairing = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

    with pytest.raises(
        ValueError,
        match=r"pairing must be antisymmetric.*maximum residual",
    ):
        build_bdg_hamiltonian(
            np.eye(2),
            pairing,
            basis=NambuBasis(n_sites=2),
        )


def test_validation_tolerance_accepts_small_numerical_residuals() -> None:
    normal_state = np.asarray([[0.0, 1.0e-12], [0.0, 0.0]], dtype=complex)

    hamiltonian = build_bdg_hamiltonian(
        normal_state,
        np.zeros((2, 2)),
        basis=NambuBasis(n_sites=2),
        validation_tolerance=1.0e-10,
    )

    assert np.allclose(
        hamiltonian,
        hamiltonian.conj().T,
        rtol=0.0,
        atol=1.0e-10,
    )


@pytest.mark.parametrize("tolerance", [-1.0, np.inf, np.nan])
def test_invalid_validation_tolerance_is_rejected(tolerance: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_bdg_hamiltonian(
            np.eye(2),
            np.zeros((2, 2)),
            basis=NambuBasis(n_sites=2),
            validation_tolerance=tolerance,
        )


@pytest.mark.parametrize("tolerance", [True, "strict"])
def test_nonreal_validation_tolerance_is_rejected(tolerance: object) -> None:
    with pytest.raises(TypeError, match="real number"):
        build_bdg_hamiltonian(
            np.eye(2),
            np.zeros((2, 2)),
            basis=NambuBasis(n_sites=2),
            validation_tolerance=tolerance,  # type: ignore[arg-type]
        )
