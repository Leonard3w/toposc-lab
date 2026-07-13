import numpy as np
import pytest

from toposc_lab.models.haldane_model import (
    HaldaneModel,
    HaldaneModelParameters,
)
from toposc_lab.observables.berry import berry_curvature, chern_number
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver


def _haldane_model(
    phase: float = np.pi / 2.0,
    sublattice_mass: float = 0.0,
) -> HaldaneModel:
    return HaldaneModel(
        HaldaneModelParameters(
            n_x=3,
            n_y=3,
            nearest_neighbor_hopping=1.0,
            next_nearest_neighbor_hopping=0.2,
            phase=phase,
            sublattice_mass=sublattice_mass,
            boundary_x="periodic",
            boundary_y="periodic",
        )
    )


def test_haldane_bloch_hamiltonian_is_hermitian() -> None:
    model = _haldane_model()

    hamiltonian = model.bloch_hamiltonian(0.37, -1.21)

    assert hamiltonian.shape == (2, 2)
    assert np.allclose(hamiltonian, hamiltonian.conj().T)


def test_bloch_spectrum_matches_periodic_real_space_spectrum() -> None:
    """Der Bloch-Operator muss den periodischen Realraum-Bulk reproduzieren."""
    model = _haldane_model()
    real_space_energies = ExactDiagonalizationSolver().solve_model(
        model
    ).eigenvalues

    bloch_energies: list[float] = []

    for index_x in range(3):
        for index_y in range(3):
            k_x = 2.0 * np.pi * index_x / 3.0
            k_y = 2.0 * np.pi * index_y / 3.0
            bloch_energies.extend(
                np.linalg.eigvalsh(
                    model.bloch_hamiltonian(k_x, k_y)
                )
            )

    assert np.allclose(
        np.sort(real_space_energies),
        np.sort(bloch_energies),
    )


def test_haldane_topological_phase_has_quantized_chern_number() -> None:
    model = _haldane_model()

    result = berry_curvature(model.bloch_hamiltonian, n_k=31)

    assert abs(result.chern_number) == pytest.approx(1.0, abs=1e-8)
    assert abs(chern_number(model.bloch_hamiltonian, n_k=31)) == 1


def test_haldane_phase_reversal_flips_chern_number() -> None:
    positive_phase = _haldane_model(phase=np.pi / 2.0)
    negative_phase = _haldane_model(phase=-np.pi / 2.0)

    positive_chern = chern_number(
        positive_phase.bloch_hamiltonian,
        n_k=31,
    )
    negative_chern = chern_number(
        negative_phase.bloch_hamiltonian,
        n_k=31,
    )

    assert positive_chern == -negative_chern


def test_large_sublattice_mass_is_topologically_trivial() -> None:
    model = _haldane_model(sublattice_mass=2.0)

    assert chern_number(model.bloch_hamiltonian, n_k=31) == 0


def test_chern_number_rejects_gapless_graphene_limit() -> None:
    graphene_limit = HaldaneModel(
        HaldaneModelParameters(
            n_x=3,
            n_y=3,
            next_nearest_neighbor_hopping=0.0,
            sublattice_mass=0.0,
        )
    )

    with pytest.raises(ValueError, match="gap closes"):
        chern_number(graphene_limit.bloch_hamiltonian, n_k=30)
