from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.geometry import Geometry, GeometryEdge, square
from toposc_lab.hamiltonians import build_chiral_p_wave_pairing
from toposc_lab.models.chiral_p_wave import (
    ChiralPWaveModel,
    ChiralPWaveParameters,
)
from toposc_lab.observables.symmetries import check_bdg_particle_hole_symmetry


def _parameters(*, chirality: int = 1) -> ChiralPWaveParameters:
    return ChiralPWaveParameters(
        hopping=1.0,
        chemical_potential=0.4,
        pairing=0.6,
        chirality=chirality,  # type: ignore[arg-type]
    )


def test_square_edges_realize_px_plus_i_py_pairing() -> None:
    geometry = square(2, 2)
    amplitude = 0.7
    pairing = build_chiral_p_wave_pairing(geometry, pairing=amplitude)

    for edge in geometry.edges:
        direction = geometry.direction(edge.source, edge.target)
        expected = amplitude * (direction[0] + 1.0j * direction[1])
        assert pairing[edge.source, edge.target] == expected
        assert pairing[edge.target, edge.source] == -expected
    assert np.array_equal(pairing, -pairing.T)


def test_negative_chirality_complex_conjugates_real_amplitude_pairing() -> None:
    geometry = square(2, 2)

    positive = build_chiral_p_wave_pairing(geometry, pairing=0.5, chirality=1)
    negative = build_chiral_p_wave_pairing(geometry, pairing=0.5, chirality=-1)

    assert np.array_equal(negative, positive.conj())


def test_diagonal_edge_uses_normalized_direction() -> None:
    geometry = Geometry(
        n_sites=2,
        edges=(GeometryEdge(0, 1, displacement=(3.0, 4.0)),),
    )

    pairing = build_chiral_p_wave_pairing(geometry, pairing=2.0)

    assert pairing[0, 1] == pytest.approx(2.0 * (0.6 + 0.8j))
    assert pairing[1, 0] == pytest.approx(-2.0 * (0.6 + 0.8j))


def test_periodic_pairing_uses_local_closing_edge_displacement() -> None:
    geometry = square(3, 2, boundary_x="periodic")
    closing_edge = next(edge for edge in geometry.edges if edge.boundary_crossing)

    pairing = build_chiral_p_wave_pairing(geometry, pairing=0.4)

    assert pairing[closing_edge.source, closing_edge.target] == 0.4


def test_coordinate_free_graph_without_displacements_is_rejected() -> None:
    geometry = Geometry(n_sites=2, edges=(GeometryEdge(0, 1),))

    with pytest.raises(ValueError, match="coordinates or explicit nonzero"):
        build_chiral_p_wave_pairing(geometry, pairing=0.5)


def test_chiral_model_builds_hermitian_particle_hole_symmetric_bdg_matrix() -> None:
    geometry = square(2, 2)
    model = ChiralPWaveModel(geometry, _parameters())

    hamiltonian = model.hamiltonian()
    symmetry = check_bdg_particle_hole_symmetry(hamiltonian, model.nambu_basis)

    assert hamiltonian.shape == (2 * geometry.n_sites, 2 * geometry.n_sites)
    assert np.array_equal(hamiltonian, hamiltonian.conj().T)
    assert symmetry.satisfied
    assert model.basis_layout == model.nambu_basis.basis_layout


def test_opposite_chiralities_are_complex_conjugate_models() -> None:
    geometry = square(3, 2)
    positive = ChiralPWaveModel(geometry, _parameters(chirality=1)).hamiltonian()
    negative = ChiralPWaveModel(geometry, _parameters(chirality=-1)).hamiltonian()

    assert np.array_equal(negative, positive.conj())
    assert np.array_equal(np.linalg.eigvalsh(negative), np.linalg.eigvalsh(positive))


@pytest.mark.parametrize("chirality", [0, 2, True])
def test_invalid_chirality_is_rejected(chirality: object) -> None:
    with pytest.raises((TypeError, ValueError), match="chirality"):
        build_chiral_p_wave_pairing(
            square(2, 2),
            pairing=0.5,
            chirality=chirality,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("axes", [(0, 0), (-1, 1)])
def test_invalid_pairing_plane_axes_are_rejected(axes: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="plane_axes"):
        build_chiral_p_wave_pairing(
            square(2, 2),
            pairing=0.5,
            plane_axes=axes,
        )


def test_pairing_plane_axis_outside_embedding_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside direction dimension"):
        build_chiral_p_wave_pairing(
            square(2, 2),
            pairing=0.5,
            plane_axes=(0, 2),
        )
