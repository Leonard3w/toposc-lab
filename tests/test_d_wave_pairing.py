from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.geometry import Geometry, GeometryEdge, square
from toposc_lab.hamiltonians import (
    NambuBasis,
    build_bdg_hamiltonian,
    build_d_wave_pairing,
    build_tight_binding_hamiltonian,
)
from toposc_lab.observables.symmetries import check_bdg_particle_hole_symmetry


def _spin_block(matrix: np.ndarray, source: int, target: int) -> np.ndarray:
    source_slice = slice(2 * source, 2 * source + 2)
    target_slice = slice(2 * target, 2 * target + 2)
    return matrix[source_slice, target_slice]


def test_square_lattice_x_and_y_bonds_have_opposite_d_wave_signs() -> None:
    geometry = square(2, 2)
    amplitude = 0.6
    pairing = build_d_wave_pairing(geometry, pairing=amplitude)
    spin_singlet = np.asarray([[0.0, 1.0], [-1.0, 0.0]])

    for edge in geometry.edges:
        direction = geometry.direction(edge.source, edge.target)
        expected_sign = 1.0 if direction[0] != 0.0 else -1.0
        expected = amplitude * expected_sign * spin_singlet
        assert np.array_equal(_spin_block(pairing, edge.source, edge.target), expected)
        assert np.array_equal(_spin_block(pairing, edge.target, edge.source), expected)
    assert np.array_equal(pairing, -pairing.T)


def test_spatial_pairing_is_even_under_edge_orientation_reversal() -> None:
    coordinates = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    forward = Geometry(
        n_sites=2,
        edges=(GeometryEdge(0, 1),),
        coordinates=coordinates,
    )
    reverse = Geometry(
        n_sites=2,
        edges=(GeometryEdge(1, 0),),
        coordinates=coordinates,
    )

    forward_pairing = build_d_wave_pairing(forward, pairing=0.5)
    reverse_pairing = build_d_wave_pairing(reverse, pairing=0.5)

    assert np.array_equal(reverse_pairing, forward_pairing)


def test_diagonal_bond_is_a_d_wave_node() -> None:
    geometry = Geometry(
        n_sites=2,
        edges=(GeometryEdge(0, 1, displacement=(1.0, 1.0)),),
    )

    pairing = build_d_wave_pairing(geometry, pairing=0.8)

    assert np.allclose(pairing, 0.0, atol=1e-15)


def test_edge_mapping_supports_complex_inhomogeneous_d_wave_pairing() -> None:
    geometry = square(2, 2)
    amplitudes = {
        edge: (index + 1) * (0.1 + 0.2j)
        for index, edge in enumerate(geometry.edges)
    }

    pairing = build_d_wave_pairing(geometry, pairing=amplitudes)

    assert np.array_equal(pairing, -pairing.T)
    for edge, amplitude in amplitudes.items():
        direction = geometry.direction(edge.source, edge.target)
        form_factor = direction[0] ** 2 - direction[1] ** 2
        assert _spin_block(pairing, edge.source, edge.target)[0, 1] == pytest.approx(
            amplitude * form_factor
        )


def test_periodic_edges_retain_local_d_wave_sign() -> None:
    geometry = square(3, 3, boundary_x="periodic", boundary_y="periodic")
    pairing = build_d_wave_pairing(geometry, pairing=0.4)
    spin_singlet = np.asarray([[0.0, 1.0], [-1.0, 0.0]])

    for edge in geometry.edges:
        if not edge.boundary_crossing:
            continue
        direction = geometry.direction(edge.source, edge.target)
        form_factor = direction[0] ** 2 - direction[1] ** 2
        expected = 0.4 * form_factor * spin_singlet
        assert np.array_equal(_spin_block(pairing, edge.source, edge.target), expected)


def test_d_wave_pairing_integrates_with_spinful_bdg_builder() -> None:
    geometry = square(2, 2)
    normal_state = build_tight_binding_hamiltonian(
        geometry,
        onsite=-0.3,
        hopping=-1.0,
        components_per_site=2,
    )
    pairing = build_d_wave_pairing(geometry, pairing=0.5)
    basis = NambuBasis(
        n_sites=geometry.n_sites,
        normal_components_per_site=2,
        normal_component_labels=("up", "down"),
    )

    hamiltonian = build_bdg_hamiltonian(normal_state, pairing, basis=basis)
    symmetry = check_bdg_particle_hole_symmetry(hamiltonian, basis)

    assert np.array_equal(hamiltonian, hamiltonian.conj().T)
    assert symmetry.satisfied


def test_coordinate_free_graph_without_displacements_is_rejected() -> None:
    geometry = Geometry(n_sites=2, edges=(GeometryEdge(0, 1),))

    with pytest.raises(ValueError, match="coordinates or explicit nonzero"):
        build_d_wave_pairing(geometry, pairing=0.5)


def test_pairing_plane_axis_outside_embedding_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside direction dimension"):
        build_d_wave_pairing(square(2, 2), pairing=0.5, plane_axes=(0, 2))


@pytest.mark.parametrize("axes", [(0, 0), (-1, 1)])
def test_invalid_d_wave_plane_axes_are_rejected(axes: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="plane_axes"):
        build_d_wave_pairing(square(2, 2), pairing=0.5, plane_axes=axes)
