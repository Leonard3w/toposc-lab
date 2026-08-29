from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.geometry import Geometry, GeometryEdge, square
from toposc_lab.hamiltonians import (
    NambuBasis,
    build_bdg_hamiltonian,
    build_onsite_s_wave_pairing,
    build_rashba_term,
    rashba_hopping_matrix,
)


def test_cartesian_directions_produce_expected_rashba_hoppings() -> None:
    coupling = 0.4

    hopping_x = rashba_hopping_matrix([1.0, 0.0], coupling=coupling)
    hopping_y = rashba_hopping_matrix([0.0, 1.0], coupling=coupling)

    assert np.array_equal(hopping_x, [[0.0, -coupling], [coupling, 0.0]])
    assert np.array_equal(
        hopping_y,
        [[0.0, 1.0j * coupling], [1.0j * coupling, 0.0]],
    )
    assert np.array_equal(hopping_x.conj().T, -hopping_x)
    assert np.array_equal(hopping_y.conj().T, -hopping_y)


def test_square_lattice_term_follows_every_oriented_edge() -> None:
    geometry = square(2, 2)
    coupling = 0.3

    term = build_rashba_term(geometry, coupling=coupling)

    for edge in geometry.edges:
        expected = rashba_hopping_matrix(
            geometry.displacement_between(edge.source, edge.target),
            coupling=coupling,
        )
        source = slice(2 * edge.source, 2 * edge.source + 2)
        target = slice(2 * edge.target, 2 * edge.target + 2)
        assert np.array_equal(term[source, target], expected)
        assert np.array_equal(term[target, source], expected.conj().T)
    assert np.array_equal(term, term.conj().T)


def test_periodic_edge_uses_explicit_local_displacement() -> None:
    geometry = square(3, 2, boundary_x="periodic")
    closing_edge = next(edge for edge in geometry.edges if edge.boundary_crossing)

    term = build_rashba_term(geometry, coupling=0.5)
    source = slice(2 * closing_edge.source, 2 * closing_edge.source + 2)
    target = slice(2 * closing_edge.target, 2 * closing_edge.target + 2)

    assert np.array_equal(
        term[source, target],
        rashba_hopping_matrix([1.0, 0.0], coupling=0.5),
    )


def test_edge_dependent_coupling_mapping_is_supported() -> None:
    geometry = square(2, 2)
    couplings = {
        edge: 0.1 * (index + 1)
        for index, edge in enumerate(geometry.edges)
    }

    term = build_rashba_term(geometry, coupling=couplings)

    for edge, coupling in couplings.items():
        source = slice(2 * edge.source, 2 * edge.source + 2)
        target = slice(2 * edge.target, 2 * edge.target + 2)
        expected = rashba_hopping_matrix(
            geometry.displacement_between(edge.source, edge.target),
            coupling=coupling,
        )
        assert np.allclose(term[source, target], expected)


def test_coordinate_free_graph_accepts_explicit_edge_displacement() -> None:
    geometry = Geometry(
        n_sites=2,
        edges=(GeometryEdge(0, 1, displacement=(0.0, 2.0)),),
    )

    term = build_rashba_term(geometry, coupling=0.6)

    assert np.array_equal(
        term[0:2, 2:4],
        rashba_hopping_matrix([0.0, 1.0], coupling=0.6),
    )


def test_coordinate_free_graph_without_displacements_is_rejected() -> None:
    geometry = Geometry(n_sites=2, edges=(GeometryEdge(0, 1),))

    with pytest.raises(ValueError, match="coordinates or explicit edge displacements"):
        build_rashba_term(geometry, coupling=0.5)


def test_selected_plane_axes_work_in_higher_dimensional_embedding() -> None:
    hopping = rashba_hopping_matrix(
        [1.0, 4.0, 1.0],
        coupling=0.5,
        plane_axes=(0, 2),
    )
    expected = rashba_hopping_matrix(
        [1.0, 1.0],
        coupling=0.5 / 3.0,
    )

    assert np.allclose(hopping, expected)


def test_rashba_term_integrates_with_spinful_s_wave_bdg_builder() -> None:
    geometry = square(2, 2)
    normal_state = build_rashba_term(geometry, coupling=0.25)
    pairing = build_onsite_s_wave_pairing(geometry, pairing=0.4)
    basis = NambuBasis(
        n_sites=geometry.n_sites,
        normal_components_per_site=2,
        normal_component_labels=("up", "down"),
    )

    hamiltonian = build_bdg_hamiltonian(normal_state, pairing, basis=basis)

    assert np.array_equal(hamiltonian, hamiltonian.conj().T)


def test_incomplete_coupling_mapping_is_rejected() -> None:
    geometry = square(2, 2)

    with pytest.raises(ValueError, match="coupling mapping has no value"):
        build_rashba_term(geometry, coupling={geometry.edges[0]: 0.2})


@pytest.mark.parametrize("axes", [(0, 0), (-1, 1)])
def test_invalid_plane_axes_are_rejected(axes: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="plane_axes"):
        rashba_hopping_matrix([1.0, 0.0], coupling=0.2, plane_axes=axes)


def test_plane_axis_outside_embedding_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside direction dimension"):
        rashba_hopping_matrix([1.0, 0.0], coupling=0.2, plane_axes=(0, 2))


@pytest.mark.parametrize("coupling", [np.nan, np.inf])
def test_nonfinite_coupling_is_rejected(coupling: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        rashba_hopping_matrix([1.0, 0.0], coupling=coupling)


def test_complex_coupling_is_rejected() -> None:
    with pytest.raises(TypeError, match="real number"):
        rashba_hopping_matrix([1.0, 0.0], coupling=0.2j)  # type: ignore[arg-type]
