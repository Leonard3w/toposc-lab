from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.evaluation import exact_geometry_id
from toposc_lab.geometry import Geometry, GeometryEdge, chain, square
from toposc_lab.hamiltonians import (
    NambuBasis,
    build_chiral_p_wave_pairing,
    build_d_wave_pairing,
    build_onsite_s_wave_pairing,
    build_spinless_p_wave_pairing,
    uniform_edge_disorder,
    uniform_site_disorder,
)
from toposc_lab.observables.symmetries import check_bdg_particle_hole_symmetry
from toposc_lab.robustness import (
    UNIFORM_PAIRING_DISORDER_KEY,
    UNIFORM_PAIRING_DISORDER_VERSION,
    PairingDisorderChannel,
    apply_uniform_pairing_disorder,
)


def _pairing_block(matrix: np.ndarray, basis: NambuBasis) -> np.ndarray:
    particles = np.asarray(basis.particle_indices, dtype=np.intp)
    holes = np.asarray(basis.hole_indices, dtype=np.intp)
    return matrix[np.ix_(particles, holes)]


@pytest.mark.parametrize("ordering", ["component_major", "site_major"])
def test_spinless_p_wave_disorder_reuses_oriented_edge_offsets(
    ordering: str,
) -> None:
    geometry = Geometry(
        n_sites=3,
        edges=(GeometryEdge(2, 0), GeometryEdge(0, 1)),
    )
    basis = NambuBasis(n_sites=3, ordering=ordering)  # type: ignore[arg-type]
    clean = np.zeros((basis.dimension, basis.dimension), dtype=np.complex128)
    offsets = uniform_edge_disorder(geometry, width=0.8, seed=17)
    expected_pairing = build_spinless_p_wave_pairing(
        geometry,
        pairing=offsets,
    )

    realization = apply_uniform_pairing_disorder(
        geometry,
        clean,
        width=0.8,
        seed=17,
        nambu_basis=basis,
        channel=PairingDisorderChannel.SPINLESS_P_WAVE,
    )

    assert isinstance(realization.state, np.ndarray)
    disordered = np.asarray(realization.state, dtype=np.complex128)
    assert np.array_equal(_pairing_block(disordered, basis), expected_pairing)
    assert expected_pairing[2, 0] == offsets[geometry.edges[0]]
    assert expected_pairing[0, 2] == -offsets[geometry.edges[0]]
    assert np.array_equal(disordered, disordered.conj().T)
    assert check_bdg_particle_hole_symmetry(disordered, basis).satisfied


def test_spinless_pairing_disorder_records_complete_contract() -> None:
    geometry = chain(3)
    basis = NambuBasis(n_sites=3)
    realization = apply_uniform_pairing_disorder(
        geometry,
        np.zeros((basis.dimension, basis.dimension)),
        width=0.4,
        seed=9,
        nambu_basis=basis,
        channel=PairingDisorderChannel.SPINLESS_P_WAVE,
    )

    assert realization.provenance.disorder_key == UNIFORM_PAIRING_DISORDER_KEY
    assert realization.provenance.disorder_version == (
        UNIFORM_PAIRING_DISORDER_VERSION
    )
    assert realization.provenance.parameters == {
        "basis_ordering": "component_major",
        "channel": "spinless_p_wave",
        "chirality": None,
        "distribution": "uniform",
        "geometry_id": exact_geometry_id(geometry),
        "n_edges": 2,
        "n_sites": 3,
        "normal_components_per_site": 1,
        "pairing_embedding": "upper_delta_lower_negative_conjugate",
        "plane_axes": None,
        "support": "edge",
        "width": 0.4,
    }


def test_onsite_s_wave_disorder_reuses_site_offsets() -> None:
    geometry = chain(3)
    basis = NambuBasis(
        n_sites=3,
        normal_components_per_site=2,
        normal_component_labels=("up", "down"),
    )
    clean = np.zeros((basis.dimension, basis.dimension), dtype=np.complex128)
    offsets = uniform_site_disorder(geometry, width=0.6, seed=31)
    expected_pairing = build_onsite_s_wave_pairing(geometry, pairing=offsets)

    realization = apply_uniform_pairing_disorder(
        geometry,
        clean,
        width=0.6,
        seed=31,
        nambu_basis=basis,
        channel=PairingDisorderChannel.ONSITE_S_WAVE,
    )

    assert isinstance(realization.state, np.ndarray)
    disordered = np.asarray(realization.state, dtype=np.complex128)
    assert np.array_equal(_pairing_block(disordered, basis), expected_pairing)
    assert check_bdg_particle_hole_symmetry(disordered, basis).satisfied
    assert realization.provenance.parameters["support"] == "site"


def test_chiral_p_wave_disorder_uses_explicit_spatial_plane() -> None:
    geometry = square(2, 2)
    basis = NambuBasis(n_sites=geometry.n_sites)
    clean = np.zeros((basis.dimension, basis.dimension), dtype=np.complex128)
    offsets = uniform_edge_disorder(geometry, width=0.5, seed=23)
    expected_pairing = build_chiral_p_wave_pairing(
        geometry,
        pairing=offsets,
        chirality=-1,
        plane_axes=(0, 1),
    )

    realization = apply_uniform_pairing_disorder(
        geometry,
        clean,
        width=0.5,
        seed=23,
        nambu_basis=basis,
        channel=PairingDisorderChannel.CHIRAL_P_WAVE,
        chirality=-1,
        plane_axes=(0, 1),
    )

    assert isinstance(realization.state, np.ndarray)
    disordered = np.asarray(realization.state, dtype=np.complex128)
    assert np.array_equal(_pairing_block(disordered, basis), expected_pairing)
    assert check_bdg_particle_hole_symmetry(disordered, basis).satisfied
    assert realization.provenance.parameters["chirality"] == -1
    assert realization.provenance.parameters["plane_axes"] == (0, 1)


def test_d_wave_disorder_uses_existing_form_factor() -> None:
    geometry = square(2, 2)
    basis = NambuBasis(
        n_sites=geometry.n_sites,
        normal_components_per_site=2,
        normal_component_labels=("up", "down"),
    )
    clean = np.zeros((basis.dimension, basis.dimension), dtype=np.complex128)
    offsets = uniform_edge_disorder(geometry, width=0.5, seed=23)
    expected_pairing = build_d_wave_pairing(
        geometry,
        pairing=offsets,
        plane_axes=(0, 1),
    )

    realization = apply_uniform_pairing_disorder(
        geometry,
        clean,
        width=0.5,
        seed=23,
        nambu_basis=basis,
        channel=PairingDisorderChannel.D_WAVE,
        plane_axes=(0, 1),
    )

    assert isinstance(realization.state, np.ndarray)
    disordered = np.asarray(realization.state, dtype=np.complex128)
    assert np.array_equal(_pairing_block(disordered, basis), expected_pairing)
    assert check_bdg_particle_hole_symmetry(disordered, basis).satisfied


@pytest.mark.parametrize(
    ("channel", "components"),
    [
        (PairingDisorderChannel.SPINLESS_P_WAVE, 2),
        (PairingDisorderChannel.CHIRAL_P_WAVE, 2),
        (PairingDisorderChannel.ONSITE_S_WAVE, 1),
        (PairingDisorderChannel.D_WAVE, 1),
    ],
)
def test_pairing_channel_requires_explicit_compatible_components(
    channel: PairingDisorderChannel,
    components: int,
) -> None:
    geometry = square(2, 2)
    basis = NambuBasis(
        n_sites=geometry.n_sites,
        normal_components_per_site=components,
    )
    with pytest.raises(ValueError, match="requires exactly"):
        apply_uniform_pairing_disorder(
            geometry,
            np.eye(basis.dimension),
            width=0.2,
            seed=1,
            nambu_basis=basis,
            channel=channel,
            chirality=1 if channel is PairingDisorderChannel.CHIRAL_P_WAVE else None,
            plane_axes=(0, 1) if channel in {
                PairingDisorderChannel.CHIRAL_P_WAVE,
                PairingDisorderChannel.D_WAVE,
            } else None,
        )


def test_spatial_pairing_options_are_explicit_and_channel_specific() -> None:
    geometry = square(2, 2)
    spinless_basis = NambuBasis(n_sites=geometry.n_sites)
    with pytest.raises(ValueError, match="requires chirality and plane_axes"):
        apply_uniform_pairing_disorder(
            geometry,
            np.eye(spinless_basis.dimension),
            width=0.2,
            seed=1,
            nambu_basis=spinless_basis,
            channel=PairingDisorderChannel.CHIRAL_P_WAVE,
        )
    with pytest.raises(ValueError, match="does not accept"):
        apply_uniform_pairing_disorder(
            geometry,
            np.eye(spinless_basis.dimension),
            width=0.2,
            seed=1,
            nambu_basis=spinless_basis,
            channel=PairingDisorderChannel.SPINLESS_P_WAVE,
            plane_axes=(0, 1),
        )


def test_chiral_pairing_does_not_invent_directions_for_abstract_graph() -> None:
    geometry = Geometry(n_sites=2, edges=(GeometryEdge(0, 1),))
    basis = NambuBasis(n_sites=2)
    with pytest.raises(ValueError, match="coordinates or explicit"):
        apply_uniform_pairing_disorder(
            geometry,
            np.eye(basis.dimension),
            width=0.2,
            seed=1,
            nambu_basis=basis,
            channel=PairingDisorderChannel.CHIRAL_P_WAVE,
            chirality=1,
            plane_axes=(0, 1),
        )


def test_zero_width_preserves_exact_matrix_snapshot() -> None:
    geometry = chain(2)
    basis = NambuBasis(n_sites=2)
    clean = np.arange(basis.dimension**2).reshape(basis.dimension, basis.dimension)

    realization = apply_uniform_pairing_disorder(
        geometry,
        clean,
        width=0.0,
        seed=3,
        nambu_basis=basis,
        channel=PairingDisorderChannel.SPINLESS_P_WAVE,
    )

    assert isinstance(realization.state, np.ndarray)
    assert np.array_equal(realization.state, clean)
    assert realization.state.dtype == clean.dtype
    assert realization.provenance.source == realization.provenance.result


def test_bdg_matrix_shape_and_geometry_basis_site_count_are_explicit() -> None:
    geometry = chain(3)
    basis = NambuBasis(n_sites=3)
    with pytest.raises(ValueError, match="shape does not match"):
        apply_uniform_pairing_disorder(
            geometry,
            np.eye(4),
            width=0.2,
            seed=1,
            nambu_basis=basis,
            channel=PairingDisorderChannel.SPINLESS_P_WAVE,
        )

    mismatched = NambuBasis(n_sites=2)
    with pytest.raises(ValueError, match="site count must match"):
        apply_uniform_pairing_disorder(
            geometry,
            np.eye(mismatched.dimension),
            width=0.2,
            seed=1,
            nambu_basis=mismatched,
            channel=PairingDisorderChannel.SPINLESS_P_WAVE,
        )
