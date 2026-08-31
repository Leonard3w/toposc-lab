from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.evaluation import exact_geometry_id
from toposc_lab.geometry import Geometry, GeometryEdge, chain
from toposc_lab.hamiltonians import (
    NambuBasis,
    sample_uniform_edge_disorder,
    uniform_edge_disorder,
)
from toposc_lab.robustness import (
    UNIFORM_HOPPING_DISORDER_KEY,
    UNIFORM_HOPPING_DISORDER_VERSION,
    apply_uniform_hopping_disorder,
)


def test_normal_hopping_disorder_reuses_existing_uniform_reference_sample() -> None:
    geometry = chain(4)
    clean = np.zeros((4, 4), dtype=np.complex128)
    expected = uniform_edge_disorder(geometry, width=0.8, seed=17)

    realization = apply_uniform_hopping_disorder(
        geometry,
        clean,
        width=0.8,
        seed=17,
    )

    assert isinstance(realization.state, np.ndarray)
    for edge, offset in expected.items():
        assert realization.state[edge.source, edge.target] == offset
        assert realization.state[edge.target, edge.source] == offset
    assert np.array_equal(realization.state, realization.state.conj().T)
    assert realization.provenance.disorder_key == UNIFORM_HOPPING_DISORDER_KEY
    assert realization.provenance.disorder_version == (
        UNIFORM_HOPPING_DISORDER_VERSION
    )
    assert realization.provenance.parameters == {
        "basis_ordering": "site_major",
        "distribution": "uniform",
        "edge_application": "oriented_forward_plus_hermitian_conjugate",
        "geometry_id": exact_geometry_id(geometry),
        "n_edges": 3,
        "n_sites": 4,
        "normal_components_per_site": 1,
        "particle_hole_embedding": "not_applicable",
        "representation": "normal_state",
        "width": 0.8,
    }


def test_supplied_rng_sampler_matches_seeded_edge_sampler_and_keeps_keys() -> None:
    geometry = Geometry(
        n_sites=3,
        edges=(GeometryEdge(2, 0), GeometryEdge(0, 1)),
    )
    rng = np.random.Generator(np.random.PCG64(29))

    sampled = sample_uniform_edge_disorder(geometry, width=1.2, rng=rng)

    assert sampled == uniform_edge_disorder(geometry, width=1.2, seed=29)
    assert tuple(sampled) == geometry.edges
    assert tuple((edge.source, edge.target) for edge in sampled) == ((2, 0), (0, 1))


def test_multicomponent_hopping_is_scalar_and_preserves_oriented_base_blocks() -> None:
    edge = GeometryEdge(2, 0, edge_type="oriented_test")
    geometry = Geometry(n_sites=3, edges=(edge,))
    clean = np.zeros((6, 6), dtype=np.complex128)
    clean[4, 0] = 1.0 + 2.0j
    clean[0, 4] = 1.0 - 2.0j
    clean[5, 1] = 3.0 + 4.0j
    clean[1, 5] = 3.0 - 4.0j
    offset = uniform_edge_disorder(geometry, width=0.6, seed=3)[edge]

    realization = apply_uniform_hopping_disorder(
        geometry,
        clean,
        width=0.6,
        seed=3,
        components_per_site=2,
    )

    assert isinstance(realization.state, np.ndarray)
    assert realization.state[4, 0] == clean[4, 0] + offset
    assert realization.state[0, 4] == clean[0, 4] + offset
    assert realization.state[5, 1] == clean[5, 1] + offset
    assert realization.state[1, 5] == clean[1, 5] + offset
    assert np.array_equal(realization.state, realization.state.conj().T)


@pytest.mark.parametrize("ordering", ["component_major", "site_major"])
def test_bdg_hopping_disorder_preserves_particle_hole_embedding(
    ordering: str,
) -> None:
    geometry = chain(3)
    basis = NambuBasis(
        n_sites=3,
        normal_components_per_site=2,
        ordering=ordering,  # type: ignore[arg-type]
    )
    clean = np.zeros((basis.dimension, basis.dimension), dtype=np.complex128)
    offsets = uniform_edge_disorder(geometry, width=0.4, seed=41)

    realization = apply_uniform_hopping_disorder(
        geometry,
        clean,
        width=0.4,
        seed=41,
        nambu_basis=basis,
    )

    assert isinstance(realization.state, np.ndarray)
    disordered = np.asarray(realization.state, dtype=np.complex128)
    for edge, offset in offsets.items():
        for component in range(2):
            particle_source = basis.particle_index(edge.source, component=component)
            particle_target = basis.particle_index(edge.target, component=component)
            hole_source = basis.hole_index(edge.source, component=component)
            hole_target = basis.hole_index(edge.target, component=component)
            assert disordered[particle_source, particle_target] == offset
            assert disordered[particle_target, particle_source] == offset
            assert disordered[hole_source, hole_target] == -offset
            assert disordered[hole_target, hole_source] == -offset
    particle_hole = basis.particle_hole_operator
    transformed = particle_hole @ disordered.conj() @ particle_hole.conj().T
    assert np.allclose(transformed, -disordered)
    assert realization.provenance.parameters["basis_ordering"] == ordering


def test_zero_width_preserves_exact_matrix_snapshot() -> None:
    geometry = chain(2)
    clean = np.asarray([[1, 2], [2, 3]], dtype=np.int64)

    realization = apply_uniform_hopping_disorder(
        geometry,
        clean,
        width=0.0,
        seed=9,
    )

    assert isinstance(realization.state, np.ndarray)
    assert np.array_equal(realization.state, clean)
    assert realization.state.dtype == clean.dtype
    assert realization.provenance.source == realization.provenance.result


@pytest.mark.parametrize("width", [-0.1, np.inf, np.nan])
def test_invalid_width_is_rejected(width: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        apply_uniform_hopping_disorder(
            chain(2),
            np.eye(2),
            width=width,
            seed=1,
        )


def test_normal_matrix_shape_must_match_geometry_and_components() -> None:
    with pytest.raises(ValueError, match="shape does not match"):
        apply_uniform_hopping_disorder(
            chain(3),
            np.eye(4),
            width=0.2,
            seed=1,
        )


def test_nambu_basis_contract_matches_onsite_disorder_contract() -> None:
    geometry = chain(3)
    mismatched = NambuBasis(n_sites=2)
    with pytest.raises(ValueError, match="site count must match"):
        apply_uniform_hopping_disorder(
            geometry,
            np.eye(mismatched.dimension),
            width=0.2,
            seed=1,
            nambu_basis=mismatched,
        )

    basis = NambuBasis(n_sites=3)
    with pytest.raises(ValueError, match="must be omitted"):
        apply_uniform_hopping_disorder(
            geometry,
            np.eye(basis.dimension),
            width=0.2,
            seed=1,
            components_per_site=1,
            nambu_basis=basis,
        )
