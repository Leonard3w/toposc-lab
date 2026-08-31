from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.evaluation import exact_geometry_id
from toposc_lab.geometry import chain
from toposc_lab.hamiltonians import (
    NambuBasis,
    sample_uniform_site_disorder,
    uniform_site_disorder,
)
from toposc_lab.robustness import (
    UNIFORM_ONSITE_DISORDER_KEY,
    UNIFORM_ONSITE_DISORDER_VERSION,
    apply_uniform_onsite_disorder,
)


def test_normal_onsite_disorder_reuses_existing_uniform_reference_sample() -> None:
    geometry = chain(4)
    clean = np.zeros((4, 4), dtype=np.complex128)
    expected = uniform_site_disorder(geometry, width=0.8, seed=17)

    realization = apply_uniform_onsite_disorder(
        geometry,
        clean,
        width=0.8,
        seed=17,
    )

    assert isinstance(realization.state, np.ndarray)
    assert np.array_equal(
        np.diag(realization.state),
        np.asarray(tuple(expected.values())),
    )
    assert realization.provenance.disorder_key == UNIFORM_ONSITE_DISORDER_KEY
    assert realization.provenance.disorder_version == UNIFORM_ONSITE_DISORDER_VERSION
    assert realization.provenance.seed == 17
    assert realization.provenance.parameters == {
        "basis_ordering": "site_major",
        "distribution": "uniform",
        "geometry_id": exact_geometry_id(geometry),
        "n_sites": 4,
        "normal_components_per_site": 1,
        "particle_hole_embedding": "not_applicable",
        "representation": "normal_state",
        "width": 0.8,
    }


def test_supplied_rng_sampler_matches_seeded_legacy_sampler() -> None:
    geometry = chain(5)
    rng = np.random.Generator(np.random.PCG64(29))

    sampled = sample_uniform_site_disorder(geometry, width=1.2, rng=rng)

    assert sampled == uniform_site_disorder(geometry, width=1.2, seed=29)


def test_multicomponent_normal_disorder_is_site_scalar_and_additive() -> None:
    geometry = chain(3)
    clean = np.asarray(
        [
            [2.0, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.1, 3.0, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.2, 4.0, 0.3, 0.0, 0.0],
            [0.0, 0.0, 0.3, 5.0, 0.4, 0.0],
            [0.0, 0.0, 0.0, 0.4, 6.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.5, 7.0],
        ]
    )
    offsets = uniform_site_disorder(geometry, width=0.6, seed=3)

    realization = apply_uniform_onsite_disorder(
        geometry,
        clean,
        width=0.6,
        seed=3,
        components_per_site=2,
    )

    assert isinstance(realization.state, np.ndarray)
    disordered = np.asarray(realization.state, dtype=np.float64)
    difference = disordered - clean
    expected_diagonal = np.repeat(np.asarray(tuple(offsets.values())), 2)
    assert np.allclose(np.diag(difference), expected_diagonal, rtol=0.0, atol=1e-15)
    assert np.count_nonzero(difference - np.diag(np.diag(difference))) == 0
    assert np.array_equal(clean, clean.conj().T)


@pytest.mark.parametrize("ordering", ["component_major", "site_major"])
def test_bdg_onsite_disorder_uses_opposite_particle_hole_signs(
    ordering: str,
) -> None:
    geometry = chain(3)
    basis = NambuBasis(
        n_sites=3,
        normal_components_per_site=2,
        ordering=ordering,  # type: ignore[arg-type]
    )
    clean = np.zeros((basis.dimension, basis.dimension), dtype=np.complex128)
    offsets = uniform_site_disorder(geometry, width=0.4, seed=41)

    realization = apply_uniform_onsite_disorder(
        geometry,
        clean,
        width=0.4,
        seed=41,
        nambu_basis=basis,
    )

    assert isinstance(realization.state, np.ndarray)
    disordered = np.asarray(realization.state, dtype=np.complex128)
    for site, offset in offsets.items():
        for component in range(2):
            particle = basis.particle_index(site, component=component)
            hole = basis.hole_index(site, component=component)
            assert disordered[particle, particle] == offset
            assert disordered[hole, hole] == -offset
    particle_hole = basis.particle_hole_operator
    transformed = particle_hole @ disordered.conj() @ particle_hole.conj().T
    assert np.allclose(transformed, -disordered)
    assert realization.provenance.parameters["basis_ordering"] == ordering
    assert realization.provenance.parameters["particle_hole_embedding"] == (
        "opposite_sign"
    )


def test_zero_width_preserves_exact_matrix_snapshot() -> None:
    geometry = chain(2)
    clean = np.asarray([[1, 2], [2, 3]], dtype=np.int64)

    realization = apply_uniform_onsite_disorder(
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
        apply_uniform_onsite_disorder(
            chain(2),
            np.eye(2),
            width=width,
            seed=1,
        )


def test_normal_matrix_shape_must_match_geometry_and_components() -> None:
    with pytest.raises(ValueError, match="shape does not match"):
        apply_uniform_onsite_disorder(
            chain(3),
            np.eye(4),
            width=0.2,
            seed=1,
        )


def test_nambu_basis_must_match_geometry_and_excludes_component_override() -> None:
    geometry = chain(3)
    mismatched = NambuBasis(n_sites=2)
    with pytest.raises(ValueError, match="site count must match"):
        apply_uniform_onsite_disorder(
            geometry,
            np.eye(mismatched.dimension),
            width=0.2,
            seed=1,
            nambu_basis=mismatched,
        )

    basis = NambuBasis(n_sites=3)
    with pytest.raises(ValueError, match="must be omitted"):
        apply_uniform_onsite_disorder(
            geometry,
            np.eye(basis.dimension),
            width=0.2,
            seed=1,
            components_per_site=1,
            nambu_basis=basis,
        )
