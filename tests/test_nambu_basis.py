from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.hamiltonians import NambuBasis, NambuState


def test_spinless_component_major_basis_matches_legacy_kitaev_order() -> None:
    basis = NambuBasis(n_sites=3)

    assert basis.normal_dimension == 3
    assert basis.nambu_components_per_site == 2
    assert basis.dimension == 6
    assert basis.particle_indices == (0, 1, 2)
    assert basis.hole_indices == (3, 4, 5)
    assert basis.component_labels == ("electron", "hole")
    assert basis.basis_layout.ordering == "component_major"
    assert basis.basis_layout.component_labels == ("electron", "hole")


def test_spinless_site_major_basis_interleaves_particle_and_hole_states() -> None:
    basis = NambuBasis(n_sites=3, ordering="site_major")

    assert basis.particle_indices == (0, 2, 4)
    assert basis.hole_indices == (1, 3, 5)
    assert basis.particle_index(1) == 2
    assert basis.hole_index(1) == 3


def test_spinful_component_major_indices_are_unambiguous() -> None:
    basis = NambuBasis(
        n_sites=2,
        normal_components_per_site=2,
        normal_component_labels=("up", "down"),
    )

    assert basis.particle_indices == (0, 2, 1, 3)
    assert basis.hole_indices == (4, 6, 5, 7)
    assert basis.particle_index(1, component=0) == 1
    assert basis.particle_index(0, component=1) == 2
    assert basis.hole_index(1, component=1) == 7
    assert basis.component_labels == (
        "particle up",
        "particle down",
        "hole up",
        "hole down",
    )


@pytest.mark.parametrize("ordering", ["component_major", "site_major"])
def test_decode_is_inverse_of_index(ordering: str) -> None:
    basis = NambuBasis(
        n_sites=3,
        normal_components_per_site=2,
        ordering=ordering,  # type: ignore[arg-type]
    )

    for index in range(basis.dimension):
        state = basis.decode(index)
        assert basis.index(
            state.site,
            component=state.component,
            sector=state.sector,
        ) == index


def test_partner_index_exchanges_particle_and_hole_sectors() -> None:
    basis = NambuBasis(
        n_sites=2,
        normal_components_per_site=2,
        ordering="site_major",
    )

    particle = basis.particle_index(1, component=1)
    hole = basis.hole_index(1, component=1)

    assert basis.partner_index(particle) == hole
    assert basis.partner_index(hole) == particle
    assert basis.decode(hole) == NambuState(site=1, component=1, sector="hole")


def test_permutation_reorders_states_and_matrices_to_site_major() -> None:
    basis = NambuBasis(n_sites=2, normal_components_per_site=2)
    states = np.arange(8)
    matrix = np.arange(64).reshape(8, 8)
    expected_permutation = np.asarray([0, 2, 4, 6, 1, 3, 5, 7])

    assert np.array_equal(basis.permutation_to("site_major"), expected_permutation)
    assert np.array_equal(
        basis.reorder_states(states, ordering="site_major"),
        states[expected_permutation],
    )
    assert np.array_equal(
        basis.reorder_matrix(matrix, ordering="site_major"),
        matrix[np.ix_(expected_permutation, expected_permutation)],
    )


def test_basis_layout_and_nambu_reordering_agree() -> None:
    basis = NambuBasis(
        n_sites=2,
        normal_components_per_site=2,
        normal_component_labels=("up", "down"),
    )
    eigenvectors = np.arange(16).reshape(8, 2)

    assert np.array_equal(
        basis.basis_layout.to_site_major(eigenvectors),
        basis.reorder_states(eigenvectors, ordering="site_major"),
    )


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"n_sites": 0}, ValueError, "n_sites must be at least one"),
        (
            {"n_sites": 2, "normal_components_per_site": 0},
            ValueError,
            "normal_components_per_site must be at least one",
        ),
        (
            {"n_sites": 2, "normal_components_per_site": 1.5},
            TypeError,
            "normal_components_per_site must be an integer",
        ),
        (
            {
                "n_sites": 2,
                "normal_components_per_site": 2,
                "normal_component_labels": ("up",),
            },
            ValueError,
            "must match",
        ),
        (
            {
                "n_sites": 2,
                "normal_components_per_site": 2,
                "normal_component_labels": ("spin", "spin"),
            },
            ValueError,
            "must be unique",
        ),
    ],
)
def test_nambu_basis_rejects_invalid_definitions(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        NambuBasis(**kwargs)  # type: ignore[arg-type]


def test_nambu_basis_rejects_invalid_state_queries() -> None:
    basis = NambuBasis(n_sites=2)

    with pytest.raises(ValueError, match="outside the Nambu basis"):
        basis.particle_index(2)

    with pytest.raises(ValueError, match="outside the Nambu basis"):
        basis.particle_index(0, component=1)

    with pytest.raises(ValueError, match="sector must be either"):
        basis.index(0, sector="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="outside the Nambu basis"):
        basis.decode(4)
