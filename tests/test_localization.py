import numpy as np
import pytest

from toposc_lab.core.results import BasisLayout, SimulationResult
from toposc_lab.observables.localization import (
    bulk_weight,
    edge_weight,
    inverse_participation_ratio,
    is_edge_localized,
    left_edge_weight,
    localization_profile,
    participation_ratio,
    right_edge_weight,
    site_probability_density,
    site_probability_density_from_result,
)


def test_localization_profile_returns_site_probabilities() -> None:
    # Ein Zustand gleichmäßig auf den ersten zwei Plätzen.
    eigenvectors = np.array(
        [
            [1.0 / np.sqrt(2.0)],
            [1.0 / np.sqrt(2.0)],
            [0.0],
            [0.0],
        ]
    )

    result = localization_profile(
        eigenvectors=eigenvectors,
        state_index=0,
        lattice_shape=(4,),
    )

    assert np.allclose(result.probability, [0.5, 0.5, 0.0, 0.0])
    assert result.inverse_participation_ratio == pytest.approx(0.5)
    assert result.participation_ratio == pytest.approx(2.0)


def test_localization_profile_sums_internal_components_per_site() -> None:
    # Zwei Plätze mit jeweils zwei internen Komponenten.
    eigenvectors = np.array(
        [
            [1.0 / np.sqrt(2.0)],
            [0.0],
            [0.0],
            [1.0 / np.sqrt(2.0)],
        ]
    )

    result = localization_profile(
        eigenvectors=eigenvectors,
        state_index=0,
        lattice_shape=(2,),
        components_per_site=2,
        component_labels=("up", "down"),
    )

    assert np.allclose(result.probability, [0.5, 0.5])
    assert result.component_probabilities.shape == (2, 2)
    assert result.component_labels == ("up", "down")
    assert result.inverse_participation_ratio == pytest.approx(0.5)


def test_localization_profile_returns_2d_shape() -> None:
    eigenvectors = np.array([[1.0], [0.0], [0.0], [0.0]])

    result = localization_profile(
        eigenvectors=eigenvectors,
        state_index=0,
        lattice_shape=(2, 2),
    )

    assert result.probability.shape == (2, 2)
    assert result.edge_weight == pytest.approx(1.0)


def test_localization_profile_rejects_wrong_dimension() -> None:
    eigenvectors = np.array([[1.0], [0.0], [0.0]])

    with pytest.raises(ValueError):
        localization_profile(
            eigenvectors=eigenvectors,
            state_index=0,
            lattice_shape=(2,),
            components_per_site=2,
        )


def test_edge_weights_detect_left_and_right_boundary_weight() -> None:
    probability = np.array([0.4, 0.1, 0.1, 0.4])

    assert left_edge_weight(probability) == pytest.approx(0.4)
    assert right_edge_weight(probability) == pytest.approx(0.4)
    assert edge_weight(probability) == pytest.approx(0.8)
    assert bulk_weight(probability) == pytest.approx(0.2)
    assert is_edge_localized(probability)


def test_edge_weight_handles_a_two_dimensional_boundary() -> None:
    probability = np.zeros((5, 5))
    probability[0, 2] = 0.6
    probability[2, 2] = 0.4

    assert edge_weight(probability) == pytest.approx(0.6)
    assert bulk_weight(probability) == pytest.approx(0.4)
    assert is_edge_localized(probability, threshold=0.5)


def test_is_edge_localized_respects_custom_threshold() -> None:
    probability = np.array([0.2, 0.3, 0.1, 0.2, 0.2])

    assert is_edge_localized(probability, threshold=0.4)
    assert not is_edge_localized(probability, threshold=0.5)


def test_ipr_is_one_for_a_completely_localized_state() -> None:
    probability = np.array([1.0, 0.0, 0.0, 0.0])

    assert inverse_participation_ratio(probability) == pytest.approx(1.0)
    assert participation_ratio(probability) == pytest.approx(1.0)


def test_ipr_is_small_for_a_uniform_state() -> None:
    probability = np.ones(4)

    assert inverse_participation_ratio(probability) == pytest.approx(0.25)
    assert participation_ratio(probability) == pytest.approx(4.0)


def test_ipr_normalizes_the_input_probability() -> None:
    probability = np.array([2.0, 2.0])

    assert inverse_participation_ratio(probability) == pytest.approx(0.5)


def test_ipr_sums_internal_components_before_squaring() -> None:
    component_probabilities = np.full((2, 2), 0.25)

    assert inverse_participation_ratio(
        component_probabilities,
        component_axis=-1,
    ) == pytest.approx(0.5)
    assert participation_ratio(
        component_probabilities,
        component_axis=-1,
    ) == pytest.approx(2.0)


def test_ipr_does_not_treat_components_as_separate_sites() -> None:
    component_probabilities = np.array([[0.5, 0.5]])

    assert inverse_participation_ratio(
        component_probabilities,
        component_axis=1,
    ) == pytest.approx(1.0)


@pytest.mark.parametrize("component_axis", [-3, 2])
def test_ipr_rejects_component_axis_outside_dimensions(component_axis: int) -> None:
    with pytest.raises(ValueError, match="component_axis"):
        inverse_participation_ratio(
            np.ones((2, 2)),
            component_axis=component_axis,
        )


def test_site_probability_density_maps_component_major_bdg_basis() -> None:
    vectors = np.zeros((6, 1), dtype=complex)
    vectors[0, 0] = 1.0 / np.sqrt(2.0)  # electron at site 0
    vectors[3, 0] = 1.0 / np.sqrt(2.0)  # hole at site 0

    density = site_probability_density(
        eigenvectors=vectors,
        state_index=0,
        basis_layout=BasisLayout(
            spatial_shape=(3,),
            components_per_site=2,
            ordering="component_major",
            component_labels=("electron", "hole"),
        ),
    )

    assert np.allclose(density.probability, [1.0, 0.0, 0.0])
    assert np.allclose(density.component_probabilities[0], [0.5, 0.5])
    assert density.component_labels == ("electron", "hole")


def test_site_probability_density_normalizes_and_retains_grid_shape() -> None:
    density = site_probability_density(
        eigenvectors=np.array([[2.0], [0.0], [0.0], [0.0]]),
        state_index=0,
        basis_layout=BasisLayout(spatial_shape=(2, 2)),
    )

    assert density.probability.shape == (2, 2)
    assert density.component_probabilities.shape == (2, 2, 1)
    assert np.sum(density.probability) == pytest.approx(1.0)


def test_site_probability_density_from_result_uses_declared_layout() -> None:
    result = SimulationResult(
        model_name="BdGTest",
        eigenvalues=np.array([0.0]),
        eigenvectors=np.array([[0.0], [1.0], [0.0], [1.0]]),
        basis_layout=BasisLayout(
            spatial_shape=(2,),
            components_per_site=2,
            ordering="component_major",
            component_labels=("electron", "hole"),
        ),
    )

    density = site_probability_density_from_result(result, state_index=0)

    assert np.allclose(density.probability, [0.0, 1.0])


def test_site_probability_density_rejects_zero_norm_state() -> None:
    with pytest.raises(ValueError, match="positive norm"):
        site_probability_density(
            eigenvectors=np.zeros((2, 1)),
            state_index=0,
            basis_layout=BasisLayout(spatial_shape=(2,)),
        )
