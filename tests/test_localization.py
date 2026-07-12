import numpy as np
import pytest

from toposc_lab.observables.localization import (
    bulk_weight,
    edge_weight,
    is_edge_localized,
    left_edge_weight,
    localization_profile,
    right_edge_weight,
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
