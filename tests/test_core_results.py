import numpy as np
import pytest

from toposc_lab.core.results import BasisLayout, SimulationResult
from toposc_lab.observables.localization import localization_profile_from_result


def test_component_major_basis_is_converted_to_site_major() -> None:
    """Die Kitaev-artige Blockbasis wird platzweise umgeordnet."""
    layout = BasisLayout(
        spatial_shape=(3,),
        components_per_site=2,
        ordering="component_major",
        component_labels=("electron", "hole"),
    )

    vectors = np.array(
        [
            [1.0],
            [2.0],
            [3.0],
            [10.0],
            [20.0],
            [30.0],
        ]
    )

    assert np.allclose(
        layout.to_site_major(vectors),
        [
            [1.0],
            [10.0],
            [2.0],
            [20.0],
            [3.0],
            [30.0],
        ],
    )


def test_simulation_result_rejects_wrong_dimension() -> None:
    layout = BasisLayout(spatial_shape=(4,))

    with pytest.raises(ValueError):
        SimulationResult(
            model_name="Test",
            eigenvalues=np.array([0.0, 1.0]),
            eigenvectors=np.eye(2),
            basis_layout=layout,
        )


def test_simulation_result_standardizes_eigenvalue_storage() -> None:
    layout = BasisLayout(spatial_shape=(2,))
    source = np.asarray([-1.0, 2.0], dtype=np.float32)

    result = SimulationResult(
        model_name="Test",
        eigenvalues=source,
        eigenvectors=np.eye(2),
        basis_layout=layout,
    )
    source[0] = 99.0

    assert result.eigenvalues.dtype == np.dtype(float)
    assert result.eigenvalues.shape == (2,)
    assert np.array_equal(result.eigenvalues, [-1.0, 2.0])
    assert not result.eigenvalues.flags.writeable
    assert result.n_states == 2

    with pytest.raises(ValueError, match="read-only"):
        result.eigenvalues[0] = 0.0


@pytest.mark.parametrize(
    ("eigenvalues", "message"),
    [
        (np.zeros((1, 1)), "one-dimensional"),
        (np.asarray([np.nan]), "must be finite"),
        (np.asarray([np.inf]), "must be finite"),
    ],
)
def test_simulation_result_rejects_malformed_eigenvalues(
    eigenvalues: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SimulationResult(
            model_name="Test",
            eigenvalues=eigenvalues,
            eigenvectors=np.eye(eigenvalues.size),
            basis_layout=BasisLayout(spatial_shape=(eigenvalues.size,)),
        )


def test_kitaev_result_uses_its_basis_layout_for_localization() -> None:
    """Elektron links und Loch rechts müssen an beiden Rändern erscheinen."""
    layout = BasisLayout(
        spatial_shape=(4,),
        components_per_site=2,
        ordering="component_major",
        component_labels=("electron", "hole"),
    )

    vectors = np.zeros((8, 1), dtype=complex)
    vectors[0, 0] = 1.0 / np.sqrt(2.0)
    vectors[7, 0] = 1.0 / np.sqrt(2.0)

    result = SimulationResult(
        model_name="KitaevChain",
        eigenvalues=np.array([0.0]),
        eigenvectors=vectors,
        basis_layout=layout,
    )

    profile = localization_profile_from_result(result, state_index=0)

    assert np.allclose(profile.probability, [0.5, 0.0, 0.0, 0.5])
    assert profile.is_edge_localized
