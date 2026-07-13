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
