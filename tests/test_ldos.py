import numpy as np
import pytest

from toposc_lab.core.results import BasisLayout, SimulationResult
from toposc_lab.observables.ldos import local_density_of_states_from_result


def test_ldos_is_largest_at_the_matching_site_and_energy() -> None:
    result = SimulationResult(
        model_name="TwoSiteTest",
        eigenvalues=np.array([-1.0, 1.0]),
        eigenvectors=np.eye(2),
        basis_layout=BasisLayout(spatial_shape=(2,)),
    )

    ldos = local_density_of_states_from_result(
        result,
        energy_values=np.array([-1.0, 1.0]),
        broadening=0.05,
    )

    assert ldos.values.shape == (2, 2)
    assert ldos.values[0, 0] > ldos.values[0, 1]
    assert ldos.values[1, 1] > ldos.values[1, 0]


def test_ldos_keeps_component_information() -> None:
    result = SimulationResult(
        model_name="TwoComponentTest",
        eigenvalues=np.array([0.0, 1.0]),
        eigenvectors=np.eye(2),
        basis_layout=BasisLayout(
            spatial_shape=(1,),
            components_per_site=2,
            component_labels=("A", "B"),
        ),
    )

    ldos = local_density_of_states_from_result(
        result,
        energy_values=np.array([0.0]),
        broadening=0.1,
    )

    assert ldos.values.shape == (1, 1)
    assert ldos.component_values.shape == (1, 1, 2)
    assert ldos.component_labels == ("A", "B")
    assert ldos.component_values[0, 0, 0] > ldos.component_values[0, 0, 1]


def test_ldos_rejects_non_positive_broadening() -> None:
    result = SimulationResult(
        model_name="Test",
        eigenvalues=np.array([0.0]),
        eigenvectors=np.array([[1.0]]),
        basis_layout=BasisLayout(spatial_shape=(1,)),
    )

    with pytest.raises(ValueError, match="broadening"):
        local_density_of_states_from_result(
            result,
            energy_values=np.array([0.0]),
            broadening=0.0,
        )
