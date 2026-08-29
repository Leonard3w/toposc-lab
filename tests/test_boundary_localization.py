import numpy as np
import pytest

from toposc_lab.core.results import BasisLayout, SimulationResult
from toposc_lab.geometry import Geometry, GeometryEdge, ring
from toposc_lab.observables.localization import (
    boundary_weight,
    boundary_weight_from_geometry,
    boundary_weight_from_result,
    is_boundary_localized,
    is_boundary_localized_from_geometry,
)


def test_boundary_weight_uses_explicit_site_indices() -> None:
    probability = np.array([0.4, 0.1, 0.2, 0.3])

    assert boundary_weight(probability, {0, 3}) == pytest.approx(0.7)
    assert is_boundary_localized(probability, {0, 3}, threshold=0.7)


def test_geometry_boundary_does_not_infer_boundary_from_low_degree() -> None:
    geometry = Geometry(
        n_sites=3,
        edges=(GeometryEdge(0, 1),),
        boundary_sites=frozenset({0}),
    )
    probability = np.array([0.0, 0.0, 1.0])

    assert geometry.degree(2) == 0
    assert boundary_weight_from_geometry(probability, geometry) == 0.0
    assert not is_boundary_localized_from_geometry(probability, geometry)


def test_boundaryless_periodic_geometry_has_zero_boundary_weight() -> None:
    geometry = ring(5)

    assert geometry.boundary_sites == frozenset()
    assert boundary_weight_from_geometry(np.ones(5), geometry) == 0.0
    assert not is_boundary_localized_from_geometry(np.ones(5), geometry)


def test_boundary_weight_from_result_maps_component_major_bdg_state() -> None:
    geometry = Geometry(n_sites=2, boundary_sites=frozenset({1}))
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

    assert boundary_weight_from_result(result, 0, geometry) == pytest.approx(1.0)


def test_geometry_boundary_rejects_site_count_mismatch() -> None:
    with pytest.raises(ValueError, match="one value per geometry site"):
        boundary_weight_from_geometry(
            np.ones(2),
            Geometry(n_sites=3, boundary_sites=frozenset({0})),
        )


@pytest.mark.parametrize("boundary_sites", [{-1}, {3}])
def test_boundary_weight_rejects_sites_outside_probability(
    boundary_sites: set[int],
) -> None:
    with pytest.raises(ValueError, match="outside"):
        boundary_weight(np.ones(3), boundary_sites)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf])
def test_boundary_localization_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        is_boundary_localized(np.ones(2), {0}, threshold=threshold)
