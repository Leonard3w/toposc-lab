from __future__ import annotations

import pytest

from toposc_lab.geometry import Geometry, GeometryBoundaryComponent


def test_geometry_boundary_components_support_overlapping_sites() -> None:
    outer = GeometryBoundaryComponent("outer", 0, {0, 1, 2})  # type: ignore[arg-type]
    hole = GeometryBoundaryComponent("hole", 0, {2, 3})  # type: ignore[arg-type]
    geometry = Geometry(
        n_sites=4,
        boundary_sites=frozenset({0, 1, 2, 3}),
        boundary_components=(outer, hole),
    )

    assert geometry.boundary_components == (outer, hole)
    assert geometry.boundary_components[0].sites == frozenset({0, 1, 2})
    assert geometry.boundary_components[1].sites == frozenset({2, 3})
    assert geometry.boundary_sites == frozenset({0, 1, 2, 3})


def test_boundary_component_defensively_freezes_site_inputs() -> None:
    sites = {0, 1}
    component = GeometryBoundaryComponent("outer", 0, sites)  # type: ignore[arg-type]
    sites.add(2)

    assert component.sites == frozenset({0, 1})


@pytest.mark.parametrize("kind", ("", "edge", "boundary"))
def test_boundary_component_rejects_unknown_kind(kind: str) -> None:
    with pytest.raises(ValueError, match="unsupported geometry boundary kind"):
        GeometryBoundaryComponent(kind, 0, frozenset({0}))  # type: ignore[arg-type]


@pytest.mark.parametrize("component_index", (-1, -4))
def test_boundary_component_rejects_negative_index(component_index: int) -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        GeometryBoundaryComponent("hole", component_index, frozenset({0}))


@pytest.mark.parametrize("component_index", (True, 1.5, "1"))
def test_boundary_component_rejects_non_integer_index(
    component_index: object,
) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        GeometryBoundaryComponent(
            "hole",
            component_index,  # type: ignore[arg-type]
            frozenset({0}),
        )


def test_boundary_component_rejects_empty_site_set() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        GeometryBoundaryComponent("outer", 0, frozenset())


def test_geometry_rejects_boundary_component_outside_geometry() -> None:
    with pytest.raises(ValueError, match="outside the geometry"):
        Geometry(
            n_sites=2,
            boundary_sites=frozenset({0, 2}),
            boundary_components=(
                GeometryBoundaryComponent("outer", 0, frozenset({0, 2})),
            ),
        )


def test_geometry_requires_boundary_sites_to_match_component_union() -> None:
    with pytest.raises(ValueError, match="must equal the union"):
        Geometry(
            n_sites=3,
            boundary_sites=frozenset({0}),
            boundary_components=(
                GeometryBoundaryComponent("outer", 0, frozenset({0, 1})),
            ),
        )


def test_geometry_rejects_duplicate_boundary_component_identity() -> None:
    with pytest.raises(ValueError, match="unique kind/index pairs"):
        Geometry(
            n_sites=3,
            boundary_sites=frozenset({0, 1}),
            boundary_components=(
                GeometryBoundaryComponent("hole", 0, frozenset({0})),
                GeometryBoundaryComponent("hole", 0, frozenset({1})),
            ),
        )
