from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pytest

from toposc_lab.geometry import Geometry, GeometryEdge


def test_geometry_defensively_copies_mutable_constructor_inputs() -> None:
    edge_notes = ["original"]
    edge = GeometryEdge(0, 1, metadata={"notes": edge_notes})
    edges = [edge]
    coordinates = np.asarray(((0.0, 0.0), (1.0, 0.0)))
    boundary_sites = {0}
    site_types = ["A", "B"]
    labels = ["left", "right"]
    weights = np.asarray((1.0, 2.0))
    metadata: dict[str, Any] = {
        "nested": {"labels": labels},
        "weights": weights,
    }

    geometry = Geometry(
        n_sites=2,
        edges=edges,  # type: ignore[arg-type]
        coordinates=coordinates,
        boundary_sites=boundary_sites,  # type: ignore[arg-type]
        site_types=site_types,  # type: ignore[arg-type]
        metadata=metadata,
    )

    edge_notes.append("changed")
    edges.append(GeometryEdge(1, 0))
    coordinates[0, 0] = 99.0
    boundary_sites.add(1)
    site_types[0] = "changed"
    labels.append("changed")
    weights[0] = 99.0
    metadata["new"] = "changed"

    assert geometry.coordinates is not None
    assert geometry.n_edges == 1
    assert np.array_equal(geometry.coordinates, ((0.0, 0.0), (1.0, 0.0)))
    assert geometry.boundary_sites == frozenset({0})
    assert geometry.site_types == ("A", "B")
    assert geometry.edges[0].metadata["notes"] == ("original",)
    assert geometry.metadata["nested"]["labels"] == ("left", "right")
    assert np.array_equal(geometry.metadata["weights"], (1.0, 2.0))
    assert "new" not in geometry.metadata


def test_geometry_snapshot_cannot_be_modified_through_public_values() -> None:
    geometry = Geometry(
        n_sites=2,
        edges=(GeometryEdge(0, 1, metadata={"nested": {"value": 1}}),),
        coordinates=np.asarray(((0.0,), (1.0,))),
        metadata={"nested": {"values": [1, 2]}, "array": np.asarray((1.0, 2.0))},
    )

    with pytest.raises(FrozenInstanceError):
        geometry.n_sites = 3  # type: ignore[misc]

    assert geometry.coordinates is not None
    with pytest.raises(ValueError):
        geometry.coordinates[0, 0] = 5.0
    with pytest.raises(ValueError):
        geometry.coordinates.setflags(write=True)

    with pytest.raises(TypeError):
        cast(dict[str, Any], geometry.metadata)["new"] = 1
    nested = cast(MappingProxyType[str, Any], geometry.metadata["nested"])
    with pytest.raises(TypeError):
        cast(dict[str, Any], nested)["new"] = 1

    metadata_array = cast(np.ndarray, geometry.metadata["array"])
    with pytest.raises(ValueError):
        metadata_array[0] = 5.0
    with pytest.raises(ValueError):
        metadata_array.setflags(write=True)

    with pytest.raises(FrozenInstanceError):
        geometry.edges[0].source = 1  # type: ignore[misc]
    edge_nested = cast(MappingProxyType[str, Any], geometry.edges[0].metadata["nested"])
    with pytest.raises(TypeError):
        cast(dict[str, Any], edge_nested)["value"] = 2


def test_position_returns_an_independent_mutable_copy() -> None:
    geometry = Geometry(
        n_sites=2,
        coordinates=np.asarray(((0.0, 1.0), (2.0, 3.0))),
    )

    position = geometry.position(1)
    position[0] = 99.0

    assert np.array_equal(geometry.position(1), (2.0, 3.0))


def test_dataclass_replace_creates_a_new_validated_snapshot() -> None:
    original = Geometry(
        n_sites=3,
        edges=(GeometryEdge(0, 1),),
        coordinates=np.asarray(((0.0,), (1.0,), (2.0,))),
        metadata={"revision": 1},
    )
    updated = replace(
        original,
        edges=original.edges + (GeometryEdge(1, 2),),
        metadata={"revision": 2},
    )

    assert original.n_edges == 1
    assert original.metadata["revision"] == 1
    assert updated.n_edges == 2
    assert updated.metadata["revision"] == 2
    assert original.coordinates is not None
    assert updated.coordinates is not None
    assert updated.coordinates is not original.coordinates
    assert np.array_equal(updated.coordinates, original.coordinates)

    with pytest.raises(ValueError):
        updated.coordinates.setflags(write=True)
