from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from toposc_lab.geometry import Geometry, GeometryEdge, GeometryFace


def _square_edges() -> tuple[GeometryEdge, ...]:
    return (
        GeometryEdge(0, 1),
        GeometryEdge(1, 2),
        GeometryEdge(2, 3),
        GeometryEdge(3, 0),
    )


def test_geometry_stores_an_ordered_typed_face() -> None:
    face = GeometryFace(
        (0, 1, 2, 3),
        face_type="square",
        metadata={"orientation": 1},
    )
    geometry = Geometry(n_sites=4, edges=_square_edges(), faces=(face,))

    assert geometry.n_faces == 1
    assert geometry.faces == (face,)
    assert geometry.faces[0].sites == (0, 1, 2, 3)
    assert geometry.faces[0].face_type == "square"
    assert geometry.faces[0].metadata["orientation"] == 1


def test_geometry_face_defensively_freezes_sites_and_metadata() -> None:
    sites = [0, 1, 2]
    labels = ["a", "b"]
    face = GeometryFace(
        sites,  # type: ignore[arg-type]
        metadata={"labels": labels},
    )

    sites.append(3)
    labels.append("c")

    assert face.sites == (0, 1, 2)
    assert face.metadata["labels"] == ("a", "b")
    with pytest.raises(FrozenInstanceError):
        face.sites = (0, 1, 2, 3)  # type: ignore[misc]
    with pytest.raises(TypeError):
        cast(dict[str, Any], face.metadata)["new"] = True


@pytest.mark.parametrize("sites", ((), (0,), (0, 1)))
def test_geometry_face_requires_at_least_three_sites(sites: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="at least three"):
        GeometryFace(sites)


def test_geometry_face_rejects_repeated_sites() -> None:
    with pytest.raises(ValueError, match="cannot repeat"):
        GeometryFace((0, 1, 2, 1))


@pytest.mark.parametrize("sites", ((True, 1, 2), (0, 1.5, 2), (0, "1", 2)))
def test_geometry_face_rejects_non_integer_sites(sites: tuple[object, ...]) -> None:
    with pytest.raises(TypeError, match="face site must be an integer"):
        GeometryFace(sites)  # type: ignore[arg-type]


def test_geometry_face_rejects_non_string_type() -> None:
    with pytest.raises(TypeError, match="face_type must be a string or None"):
        GeometryFace((0, 1, 2), face_type=3)  # type: ignore[arg-type]


def test_geometry_rejects_face_site_outside_geometry() -> None:
    with pytest.raises(ValueError, match="outside the geometry"):
        Geometry(
            n_sites=4,
            edges=_square_edges(),
            faces=(GeometryFace((0, 1, 2, 4)),),
        )


def test_geometry_rejects_face_with_missing_boundary_edge() -> None:
    with pytest.raises(ValueError, match="boundary must consist"):
        Geometry(
            n_sites=4,
            edges=_square_edges(),
            faces=(GeometryFace((0, 1, 3)),),
        )


def test_geometry_rejects_duplicate_face_with_different_orientation() -> None:
    with pytest.raises(ValueError, match="duplicate geometry face"):
        Geometry(
            n_sites=4,
            edges=_square_edges(),
            faces=(
                GeometryFace((0, 1, 2, 3)),
                GeometryFace((3, 2, 1, 0)),
            ),
        )


def test_geometry_rejects_non_face_entries() -> None:
    with pytest.raises(TypeError, match="GeometryFace"):
        Geometry(n_sites=1, faces=("face",))  # type: ignore[arg-type]
