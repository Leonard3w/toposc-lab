from __future__ import annotations

import pytest

from toposc_lab.geometry import Geometry, GeometryEdge, RootedTreeStructure


def test_rooted_tree_exposes_parent_child_depth_and_leaf_queries() -> None:
    structure = RootedTreeStructure(
        root_site=0,
        parents=(None, 0, 0, 1, 1, 3),
    )

    assert structure.n_sites == 6
    assert structure.root_site == 0
    assert structure.parent(0) is None
    assert structure.parent(5) == 3
    assert structure.children(0) == (1, 2)
    assert structure.children(1) == (3, 4)
    assert structure.children(2) == ()
    assert tuple(structure.depth(site) for site in range(6)) == (0, 1, 1, 2, 2, 3)
    assert structure.leaf_sites == frozenset({2, 4, 5})


def test_singleton_root_is_a_leaf() -> None:
    structure = RootedTreeStructure(root_site=0, parents=(None,))

    assert structure.leaf_sites == frozenset({0})
    assert structure.children(0) == ()
    assert structure.depth(0) == 0


def test_deep_parent_chain_uses_linear_memoized_depth_resolution() -> None:
    n_sites = 10_000
    structure = RootedTreeStructure(
        root_site=0,
        parents=(None, *(site - 1 for site in range(1, n_sites))),
    )

    assert structure.depth(n_sites - 1) == n_sites - 1
    assert structure.leaf_sites == frozenset({n_sites - 1})


def test_rooted_tree_supports_parent_labels_after_their_children() -> None:
    structure = RootedTreeStructure(
        root_site=2,
        parents=(2, 2, None, 1),
    )

    assert tuple(structure.depth(site) for site in range(4)) == (1, 1, 0, 2)
    assert structure.children(2) == (0, 1)
    assert structure.leaf_sites == frozenset({0, 3})


def test_rooted_tree_defensively_copies_parent_input() -> None:
    parents: list[int | None] = [None, 0]
    structure = RootedTreeStructure(0, parents)  # type: ignore[arg-type]
    parents[1] = None

    assert structure.parents == (None, 0)


@pytest.mark.parametrize(
    ("root_site", "parents", "message"),
    (
        (0, (), "at least one"),
        (2, (None, 0), "root site is outside"),
        (0, (1, 0), "root site must have no parent"),
        (0, (None, None), "only the root"),
        (0, (None, 1), "own parent"),
        (0, (None, 2), "parent is outside"),
        (0, (None, 2, 1), "contain a cycle"),
    ),
)
def test_rooted_tree_rejects_invalid_hierarchy(
    root_site: int,
    parents: tuple[int | None, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RootedTreeStructure(root_site, parents)


@pytest.mark.parametrize("root_site", (True, 1.5, "0"))
def test_rooted_tree_rejects_non_integer_root(root_site: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        RootedTreeStructure(
            root_site,  # type: ignore[arg-type]
            (None, 0),
        )


@pytest.mark.parametrize("parent", (True, 1.5, "0"))
def test_rooted_tree_rejects_non_integer_parent(parent: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        RootedTreeStructure(0, (None, parent))  # type: ignore[arg-type]


def test_rooted_tree_queries_validate_site() -> None:
    structure = RootedTreeStructure(0, (None, 0))

    with pytest.raises(ValueError, match="outside the rooted tree"):
        structure.parent(2)
    with pytest.raises(TypeError, match="must be an integer"):
        structure.depth(True)


def test_geometry_accepts_exact_tree_parent_edges() -> None:
    structure = RootedTreeStructure(0, (None, 0, 0, 1))
    geometry = Geometry(
        n_sites=4,
        edges=(GeometryEdge(0, 1), GeometryEdge(0, 2), GeometryEdge(1, 3)),
        rooted_tree=structure,
    )

    assert geometry.rooted_tree is structure


def test_geometry_rejects_rooted_tree_with_wrong_site_count() -> None:
    with pytest.raises(ValueError, match="one parent per geometry site"):
        Geometry(
            n_sites=3,
            rooted_tree=RootedTreeStructure(0, (None, 0)),
        )


def test_geometry_requires_exact_parent_edge_set_for_rooted_tree() -> None:
    structure = RootedTreeStructure(0, (None, 0, 0))

    with pytest.raises(ValueError, match="exactly match"):
        Geometry(
            n_sites=3,
            edges=(GeometryEdge(0, 1), GeometryEdge(1, 2)),
            rooted_tree=structure,
        )

    with pytest.raises(ValueError, match="exactly match"):
        Geometry(
            n_sites=3,
            edges=(
                GeometryEdge(0, 1),
                GeometryEdge(0, 2),
                GeometryEdge(1, 2),
            ),
            rooted_tree=structure,
        )
