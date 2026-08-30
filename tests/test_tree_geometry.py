from __future__ import annotations

import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    Geometry,
    RootedTreeStructure,
    tree,
)


def test_tree_builds_reference_parent_hierarchy() -> None:
    geometry = tree((0, 0, 1, 1, 3))

    assert isinstance(geometry, Geometry)
    assert geometry.n_sites == 6
    assert tuple((edge.source, edge.target) for edge in geometry.edges) == (
        (0, 1),
        (0, 2),
        (1, 3),
        (1, 4),
        (3, 5),
    )
    assert all(edge.edge_type == "tree_branch" for edge in geometry.edges)
    assert geometry.rooted_tree == RootedTreeStructure(
        root_site=0,
        parents=(None, 0, 0, 1, 1, 3),
    )
    assert geometry.boundary_sites == frozenset({2, 4, 5})
    assert geometry.site_types == (
        "root",
        "internal",
        "leaf",
        "internal",
        "leaf",
        "leaf",
    )


def test_tree_has_n_minus_one_edges_and_is_connected() -> None:
    geometry = tree((0, 0, 1, 1, 2, 2, 4, 6, 6))
    visited = {0}
    pending = [0]
    while pending:
        site = pending.pop()
        for neighbor in geometry.neighbors(site):
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)

    assert geometry.n_edges == geometry.n_sites - 1
    assert visited == set(geometry.site_indices)


def test_singleton_tree_marks_root_as_boundary_leaf() -> None:
    geometry = tree(())

    assert geometry.n_sites == 1
    assert geometry.n_edges == 0
    assert geometry.boundary_sites == frozenset({0})
    assert geometry.site_types == ("root",)
    assert geometry.rooted_tree is not None
    assert geometry.rooted_tree.leaf_sites == frozenset({0})
    assert geometry.metadata["max_depth"] == 0


def test_path_root_is_not_a_leaf_despite_graph_degree_one() -> None:
    geometry = tree((0, 1, 2, 3))

    assert geometry.boundary_sites == frozenset({4})
    assert geometry.degree(0) == 1
    assert 0 not in geometry.boundary_sites
    assert geometry.site_types == (
        "root",
        "internal",
        "internal",
        "internal",
        "leaf",
    )


def test_star_tree_marks_every_nonroot_site_as_leaf() -> None:
    geometry = tree((0, 0, 0, 0, 0))

    assert geometry.boundary_sites == frozenset({1, 2, 3, 4, 5})
    assert geometry.degree(0) == 5
    assert geometry.metadata["max_depth"] == 1


def test_tree_defensively_copies_parent_sequence() -> None:
    parents = [0, 0, 1]
    geometry = tree(parents)
    parents[2] = 2

    assert geometry.rooted_tree is not None
    assert geometry.rooted_tree.parents == (None, 0, 0, 1)


def test_tree_generator_is_deterministic() -> None:
    first = tree((0, 0, 1, 1, 3))
    second = tree((0, 0, 1, 1, 3))

    assert first.edges == second.edges
    assert first.boundary_sites == second.boundary_sites
    assert first.site_types == second.site_types
    assert first.rooted_tree == second.rooted_tree
    assert first.metadata == second.metadata


def test_abstract_tree_has_no_invented_spatial_or_dimension_data() -> None:
    geometry = tree((0, 0, 1))

    assert geometry.coordinates is None
    assert geometry.embedding_dimension is None
    assert geometry.boundary_components == ()
    assert geometry.dimension_records == ()


def test_tree_records_hierarchy_metadata_without_duplicating_parents() -> None:
    geometry = tree((0, 0, 1, 1, 3))

    assert geometry.metadata == {
        "generator": "tree",
        "root_site": 0,
        "construction_order": "parent_before_child",
        "leaf_definition": "no_children",
        "max_depth": 3,
    }
    assert "parents" not in geometry.metadata


def test_tree_is_available_through_common_registry() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "tree",
        parameters={"parents": [0, 0, 1, 1, 3]},
    )

    assert geometry.rooted_tree is not None
    assert geometry.rooted_tree.parents == (None, 0, 0, 1, 1, 3)
    assert geometry.metadata["generation"] == {
        "generator_key": "tree",
        "generator_version": 1,
        "parameters": {"parents": (0, 0, 1, 1, 3)},
        "seed": None,
    }


@pytest.mark.parametrize(
    "parents",
    ((1,), (-1,), (0, 2), (0, 0, 3)),
)
def test_tree_rejects_parent_that_is_not_an_earlier_site(
    parents: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="must reference an earlier site"):
        tree(parents)


@pytest.mark.parametrize("parents", ((True,), (0, 1.5), (0, "1")))
def test_tree_rejects_non_integer_parent(
    parents: tuple[object, ...],
) -> None:
    with pytest.raises(TypeError, match="contain only integers"):
        tree(parents)  # type: ignore[arg-type]


@pytest.mark.parametrize("parents", (None, 4, "001", b"001"))
def test_tree_rejects_non_sequence_parent_input(parents: object) -> None:
    with pytest.raises(TypeError, match="sequence of integers"):
        tree(parents)  # type: ignore[arg-type]
