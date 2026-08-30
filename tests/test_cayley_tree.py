from __future__ import annotations

import pytest

from toposc_lab.geometry import (
    BUILTIN_GEOMETRY_GENERATORS,
    DEFAULT_CAYLEY_TREE_MAX_SITES,
    Geometry,
    RootedTreeStructure,
    cayley_tree,
)


def test_cayley_tree_has_exact_reference_shells() -> None:
    geometry = cayley_tree(3, 3)

    assert isinstance(geometry, Geometry)
    assert geometry.n_sites == 22
    assert geometry.n_edges == 21
    assert geometry.metadata["shell_populations"] == (1, 3, 6, 12)
    assert geometry.rooted_tree is not None
    assert tuple(
        sum(
            geometry.rooted_tree.depth(site) == depth
            for site in geometry.site_indices
        )
        for depth in range(4)
    ) == (1, 3, 6, 12)


@pytest.mark.parametrize(
    ("coordination", "shells", "expected_sites"),
    ((2, 4, 9), (3, 4, 46), (4, 3, 53), (5, 2, 26)),
)
def test_cayley_site_count_follows_coordination_formula(
    coordination: int,
    shells: int,
    expected_sites: int,
) -> None:
    geometry = cayley_tree(coordination, shells)

    assert geometry.n_sites == expected_sites
    assert geometry.n_edges == expected_sites - 1


def test_root_and_internal_sites_have_requested_coordination() -> None:
    coordination = 4
    shells = 4
    geometry = cayley_tree(coordination, shells)
    assert geometry.rooted_tree is not None

    for site in geometry.site_indices:
        depth = geometry.rooted_tree.depth(site)
        expected_degree = 1 if depth == shells else coordination
        assert geometry.degree(site) == expected_degree


def test_outer_shell_is_exactly_the_boundary_leaf_set() -> None:
    coordination = 3
    shells = 4
    geometry = cayley_tree(coordination, shells)
    assert geometry.rooted_tree is not None
    outer_shell = frozenset(
        site
        for site in geometry.site_indices
        if geometry.rooted_tree.depth(site) == shells
    )

    assert geometry.boundary_sites == outer_shell
    assert geometry.rooted_tree.leaf_sites == outer_shell
    assert len(outer_shell) == coordination * (coordination - 1) ** (shells - 1)
    assert geometry.site_types is not None
    assert all(geometry.site_types[site] == "leaf" for site in outer_shell)


def test_coordination_two_is_a_centered_path_with_two_arms() -> None:
    geometry = cayley_tree(2, 4)

    assert geometry.n_sites == 9
    assert geometry.n_edges == 8
    assert geometry.boundary_sites == frozenset({7, 8})
    assert [geometry.degree(site) for site in geometry.site_indices] == [
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        1,
        1,
    ]
    assert geometry.rooted_tree == RootedTreeStructure(
        root_site=0,
        parents=(None, 0, 0, 1, 2, 3, 4, 5, 6),
    )


def test_zero_shells_is_a_singleton_boundary_root() -> None:
    geometry = cayley_tree(7, 0)

    assert geometry.n_sites == 1
    assert geometry.n_edges == 0
    assert geometry.boundary_sites == frozenset({0})
    assert geometry.site_types == ("root",)
    assert geometry.metadata["shell_populations"] == (1,)
    assert geometry.metadata["root_degree"] == 0


def test_cayley_uses_breadth_first_parent_order() -> None:
    geometry = cayley_tree(3, 2)

    assert geometry.rooted_tree is not None
    assert geometry.rooted_tree.parents == (
        None,
        0,
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    )
    assert tuple((edge.source, edge.target) for edge in geometry.edges) == tuple(
        (parent, child)
        for child, parent in enumerate(geometry.rooted_tree.parents)
        if parent is not None
    )
    assert all(edge.edge_type == "tree_branch" for edge in geometry.edges)


def test_cayley_generator_is_deterministic() -> None:
    first = cayley_tree(4, 3)
    second = cayley_tree(4, 3)

    assert first.edges == second.edges
    assert first.boundary_sites == second.boundary_sites
    assert first.site_types == second.site_types
    assert first.rooted_tree == second.rooted_tree
    assert first.metadata == second.metadata


def test_abstract_cayley_tree_has_no_invented_geometry_data() -> None:
    geometry = cayley_tree(3, 3)

    assert geometry.coordinates is None
    assert geometry.embedding_dimension is None
    assert geometry.boundary_components == ()
    assert geometry.dimension_records == ()


def test_cayley_tree_records_coordination_and_shell_metadata() -> None:
    geometry = cayley_tree(3, 3)

    assert geometry.metadata == {
        "generator": "cayley_tree",
        "family": "finite_bethe_like_tree",
        "coordination": 3,
        "shells": 3,
        "shell_populations": (1, 3, 6, 12),
        "root_site": 0,
        "root_degree": 3,
        "leaf_definition": "outer_shell",
        "construction_order": "breadth_first_shells",
        "max_sites": DEFAULT_CAYLEY_TREE_MAX_SITES,
    }


def test_cayley_tree_is_available_through_common_registry() -> None:
    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "cayley_tree",
        parameters={"coordination": 3, "shells": 3},
    )

    assert geometry.n_sites == 22
    assert geometry.metadata["generation"] == {
        "generator_key": "cayley_tree",
        "generator_version": 1,
        "parameters": {"coordination": 3, "shells": 3},
        "seed": None,
    }


def test_site_budget_accepts_exact_count_and_rejects_next_site() -> None:
    assert cayley_tree(3, 3, max_sites=22).n_sites == 22
    with pytest.raises(ValueError, match=r"exceeds max_sites=21 at shell 3"):
        cayley_tree(3, 3, max_sites=21)


def test_default_budget_stops_exponential_growth() -> None:
    with pytest.raises(ValueError, match=r"max_sites=25000 at shell 14"):
        cayley_tree(3, 14)


def test_site_budget_can_be_explicitly_disabled() -> None:
    geometry = cayley_tree(3, 3, max_sites=None)

    assert geometry.n_sites == 22
    assert geometry.metadata["max_sites"] is None


@pytest.mark.parametrize("coordination", (1, 0, -1, -10))
def test_cayley_rejects_coordination_below_two(coordination: int) -> None:
    with pytest.raises(ValueError, match="must be at least two"):
        cayley_tree(coordination, 2)


@pytest.mark.parametrize("coordination", (True, 2.5, "3"))
def test_cayley_rejects_non_integer_coordination(coordination: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        cayley_tree(coordination, 2)  # type: ignore[arg-type]


@pytest.mark.parametrize("shells", (-1, -5))
def test_cayley_rejects_negative_shell_count(shells: int) -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        cayley_tree(3, shells)


@pytest.mark.parametrize("shells", (True, 2.5, "3"))
def test_cayley_rejects_non_integer_shell_count(shells: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        cayley_tree(3, shells)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_sites", (0, -1, -10))
def test_cayley_rejects_nonpositive_site_budget(max_sites: int) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        cayley_tree(3, 2, max_sites=max_sites)


@pytest.mark.parametrize("max_sites", (True, 1.5, "10"))
def test_cayley_rejects_non_integer_site_budget(max_sites: object) -> None:
    with pytest.raises(TypeError, match="must be an integer or None"):
        cayley_tree(3, 2, max_sites=max_sites)  # type: ignore[arg-type]
