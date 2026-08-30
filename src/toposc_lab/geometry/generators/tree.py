"""Deterministic general rooted-tree geometry generator."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

from toposc_lab.geometry.base import Geometry, GeometryEdge, RootedTreeStructure


def tree(parents: Sequence[int]) -> Geometry:
    """Create a rooted tree from a canonical parent-before-child sequence.

    Site zero is the root. ``parents[i - 1]`` is the parent of site ``i`` and
    must refer to an earlier site. The empty sequence creates a singleton tree.
    """
    if isinstance(parents, (str, bytes)):
        raise TypeError("parents must be a sequence of integers")
    try:
        parent_inputs = tuple(parents)
    except TypeError as error:
        raise TypeError("parents must be a sequence of integers") from error

    normalized_parents: list[int] = []
    for child_site, parent in enumerate(parent_inputs, start=1):
        if isinstance(parent, bool) or not isinstance(parent, Integral):
            raise TypeError("parents must contain only integers")
        normalized_parent = int(parent)
        if not 0 <= normalized_parent < child_site:
            raise ValueError(
                f"parent for site {child_site} must reference an earlier site"
            )
        normalized_parents.append(normalized_parent)

    rooted_tree = RootedTreeStructure(
        root_site=0,
        parents=(None, *normalized_parents),
    )
    boundary_sites = rooted_tree.leaf_sites
    return Geometry(
        n_sites=rooted_tree.n_sites,
        edges=tuple(
            GeometryEdge(parent, child, edge_type="tree_branch")
            for child, parent in enumerate(rooted_tree.parents)
            if parent is not None
        ),
        boundary_sites=boundary_sites,
        site_types=tuple(
            "root"
            if site == rooted_tree.root_site
            else "leaf"
            if site in boundary_sites
            else "internal"
            for site in range(rooted_tree.n_sites)
        ),
        rooted_tree=rooted_tree,
        metadata={
            "generator": "tree",
            "root_site": rooted_tree.root_site,
            "construction_order": "parent_before_child",
            "leaf_definition": "no_children",
            "max_depth": max(
                rooted_tree.depth(site) for site in range(rooted_tree.n_sites)
            ),
        },
    )
