"""Deterministic finite Cayley-tree geometry generator."""

from __future__ import annotations

from dataclasses import replace
from numbers import Integral

from toposc_lab.geometry.base import Geometry
from toposc_lab.geometry.generators._validation import validate_optional_budget
from toposc_lab.geometry.generators.tree import tree

DEFAULT_CAYLEY_TREE_MAX_SITES = 25_000


def cayley_tree(
    coordination: int,
    shells: int,
    *,
    max_sites: int | None = DEFAULT_CAYLEY_TREE_MAX_SITES,
) -> Geometry:
    """Create a finite rooted Cayley tree with Bethe coordination ``z``.

    The root has ``coordination`` children. Every nonroot site before the outer
    shell has ``coordination - 1`` children, giving it total graph degree
    ``coordination`` after including its parent. ``shells`` counts generations
    outside the root.
    """
    coordination = _validate_coordination(coordination)
    shells = _validate_shells(shells)
    max_sites = validate_optional_budget(max_sites, name="max_sites")
    shell_populations = _shell_populations(
        coordination=coordination,
        shells=shells,
        max_sites=max_sites,
    )

    parents: list[int] = []
    current_shell = [0]
    next_site = 1
    for depth in range(shells):
        children_per_site = coordination if depth == 0 else coordination - 1
        next_shell: list[int] = []
        for parent in current_shell:
            for _ in range(children_per_site):
                parents.append(parent)
                next_shell.append(next_site)
                next_site += 1
        current_shell = next_shell

    geometry = tree(parents)
    return replace(
        geometry,
        metadata={
            "generator": "cayley_tree",
            "family": "finite_bethe_like_tree",
            "coordination": coordination,
            "shells": shells,
            "shell_populations": shell_populations,
            "root_site": 0,
            "root_degree": coordination if shells > 0 else 0,
            "leaf_definition": "outer_shell",
            "construction_order": "breadth_first_shells",
            "max_sites": max_sites,
        },
    )


def _validate_coordination(coordination: int) -> int:
    if isinstance(coordination, bool) or not isinstance(coordination, Integral):
        raise TypeError("coordination must be an integer")
    result = int(coordination)
    if result < 2:
        raise ValueError("coordination must be at least two")
    return result


def _validate_shells(shells: int) -> int:
    if isinstance(shells, bool) or not isinstance(shells, Integral):
        raise TypeError("shells must be an integer")
    result = int(shells)
    if result < 0:
        raise ValueError("shells must be nonnegative")
    return result


def _shell_populations(
    *,
    coordination: int,
    shells: int,
    max_sites: int | None,
) -> tuple[int, ...]:
    populations = [1]
    total_sites = 1
    shell_population = 1
    for shell in range(1, shells + 1):
        shell_population = (
            coordination
            if shell == 1
            else shell_population * (coordination - 1)
        )
        total_sites += shell_population
        if max_sites is not None and total_sites > max_sites:
            raise ValueError(
                f"cayley_tree exceeds max_sites={max_sites} at shell {shell}"
            )
        populations.append(shell_population)
    return tuple(populations)
