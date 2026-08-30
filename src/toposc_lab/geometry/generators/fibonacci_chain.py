"""Deterministic one-dimensional Fibonacci substitution-tiling generator."""

from __future__ import annotations

from numbers import Real

import numpy as np

from toposc_lab.geometry.base import Geometry, GeometryDimension, GeometryEdge
from toposc_lab.geometry.generators._validation import (
    validate_optional_budget,
    validate_recursion_order,
    validate_spacing,
)

FIBONACCI_GOLDEN_RATIO = (1.0 + np.sqrt(5.0)) / 2.0
DEFAULT_FIBONACCI_CHAIN_MAX_SITES = 100_000


def fibonacci_chain(
    order: int,
    *,
    spacing: float = 1.0,
    long_short_ratio: float = FIBONACCI_GOLDEN_RATIO,
    max_sites: int | None = DEFAULT_FIBONACCI_CHAIN_MAX_SITES,
) -> Geometry:
    """Create an open Fibonacci chain of long and short geometric bonds.

    Starting from ``L``, each substitution step applies ``L -> LS`` and
    ``S -> L`` simultaneously. ``spacing`` is the short-bond length and
    ``long_short_ratio * spacing`` is the long-bond length. The default golden
    ratio yields the exactly self-similar inflation geometry.
    """
    order = validate_recursion_order(order)
    spacing = validate_spacing(spacing)
    long_short_ratio = _validate_long_short_ratio(long_short_ratio)
    max_sites = validate_optional_budget(max_sites, name="max_sites")
    n_long_bonds, n_short_bonds = _fibonacci_bond_counts(
        order=order,
        max_sites=max_sites,
    )

    word = "L"
    for _ in range(order):
        word = "".join("LS" if bond == "L" else "L" for bond in word)

    short_length = spacing
    long_length = spacing * long_short_ratio
    bond_lengths = np.fromiter(
        (long_length if bond == "L" else short_length for bond in word),
        dtype=float,
        count=len(word),
    )
    coordinates = np.concatenate(
        (np.asarray([0.0]), np.cumsum(bond_lengths))
    ).reshape(-1, 1)
    edges = tuple(
        GeometryEdge(
            site,
            site + 1,
            edge_type=(
                "fibonacci_long" if bond == "L" else "fibonacci_short"
            ),
            displacement=(float(bond_lengths[site]),),
        )
        for site, bond in enumerate(word)
    )

    return Geometry(
        n_sites=len(word) + 1,
        edges=edges,
        coordinates=coordinates,
        boundary_sites=frozenset({0, len(word)}),
        dimension_records=(
            GeometryDimension(
                kind="topological",
                value=1.0,
                scope="infinite_family",
                method="covering_dimension",
                exact=True,
            ),
        ),
        metadata={
            "generator": "fibonacci_chain",
            "family": "fibonacci_substitution_tiling",
            "order": order,
            "substitution_seed": "L",
            "substitution_rule": {"L": "LS", "S": "L"},
            "boundary_condition": "open",
            "spacing": spacing,
            "long_short_ratio": long_short_ratio,
            "short_length": short_length,
            "long_length": long_length,
            "n_long_bonds": n_long_bonds,
            "n_short_bonds": n_short_bonds,
            "geometric_regime": (
                "golden_ratio_self_similar"
                if long_short_ratio == FIBONACCI_GOLDEN_RATIO
                else "fibonacci_symbolic_nongolden_lengths"
            ),
            "max_sites": max_sites,
        },
    )


def _validate_long_short_ratio(long_short_ratio: float) -> float:
    if isinstance(long_short_ratio, bool) or not isinstance(long_short_ratio, Real):
        raise TypeError("long_short_ratio must be a real number")
    result = float(long_short_ratio)
    if not np.isfinite(result) or result <= 1.0:
        raise ValueError("long_short_ratio must be finite and greater than one")
    return result


def _fibonacci_bond_counts(
    *,
    order: int,
    max_sites: int | None,
) -> tuple[int, int]:
    n_long_bonds = 1
    n_short_bonds = 0
    if max_sites is not None and 2 > max_sites:
        raise ValueError(
            f"fibonacci_chain order {order} requires 2 sites, "
            f"exceeding max_sites={max_sites}"
        )
    for step in range(1, order + 1):
        n_long_bonds, n_short_bonds = (
            n_long_bonds + n_short_bonds,
            n_long_bonds,
        )
        n_sites = n_long_bonds + n_short_bonds + 1
        if max_sites is not None and n_sites > max_sites:
            raise ValueError(
                f"fibonacci_chain order {order} requires at least {n_sites} "
                f"sites at substitution step {step}, exceeding "
                f"max_sites={max_sites}"
            )
    return n_long_bonds, n_short_bonds
