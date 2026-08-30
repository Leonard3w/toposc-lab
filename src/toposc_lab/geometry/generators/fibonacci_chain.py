"""Deterministic one-dimensional Fibonacci substitution-tiling generator."""

from __future__ import annotations

from math import sqrt

from toposc_lab.geometry.base import Geometry
from toposc_lab.geometry.generators._substitution_chain import (
    BinarySubstitutionChainSpec,
    build_binary_substitution_chain,
)

FIBONACCI_GOLDEN_RATIO = (1.0 + sqrt(5.0)) / 2.0
DEFAULT_FIBONACCI_CHAIN_MAX_SITES = 100_000

_FIBONACCI_SPEC = BinarySubstitutionChainSpec(
    generator="fibonacci_chain",
    family="fibonacci_substitution_tiling",
    seed="L",
    long_replacement="LS",
    short_replacement="L",
    long_edge_type="fibonacci_long",
    short_edge_type="fibonacci_short",
    self_similar_ratio=FIBONACCI_GOLDEN_RATIO,
    self_similar_regime="golden_ratio_self_similar",
    non_self_similar_regime="fibonacci_symbolic_nongolden_lengths",
)


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
    return build_binary_substitution_chain(
        order,
        spec=_FIBONACCI_SPEC,
        spacing=spacing,
        long_short_ratio=long_short_ratio,
        max_sites=max_sites,
    )
