"""Deterministic one-dimensional silver-mean substitution-tiling generator."""

from __future__ import annotations

from math import sqrt

from toposc_lab.geometry.base import Geometry
from toposc_lab.geometry.generators._substitution_chain import (
    BinarySubstitutionChainSpec,
    build_binary_substitution_chain,
)

SILVER_MEAN_RATIO = 1.0 + sqrt(2.0)
DEFAULT_SILVER_MEAN_CHAIN_MAX_SITES = 100_000

_SILVER_MEAN_SPEC = BinarySubstitutionChainSpec(
    generator="silver_mean_chain",
    family="silver_mean_substitution_tiling",
    seed="S",
    long_replacement="LSL",
    short_replacement="L",
    long_edge_type="silver_mean_long",
    short_edge_type="silver_mean_short",
    self_similar_ratio=SILVER_MEAN_RATIO,
    self_similar_regime="silver_mean_self_similar",
    non_self_similar_regime="octonacci_symbolic_nonsilver_lengths",
)


def silver_mean_chain(
    order: int,
    *,
    spacing: float = 1.0,
    long_short_ratio: float = SILVER_MEAN_RATIO,
    max_sites: int | None = DEFAULT_SILVER_MEAN_CHAIN_MAX_SITES,
) -> Geometry:
    """Create an open silver-mean, or octonacci, geometric bond chain.

    Starting from ``S``, each substitution step simultaneously applies
    ``L -> LSL`` and ``S -> L``. ``spacing`` is the short-bond length and
    ``long_short_ratio * spacing`` is the long-bond length. The default silver
    mean yields exact inflation symmetry and palindromic finite approximants.
    """
    return build_binary_substitution_chain(
        order,
        spec=_SILVER_MEAN_SPEC,
        spacing=spacing,
        long_short_ratio=long_short_ratio,
        max_sites=max_sites,
        extra_metadata={
            "alternative_name": "octonacci_chain",
            "word_symmetry": "palindromic",
        },
    )
