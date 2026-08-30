"""Private construction engine for geometric two-symbol substitution chains."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np

from toposc_lab.geometry.base import Geometry, GeometryDimension, GeometryEdge
from toposc_lab.geometry.generators._validation import (
    validate_optional_budget,
    validate_recursion_order,
    validate_spacing,
)


@dataclass(frozen=True, slots=True)
class BinarySubstitutionChainSpec:
    """Internal immutable definition of a long/short substitution family."""

    generator: str
    family: str
    seed: str
    long_replacement: str
    short_replacement: str
    long_edge_type: str
    short_edge_type: str
    self_similar_ratio: float
    self_similar_regime: str
    non_self_similar_regime: str

    def __post_init__(self) -> None:
        words = (self.seed, self.long_replacement, self.short_replacement)
        if any(not word or set(word) - {"L", "S"} for word in words):
            raise ValueError("binary substitution words must contain only L and S")
        if self.long_edge_type == self.short_edge_type:
            raise ValueError("long and short edge types must differ")

    @property
    def substitution_rule(self) -> dict[str, str]:
        """Return the simultaneous substitution rule."""
        return {"L": self.long_replacement, "S": self.short_replacement}


def build_binary_substitution_chain(
    order: int,
    *,
    spec: BinarySubstitutionChainSpec,
    spacing: float,
    long_short_ratio: float,
    max_sites: int | None,
    extra_metadata: Mapping[str, object] | None = None,
) -> Geometry:
    """Build an open embedded chain from a validated binary substitution spec."""
    order = validate_recursion_order(order)
    spacing = validate_spacing(spacing)
    long_short_ratio = _validate_long_short_ratio(long_short_ratio)
    max_sites = validate_optional_budget(max_sites, name="max_sites")
    n_long_bonds, n_short_bonds = _binary_symbol_counts(
        order=order,
        spec=spec,
        max_sites=max_sites,
    )

    rule = spec.substitution_rule
    word = spec.seed
    for _ in range(order):
        word = "".join(rule[bond] for bond in word)

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
                spec.long_edge_type if bond == "L" else spec.short_edge_type
            ),
            displacement=(float(bond_lengths[site]),),
        )
        for site, bond in enumerate(word)
    )

    metadata: dict[str, object] = {
        "generator": spec.generator,
        "family": spec.family,
        "order": order,
        "substitution_seed": spec.seed,
        "substitution_rule": rule,
        "boundary_condition": "open",
        "spacing": spacing,
        "long_short_ratio": long_short_ratio,
        "short_length": short_length,
        "long_length": long_length,
        "n_long_bonds": n_long_bonds,
        "n_short_bonds": n_short_bonds,
        "geometric_regime": (
            spec.self_similar_regime
            if long_short_ratio == spec.self_similar_ratio
            else spec.non_self_similar_regime
        ),
        "max_sites": max_sites,
    }
    if extra_metadata is not None:
        metadata.update(extra_metadata)

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
        metadata=metadata,
    )


def _validate_long_short_ratio(long_short_ratio: float) -> float:
    if isinstance(long_short_ratio, bool) or not isinstance(long_short_ratio, Real):
        raise TypeError("long_short_ratio must be a real number")
    result = float(long_short_ratio)
    if not np.isfinite(result) or result <= 1.0:
        raise ValueError("long_short_ratio must be finite and greater than one")
    return result


def _binary_symbol_counts(
    *,
    order: int,
    spec: BinarySubstitutionChainSpec,
    max_sites: int | None,
) -> tuple[int, int]:
    n_long_bonds = spec.seed.count("L")
    n_short_bonds = spec.seed.count("S")
    _enforce_site_budget(
        n_long_bonds=n_long_bonds,
        n_short_bonds=n_short_bonds,
        order=order,
        step=0,
        spec=spec,
        max_sites=max_sites,
    )

    long_to_long = spec.long_replacement.count("L")
    long_to_short = spec.long_replacement.count("S")
    short_to_long = spec.short_replacement.count("L")
    short_to_short = spec.short_replacement.count("S")
    for step in range(1, order + 1):
        n_long_bonds, n_short_bonds = (
            n_long_bonds * long_to_long + n_short_bonds * short_to_long,
            n_long_bonds * long_to_short + n_short_bonds * short_to_short,
        )
        _enforce_site_budget(
            n_long_bonds=n_long_bonds,
            n_short_bonds=n_short_bonds,
            order=order,
            step=step,
            spec=spec,
            max_sites=max_sites,
        )
    return n_long_bonds, n_short_bonds


def _enforce_site_budget(
    *,
    n_long_bonds: int,
    n_short_bonds: int,
    order: int,
    step: int,
    spec: BinarySubstitutionChainSpec,
    max_sites: int | None,
) -> None:
    n_sites = n_long_bonds + n_short_bonds + 1
    if max_sites is not None and n_sites > max_sites:
        step_text = f" at substitution step {step}" if step else ""
        raise ValueError(
            f"{spec.generator} order {order} requires at least {n_sites} "
            f"sites{step_text}, exceeding max_sites={max_sites}"
        )
