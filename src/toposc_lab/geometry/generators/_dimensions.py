"""Dimension records shared by regular geometry generators."""

from __future__ import annotations

from toposc_lab.geometry.base import GeometryDimension


def regular_lattice_dimensions(
    dimension: int,
) -> tuple[GeometryDimension, ...]:
    """Describe the translation rank of an ideal regular-lattice family."""
    return (
        GeometryDimension(
            kind="lattice",
            value=float(dimension),
            scope="infinite_family",
            method="translation_rank",
            exact=True,
        ),
    )
