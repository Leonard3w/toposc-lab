from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from toposc_lab.geometry.base import Geometry, GeometryEdge
from toposc_lab.geometry.generators._dimensions import regular_lattice_dimensions


def chain(n_sites: int, *, spacing: float = 1.0) -> Geometry:
    """Create an open one-dimensional nearest-neighbor chain.

    Parameters
    ----------
    n_sites:
        Number of sites. A chain requires at least two sites.
    spacing:
        Positive distance between neighboring sites.
    """
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral):
        raise TypeError("n_sites must be an integer")
    n_sites = int(n_sites)
    if n_sites < 2:
        raise ValueError("n_sites must be at least two for a chain")

    if isinstance(spacing, bool) or not isinstance(spacing, Real):
        raise TypeError("spacing must be a real number")
    spacing = float(spacing)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")

    coordinates = spacing * np.arange(n_sites, dtype=float).reshape(-1, 1)
    edges = tuple(
        GeometryEdge(
            site,
            site + 1,
            edge_type="nearest_neighbor",
            displacement=(spacing,),
        )
        for site in range(n_sites - 1)
    )

    return Geometry(
        n_sites=n_sites,
        edges=edges,
        coordinates=coordinates,
        boundary_sites=frozenset({0, n_sites - 1}),
        dimension_records=regular_lattice_dimensions(1),
        metadata={
            "generator": "chain",
            "boundary_condition": "open",
            "spacing": spacing,
        },
    )
