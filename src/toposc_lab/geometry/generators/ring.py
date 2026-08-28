from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from toposc_lab.geometry.base import Geometry, GeometryEdge


def ring(n_sites: int, *, radius: float = 1.0) -> Geometry:
    """Create a periodic one-dimensional cycle embedded in two dimensions.

    Parameters
    ----------
    n_sites:
        Number of sites. A simple ring requires at least three sites.
    radius:
        Positive radius of the circular embedding.
    """
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral):
        raise TypeError("n_sites must be an integer")
    n_sites = int(n_sites)
    if n_sites < 3:
        raise ValueError("n_sites must be at least three for a ring")

    if isinstance(radius, bool) or not isinstance(radius, Real):
        raise TypeError("radius must be a real number")
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")

    angles = 2.0 * np.pi * np.arange(n_sites, dtype=float) / n_sites
    coordinates = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    edges = tuple(
        GeometryEdge(
            site,
            (site + 1) % n_sites,
            edge_type="nearest_neighbor",
            boundary_crossing=site == n_sites - 1,
        )
        for site in range(n_sites)
    )

    return Geometry(
        n_sites=n_sites,
        edges=edges,
        coordinates=coordinates,
        metadata={
            "generator": "ring",
            "boundary_condition": "periodic",
            "intrinsic_dimension": 1,
            "radius": radius,
        },
    )
