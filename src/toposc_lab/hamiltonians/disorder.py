"""Reproducible disorder maps for graph-based Hamiltonian parameters."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from toposc_lab.geometry import Geometry, GeometryEdge


def uniform_site_disorder(
    geometry: Geometry,
    *,
    width: float,
    seed: int | None,
) -> dict[int, float]:
    r"""Sample independent onsite offsets from ``[-width / 2, width / 2]``.

    The returned mapping is keyed by canonical site indices and can be passed
    directly as an onsite parameter map. A fixed ``seed`` reproduces the same
    realization without modifying NumPy's global random state.
    """
    width = _validate_width(width)
    generator = np.random.default_rng(_validate_seed(seed))
    return sample_uniform_site_disorder(
        geometry,
        width=width,
        rng=generator,
    )


def sample_uniform_site_disorder(
    geometry: Geometry,
    *,
    width: float,
    rng: np.random.Generator,
) -> dict[int, float]:
    r"""Sample onsite offsets using an explicitly owned NumPy generator.

    This is the shared sampling primitive behind the seeded convenience API
    and the Phase-8 reproducible disorder executor. It never creates or reads
    random state itself.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be Geometry")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    width = _validate_width(width)
    values = rng.uniform(-width / 2.0, width / 2.0, geometry.n_sites)
    return dict(zip(geometry.site_indices, values.tolist(), strict=True))


def uniform_edge_disorder(
    geometry: Geometry,
    *,
    width: float,
    seed: int | None,
) -> dict[GeometryEdge, float]:
    r"""Sample independent edge offsets from ``[-width / 2, width / 2]``.

    The returned mapping uses the geometry's immutable, oriented edge objects
    as keys and can be passed directly as a hopping parameter map. A fixed
    ``seed`` reproduces the same realization without modifying NumPy's global
    random state.
    """
    width = _validate_width(width)
    generator = np.random.default_rng(_validate_seed(seed))
    values = generator.uniform(-width / 2.0, width / 2.0, geometry.n_edges)
    return dict(zip(geometry.edges, values.tolist(), strict=True))


def _validate_width(width: float) -> float:
    if isinstance(width, bool) or not isinstance(width, Real):
        raise TypeError("width must be a real number")
    width = float(width)
    if not np.isfinite(width) or width < 0.0:
        raise ValueError("width must be finite and non-negative")
    return width


def _validate_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer or None")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return seed
