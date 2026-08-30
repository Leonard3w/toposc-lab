"""Private validation shared by geometry builders operating on point clouds."""

from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_point_coordinates(
    coordinates: ArrayLike,
) -> NDArray[np.float64]:
    """Return a finite, nonempty rectangular ``(n_sites, dimension)`` array."""
    try:
        source = np.asarray(coordinates)
        if np.iscomplexobj(source):
            raise ValueError("complex coordinates are not supported")
        result = np.array(coordinates, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "coordinates must be a rectangular array of real numbers"
        ) from error
    if result.ndim != 2:
        raise ValueError("coordinates must be a two-dimensional array")
    if result.shape[0] < 1:
        raise ValueError("coordinates must contain at least one site")
    if result.shape[1] < 1:
        raise ValueError("coordinates must contain at least one dimension")
    if not np.all(np.isfinite(result)):
        raise ValueError("coordinates must contain only finite values")
    return np.ascontiguousarray(result, dtype=float)


def validate_positive_distance(value: float, *, name: str) -> float:
    """Return a finite strictly positive distance parameter."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result
