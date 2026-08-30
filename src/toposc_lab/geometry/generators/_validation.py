"""Shared validation helpers for regular geometry generators."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np


def validate_axis_size(
    value: int,
    *,
    name: str,
    geometry_name: str,
) -> int:
    """Return an integer axis size of at least two sites."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    size = int(value)
    if size < 2:
        raise ValueError(f"{name} must be at least two for a {geometry_name}")
    return size


def validate_spacing(spacing: float) -> float:
    """Return a finite positive lattice spacing."""
    if isinstance(spacing, bool) or not isinstance(spacing, Real):
        raise TypeError("spacing must be a real number")
    result = float(spacing)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("spacing must be finite and positive")
    return result


def validate_boundary(boundary: str, *, name: str) -> str:
    """Return a supported regular-lattice boundary condition."""
    if boundary not in ("open", "periodic"):
        raise ValueError(f"{name} must be either open or periodic")
    return boundary


def resolve_axis_index(
    index: int,
    *,
    size: int,
    boundary: str,
) -> tuple[int, bool] | None:
    """Resolve an axis index and report whether it crossed a periodic boundary."""
    if 0 <= index < size:
        return index, False
    if boundary == "open":
        return None
    return index % size, True
