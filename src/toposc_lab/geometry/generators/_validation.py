"""Shared validation helpers for geometry generators."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np


def validate_recursion_order(order: int) -> int:
    """Return a nonnegative integer recursion order."""
    if isinstance(order, bool) or not isinstance(order, Integral):
        raise TypeError("order must be an integer")
    result = int(order)
    if result < 0:
        raise ValueError("order must be nonnegative")
    return result


def validate_graph_size(n_sites: int) -> int:
    """Return a positive integer number of graph sites."""
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral):
        raise TypeError("n_sites must be an integer")
    result = int(n_sites)
    if result < 1:
        raise ValueError("n_sites must be positive")
    return result


def validate_probability(probability: float, *, name: str) -> float:
    """Return a finite probability in the closed unit interval."""
    if isinstance(probability, bool) or not isinstance(probability, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(probability)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")
    return result


def validate_seed(seed: int) -> int:
    """Return a nonnegative integer random seed."""
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    result = int(seed)
    if result < 0:
        raise ValueError("seed must be nonnegative")
    return result


def validate_optional_budget(value: int | None, *, name: str) -> int | None:
    """Return a positive integer resource budget or ``None``."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer or None")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


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


def validate_periodic_axis_size(
    size: int,
    boundary: str,
    *,
    name: str,
    geometry_name: str,
) -> None:
    """Reject periodic axes that require parallel edges in a simple graph."""
    if boundary == "periodic" and size < 3:
        raise ValueError(
            f"{name} must be at least three for a periodic {geometry_name}"
        )


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
