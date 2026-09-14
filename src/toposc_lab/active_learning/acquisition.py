"""Deterministic acquisition; OOD values cannot drive exploitation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from toposc_lab.active_learning.prediction import Prediction


def exploit(predictions: tuple[Prediction, ...], count: int) -> tuple[str, ...]:
    """Maximize in-distribution predicted value; break ties by candidate identity."""
    if count < 0:
        raise ValueError("count must be nonnegative")
    return tuple(
        p.candidate_id
        for p in sorted(
            (p for p in predictions if p.valid and not p.is_ood),
            key=lambda p: (-p.value, p.candidate_id),
        )[:count]
    )


def explore(
    predictions: tuple[Prediction, ...],
    count: int,
    *,
    uncertainty_weight: float = 1.0,
    novelty_weight: float = 1.0,
) -> tuple[str, ...]:
    """Bound both signals to [0,1); extreme OOD distance cannot dominate unboundedly.

    OOD candidates remain eligible for exact exploration, never pseudo-labeling.
    """
    if count < 0 or any(
        not math.isfinite(w) or w < 0 for w in (uncertainty_weight, novelty_weight)
    ):
        raise ValueError("invalid exploration configuration")
    if uncertainty_weight + novelty_weight == 0:
        raise ValueError("exploration requires an informative signal")
    return tuple(
        p.candidate_id
        for p in sorted(
            (p for p in predictions if p.valid),
            key=lambda p: (
                -(
                    uncertainty_weight * p.uncertainty / (1 + p.uncertainty)
                    + novelty_weight * p.novelty / (1 + p.novelty)
                ),
                p.candidate_id,
            ),
        )[:count]
    )


@dataclass(frozen=True, slots=True)
class AcquisitionConfig:
    batch_size: int = 4
    exploration_fraction: float = 0.25
    value_weight: float = 1.0
    uncertainty_weight: float = 0.25
    novelty_weight: float = 0.25
    scaling: str = "legacy"
    diversity_radius_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.scaling not in {"legacy", "target_scale"}:
            raise ValueError("unsupported acquisition scaling")
        if not math.isfinite(self.diversity_radius_fraction) or self.diversity_radius_fraction < 0:
            raise ValueError("diversity radius fraction must be finite and nonnegative")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("batch_size must be an integer")
        if self.batch_size < 2 or not 0 < self.exploration_fraction < 1:
            raise ValueError("reserve both exploration and exploitation slots")
        weights = (self.value_weight, self.uncertainty_weight, self.novelty_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("weights must be finite and nonnegative")
        if self.value_weight == 0 or self.uncertainty_weight + self.novelty_weight == 0:
            raise ValueError("both value and exploration signals are required")


@dataclass(frozen=True, slots=True)
class Acquisition:
    candidate_id: str
    strategy: str


def acquire(
    predictions: tuple[Prediction, ...],
    config: AcquisitionConfig,
    *,
    feature_coordinates: dict[str, np.ndarray] | None = None,
    reference_spacing: float | None = None,
) -> tuple[Acquisition, ...]:
    """Reserve exploration slots; use bounded UCB for in-distribution exploitation.

    If no trustworthy exploitation exists, fill remaining slots with exploration
    and record that decision. Invalid predictions consume no simulation slots.
    """
    if len({p.candidate_id for p in predictions}) != len(predictions):
        raise ValueError("duplicate prediction identities")
    if config.scaling != "legacy" or config.diversity_radius_fraction:
        return _scaled_diverse_acquire(predictions, config, feature_coordinates, reference_spacing)
    explore_count = min(
        config.batch_size - 1, max(1, math.ceil(config.batch_size * config.exploration_fraction))
    )
    combined = tuple(
        replace(
            p,
            value=(
                config.value_weight * p.value
                + config.uncertainty_weight * p.uncertainty / (1 + p.uncertainty)
                + config.novelty_weight * p.novelty / (1 + p.novelty)
            ),
        )
        for p in predictions
        if p.valid
    )
    exploitation = exploit(combined, config.batch_size - explore_count)
    exploration = explore(
        tuple(p for p in predictions if p.candidate_id not in exploitation),
        config.batch_size - len(exploitation),
        uncertainty_weight=config.uncertainty_weight,
        novelty_weight=config.novelty_weight,
    )
    return tuple(Acquisition(i, "exploitation") for i in exploitation) + tuple(
        Acquisition(i, "exploration") for i in exploration
    )


def _scaled_diverse_acquire(
    predictions: tuple[Prediction, ...],
    config: AcquisitionConfig,
    coordinates: dict[str, np.ndarray] | None,
    spacing: float | None,
) -> tuple[Acquisition, ...]:
    valid = tuple(p for p in predictions if p.valid)
    if not valid:
        return ()
    radius = 0.0
    if config.diversity_radius_fraction:
        if coordinates is None or spacing is None or not math.isfinite(spacing) or spacing < 0:
            raise ValueError("diversity requires finite feature coordinates and reference spacing")
        if not {p.candidate_id for p in valid}.issubset(coordinates):
            raise ValueError("diversity coordinates must cover every valid prediction")
        vectors = [np.asarray(coordinates[p.candidate_id], dtype=float) for p in valid]
        if any(v.ndim != 1 or not len(v) or not np.all(np.isfinite(v)) for v in vectors):
            raise ValueError("diversity coordinates must be finite vectors")
        if len({v.shape for v in vectors}) != 1:
            raise ValueError("diversity feature dimensions must agree")
        radius = spacing * config.diversity_radius_fraction
    trusted_values = [p.value for p in valid if not p.is_ood]
    values = trusted_values or [p.value for p in valid]
    scale = float(np.quantile(values, 0.9) - np.quantile(values, 0.1))
    if scale <= 0:
        scale = max(float(np.median([p.uncertainty for p in valid])), np.finfo(float).tiny)

    def utility(p):
        if config.scaling == "legacy":
            return (
                config.value_weight * p.value
                + config.uncertainty_weight * p.uncertainty / (1 + p.uncertainty)
                + config.novelty_weight * p.novelty / (1 + p.novelty)
            )
        # All bonuses use target units, and no uncertainty bonus exceeds its
        # configured fraction of the robust prediction range.
        return (
            config.value_weight * p.value
            + config.uncertainty_weight * min(p.uncertainty, scale)
            + config.novelty_weight * scale * p.novelty / (1 + p.novelty)
        )

    exploit_order = sorted(
        (p for p in valid if not p.is_ood), key=lambda p: (-utility(p), p.candidate_id)
    )
    explore_order = sorted(
        valid,
        key=lambda p: (
            -(
                config.uncertainty_weight * (p.uncertainty / (scale + p.uncertainty))
                + config.novelty_weight * p.novelty / (1 + p.novelty)
            ),
            p.candidate_id,
        ),
    )
    chosen: list[Acquisition] = []
    chosen_ids = set()

    def fill(order, target_count, strategy):
        deferred = []
        for p in order:
            if len(chosen) >= target_count:
                break
            if p.candidate_id in chosen_ids:
                continue
            separated = not radius or all(
                np.linalg.norm(coordinates[p.candidate_id] - coordinates[c.candidate_id])
                / np.sqrt(len(coordinates[p.candidate_id]))
                >= radius
                for c in chosen
            )
            if not separated:
                deferred.append(p)
                continue
            chosen.append(Acquisition(p.candidate_id, strategy))
            chosen_ids.add(p.candidate_id)
        # Finite/clustered catalogs may lack enough separated candidates. Retain
        # the exact budget with explicit relaxation, never silently drop a slot.
        for p in deferred:
            if len(chosen) >= target_count:
                break
            chosen.append(Acquisition(p.candidate_id, strategy + "_diversity_relaxed"))
            chosen_ids.add(p.candidate_id)

    exploration_count = min(
        config.batch_size - 1, max(1, math.ceil(config.batch_size * config.exploration_fraction))
    )
    fill(exploit_order, config.batch_size - exploration_count, "exploitation")
    fill(explore_order, config.batch_size, "exploration")
    return tuple(chosen)
