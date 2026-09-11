"""Feature-space OOD flags for generated geometries, without physics claims."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

OOD_CAUTION = (
    "Feature-space OOD is an acquisition warning only; it is not evidence of "
    "invalid or unfavorable physics. Exact simulation remains required."
)


@dataclass(frozen=True, slots=True)
class OODAssessment:
    record_ids: tuple[str, ...]
    scores: NDArray[np.float64]
    is_ood: NDArray[np.bool_]
    threshold: float
    reference_quantile: float
    caution: str = OOD_CAUTION

    def __post_init__(self) -> None:
        scores = np.array(self.scores, dtype=float, copy=True)
        flags = np.array(self.is_ood, dtype=bool, copy=True)
        if scores.shape != (len(self.record_ids),) or flags.shape != scores.shape:
            raise ValueError("OOD arrays must align with record_ids")
        if not np.all(np.isfinite(scores)) or np.any(scores < 0.0):
            raise ValueError("OOD scores must be finite and nonnegative")
        if self.caution != OOD_CAUTION:
            raise ValueError("OOD caution cannot be weakened or replaced")
        scores.setflags(write=False)
        flags.setflags(write=False)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "is_ood", flags)
        object.__setattr__(self, "record_ids", tuple(self.record_ids))


class FeatureOODDetector:
    """Nearest-reference distance with a leave-one-out training threshold."""

    def __init__(self, *, reference_quantile: float = 0.99) -> None:
        self.reference_quantile = _quantile(reference_quantile)
        self._mean: NDArray[np.float64] | None = None
        self._scale: NDArray[np.float64] | None = None
        self._reference: NDArray[np.float64] | None = None
        self._threshold: float | None = None

    def fit(self, features: ArrayLike) -> Self:
        matrix = _matrix(features)
        if len(matrix) < 3:
            raise ValueError("OOD fitting requires at least three training samples")
        mean = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        reference = (matrix - mean) / scale
        pairwise = _distances(reference, reference)
        np.fill_diagonal(pairwise, np.inf)
        leave_one_out = np.min(pairwise, axis=1)
        threshold = float(np.quantile(leave_one_out, self.reference_quantile))
        self._mean = mean
        self._scale = scale
        self._reference = reference
        self._threshold = max(threshold, np.finfo(float).eps)
        return self

    def assess(self, features: ArrayLike, *, record_ids: tuple[str, ...]) -> OODAssessment:
        if (
            self._mean is None
            or self._scale is None
            or self._reference is None
            or self._threshold is None
        ):
            raise RuntimeError("OOD detector must be fitted before assessment")
        matrix = _matrix(features)
        if matrix.shape[1] != len(self._mean):
            raise ValueError("OOD feature count does not match fitted data")
        if len(record_ids) != len(matrix):
            raise ValueError("record_ids must align with feature rows")
        standardized = (matrix - self._mean) / self._scale
        scores = np.min(_distances(standardized, self._reference), axis=1)
        return OODAssessment(
            record_ids=record_ids,
            scores=scores,
            is_ood=scores > self._threshold,
            threshold=self._threshold,
            reference_quantile=self.reference_quantile,
        )


def _distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.sqrt(
        np.sum(np.square(first[:, None, :] - second[None, :, :]), axis=2)
        / max(1, first.shape[1])
    )


def _matrix(values: ArrayLike) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not len(matrix) or not np.all(np.isfinite(matrix)):
        raise ValueError("features must be a non-empty finite matrix")
    return matrix


def _quantile(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("reference_quantile must be a real number")
    result = float(value)
    if not np.isfinite(result) or not 0.5 <= result < 1.0:
        raise ValueError("reference_quantile must be in [0.5, 1)")
    return result
