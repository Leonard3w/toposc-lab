"""Statistical uncertainty for empirical robustness success fractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Real
from statistics import NormalDist

from toposc_lab.robustness.metrics import RobustnessFractionMetric

ROBUSTNESS_UNCERTAINTY_VERSION = 1


class RobustnessUncertaintyMethod(str, Enum):
    """Supported confidence-interval construction for a success fraction."""

    WILSON_SCORE = "wilson_score"


@dataclass(frozen=True, slots=True)
class RobustnessUncertaintyEstimate:
    """Binomial standard error and bounded interval for one robustness metric."""

    metric: RobustnessFractionMetric
    confidence_level: float = 0.95
    method: RobustnessUncertaintyMethod = RobustnessUncertaintyMethod.WILSON_SCORE
    uncertainty_version: int = field(
        default=ROBUSTNESS_UNCERTAINTY_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.metric, RobustnessFractionMetric):
            raise TypeError("metric must be RobustnessFractionMetric")
        object.__setattr__(
            self,
            "confidence_level",
            _confidence_level(self.confidence_level),
        )
        if not isinstance(self.method, RobustnessUncertaintyMethod):
            raise TypeError("method must be RobustnessUncertaintyMethod")

    @property
    def sample_size(self) -> int:
        """Number of requested ensemble members used in the estimate."""
        return self.metric.total_count

    @property
    def successful_count(self) -> int:
        """Number of successful ensemble members used in the estimate."""
        return self.metric.successful_count

    @property
    def observed_fraction(self) -> float:
        """Empirical robustness success fraction."""
        return self.metric.value

    @property
    def standard_error(self) -> float:
        """Return the plug-in binomial standard error of the observed fraction."""
        probability = self.observed_fraction
        return math.sqrt(probability * (1.0 - probability) / self.sample_size)

    @property
    def critical_value(self) -> float:
        """Two-sided standard-normal critical value for the confidence level."""
        lower_tail_probability = (1.0 - self.confidence_level) / 2.0
        return -NormalDist().inv_cdf(lower_tail_probability)

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """Return the two-sided Wilson score confidence interval."""
        probability = self.observed_fraction
        sample_size = self.sample_size
        critical_value = self.critical_value
        squared_critical_value = critical_value * critical_value
        denominator = 1.0 + squared_critical_value / sample_size
        center = (
            probability + squared_critical_value / (2.0 * sample_size)
        ) / denominator
        half_width = (
            critical_value
            * math.sqrt(
                probability * (1.0 - probability) / sample_size
                + squared_critical_value / (4.0 * sample_size * sample_size)
            )
            / denominator
        )
        return max(0.0, center - half_width), min(1.0, center + half_width)

    @property
    def lower_bound(self) -> float:
        """Lower Wilson score bound."""
        return self.confidence_interval[0]

    @property
    def upper_bound(self) -> float:
        """Upper Wilson score bound."""
        return self.confidence_interval[1]


def estimate_robustness_uncertainty(
    metric: RobustnessFractionMetric,
    *,
    confidence_level: float = 0.95,
    method: RobustnessUncertaintyMethod = RobustnessUncertaintyMethod.WILSON_SCORE,
) -> RobustnessUncertaintyEstimate:
    """Estimate binomial uncertainty for an existing robustness fraction."""
    return RobustnessUncertaintyEstimate(
        metric=metric,
        confidence_level=confidence_level,
        method=method,
    )


def _confidence_level(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("confidence_level must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError("confidence_level must be finite and strictly between 0 and 1")
    return result
