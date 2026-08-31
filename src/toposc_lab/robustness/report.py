"""Validated report composition over separated robustness result objects."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from toposc_lab.robustness.ensemble import DisorderEnsembleResult
from toposc_lab.robustness.family import CrossSizeGeometryFamily
from toposc_lab.robustness.finite_size import FiniteSizeScalingResult
from toposc_lab.robustness.metrics import RobustnessFractionMetric
from toposc_lab.robustness.uncertainty import RobustnessUncertaintyEstimate

ROBUSTNESS_REPORT_VERSION = 1

_TECHNICAL_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True, eq=False)
class RobustnessReportEntry:
    """One ensemble and its unchanged metric and uncertainty results."""

    ensemble: DisorderEnsembleResult = field(repr=False, compare=False)
    metric: RobustnessFractionMetric
    uncertainty: RobustnessUncertaintyEstimate

    def __post_init__(self) -> None:
        if not isinstance(self.ensemble, DisorderEnsembleResult):
            raise TypeError("ensemble must be DisorderEnsembleResult")
        if not isinstance(self.metric, RobustnessFractionMetric):
            raise TypeError("metric must be RobustnessFractionMetric")
        if not isinstance(self.uncertainty, RobustnessUncertaintyEstimate):
            raise TypeError("uncertainty must be RobustnessUncertaintyEstimate")
        if self.metric.request != self.ensemble.request:
            raise ValueError("metric request must match the ensemble request exactly")
        expected_failure_indices = tuple(
            member.sample_index
            for member in self.ensemble.members
            if member.failure is not None
        )
        if self.metric.execution_failure_indices != expected_failure_indices:
            raise ValueError(
                "metric execution failures must match the ensemble exactly"
            )
        if self.uncertainty.metric is not self.metric:
            raise ValueError("uncertainty must reference the exact entry metric")

    @property
    def successful_count(self) -> int:
        """Successful realizations under the explicit metric criterion."""
        return self.metric.successful_count

    @property
    def total_count(self) -> int:
        """All explicitly requested ensemble realizations."""
        return self.metric.total_count

    @property
    def robustness_fraction(self) -> float:
        """Observed success fraction for this entry."""
        return self.metric.value

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """Phase-8.11 confidence interval for this entry."""
        return self.uncertainty.confidence_interval

    @property
    def execution_failure_count(self) -> int:
        """Operational failures retained in the unsuccessful population."""
        return len(self.metric.execution_failure_indices)


@dataclass(frozen=True, slots=True, eq=False)
class RobustnessReport:
    """Single-size or provenance-linked cross-size robustness report."""

    report_key: str
    description: str
    entries: tuple[RobustnessReportEntry, ...]
    scaling: FiniteSizeScalingResult | None = None
    geometry_family: CrossSizeGeometryFamily | None = None
    report_version: int = field(
        default=ROBUSTNESS_REPORT_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "report_key",
            _technical_key(self.report_key, name="report_key"),
        )
        object.__setattr__(
            self,
            "description",
            _description(self.description, name="description"),
        )
        entries = tuple(self.entries)
        if not entries:
            raise ValueError("entries must not be empty")
        if any(not isinstance(entry, RobustnessReportEntry) for entry in entries):
            raise TypeError("entries must contain only RobustnessReportEntry values")
        if self.scaling is not None and not isinstance(
            self.scaling,
            FiniteSizeScalingResult,
        ):
            raise TypeError("scaling must be FiniteSizeScalingResult or None")
        if self.geometry_family is not None and not isinstance(
            self.geometry_family,
            CrossSizeGeometryFamily,
        ):
            raise TypeError(
                "geometry_family must be CrossSizeGeometryFamily or None"
            )

        if self.scaling is None and self.geometry_family is None:
            if len(entries) != 1:
                raise ValueError(
                    "a single-size report must contain exactly one entry"
                )
        elif self.scaling is None or self.geometry_family is None:
            raise ValueError(
                "cross-size reports require both scaling and geometry_family"
            )
        else:
            if len(entries) != len(self.scaling.points):
                raise ValueError(
                    "cross-size reports require one entry per scaling point"
                )
            if len({id(entry) for entry in entries}) != len(entries):
                raise ValueError(
                    "cross-size reports require distinct entry objects per size"
                )
            if len({id(entry.ensemble) for entry in entries}) != len(entries):
                raise ValueError(
                    "cross-size reports require distinct ensemble objects per size"
                )
            for entry, point in zip(entries, self.scaling.points, strict=True):
                if entry.uncertainty is not point.uncertainty:
                    raise ValueError(
                        "report entries must match scaling points in exact order"
                    )
            if self.geometry_family.scaling is not self.scaling:
                raise ValueError(
                    "geometry_family must reference the report scaling result"
                )
            _validate_cross_size_disorder_contract(entries)

        object.__setattr__(self, "entries", entries)

    @property
    def is_cross_size(self) -> bool:
        """Whether the report includes scaling and geometry-family provenance."""
        return self.scaling is not None

    @property
    def criterion_key(self) -> str:
        """Stable key of the explicit success criterion."""
        return self.entries[0].metric.criterion_key

    @property
    def criterion_description(self) -> str:
        """Human-readable success criterion used by the report."""
        return self.entries[0].metric.criterion_description

    @property
    def robustness_fractions(self) -> tuple[float, ...]:
        """Per-entry fractions without aggregation across system sizes."""
        return tuple(entry.robustness_fraction for entry in self.entries)

    @property
    def confidence_intervals(self) -> tuple[tuple[float, float], ...]:
        """Per-entry confidence intervals in report order."""
        return tuple(entry.confidence_interval for entry in self.entries)

    @property
    def successful_counts(self) -> tuple[int, ...]:
        """Per-entry successful realization counts."""
        return tuple(entry.successful_count for entry in self.entries)

    @property
    def total_counts(self) -> tuple[int, ...]:
        """Per-entry requested realization counts."""
        return tuple(entry.total_count for entry in self.entries)

    @property
    def execution_failure_counts(self) -> tuple[int, ...]:
        """Per-entry operational failure counts."""
        return tuple(entry.execution_failure_count for entry in self.entries)

    @property
    def total_execution_failure_count(self) -> int:
        """Total operational failures, without redefining scientific success."""
        return sum(self.execution_failure_counts)

    @property
    def system_sizes(self) -> tuple[float, ...] | None:
        """Explicit cross-size coordinates, or ``None`` for a single-size report."""
        if self.scaling is None:
            return None
        return self.scaling.system_sizes

    @property
    def warnings(self) -> tuple[str, ...]:
        """Deterministic interpretation warnings without flattening source warnings."""
        warnings = [
            "Robustness uses the report's explicit success criterion and does not "
            "by itself establish topology or protected boundary modes."
        ]
        if self.total_execution_failure_count:
            warnings.append(
                f"{self.total_execution_failure_count} ensemble execution failure(s) "
                "remain in the unsuccessful population; inspect the source entries."
            )
        if self.scaling is None:
            warnings.append(
                "A single finite-size robustness result cannot establish "
                "thermodynamic behavior."
            )
        else:
            warnings.append(
                "The scaling fit and declared geometry family record assumptions; "
                "they do not prove a thermodynamic topological phase."
            )
            warnings.append(
                "Cross-size disorder validation covers transform identity, RNG, "
                "target, snapshot schemes, and parameter schema; inspect each "
                "entry's provenance for parameter values and source identities."
            )
            if not self.scaling.infinite_size_intercept_in_unit_interval:
                warnings.append(
                    "The fitted infinite-size robustness intercept lies outside "
                    "the physical unit interval and must not be clipped silently."
                )
            missing_provenance = sum(
                not _entry_has_disorder_provenance(entry)
                for entry in self.entries
            )
            if missing_provenance:
                warnings.append(
                    f"{missing_provenance} cross-size entry or entries contain no "
                    "completed disorder realization from which to verify the "
                    "disorder protocol."
                )
        return tuple(warnings)


def create_robustness_report(
    report_key: str,
    *,
    description: str,
    entries: tuple[RobustnessReportEntry, ...],
    scaling: FiniteSizeScalingResult | None = None,
    geometry_family: CrossSizeGeometryFamily | None = None,
) -> RobustnessReport:
    """Compose already-computed robustness records into one validated report."""
    return RobustnessReport(
        report_key=report_key,
        description=description,
        entries=entries,
        scaling=scaling,
        geometry_family=geometry_family,
    )


def _validate_cross_size_disorder_contract(
    entries: tuple[RobustnessReportEntry, ...],
) -> None:
    contracts = tuple(
        contract
        for entry in entries
        if (contract := _entry_disorder_contract(entry)) is not None
    )
    if contracts and any(
        contract != contracts[0] for contract in contracts[1:]
    ):
        raise ValueError(
            "cross-size entries must share one disorder transform, RNG, target, "
            "snapshot scheme, and parameter schema"
        )


def _entry_disorder_contract(entry: RobustnessReportEntry) -> tuple[object, ...] | None:
    for member in entry.ensemble.members:
        if member.disorder is not None:
            provenance = member.disorder.provenance
            return (
                provenance.disorder_key,
                provenance.disorder_version,
                provenance.rng_algorithm,
                provenance.source.target,
                provenance.source.scheme,
                provenance.result.scheme,
                tuple(provenance.parameters),
            )
    return None


def _entry_has_disorder_provenance(entry: RobustnessReportEntry) -> bool:
    return _entry_disorder_contract(entry) is not None


def _technical_key(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _TECHNICAL_KEY_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{name} must start with a lowercase letter and contain only lowercase "
            "letters, digits, and underscores"
        )
    return value


def _description(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()
