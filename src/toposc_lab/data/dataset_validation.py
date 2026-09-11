"""Cross-field and scientific-claim validation for dataset records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral

from toposc_lab.data.dataset_codec import create_dataset_record
from toposc_lab.data.dataset_schema import DatasetRecord, TopologyValidity
from toposc_lab.data.dataset_storage import ExactPhysicsDataset, dataset_from_bytes
from toposc_lab.geometry import validate_geometry


class DatasetValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class DatasetValidationIssue:
    severity: DatasetValidationSeverity
    code: str
    message: str
    record_id: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    issues: tuple[DatasetValidationIssue, ...]
    record_count: int

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity is DatasetValidationSeverity.ERROR for issue in self.issues)

    def raise_for_errors(self) -> None:
        errors = [issue for issue in self.issues if issue.severity is DatasetValidationSeverity.ERROR]
        if errors:
            summary = "; ".join(f"{issue.code}: {issue.message}" for issue in errors)
            raise ValueError(f"dataset validation failed: {summary}")


def validate_dataset_record(record: DatasetRecord) -> DatasetValidationReport:
    """Validate cross-field contracts not owned by individual typed records."""
    if not isinstance(record, DatasetRecord):
        raise TypeError("record must be DatasetRecord")
    issues: list[DatasetValidationIssue] = []

    geometry = record.geometry.to_geometry()
    geometry_report = validate_geometry(geometry)
    for issue in geometry_report.issues:
        if issue.severity.value == "error":
            issues.append(_issue("malformed_geometry", issue.message, record))

    if record.spectrum.basis_size % geometry.n_sites != 0:
        issues.append(
            _issue(
                "inconsistent_basis_dimension",
                "spectrum basis_size must be an integer multiple of geometry.n_sites",
                record,
            )
        )
    reference = record.spectrum.state_reference
    if (
        reference is not None
        and reference.shape is not None
        and reference.shape[0] != record.spectrum.basis_size
    ):
        issues.append(
            _issue(
                "inconsistent_state_shape",
                "state artifact leading dimension must equal spectrum basis_size",
                record,
            )
        )

    model_n_sites = record.model.parameters.get("n_sites")
    if model_n_sites is not None and (
        isinstance(model_n_sites, bool)
        or not isinstance(model_n_sites, Integral)
        or int(model_n_sites) != geometry.n_sites
    ):
        issues.append(
            _issue(
                "inconsistent_model_site_count",
                "model parameter n_sites must equal the stored geometry size",
                record,
            )
        )
    model_dimension = record.model.parameters.get("embedding_dimension")
    if model_dimension is not None and model_dimension != geometry.embedding_dimension:
        issues.append(
            _issue(
                "inconsistent_embedding_dimension",
                "model embedding_dimension must equal geometry.embedding_dimension",
                record,
            )
        )

    for topology in record.topology:
        if topology.validity is not TopologyValidity.VALID and topology.is_topological is not None:
            issues.append(
                _issue(
                    "invalid_topology_claim",
                    f"{topology.method} makes a claim although validity is {topology.validity.value}",
                    record,
                )
            )
        if topology.validity is TopologyValidity.VALID and not topology.tolerances:
            issues.append(
                _issue(
                    "missing_topology_tolerances",
                    f"{topology.method} is valid but records no numerical tolerances",
                    record,
                )
            )

    for robustness in record.robustness:
        sample_count = robustness.statistics.get("sample_count")
        if sample_count is not None and sample_count != len(robustness.seeds):
            issues.append(
                _issue(
                    "inconsistent_robustness_sample_count",
                    f"{robustness.protocol} sample_count differs from its explicit seeds",
                    record,
                )
            )

    rebuilt = create_dataset_record(
        geometry=record.geometry,
        model=record.model,
        spectrum=record.spectrum,
        observables=record.observables,
        topology=record.topology,
        robustness=record.robustness,
        provenance=record.provenance,
    )
    if rebuilt.record_id != record.record_id:
        issues.append(_issue("record_identity_mismatch", "record_id does not cover current contents", record))
    return DatasetValidationReport(tuple(issues), record_count=1)


def validate_dataset(dataset: ExactPhysicsDataset) -> DatasetValidationReport:
    """Validate all records and cross-record uniqueness."""
    if not isinstance(dataset, ExactPhysicsDataset):
        raise TypeError("dataset must be ExactPhysicsDataset")
    issues: list[DatasetValidationIssue] = []
    exact_geometry_owners: dict[str, str] = {}
    for record in dataset.records:
        issues.extend(validate_dataset_record(record).issues)
        previous = exact_geometry_owners.get(record.geometry.exact_id)
        if previous is not None:
            issues.append(
                DatasetValidationIssue(
                    severity=DatasetValidationSeverity.WARNING,
                    code="repeated_exact_geometry",
                    message=f"geometry also occurs in record {previous}",
                    record_id=record.record_id,
                )
            )
        else:
            exact_geometry_owners[record.geometry.exact_id] = record.record_id
    return DatasetValidationReport(tuple(issues), record_count=len(dataset.records))


def validate_dataset_bytes(payload: bytes) -> DatasetValidationReport:
    """Strictly decode bytes before running semantic validation."""
    dataset = dataset_from_bytes(payload)
    return validate_dataset(dataset)


def _issue(code: str, message: str, record: DatasetRecord) -> DatasetValidationIssue:
    return DatasetValidationIssue(
        severity=DatasetValidationSeverity.ERROR,
        code=code,
        message=message,
        record_id=record.record_id,
    )
