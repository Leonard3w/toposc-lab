"""Dimension- and symmetry-aware dispatch for topology calculations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from types import MappingProxyType

from toposc_lab.observables.results import ObservableRecord
from toposc_lab.topology.results import (
    TopologyDiagnosticResult,
    TopologyMethod,
    TopologyResult,
    unify_topology_result,
)
from toposc_lab.topology.symmetry import (
    AltlandZirnbauerClass,
    SymmetryClassification,
)


class TopologyCapability(str, Enum):
    """Explicit physical evidence or inputs available to the dispatcher."""

    TRANSLATION_INVARIANT_BULK = "translation_invariant_bulk"
    BULK_GAP_EVIDENCE = "bulk_gap_evidence"
    BLOCH_PARTICLE_HOLE_ENDPOINTS = "bloch_particle_hole_endpoints"
    CHIRAL_OPERATOR = "chiral_operator"
    BASIS_COORDINATES = "basis_coordinates"
    BULK_MASK = "bulk_mask"
    COORDINATE_PERIODS = "coordinate_periods"
    POSITION_AREAS = "position_areas"
    LOCALIZER_PROBE = "localizer_probe"


@dataclass(frozen=True, slots=True)
class TopologyDispatchContext:
    """Physical dimension, symmetry class, and available method prerequisites.

    ``physical_dimension`` is the integer intrinsic/topological dimension of
    the problem. It is deliberately separate from ``embedding_dimension`` and
    must not be inferred from the number of coordinate columns.
    """

    physical_dimension: int
    classification: SymmetryClassification
    capabilities: frozenset[TopologyCapability] = frozenset()
    embedding_dimension: int | None = None

    def __post_init__(self) -> None:
        physical_dimension = _positive_integer(
            self.physical_dimension,
            name="physical_dimension",
        )
        if not isinstance(self.classification, SymmetryClassification):
            raise TypeError("classification must be a SymmetryClassification")
        capabilities = frozenset(self.capabilities)
        if not all(isinstance(value, TopologyCapability) for value in capabilities):
            raise TypeError("capabilities must contain only TopologyCapability values")
        embedding_dimension = self.embedding_dimension
        if embedding_dimension is not None:
            embedding_dimension = _positive_integer(
                embedding_dimension,
                name="embedding_dimension",
            )
        object.__setattr__(self, "physical_dimension", physical_dimension)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "embedding_dimension", embedding_dimension)


@dataclass(frozen=True, slots=True)
class TopologyDispatchDecision:
    """Applicable methods and explicit rejection reasons for all others."""

    context: TopologyDispatchContext
    applicable_methods: tuple[TopologyMethod, ...]
    rejected_methods: Mapping[TopologyMethod, tuple[str, ...]]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.context, TopologyDispatchContext):
            raise TypeError("context must be a TopologyDispatchContext")
        methods = tuple(self.applicable_methods)
        if not all(isinstance(method, TopologyMethod) for method in methods):
            raise TypeError("applicable_methods must contain TopologyMethod values")
        if len(set(methods)) != len(methods):
            raise ValueError("applicable_methods must not contain duplicates")
        rejected: dict[TopologyMethod, tuple[str, ...]] = {}
        for method, reasons in self.rejected_methods.items():
            if not isinstance(method, TopologyMethod):
                raise TypeError("rejected method keys must be TopologyMethod values")
            reason_tuple = tuple(reasons)
            if not reason_tuple or not all(
                isinstance(reason, str) and bool(reason.strip())
                for reason in reason_tuple
            ):
                raise ValueError("each rejected method must have non-empty reasons")
            rejected[method] = reason_tuple
        if set(methods) & set(rejected):
            raise ValueError("a method cannot be both applicable and rejected")
        if set(methods) | set(rejected) != set(TopologyMethod):
            raise ValueError("the dispatch decision must account for every method")
        warnings = tuple(self.warnings)
        if not all(
            isinstance(warning, str) and bool(warning.strip()) for warning in warnings
        ):
            raise TypeError("warnings must contain only non-empty strings")
        object.__setattr__(self, "applicable_methods", methods)
        object.__setattr__(self, "rejected_methods", MappingProxyType(rejected))
        object.__setattr__(self, "warnings", warnings)

    def is_applicable(self, method: TopologyMethod) -> bool:
        """Return whether ``method`` is valid for this context."""
        if not isinstance(method, TopologyMethod):
            raise TypeError("method must be a TopologyMethod")
        return method in self.applicable_methods

    def require(self, method: TopologyMethod) -> None:
        """Raise with all reasons if ``method`` is not applicable."""
        if not isinstance(method, TopologyMethod):
            raise TypeError("method must be a TopologyMethod")
        if method in self.applicable_methods:
            return
        reasons = "; ".join(self.rejected_methods[method])
        raise ValueError(f"topology method {method.value} is not applicable: {reasons}")

    def to_observable_record(self) -> ObservableRecord:
        """Return a reproducible record of the dispatch decision."""
        return ObservableRecord(
            kind="topology_dispatch",
            scalars={
                "physical_dimension": self.context.physical_dimension,
                "embedding_dimension": self.context.embedding_dimension,
                "applicable_method_count": len(self.applicable_methods),
            },
            metadata={
                "az_class": self.context.classification.altland_zirnbauer_class.value,
                "capabilities": sorted(
                    capability.value for capability in self.context.capabilities
                ),
                "applicable_methods": [
                    method.value for method in self.applicable_methods
                ],
                "rejected_methods": {
                    method.value: list(reasons)
                    for method, reasons in self.rejected_methods.items()
                },
                "warnings": list(self.warnings),
                "dimension_semantics": "explicit_physical_not_inferred_from_embedding",
            },
        )


@dataclass(frozen=True, slots=True)
class _TopologyMethodSpecification:
    physical_dimension: int
    az_classes: frozenset[AltlandZirnbauerClass]
    required_capabilities: frozenset[TopologyCapability]


_METHOD_SPECIFICATIONS: Mapping[TopologyMethod, _TopologyMethodSpecification] = (
    MappingProxyType(
        {
            TopologyMethod.PFAFFIAN_1D: _TopologyMethodSpecification(
                physical_dimension=1,
                az_classes=frozenset(
                    {AltlandZirnbauerClass.D, AltlandZirnbauerClass.BDI}
                ),
                required_capabilities=frozenset(
                    {
                        TopologyCapability.TRANSLATION_INVARIANT_BULK,
                        TopologyCapability.BULK_GAP_EVIDENCE,
                        TopologyCapability.BLOCH_PARTICLE_HOLE_ENDPOINTS,
                    }
                ),
            ),
            TopologyMethod.REAL_SPACE_WINDING_1D: _TopologyMethodSpecification(
                physical_dimension=1,
                az_classes=frozenset(
                    {
                        AltlandZirnbauerClass.AIII,
                        AltlandZirnbauerClass.BDI,
                        AltlandZirnbauerClass.CII,
                    }
                ),
                required_capabilities=frozenset(
                    {
                        TopologyCapability.BULK_GAP_EVIDENCE,
                        TopologyCapability.CHIRAL_OPERATOR,
                        TopologyCapability.BASIS_COORDINATES,
                        TopologyCapability.BULK_MASK,
                    }
                ),
            ),
            TopologyMethod.BOTT_2D: _TopologyMethodSpecification(
                physical_dimension=2,
                az_classes=frozenset(
                    {
                        AltlandZirnbauerClass.A,
                        AltlandZirnbauerClass.C,
                        AltlandZirnbauerClass.D,
                    }
                ),
                required_capabilities=frozenset(
                    {
                        TopologyCapability.BULK_GAP_EVIDENCE,
                        TopologyCapability.BASIS_COORDINATES,
                        TopologyCapability.COORDINATE_PERIODS,
                    }
                ),
            ),
            TopologyMethod.LOCAL_CHERN_MARKER_2D: _TopologyMethodSpecification(
                physical_dimension=2,
                az_classes=frozenset(
                    {
                        AltlandZirnbauerClass.A,
                        AltlandZirnbauerClass.C,
                        AltlandZirnbauerClass.D,
                    }
                ),
                required_capabilities=frozenset(
                    {
                        TopologyCapability.BULK_GAP_EVIDENCE,
                        TopologyCapability.BASIS_COORDINATES,
                        TopologyCapability.POSITION_AREAS,
                        TopologyCapability.BULK_MASK,
                    }
                ),
            ),
            TopologyMethod.SPECTRAL_LOCALIZER_2D: _TopologyMethodSpecification(
                physical_dimension=2,
                az_classes=frozenset(
                    {
                        AltlandZirnbauerClass.A,
                        AltlandZirnbauerClass.C,
                        AltlandZirnbauerClass.D,
                    }
                ),
                required_capabilities=frozenset(
                    {
                        TopologyCapability.BASIS_COORDINATES,
                        TopologyCapability.LOCALIZER_PROBE,
                    }
                ),
            ),
        }
    )
)


def dispatch_topology_methods(
    context: TopologyDispatchContext,
) -> TopologyDispatchDecision:
    """Select only methods whose dimension, AZ class, and inputs all apply."""
    if not isinstance(context, TopologyDispatchContext):
        raise TypeError("context must be a TopologyDispatchContext")
    applicable: list[TopologyMethod] = []
    rejected: dict[TopologyMethod, tuple[str, ...]] = {}
    symmetry_class = context.classification.altland_zirnbauer_class
    for method in TopologyMethod:
        specification = _METHOD_SPECIFICATIONS[method]
        reasons: list[str] = []
        if context.physical_dimension != specification.physical_dimension:
            reasons.append(
                "requires physical_dimension="
                f"{specification.physical_dimension}, received "
                f"{context.physical_dimension}"
            )
        if symmetry_class not in specification.az_classes:
            supported = ", ".join(
                sorted(value.value for value in specification.az_classes)
            )
            reasons.append(
                f"AZ class {symmetry_class.value} is unsupported; expected {supported}"
            )
        missing = specification.required_capabilities - context.capabilities
        if missing:
            reasons.append(
                "missing capabilities: "
                + ", ".join(sorted(capability.value for capability in missing))
            )
        if reasons:
            rejected[method] = tuple(reasons)
        else:
            applicable.append(method)

    warnings: list[str] = []
    if (
        context.embedding_dimension is not None
        and context.embedding_dimension != context.physical_dimension
    ):
        warnings.append(
            "Embedding and physical dimensions differ; dispatch used only the explicit "
            "physical dimension."
        )
    if not applicable:
        warnings.append(
            "No implemented topology method satisfies the supplied dimension, symmetry, "
            "and capabilities."
        )
    return TopologyDispatchDecision(
        context=context,
        applicable_methods=tuple(applicable),
        rejected_methods=rejected,
        warnings=tuple(warnings),
    )


def dispatch_topology_calculation(
    context: TopologyDispatchContext,
    method: TopologyMethod,
    calculation: Callable[[], TopologyDiagnosticResult],
    *,
    convergence_checked: bool = False,
) -> TopologyResult:
    """Validate, execute, type-check, and unify one topology calculation."""
    decision = dispatch_topology_methods(context)
    decision.require(method)
    if not callable(calculation):
        raise TypeError("calculation must be callable")
    specialized_result = calculation()
    unified = unify_topology_result(
        specialized_result,
        convergence_checked=convergence_checked,
    )
    if unified.method is not method:
        raise ValueError(
            "calculation returned "
            f"{unified.method.value}, but dispatch requested {method.value}"
        )
    return unified


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be at least one")
    return result
