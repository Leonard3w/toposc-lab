"""Fixed-scaffold edge-locus crossover for compatible geometry genomes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from numbers import Integral
from typing import Any

import numpy as np

from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.geometry import GeometryDimension, GeometryEdge, GeometryFace
from toposc_lab.search.geometry_genome import GeometryGenome, geometry_from_genome
from toposc_lab.search.mutation_validity import (
    MutationValidityPolicy,
    MutationValidityReport,
    validate_geometry_mutation,
)

FIXED_SCAFFOLD_EDGE_CROSSOVER_VERSION = 1
FIXED_SCAFFOLD_EDGE_CROSSOVER_RNG_ALGORITHM = "numpy.random.PCG64"

_CROSSOVER_WARNINGS = (
    (
        "Crossover is an engineering search operation, not scientific evidence or "
        "a discovery claim."
    ),
    (
        "The operator is defined only for exactly aligned explicit spatial scaffolds; "
        "it is not a general graph or subgraph crossover."
    ),
    (
        "Every inherited edge allele retains its parent's complete orientation and "
        "attributes; no displacement or physical meaning is inferred."
    ),
    (
        "Candidate validity is reported without repair, retry, filtering, or fitness "
        "evaluation."
    ),
    (
        "Resource fairness, model compatibility, topology, robustness, diversity, "
        "and next-generation assembly remain separate contracts."
    ),
)


@dataclass(frozen=True, slots=True)
class CrossoverCompatibilityIssue:
    """One stable reason two parents lack a shared physical site scaffold."""

    code: str
    message: str
    path: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.isidentifier():
            raise ValueError("compatibility issue code must be a Python identifier")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("compatibility issue message must be non-empty")
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("compatibility issue path must be non-empty")
        object.__setattr__(self, "message", self.message.strip())
        object.__setattr__(self, "path", self.path.strip())


@dataclass(frozen=True, slots=True)
class CrossoverCompatibilityReport:
    """Structured fixed-scaffold compatibility result without parent alignment."""

    issues: tuple[CrossoverCompatibilityIssue, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.issues, (str, bytes, bytearray)) or not isinstance(
            self.issues,
            Iterable,
        ):
            raise TypeError("issues must be an iterable of compatibility issues")
        issues = tuple(self.issues)
        if not all(isinstance(issue, CrossoverCompatibilityIssue) for issue in issues):
            raise TypeError("issues must contain only CrossoverCompatibilityIssue values")
        object.__setattr__(self, "issues", issues)

    @property
    def is_compatible(self) -> bool:
        """Whether the parents admit the fixed-scaffold operator."""
        return not self.issues

    def raise_for_errors(self) -> None:
        """Raise ``IncompatibleCrossoverParentsError`` when incompatible."""
        if self.issues:
            raise IncompatibleCrossoverParentsError(self)


class IncompatibleCrossoverParentsError(ValueError):
    """Raised when fixed-scaffold crossover cannot align two parents exactly."""

    def __init__(self, report: CrossoverCompatibilityReport) -> None:
        if not isinstance(report, CrossoverCompatibilityReport):
            raise TypeError("report must be a CrossoverCompatibilityReport")
        if report.is_compatible:
            raise ValueError("an incompatible-parent error requires compatibility issues")
        self.report = report
        details = "; ".join(
            f"{issue.code} at {issue.path}: {issue.message}" for issue in report.issues
        )
        super().__init__(f"fixed-scaffold crossover parents are incompatible: {details}")


class CrossoverParent(str, Enum):
    """Parent supplying the first candidate's allele at one differing locus."""

    FIRST = "first"
    SECOND = "second"


@dataclass(frozen=True, slots=True)
class EdgeLocusDecision:
    """Seeded inheritance decision for one differing undirected edge locus."""

    edge_locus: tuple[int, int]
    first_candidate_source: CrossoverParent

    def __post_init__(self) -> None:
        if not isinstance(self.edge_locus, tuple) or len(self.edge_locus) != 2:
            raise TypeError("edge_locus must be a pair of site indices")
        first = _nonnegative_integer(self.edge_locus[0], name="edge_locus[0]")
        second = _nonnegative_integer(self.edge_locus[1], name="edge_locus[1]")
        if first >= second:
            raise ValueError("edge_locus must be in strictly ascending endpoint order")
        if not isinstance(self.first_candidate_source, CrossoverParent):
            raise TypeError("first_candidate_source must be CrossoverParent")
        object.__setattr__(self, "edge_locus", (first, second))

    @property
    def second_candidate_source(self) -> CrossoverParent:
        """Complementary source used by the second candidate."""
        if self.first_candidate_source is CrossoverParent.FIRST:
            return CrossoverParent.SECOND
        return CrossoverParent.FIRST


@dataclass(frozen=True, slots=True)
class GeometryCrossoverResult:
    """Auditable complementary candidates without next-generation construction."""

    first_parent: GeometryGenome
    second_parent: GeometryGenome
    seed: int
    validity_policy: MutationValidityPolicy
    decisions: tuple[EdgeLocusDecision, ...]
    first_candidate: GeometryGenome
    second_candidate: GeometryGenome
    first_validity: MutationValidityReport
    second_validity: MutationValidityReport
    rng_algorithm: str = field(
        default=FIXED_SCAFFOLD_EDGE_CROSSOVER_RNG_ALGORITHM,
        init=False,
    )
    version: int = field(default=FIXED_SCAFFOLD_EDGE_CROSSOVER_VERSION, init=False)
    warnings: tuple[str, ...] = field(default=_CROSSOVER_WARNINGS, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.first_parent, GeometryGenome):
            raise TypeError("first_parent must be a GeometryGenome")
        if not isinstance(self.second_parent, GeometryGenome):
            raise TypeError("second_parent must be a GeometryGenome")
        seed = _nonnegative_integer(self.seed, name="seed")
        if not isinstance(self.validity_policy, MutationValidityPolicy):
            raise TypeError("validity_policy must be MutationValidityPolicy")
        validate_fixed_scaffold_crossover_parents(
            self.first_parent,
            self.second_parent,
        ).raise_for_errors()
        if isinstance(self.decisions, (str, bytes, bytearray)) or not isinstance(
            self.decisions,
            Iterable,
        ):
            raise TypeError("decisions must be an iterable of EdgeLocusDecision values")
        decisions = tuple(self.decisions)
        if not decisions:
            raise ValueError("decisions must contain at least one differing edge locus")
        if not all(isinstance(item, EdgeLocusDecision) for item in decisions):
            raise TypeError("decisions must contain only EdgeLocusDecision values")
        if not isinstance(self.first_candidate, GeometryGenome) or not isinstance(
            self.second_candidate,
            GeometryGenome,
        ):
            raise TypeError("candidates must be GeometryGenome values")
        if not isinstance(self.first_validity, MutationValidityReport) or not isinstance(
            self.second_validity,
            MutationValidityReport,
        ):
            raise TypeError("candidate validity values must be MutationValidityReport")

        expected = _perform_fixed_scaffold_edge_crossover(
            self.first_parent,
            self.second_parent,
            seed=seed,
            validity_policy=self.validity_policy,
        )
        expected_decisions, first_candidate, second_candidate, first_report, second_report = (
            expected
        )
        if decisions != expected_decisions:
            raise ValueError("decisions do not match the seeded edge-locus crossover")
        if _genome_id(self.first_candidate) != _genome_id(first_candidate):
            raise ValueError("first_candidate does not match the crossover decisions")
        if _genome_id(self.second_candidate) != _genome_id(second_candidate):
            raise ValueError("second_candidate does not match the crossover decisions")
        if self.first_validity != first_report or self.second_validity != second_report:
            raise ValueError("candidate validity reports do not match the configured policy")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "decisions", decisions)

    @property
    def candidates(self) -> tuple[GeometryGenome, GeometryGenome]:
        """Complementary candidate genomes in stable first/second order."""
        return (self.first_candidate, self.second_candidate)

    @property
    def validity_reports(self) -> tuple[MutationValidityReport, MutationValidityReport]:
        """Policy reports corresponding exactly to ``candidates``."""
        return (self.first_validity, self.second_validity)

    @property
    def all_candidates_valid(self) -> bool:
        """Whether both candidates satisfy the supplied validity policy."""
        return self.first_validity.is_valid and self.second_validity.is_valid


def validate_fixed_scaffold_crossover_parents(
    first_parent: GeometryGenome,
    second_parent: GeometryGenome,
) -> CrossoverCompatibilityReport:
    """Check exact physical-site alignment without inventing a correspondence."""
    if not isinstance(first_parent, GeometryGenome):
        raise TypeError("first_parent must be a GeometryGenome")
    if not isinstance(second_parent, GeometryGenome):
        raise TypeError("second_parent must be a GeometryGenome")

    issues: list[CrossoverCompatibilityIssue] = []
    for role, parent in (("first_parent", first_parent), ("second_parent", second_parent)):
        try:
            geometry_from_genome(parent)
        except (TypeError, ValueError) as error:
            issues.append(
                _issue(
                    f"invalid_{role}",
                    str(error).strip() or "parent cannot materialize a valid geometry",
                    role,
                )
            )
    if issues:
        return CrossoverCompatibilityReport(tuple(issues))

    if first_parent.coordinates is None:
        issues.append(
            _issue(
                "missing_first_parent_coordinates",
                "fixed-scaffold crossover requires explicit site coordinates",
                "first_parent.coordinates",
            )
        )
    if second_parent.coordinates is None:
        issues.append(
            _issue(
                "missing_second_parent_coordinates",
                "fixed-scaffold crossover requires explicit site coordinates",
                "second_parent.coordinates",
            )
        )
    _append_mismatch(
        issues,
        matches=first_parent.n_sites == second_parent.n_sites,
        code="site_count_mismatch",
        message="parents must contain the same indexed physical sites",
        path="parents.n_sites",
    )
    _append_mismatch(
        issues,
        matches=_exact_array_equal(first_parent.coordinates, second_parent.coordinates),
        code="coordinates_mismatch",
        message="parent coordinate tables must match exactly, including row order",
        path="parents.coordinates",
    )
    _append_mismatch(
        issues,
        matches=first_parent.embedding_dimension == second_parent.embedding_dimension,
        code="embedding_dimension_mismatch",
        message="parent embedding dimensions must match",
        path="parents.embedding_dimension",
    )
    _append_mismatch(
        issues,
        matches=first_parent.boundary_sites == second_parent.boundary_sites,
        code="boundary_sites_mismatch",
        message="parent boundary-site declarations must match",
        path="parents.boundary_sites",
    )
    _append_mismatch(
        issues,
        matches=first_parent.boundary_components == second_parent.boundary_components,
        code="boundary_components_mismatch",
        message="parent boundary-component records and order must match",
        path="parents.boundary_components",
    )
    _append_mismatch(
        issues,
        matches=first_parent.site_types == second_parent.site_types,
        code="site_types_mismatch",
        message="parent site-type columns must match",
        path="parents.site_types",
    )
    _append_mismatch(
        issues,
        matches=_dimension_records_equal(
            first_parent.dimension_records,
            second_parent.dimension_records,
        ),
        code="dimension_records_mismatch",
        message="parent sourced dimension records and order must match exactly",
        path="parents.dimension_records",
    )
    _append_mismatch(
        issues,
        matches=first_parent.rooted_tree == second_parent.rooted_tree,
        code="rooted_tree_mismatch",
        message="parent rooted-tree structures must match",
        path="parents.rooted_tree",
    )
    _append_mismatch(
        issues,
        matches=_exact_value_equal(first_parent.metadata, second_parent.metadata),
        code="metadata_mismatch",
        message="parent geometry metadata must match exactly",
        path="parents.metadata",
    )
    _append_mismatch(
        issues,
        matches=_faces_equal(first_parent.faces, second_parent.faces),
        code="faces_mismatch",
        message="parent faces, order, and face metadata must match exactly",
        path="parents.faces",
    )
    if not _differing_edge_loci(first_parent, second_parent):
        issues.append(
            _issue(
                "no_differing_edge_locus",
                "parents must differ at one or more complete edge loci",
                "parents.edges",
            )
        )
    return CrossoverCompatibilityReport(tuple(issues))


def fixed_scaffold_edge_crossover(
    first_parent: GeometryGenome,
    second_parent: GeometryGenome,
    *,
    seed: int,
    validity_policy: MutationValidityPolicy | None = None,
) -> GeometryCrossoverResult:
    """Create two complementary candidates from aligned complete edge alleles.

    One PCG64 decision is drawn for every differing undirected edge locus in
    ascending endpoint order. The first candidate receives the selected parent
    allele and the second receives the complementary allele. An allele is the
    full oriented ``GeometryEdge`` record or absence of an edge.
    """
    if not isinstance(first_parent, GeometryGenome):
        raise TypeError("first_parent must be a GeometryGenome")
    if not isinstance(second_parent, GeometryGenome):
        raise TypeError("second_parent must be a GeometryGenome")
    prepared_seed = _nonnegative_integer(seed, name="seed")
    if validity_policy is None:
        validity_policy = MutationValidityPolicy()
    elif not isinstance(validity_policy, MutationValidityPolicy):
        raise TypeError("validity_policy must be MutationValidityPolicy or None")

    compatibility = validate_fixed_scaffold_crossover_parents(
        first_parent,
        second_parent,
    )
    compatibility.raise_for_errors()
    decisions, first_candidate, second_candidate, first_report, second_report = (
        _perform_fixed_scaffold_edge_crossover(
            first_parent,
            second_parent,
            seed=prepared_seed,
            validity_policy=validity_policy,
        )
    )
    return GeometryCrossoverResult(
        first_parent=first_parent,
        second_parent=second_parent,
        seed=prepared_seed,
        validity_policy=validity_policy,
        decisions=decisions,
        first_candidate=first_candidate,
        second_candidate=second_candidate,
        first_validity=first_report,
        second_validity=second_report,
    )


def _perform_fixed_scaffold_edge_crossover(
    first_parent: GeometryGenome,
    second_parent: GeometryGenome,
    *,
    seed: int,
    validity_policy: MutationValidityPolicy,
) -> tuple[
    tuple[EdgeLocusDecision, ...],
    GeometryGenome,
    GeometryGenome,
    MutationValidityReport,
    MutationValidityReport,
]:
    first_edges = _edge_locus_map(first_parent)
    second_edges = _edge_locus_map(second_parent)
    all_loci = tuple(sorted(first_edges.keys() | second_edges.keys()))
    differing_loci = frozenset(_differing_edge_loci(first_parent, second_parent))
    random_number_generator = np.random.Generator(np.random.PCG64(seed))
    decisions: list[EdgeLocusDecision] = []
    first_candidate_edges: list[GeometryEdge] = []
    second_candidate_edges: list[GeometryEdge] = []

    for locus in all_loci:
        first_edge = first_edges.get(locus)
        second_edge = second_edges.get(locus)
        if locus not in differing_loci:
            assert first_edge is not None and second_edge is not None
            first_candidate_edges.append(first_edge)
            second_candidate_edges.append(second_edge)
            continue

        source = (CrossoverParent.FIRST, CrossoverParent.SECOND)[
            int(random_number_generator.integers(2))
        ]
        decision = EdgeLocusDecision(locus, source)
        decisions.append(decision)
        if source is CrossoverParent.FIRST:
            selected_first, selected_second = first_edge, second_edge
        else:
            selected_first, selected_second = second_edge, first_edge
        if selected_first is not None:
            first_candidate_edges.append(selected_first)
        if selected_second is not None:
            second_candidate_edges.append(selected_second)

    first_candidate = replace(first_parent, edges=tuple(first_candidate_edges))
    second_candidate = replace(second_parent, edges=tuple(second_candidate_edges))
    geometry_from_genome(first_candidate)
    geometry_from_genome(second_candidate)
    first_report = validate_geometry_mutation(
        first_parent,
        first_candidate,
        policy=validity_policy,
    )
    second_report = validate_geometry_mutation(
        second_parent,
        second_candidate,
        policy=validity_policy,
    )
    return (
        tuple(decisions),
        first_candidate,
        second_candidate,
        first_report,
        second_report,
    )


def _differing_edge_loci(
    first_parent: GeometryGenome,
    second_parent: GeometryGenome,
) -> tuple[tuple[int, int], ...]:
    first_edges = _edge_locus_map(first_parent)
    second_edges = _edge_locus_map(second_parent)
    return tuple(
        locus
        for locus in sorted(first_edges.keys() | second_edges.keys())
        if not _edge_record_equal(first_edges.get(locus), second_edges.get(locus))
    )


def _edge_locus_map(genome: GeometryGenome) -> dict[tuple[int, int], GeometryEdge]:
    return {_edge_locus(edge): edge for edge in genome.edges}


def _edge_locus(edge: GeometryEdge) -> tuple[int, int]:
    return (
        (edge.source, edge.target)
        if edge.source < edge.target
        else (edge.target, edge.source)
    )


def _edge_record_equal(first: GeometryEdge | None, second: GeometryEdge | None) -> bool:
    if first is None or second is None:
        return first is second
    return (
        first.source == second.source
        and first.target == second.target
        and first.edge_type == second.edge_type
        and first.boundary_crossing is second.boundary_crossing
        and _exact_value_equal(first.displacement, second.displacement)
        and _exact_value_equal(first.metadata, second.metadata)
    )


def _dimension_records_equal(
    first: tuple[GeometryDimension, ...],
    second: tuple[GeometryDimension, ...],
) -> bool:
    return len(first) == len(second) and all(
        first_record.kind == second_record.kind
        and first_record.value.hex() == second_record.value.hex()
        and first_record.scope == second_record.scope
        and first_record.method == second_record.method
        and first_record.exact is second_record.exact
        for first_record, second_record in zip(first, second, strict=True)
    )


def _faces_equal(
    first: tuple[GeometryFace, ...],
    second: tuple[GeometryFace, ...],
) -> bool:
    return len(first) == len(second) and all(
        first_face.sites == second_face.sites
        and first_face.face_type == second_face.face_type
        and _exact_value_equal(first_face.metadata, second_face.metadata)
        for first_face, second_face in zip(first, second, strict=True)
    )


def _exact_array_equal(first: np.ndarray | None, second: np.ndarray | None) -> bool:
    if first is None or second is None:
        return first is second
    return (
        first.dtype == second.dtype
        and first.shape == second.shape
        and first.tobytes(order="C") == second.tobytes(order="C")
    )


def _exact_value_equal(first: Any, second: Any) -> bool:
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return (
            isinstance(first, np.ndarray)
            and isinstance(second, np.ndarray)
            and _exact_array_equal(first, second)
        )
    if isinstance(first, np.generic) or isinstance(second, np.generic):
        return (
            isinstance(first, np.generic)
            and isinstance(second, np.generic)
            and first.dtype == second.dtype
            and first.tobytes() == second.tobytes()
        )
    if type(first) is not type(second):
        return False
    if isinstance(first, Mapping):
        if tuple(first) != tuple(second):
            return False
        return all(_exact_value_equal(first[key], second[key]) for key in first)
    if isinstance(first, tuple):
        return len(first) == len(second) and all(
            _exact_value_equal(first_item, second_item)
            for first_item, second_item in zip(first, second, strict=True)
        )
    if isinstance(first, float) and isinstance(second, float):
        return first.hex() == second.hex()
    if isinstance(first, complex) and isinstance(second, complex):
        return (
            first.real.hex() == second.real.hex()
            and first.imag.hex() == second.imag.hex()
        )
    return bool(first == second)


def _genome_id(genome: GeometryGenome) -> str:
    return exact_geometry_id(genome.to_geometry())


def _append_mismatch(
    issues: list[CrossoverCompatibilityIssue],
    *,
    matches: bool,
    code: str,
    message: str,
    path: str,
) -> None:
    if not matches:
        issues.append(_issue(code, message, path))


def _issue(code: str, message: str, path: str) -> CrossoverCompatibilityIssue:
    return CrossoverCompatibilityIssue(code=code, message=message, path=path)


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result
