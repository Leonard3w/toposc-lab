"""Unified, assumption-aware summaries of specialized topology diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from types import MappingProxyType
from typing import Literal, TypeAlias

import numpy as np

from toposc_lab.observables.results import ObservableRecord, ObservableScalar
from toposc_lab.topology.bott import BottIndexResult
from toposc_lab.topology.local_chern import LocalChernMarkerResult
from toposc_lab.topology.pfaffian import PfaffianInvariantResult
from toposc_lab.topology.spectral_localizer import SpectralLocalizerResult
from toposc_lab.topology.winding import RealSpaceWindingResult

InvariantGroup: TypeAlias = Literal["Z", "Z2"]
TopologyDiagnosticResult: TypeAlias = (
    PfaffianInvariantResult
    | RealSpaceWindingResult
    | BottIndexResult
    | LocalChernMarkerResult
    | SpectralLocalizerResult
)


class TopologyMethod(str, Enum):
    """Stable identifiers for implemented invariant calculations."""

    PFAFFIAN_1D = "pfaffian_1d"
    REAL_SPACE_WINDING_1D = "real_space_winding_1d"
    BOTT_2D = "bott_2d"
    LOCAL_CHERN_MARKER_2D = "local_chern_marker_2d"
    SPECTRAL_LOCALIZER_2D = "spectral_localizer_2d"


@dataclass(frozen=True, slots=True)
class NumericalConfidence:
    """Method-independent numerical reliability and convergence information."""

    is_resolved: bool
    is_quantized: bool | None
    minimum_gap: float | None
    gap_kind: str | None
    quantization_error: float | None
    maximum_residual: float
    convergence_checked: bool
    diagnostics: Mapping[str, ObservableScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.is_resolved, bool):
            raise TypeError("is_resolved must be a boolean")
        if self.is_quantized is not None and not isinstance(self.is_quantized, bool):
            raise TypeError("is_quantized must be a boolean or None")
        minimum_gap = _optional_nonnegative_real(self.minimum_gap, name="minimum_gap")
        quantization_error = _optional_nonnegative_real(
            self.quantization_error,
            name="quantization_error",
        )
        maximum_residual = _nonnegative_real(
            self.maximum_residual,
            name="maximum_residual",
        )
        if self.gap_kind is not None and not self.gap_kind.isidentifier():
            raise ValueError("gap_kind must be a Python-style identifier or None")
        if (minimum_gap is None) != (self.gap_kind is None):
            raise ValueError("minimum_gap and gap_kind must be supplied together")
        if not self.is_resolved and self.is_quantized is True:
            raise ValueError("an unresolved invariant cannot be quantized")
        if not isinstance(self.convergence_checked, bool):
            raise TypeError("convergence_checked must be a boolean")

        prepared_diagnostics: dict[str, ObservableScalar] = {}
        for name, value in self.diagnostics.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("diagnostic names must be Python-style identifiers")
            prepared_diagnostics[name] = _observable_scalar(value, name=name)
        object.__setattr__(self, "minimum_gap", minimum_gap)
        object.__setattr__(self, "quantization_error", quantization_error)
        object.__setattr__(self, "maximum_residual", maximum_residual)
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(prepared_diagnostics),
        )


@dataclass(frozen=True, slots=True)
class TopologyResult:
    """Unified topology value with applicability, confidence, and warnings."""

    invariant_value: int | float | None
    is_topological: bool | None
    invariant_group: InvariantGroup
    method: TopologyMethod
    applicability_assumptions: tuple[str, ...]
    confidence: NumericalConfidence
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        invariant_value = _optional_invariant_value(self.invariant_value)
        if self.is_topological is not None and not isinstance(self.is_topological, bool):
            raise TypeError("is_topological must be a boolean or None")
        if self.invariant_group not in ("Z", "Z2"):
            raise ValueError("invariant_group must be 'Z' or 'Z2'")
        if not isinstance(self.method, TopologyMethod):
            raise TypeError("method must be a TopologyMethod")
        if not isinstance(self.confidence, NumericalConfidence):
            raise TypeError("confidence must be NumericalConfidence")
        assumptions = _messages(
            self.applicability_assumptions,
            name="applicability_assumptions",
            allow_empty=False,
        )
        warnings = _messages(self.warnings, name="warnings", allow_empty=True)
        if self.confidence.is_resolved != (invariant_value is not None):
            raise ValueError("is_resolved must agree with invariant_value availability")
        if invariant_value is None and self.is_topological is not None:
            raise ValueError("is_topological must be None for an unresolved invariant")
        if self.confidence.is_resolved and self.confidence.is_quantized is False:
            raise ValueError("a resolved integer invariant cannot be nonquantized")
        object.__setattr__(self, "invariant_value", invariant_value)
        object.__setattr__(self, "applicability_assumptions", assumptions)
        object.__setattr__(self, "warnings", warnings)

    def to_observable_record(self) -> ObservableRecord:
        """Return a standardized, dataset-ready unified topology record."""
        scalars: dict[str, ObservableScalar] = {
            "invariant_value": self.invariant_value,
            "is_topological": self.is_topological,
            "is_resolved": self.confidence.is_resolved,
            "is_quantized": self.confidence.is_quantized,
            "minimum_gap": self.confidence.minimum_gap,
            "quantization_error": self.confidence.quantization_error,
            "maximum_residual": self.confidence.maximum_residual,
            "convergence_checked": self.confidence.convergence_checked,
        }
        scalars.update(
            {
                f"diagnostic_{name}": value
                for name, value in self.confidence.diagnostics.items()
            }
        )
        return ObservableRecord(
            kind="topology_result",
            scalars=scalars,
            metadata={
                "method": self.method.value,
                "invariant_group": self.invariant_group,
                "gap_kind": self.confidence.gap_kind,
                "applicability_assumptions": list(self.applicability_assumptions),
                "warnings": list(self.warnings),
            },
        )


def unify_topology_result(
    result: TopologyDiagnosticResult,
    *,
    convergence_checked: bool = False,
) -> TopologyResult:
    """Convert one specialized topology result to the unified summary."""
    if not isinstance(convergence_checked, bool):
        raise TypeError("convergence_checked must be a boolean")
    if isinstance(result, PfaffianInvariantResult):
        return _unify_pfaffian(result, convergence_checked=convergence_checked)
    if isinstance(result, RealSpaceWindingResult):
        return _unify_winding(result, convergence_checked=convergence_checked)
    if isinstance(result, BottIndexResult):
        return _unify_bott(result, convergence_checked=convergence_checked)
    if isinstance(result, LocalChernMarkerResult):
        return _unify_local_chern(result, convergence_checked=convergence_checked)
    if isinstance(result, SpectralLocalizerResult):
        return _unify_localizer(result, convergence_checked=convergence_checked)
    raise TypeError("result must be a supported specialized topology result")


def _unify_pfaffian(
    result: PfaffianInvariantResult,
    *,
    convergence_checked: bool,
) -> TopologyResult:
    return TopologyResult(
        invariant_value=result.invariant,
        is_topological=result.is_topological,
        invariant_group="Z2",
        method=TopologyMethod.PFAFFIAN_1D,
        applicability_assumptions=(
            "The system is a translation-invariant one-dimensional periodic bulk.",
            "The supplied matrices are H(k=0) and H(k=pi).",
            "The real particle-hole convention has C^2=+1.",
            "A full Brillouin-zone bulk gap is established independently.",
        ),
        confidence=NumericalConfidence(
            is_resolved=True,
            is_quantized=True,
            minimum_gap=result.minimum_endpoint_abs_energy,
            gap_kind="endpoint_energy_gap",
            quantization_error=0.0,
            maximum_residual=max(
                result.maximum_particle_hole_residual,
                result.maximum_antisymmetry_residual,
            ),
            convergence_checked=convergence_checked,
            diagnostics={
                "pfaffian_product": result.pfaffian_product,
            },
        ),
        warnings=_standard_warnings(
            "Only k=0 and k=pi were checked; the result does not establish a full bulk gap.",
            convergence_checked=convergence_checked,
        ),
    )


def _unify_winding(
    result: RealSpaceWindingResult,
    *,
    convergence_checked: bool,
) -> TopologyResult:
    warnings = [
        "The winding sign depends on the position and chiral-eigenspace orientations.",
        "The bulk mask must exclude boundary-contaminated positions.",
    ]
    if result.zero_mode_count:
        warnings.append("Numerical zero modes were omitted from spectral flattening.")
    if not result.is_quantized:
        warnings.append("The selected bulk winding estimate is not integer-quantized.")
    return TopologyResult(
        invariant_value=result.winding_number,
        is_topological=(
            None if result.winding_number is None else result.winding_number != 0
        ),
        invariant_group="Z",
        method=TopologyMethod.REAL_SPACE_WINDING_1D,
        applicability_assumptions=(
            "The system is one-dimensional and has a gapped chiral bulk.",
            "The supplied chiral operator is the physical onsite grading.",
            "The explicit bulk mask represents the thermodynamic interior.",
        ),
        confidence=NumericalConfidence(
            is_resolved=result.winding_number is not None,
            is_quantized=result.is_quantized,
            minimum_gap=result.minimum_nonzero_abs_energy,
            gap_kind="minimum_nonzero_energy",
            quantization_error=result.quantization_error,
            maximum_residual=max(
                result.maximum_chiral_residual,
                result.marker_imaginary_residual,
            ),
            convergence_checked=convergence_checked,
            diagnostics={
                "winding_estimate": result.winding_estimate,
                "zero_mode_count": result.zero_mode_count,
            },
        ),
        warnings=_standard_warnings(
            *warnings,
            convergence_checked=convergence_checked,
        ),
    )


def _unify_bott(
    result: BottIndexResult,
    *,
    convergence_checked: bool,
) -> TopologyResult:
    warnings = [
        "The Bott sign depends on the supplied x-y orientation.",
        "One finite Hamiltonian does not establish a thermodynamic spectral or mobility gap.",
        "Coordinate periods must represent the intended finite geometry.",
    ]
    if not result.is_quantized:
        warnings.append("The Bott trace-log estimate is not integer-quantized.")
    return TopologyResult(
        invariant_value=result.bott_index,
        is_topological=None if result.bott_index is None else result.bott_index != 0,
        invariant_group="Z",
        method=TopologyMethod.BOTT_2D,
        applicability_assumptions=(
            "The system is two-dimensional and belongs to class A, C, or D.",
            "The Fermi level lies in a spectral or mobility gap.",
            "The supplied coordinate periods define valid projected phase operators.",
        ),
        confidence=NumericalConfidence(
            is_resolved=result.bott_index is not None,
            is_quantized=result.is_quantized,
            minimum_gap=result.minimum_fermi_distance,
            gap_kind="finite_fermi_distance",
            quantization_error=result.quantization_error,
            maximum_residual=max(
                result.maximum_hermiticity_residual,
                result.maximum_unitarity_residual,
            ),
            convergence_checked=convergence_checked,
            diagnostics={
                "bott_estimate": result.bott_estimate,
                "minimum_projected_position_singular_value": (
                    result.minimum_projected_position_singular_value
                ),
                "minimum_branch_cut_distance": result.minimum_branch_cut_distance,
            },
        ),
        warnings=_standard_warnings(
            *warnings,
            convergence_checked=convergence_checked,
        ),
    )


def _unify_local_chern(
    result: LocalChernMarkerResult,
    *,
    convergence_checked: bool,
) -> TopologyResult:
    warnings = [
        "The marker sign depends on the supplied x-y orientation.",
        "The bulk mask and position areas must represent the physical interior.",
        "The total marker of a finite sample cancels between bulk and boundary.",
        "One finite Hamiltonian does not establish a thermodynamic spectral or mobility gap.",
    ]
    if not result.is_quantized:
        warnings.append("The selected bulk Chern estimate is not integer-quantized.")
    return TopologyResult(
        invariant_value=result.chern_number,
        is_topological=None if result.chern_number is None else result.chern_number != 0,
        invariant_group="Z",
        method=TopologyMethod.LOCAL_CHERN_MARKER_2D,
        applicability_assumptions=(
            "The system is two-dimensional and belongs to class A, C, or D.",
            "The Fermi level lies in a spectral or mobility gap.",
            "Position areas and the explicit bulk mask approximate the thermodynamic bulk.",
        ),
        confidence=NumericalConfidence(
            is_resolved=result.chern_number is not None,
            is_quantized=result.is_quantized,
            minimum_gap=result.minimum_fermi_distance,
            gap_kind="finite_fermi_distance",
            quantization_error=result.quantization_error,
            maximum_residual=max(
                result.maximum_hermiticity_residual,
                result.maximum_projector_residual,
            ),
            convergence_checked=convergence_checked,
            diagnostics={
                "bulk_chern_estimate": result.bulk_chern_estimate,
                "finite_sample_trace_residual": (
                    result.finite_sample_trace_residual
                ),
            },
        ),
        warnings=_standard_warnings(
            *warnings,
            convergence_checked=convergence_checked,
        ),
    )


def _unify_localizer(
    result: SpectralLocalizerResult,
    *,
    convergence_checked: bool,
) -> TopologyResult:
    warnings = [
        "The local Chern sign depends on the supplied x-y orientation.",
        "The result is local to the supplied position, energy, and kappa.",
        "Kappa must be checked over a physically stable range.",
    ]
    if not result.is_invertible:
        warnings.append("The localizer gap is unresolved, so the local index is undefined.")
    return TopologyResult(
        invariant_value=result.local_chern_number,
        is_topological=(
            None
            if result.local_chern_number is None
            else result.local_chern_number != 0
        ),
        invariant_group="Z",
        method=TopologyMethod.SPECTRAL_LOCALIZER_2D,
        applicability_assumptions=(
            "The system is two-dimensional and belongs to class A, C, or D.",
            "The localizer probe and kappa resolve the intended spatial and energy scale.",
            "A nonzero localizer gap protects the half-signature index.",
        ),
        confidence=NumericalConfidence(
            is_resolved=result.local_chern_number is not None,
            is_quantized=result.is_invertible,
            minimum_gap=result.localizer_gap,
            gap_kind="localizer_gap",
            quantization_error=0.0 if result.is_invertible else None,
            maximum_residual=max(
                result.maximum_hamiltonian_hermiticity_residual,
                result.maximum_localizer_hermiticity_residual,
            ),
            convergence_checked=convergence_checked,
            diagnostics={
                "signature": result.signature,
                "minimum_energy_distance": result.minimum_energy_distance,
                "kappa": result.kappa,
            },
        ),
        warnings=_standard_warnings(
            *warnings,
            convergence_checked=convergence_checked,
        ),
    )


def _standard_warnings(
    *warnings: str,
    convergence_checked: bool,
) -> tuple[str, ...]:
    values = list(warnings)
    if not convergence_checked:
        values.append("No independent size, grid, or parameter-convergence study was supplied.")
    return tuple(values)


def _messages(
    values: tuple[str, ...],
    *,
    name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not all(
        isinstance(value, str) and bool(value.strip()) for value in values
    ):
        raise TypeError(f"{name} must be a tuple of non-empty strings")
    if not allow_empty and not values:
        raise ValueError(f"{name} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _optional_invariant_value(value: int | float | None) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (Integral, Real)):
        raise TypeError("invariant_value must be a finite real number or None")
    if isinstance(value, Integral):
        return int(value)
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("invariant_value must be finite or None")
    return result


def _observable_scalar(value: object, *, name: str) -> ObservableScalar:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if np.isfinite(result):
            return result
    raise ValueError(f"diagnostic {name!r} must be a finite scalar or None")


def _optional_nonnegative_real(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    return _nonnegative_real(value, name=name)


def _nonnegative_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result
