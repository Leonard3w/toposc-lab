from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.evaluation import GeometryEvaluation, integrate_topology_results
from toposc_lab.topology import (
    BottIndexResult,
    NumericalConfidence,
    PfaffianInvariantResult,
    RealSpaceWindingResult,
    TopologyCapability,
    TopologyDispatchContext,
    TopologyDispatchDecision,
    TopologyMethod,
    TopologyResult,
    dispatch_topology_methods,
)
from toposc_lab.topology.symmetry import SymmetryClassification


def _bdi_decision(
    *,
    embedding_dimension: int = 1,
) -> TopologyDispatchDecision:
    context = TopologyDispatchContext(
        physical_dimension=1,
        embedding_dimension=embedding_dimension,
        classification=SymmetryClassification.from_signature(
            time_reversal_square=1,
            particle_hole_square=1,
            chiral_symmetry=True,
        ),
        capabilities=frozenset(
            {
                TopologyCapability.TRANSLATION_INVARIANT_BULK,
                TopologyCapability.BULK_GAP_EVIDENCE,
                TopologyCapability.BLOCH_PARTICLE_HOLE_ENDPOINTS,
                TopologyCapability.CHIRAL_OPERATOR,
                TopologyCapability.BASIS_COORDINATES,
                TopologyCapability.BULK_MASK,
            }
        ),
    )
    return dispatch_topology_methods(context)


def _pfaffian_result() -> PfaffianInvariantResult:
    return PfaffianInvariantResult(
        invariant=-1,
        is_topological=True,
        pfaffian_zero=-1.0 + 0.0j,
        pfaffian_pi=1.0 + 0.0j,
        pfaffian_product=-1.0,
        minimum_endpoint_abs_energy=0.8,
        maximum_particle_hole_residual=1.0e-14,
        maximum_antisymmetry_residual=2.0e-14,
        tolerance=1.0e-10,
    )


def _winding_result(winding_number: int | None = 1) -> RealSpaceWindingResult:
    estimate = 0.4 if winding_number is None else float(winding_number)
    return RealSpaceWindingResult(
        positions=np.asarray([0.0, 1.0]),
        local_marker=np.asarray([estimate, estimate]),
        bulk_mask=np.asarray([True, True]),
        winding_estimate=estimate,
        winding_number=winding_number,
        quantization_error=0.4 if winding_number is None else 0.0,
        is_quantized=winding_number is not None,
        zero_mode_count=0,
        minimum_nonzero_abs_energy=0.8,
        maximum_chiral_residual=1.0e-14,
        marker_imaginary_residual=0.0,
        tolerance=1.0e-10,
        quantization_tolerance=1.0e-3,
    )


def _bott_result() -> BottIndexResult:
    return BottIndexResult(
        bott_estimate=1.0,
        bott_index=1,
        quantization_error=0.0,
        is_quantized=True,
        occupied_state_count=2,
        unoccupied_state_count=2,
        fermi_energy=0.0,
        minimum_fermi_distance=0.7,
        minimum_projected_position_singular_value=0.6,
        minimum_branch_cut_distance=0.5,
        maximum_hermiticity_residual=0.0,
        maximum_unitarity_residual=1.0e-14,
        commutator_eigenphases=np.asarray([0.0, 2.0 * np.pi]),
        coordinate_periods=np.asarray([2.0, 2.0]),
        tolerance=1.0e-10,
        quantization_tolerance=1.0e-6,
    )


def test_specialized_results_are_unified_and_deterministically_ordered() -> None:
    result = integrate_topology_results(
        GeometryEvaluation(),
        [_winding_result(), _pfaffian_result()],
        dispatch_decision=_bdi_decision(),
        convergence_checked=True,
    )

    assert tuple(item.method for item in result.topology) == (
        TopologyMethod.PFAFFIAN_1D,
        TopologyMethod.REAL_SPACE_WINDING_1D,
    )
    assert all(item.is_topological for item in result.topology)
    assert all(item.confidence.convergence_checked for item in result.topology)
    assert not any("cross-validation is available" in item for item in result.warnings)


def test_already_unified_result_retains_its_convergence_status() -> None:
    unified = TopologyResult(
        invariant_value=-1,
        is_topological=True,
        invariant_group="Z2",
        method=TopologyMethod.PFAFFIAN_1D,
        applicability_assumptions=("Valid test assumption.",),
        confidence=NumericalConfidence(
            is_resolved=True,
            is_quantized=True,
            minimum_gap=0.8,
            gap_kind="endpoint_energy_gap",
            quantization_error=0.0,
            maximum_residual=0.0,
            convergence_checked=True,
        ),
        warnings=(),
    )

    result = integrate_topology_results(
        GeometryEvaluation(),
        [unified],
        dispatch_decision=_bdi_decision(),
        convergence_checked=False,
    )

    assert result.topology == (unified,)
    assert result.topology[0].confidence.convergence_checked


def test_single_result_reports_missing_method_and_cross_validation_limit() -> None:
    result = integrate_topology_results(
        GeometryEvaluation(),
        [_pfaffian_result()],
        dispatch_decision=_bdi_decision(),
    )

    assert any(
        "applicable methods without supplied results: real_space_winding_1d" in item
        for item in result.warnings
    )
    assert any("no independent topology cross-validation" in item for item in result.warnings)


def test_inapplicable_result_is_rejected_by_dispatch() -> None:
    with pytest.raises(ValueError, match="physical_dimension=2"):
        integrate_topology_results(
            GeometryEvaluation(),
            [_bott_result()],
            dispatch_decision=_bdi_decision(),
        )


def test_duplicate_methods_are_rejected_after_unification() -> None:
    with pytest.raises(ValueError, match="at most one result per method"):
        integrate_topology_results(
            GeometryEvaluation(),
            [_pfaffian_result(), _pfaffian_result()],
            dispatch_decision=_bdi_decision(),
        )


def test_disagreement_remains_explicit_instead_of_being_collapsed() -> None:
    result = integrate_topology_results(
        GeometryEvaluation(),
        [_pfaffian_result(), _winding_result(winding_number=0)],
        dispatch_decision=_bdi_decision(),
        convergence_checked=True,
    )

    assert tuple(item.is_topological for item in result.topology) == (True, False)
    assert any("disagree on the topological classification" in item for item in result.warnings)


def test_unresolved_result_is_retained_with_warning() -> None:
    result = integrate_topology_results(
        GeometryEvaluation(),
        [_pfaffian_result(), _winding_result(winding_number=None)],
        dispatch_decision=_bdi_decision(),
    )

    assert result.topology[1].is_topological is None
    assert any("numerically unresolved" in item for item in result.warnings)
    assert any("not integer-quantized" in item for item in result.warnings)


def test_dispatch_dimension_warning_and_rejections_are_preserved() -> None:
    result = integrate_topology_results(
        GeometryEvaluation(),
        [_pfaffian_result()],
        dispatch_decision=_bdi_decision(embedding_dimension=2),
    )

    assert any("Embedding and physical dimensions differ" in item for item in result.warnings)
    assert any("bott_2d not applied" in item for item in result.warnings)


def test_no_applicable_method_is_a_valid_explicit_empty_result() -> None:
    decision = dispatch_topology_methods(
        TopologyDispatchContext(
            physical_dimension=3,
            embedding_dimension=3,
            classification=SymmetryClassification.from_signature(
                time_reversal_square=None,
                particle_hole_square=1,
                chiral_symmetry=False,
            ),
            capabilities=frozenset(TopologyCapability),
        )
    )

    result = integrate_topology_results(
        GeometryEvaluation(),
        [],
        dispatch_decision=decision,
    )

    assert result.topology == ()
    assert any("No implemented topology method" in item for item in result.warnings)
    assert all(item.startswith("[topology integration]") for item in result.warnings)


def test_reintegration_replaces_only_prior_topology_warnings() -> None:
    initial = GeometryEvaluation(warnings=("Existing evaluation warning.",))
    first = integrate_topology_results(
        initial,
        [_pfaffian_result()],
        dispatch_decision=_bdi_decision(),
    )
    second = integrate_topology_results(
        first,
        [_pfaffian_result()],
        dispatch_decision=_bdi_decision(),
    )

    assert second.warnings == first.warnings
    assert second.warnings[0] == "Existing evaluation warning."


def test_generator_input_is_supported() -> None:
    result = integrate_topology_results(
        GeometryEvaluation(),
        (item for item in (_pfaffian_result(), _winding_result())),
        dispatch_decision=_bdi_decision(),
    )

    assert len(result.topology) == 2


def test_unsupported_topology_objects_are_rejected() -> None:
    with pytest.raises(TypeError, match="supported specialized"):
        integrate_topology_results(
            GeometryEvaluation(),
            [object()],  # type: ignore[list-item]
            dispatch_decision=_bdi_decision(),
        )
