from collections.abc import Iterable

import numpy as np
import pytest

from toposc_lab.observables.results import StandardizedObservable
from toposc_lab.topology import (
    BottIndexResult,
    LocalChernMarkerResult,
    NumericalConfidence,
    PfaffianInvariantResult,
    RealSpaceWindingResult,
    SpectralLocalizerResult,
    TopologyDiagnosticResult,
    TopologyMethod,
    TopologyResult,
    unify_topology_result,
)


def _specialized_results() -> Iterable[tuple[TopologyDiagnosticResult, TopologyMethod]]:
    yield (
        PfaffianInvariantResult(
            invariant=-1,
            is_topological=True,
            pfaffian_zero=-1.0 + 0.0j,
            pfaffian_pi=1.0 + 0.0j,
            pfaffian_product=-1.0,
            minimum_endpoint_abs_energy=1.0,
            maximum_particle_hole_residual=1.0e-14,
            maximum_antisymmetry_residual=2.0e-14,
            tolerance=1.0e-10,
        ),
        TopologyMethod.PFAFFIAN_1D,
    )
    yield (
        RealSpaceWindingResult(
            positions=np.array([0.0, 1.0]),
            local_marker=np.array([1.0, 1.0]),
            bulk_mask=np.array([True, True]),
            winding_estimate=1.0,
            winding_number=1,
            quantization_error=0.0,
            is_quantized=True,
            zero_mode_count=2,
            minimum_nonzero_abs_energy=0.8,
            maximum_chiral_residual=1.0e-14,
            marker_imaginary_residual=0.0,
            tolerance=1.0e-10,
            quantization_tolerance=1.0e-3,
        ),
        TopologyMethod.REAL_SPACE_WINDING_1D,
    )
    yield (
        BottIndexResult(
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
            maximum_unitarity_residual=2.0e-14,
            commutator_eigenphases=np.array([0.0, 2.0 * np.pi]),
            coordinate_periods=np.array([2.0, 2.0]),
            tolerance=1.0e-10,
            quantization_tolerance=1.0e-6,
        ),
        TopologyMethod.BOTT_2D,
    )
    yield (
        LocalChernMarkerResult(
            positions=np.array([[0.0, 0.0]]),
            local_marker=np.array([1.0]),
            position_areas=np.array([1.0]),
            bulk_mask=np.array([True]),
            bulk_chern_estimate=1.0,
            chern_number=1,
            quantization_error=0.0,
            is_quantized=True,
            occupied_state_count=1,
            unoccupied_state_count=1,
            fermi_energy=0.0,
            minimum_fermi_distance=0.6,
            finite_sample_trace_residual=1.0e-14,
            maximum_hermiticity_residual=0.0,
            maximum_projector_residual=1.0e-14,
            tolerance=1.0e-10,
            quantization_tolerance=5.0e-3,
        ),
        TopologyMethod.LOCAL_CHERN_MARKER_2D,
    )
    yield (
        SpectralLocalizerResult(
            probe_position=np.array([0.5, 0.5]),
            energy=0.0,
            kappa=0.2,
            signature=2,
            local_chern_number=1,
            is_invertible=True,
            localizer_gap=0.4,
            positive_eigenvalue_count=3,
            negative_eigenvalue_count=1,
            zero_eigenvalue_count=0,
            minimum_energy_distance=0.2,
            maximum_hamiltonian_hermiticity_residual=0.0,
            maximum_localizer_hermiticity_residual=0.0,
            localizer_eigenvalues=np.array([-1.0, 0.4, 0.8, 1.2]),
            tolerance=1.0e-10,
        ),
        TopologyMethod.SPECTRAL_LOCALIZER_2D,
    )


@pytest.mark.parametrize(
    ("source", "expected_method"),
    tuple(_specialized_results()),
)
def test_all_specialized_results_convert_without_losing_method_identity(
    source: TopologyDiagnosticResult,
    expected_method: TopologyMethod,
) -> None:
    result = unify_topology_result(source)

    assert result.method is expected_method
    assert result.invariant_value in (-1, 1)
    assert result.is_topological is True
    assert result.confidence.is_resolved
    assert result.confidence.is_quantized
    assert result.confidence.minimum_gap is not None
    assert result.confidence.gap_kind is not None
    assert result.applicability_assumptions
    assert result.warnings


def test_pfaffian_uses_z2_group_and_explicit_endpoint_gap_warning() -> None:
    source, _ = next(iter(_specialized_results()))

    result = unify_topology_result(source)

    assert result.invariant_group == "Z2"
    assert any("k=0 and k=pi" in warning for warning in result.warnings)
    assert result.confidence.gap_kind == "endpoint_energy_gap"


def test_unresolved_winding_preserves_estimate_and_warning() -> None:
    source = RealSpaceWindingResult(
        positions=np.array([0.0]),
        local_marker=np.array([0.4]),
        bulk_mask=np.array([True]),
        winding_estimate=0.4,
        winding_number=None,
        quantization_error=0.4,
        is_quantized=False,
        zero_mode_count=0,
        minimum_nonzero_abs_energy=0.1,
        maximum_chiral_residual=0.0,
        marker_imaginary_residual=0.0,
        tolerance=1.0e-10,
        quantization_tolerance=1.0e-3,
    )

    result = unify_topology_result(source)

    assert result.invariant_value is None
    assert result.is_topological is None
    assert not result.confidence.is_resolved
    assert result.confidence.diagnostics["winding_estimate"] == 0.4
    assert any("not integer-quantized" in warning for warning in result.warnings)


def test_convergence_status_is_explicit_and_controls_default_warning() -> None:
    source, _ = next(iter(_specialized_results()))

    unchecked = unify_topology_result(source)
    checked = unify_topology_result(source, convergence_checked=True)

    assert not unchecked.confidence.convergence_checked
    assert any("convergence study" in warning for warning in unchecked.warnings)
    assert checked.confidence.convergence_checked
    assert not any("convergence study" in warning for warning in checked.warnings)


def test_unified_result_has_standardized_dataset_output() -> None:
    source, _ = next(iter(_specialized_results()))
    result = unify_topology_result(source)

    assert isinstance(result, StandardizedObservable)
    record = result.to_observable_record()
    assert record.kind == "topology_result"
    assert record.scalars["invariant_value"] == -1
    assert record.scalars["diagnostic_pfaffian_product"] == -1.0
    assert record.metadata["method"] == "pfaffian_1d"
    assert record.metadata["invariant_group"] == "Z2"
    assert record.metadata["applicability_assumptions"]
    assert record.metadata["warnings"]


def test_confidence_defensively_copies_diagnostics() -> None:
    diagnostics = {"estimate": 1.0}
    confidence = NumericalConfidence(
        is_resolved=True,
        is_quantized=True,
        minimum_gap=0.5,
        gap_kind="test_gap",
        quantization_error=0.0,
        maximum_residual=0.0,
        convergence_checked=True,
        diagnostics=diagnostics,
    )
    diagnostics["estimate"] = 2.0

    assert confidence.diagnostics["estimate"] == 1.0
    with pytest.raises(TypeError):
        confidence.diagnostics["estimate"] = 3.0  # type: ignore[index]


def test_unified_result_rejects_inconsistent_resolution() -> None:
    confidence = NumericalConfidence(
        is_resolved=False,
        is_quantized=False,
        minimum_gap=0.0,
        gap_kind="test_gap",
        quantization_error=0.2,
        maximum_residual=0.0,
        convergence_checked=False,
    )

    with pytest.raises(ValueError, match="must agree"):
        TopologyResult(
            invariant_value=1,
            is_topological=True,
            invariant_group="Z",
            method=TopologyMethod.BOTT_2D,
            applicability_assumptions=("A valid test assumption.",),
            confidence=confidence,
            warnings=(),
        )


def test_confidence_rejects_gap_without_gap_kind() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        NumericalConfidence(
            is_resolved=True,
            is_quantized=True,
            minimum_gap=0.5,
            gap_kind=None,
            quantization_error=0.0,
            maximum_residual=0.0,
            convergence_checked=False,
        )


def test_unify_rejects_unsupported_objects() -> None:
    with pytest.raises(TypeError, match="supported specialized"):
        unify_topology_result(object())  # type: ignore[arg-type]
