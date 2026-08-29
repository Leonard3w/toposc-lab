import numpy as np
import pytest

from toposc_lab.observables.results import StandardizedObservable
from toposc_lab.topology import (
    RealSpaceWindingResult,
    TopologyCapability,
    TopologyDispatchContext,
    TopologyMethod,
    dispatch_topology_calculation,
    dispatch_topology_methods,
)
from toposc_lab.topology.symmetry import SymmetryClassification


def _classification(
    az_class: str,
) -> SymmetryClassification:
    signatures = {
        "A": (None, None, False),
        "AII": (-1, None, False),
        "BDI": (1, 1, True),
        "D": (None, 1, False),
    }
    time_reversal, particle_hole, chiral = signatures[az_class]
    return SymmetryClassification.from_signature(
        time_reversal_square=time_reversal,
        particle_hole_square=particle_hole,
        chiral_symmetry=chiral,
    )


def _one_dimensional_bdi_context(
    *,
    embedding_dimension: int = 1,
) -> TopologyDispatchContext:
    return TopologyDispatchContext(
        physical_dimension=1,
        embedding_dimension=embedding_dimension,
        classification=_classification("BDI"),
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


def _two_dimensional_class_a_context(
    *,
    embedding_dimension: int = 2,
) -> TopologyDispatchContext:
    return TopologyDispatchContext(
        physical_dimension=2,
        embedding_dimension=embedding_dimension,
        classification=_classification("A"),
        capabilities=frozenset(
            {
                TopologyCapability.BULK_GAP_EVIDENCE,
                TopologyCapability.BASIS_COORDINATES,
                TopologyCapability.BULK_MASK,
                TopologyCapability.COORDINATE_PERIODS,
                TopologyCapability.POSITION_AREAS,
                TopologyCapability.LOCALIZER_PROBE,
            }
        ),
    )


def test_one_dimensional_bdi_dispatch_selects_only_1d_methods() -> None:
    decision = dispatch_topology_methods(_one_dimensional_bdi_context())

    assert decision.applicable_methods == (
        TopologyMethod.PFAFFIAN_1D,
        TopologyMethod.REAL_SPACE_WINDING_1D,
    )
    assert not decision.is_applicable(TopologyMethod.BOTT_2D)
    assert "physical_dimension=2" in decision.rejected_methods[
        TopologyMethod.BOTT_2D
    ][0]


def test_two_dimensional_class_a_dispatch_selects_only_2d_chern_methods() -> None:
    decision = dispatch_topology_methods(_two_dimensional_class_a_context())

    assert decision.applicable_methods == (
        TopologyMethod.BOTT_2D,
        TopologyMethod.LOCAL_CHERN_MARKER_2D,
        TopologyMethod.SPECTRAL_LOCALIZER_2D,
    )
    assert not decision.is_applicable(TopologyMethod.PFAFFIAN_1D)
    assert not decision.is_applicable(TopologyMethod.REAL_SPACE_WINDING_1D)


def test_embedding_dimension_never_overrides_physical_dimension() -> None:
    ring_in_plane = dispatch_topology_methods(
        _one_dimensional_bdi_context(embedding_dimension=2)
    )
    surface_in_space = dispatch_topology_methods(
        _two_dimensional_class_a_context(embedding_dimension=3)
    )

    assert ring_in_plane.is_applicable(TopologyMethod.PFAFFIAN_1D)
    assert not ring_in_plane.is_applicable(TopologyMethod.BOTT_2D)
    assert surface_in_space.is_applicable(TopologyMethod.BOTT_2D)
    assert all("Embedding and physical" in item for item in ring_in_plane.warnings)
    assert all("Embedding and physical" in item for item in surface_in_space.warnings)


def test_missing_inputs_reject_method_with_explicit_capabilities() -> None:
    context = TopologyDispatchContext(
        physical_dimension=1,
        classification=_classification("BDI"),
        capabilities=frozenset(
            {
                TopologyCapability.BULK_GAP_EVIDENCE,
                TopologyCapability.CHIRAL_OPERATOR,
                TopologyCapability.BASIS_COORDINATES,
                TopologyCapability.BULK_MASK,
            }
        ),
    )

    decision = dispatch_topology_methods(context)

    assert decision.applicable_methods == (TopologyMethod.REAL_SPACE_WINDING_1D,)
    pfaffian_reasons = " ".join(
        decision.rejected_methods[TopologyMethod.PFAFFIAN_1D]
    )
    assert "bloch_particle_hole_endpoints" in pfaffian_reasons
    assert "translation_invariant_bulk" in pfaffian_reasons


def test_az_class_filters_dimensionally_valid_methods() -> None:
    class_d = TopologyDispatchContext(
        physical_dimension=1,
        classification=_classification("D"),
        capabilities=_one_dimensional_bdi_context().capabilities,
    )

    decision = dispatch_topology_methods(class_d)

    assert decision.applicable_methods == (TopologyMethod.PFAFFIAN_1D,)
    winding_reasons = " ".join(
        decision.rejected_methods[TopologyMethod.REAL_SPACE_WINDING_1D]
    )
    assert "AZ class D is unsupported" in winding_reasons


def test_unsupported_dimension_or_symmetry_returns_no_method() -> None:
    three_dimensional = TopologyDispatchContext(
        physical_dimension=3,
        classification=_classification("D"),
        capabilities=frozenset(TopologyCapability),
    )
    class_aii_2d = TopologyDispatchContext(
        physical_dimension=2,
        classification=_classification("AII"),
        capabilities=frozenset(TopologyCapability),
    )

    dimension_decision = dispatch_topology_methods(three_dimensional)
    symmetry_decision = dispatch_topology_methods(class_aii_2d)

    assert dimension_decision.applicable_methods == ()
    assert symmetry_decision.applicable_methods == ()
    assert any("No implemented" in item for item in dimension_decision.warnings)
    assert any("No implemented" in item for item in symmetry_decision.warnings)


def test_coordinate_free_2d_context_does_not_apply_coordinate_methods() -> None:
    context = TopologyDispatchContext(
        physical_dimension=2,
        classification=_classification("A"),
        capabilities=frozenset({TopologyCapability.BULK_GAP_EVIDENCE}),
    )

    decision = dispatch_topology_methods(context)

    assert decision.applicable_methods == ()
    for method in (
        TopologyMethod.BOTT_2D,
        TopologyMethod.LOCAL_CHERN_MARKER_2D,
        TopologyMethod.SPECTRAL_LOCALIZER_2D,
    ):
        assert "basis_coordinates" in " ".join(decision.rejected_methods[method])


def test_requiring_inapplicable_method_reports_reasons() -> None:
    decision = dispatch_topology_methods(_one_dimensional_bdi_context())

    with pytest.raises(ValueError, match="physical_dimension=2"):
        decision.require(TopologyMethod.BOTT_2D)


def _resolved_winding() -> RealSpaceWindingResult:
    return RealSpaceWindingResult(
        positions=np.array([0.0]),
        local_marker=np.array([1.0]),
        bulk_mask=np.array([True]),
        winding_estimate=1.0,
        winding_number=1,
        quantization_error=0.0,
        is_quantized=True,
        zero_mode_count=0,
        minimum_nonzero_abs_energy=1.0,
        maximum_chiral_residual=0.0,
        marker_imaginary_residual=0.0,
        tolerance=1.0e-10,
        quantization_tolerance=1.0e-3,
    )


def test_guarded_calculation_returns_unified_result() -> None:
    result = dispatch_topology_calculation(
        _one_dimensional_bdi_context(),
        TopologyMethod.REAL_SPACE_WINDING_1D,
        _resolved_winding,
        convergence_checked=True,
    )

    assert result.method is TopologyMethod.REAL_SPACE_WINDING_1D
    assert result.invariant_value == 1
    assert result.confidence.convergence_checked


def test_inapplicable_calculation_is_not_executed() -> None:
    executed = False

    def calculation() -> RealSpaceWindingResult:
        nonlocal executed
        executed = True
        return _resolved_winding()

    with pytest.raises(ValueError, match="physical_dimension=2"):
        dispatch_topology_calculation(
            _one_dimensional_bdi_context(),
            TopologyMethod.BOTT_2D,
            calculation,
        )

    assert not executed


def test_calculation_result_must_match_dispatched_method() -> None:
    with pytest.raises(ValueError, match="dispatch requested pfaffian_1d"):
        dispatch_topology_calculation(
            _one_dimensional_bdi_context(),
            TopologyMethod.PFAFFIAN_1D,
            _resolved_winding,
        )


def test_dispatch_decision_has_standardized_output() -> None:
    decision = dispatch_topology_methods(_two_dimensional_class_a_context())

    assert isinstance(decision, StandardizedObservable)
    record = decision.to_observable_record()
    assert record.kind == "topology_dispatch"
    assert record.scalars["physical_dimension"] == 2
    assert record.scalars["applicable_method_count"] == 3
    assert record.metadata["applicable_methods"] == [
        "bott_2d",
        "local_chern_marker_2d",
        "spectral_localizer_2d",
    ]
    assert record.metadata["dimension_semantics"].startswith("explicit_physical")


@pytest.mark.parametrize("physical_dimension", [True, 0, -1, 1.5])
def test_dispatch_context_rejects_invalid_physical_dimension(
    physical_dimension: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        TopologyDispatchContext(
            physical_dimension=physical_dimension,  # type: ignore[arg-type]
            classification=_classification("A"),
        )
