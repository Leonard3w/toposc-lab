from __future__ import annotations

from collections.abc import Callable
import numpy as np
import pytest

from toposc_lab.geometry import (
    Geometry,
    GeometryDimension,
    body_centered_cubic,
    chain,
    cubic,
    honeycomb,
    irregular_cluster,
    kagome,
    ring,
    square,
    triangular,
)


def test_geometry_dimension_normalizes_and_preserves_provenance() -> None:
    record = GeometryDimension(
        kind="hausdorff",
        value=np.float64(1.5),
        scope="infinite_family",
        method="  analytic self similarity  ",
        exact=True,
    )

    assert record.value == 1.5
    assert type(record.value) is float
    assert record.method == "analytic self similarity"
    assert record.exact is True


@pytest.mark.parametrize("value", (True, "2", object()))
def test_geometry_dimension_rejects_non_real_values(value: object) -> None:
    with pytest.raises(TypeError, match="real number"):
        GeometryDimension(
            kind="spectral",
            value=value,  # type: ignore[arg-type]
            scope="finite_geometry",
            method="estimate",
        )


@pytest.mark.parametrize("value", (-1.0, np.inf, np.nan))
def test_geometry_dimension_rejects_invalid_numeric_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        GeometryDimension(
            kind="spectral",
            value=value,
            scope="finite_geometry",
            method="estimate",
        )


def test_geometry_dimension_rejects_unknown_kind_scope_or_empty_method() -> None:
    with pytest.raises(ValueError, match="kind"):
        GeometryDimension(
            kind="embedding",  # type: ignore[arg-type]
            value=2.0,
            scope="finite_geometry",
            method="coordinates",
        )
    with pytest.raises(ValueError, match="scope"):
        GeometryDimension(
            kind="box_counting",
            value=1.5,
            scope="unknown",  # type: ignore[arg-type]
            method="fit",
        )
    with pytest.raises(ValueError, match="method"):
        GeometryDimension(
            kind="box_counting",
            value=1.5,
            scope="finite_geometry",
            method="  ",
        )


def test_geometry_rejects_invalid_or_duplicate_dimension_records() -> None:
    record = GeometryDimension(
        kind="lattice",
        value=2.0,
        scope="infinite_family",
        method="translation_rank",
        exact=True,
    )

    with pytest.raises(TypeError, match="GeometryDimension"):
        Geometry(n_sites=1, dimension_records=("2D",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate"):
        Geometry(n_sites=1, dimension_records=(record, record))


@pytest.mark.parametrize(
    ("factory", "expected_dimension"),
    (
        (lambda: chain(3), 1.0),
        (lambda: ring(3), 1.0),
        (lambda: square(2, 2), 2.0),
        (lambda: triangular(2, 2), 2.0),
        (lambda: honeycomb(2, 2), 2.0),
        (lambda: kagome(2, 2), 2.0),
        (lambda: cubic(2, 2, 2), 3.0),
        (lambda: body_centered_cubic(2, 2, 2), 3.0),
    ),
)
def test_regular_generators_use_exact_lattice_dimension_records(
    factory: Callable[[], Geometry],
    expected_dimension: float,
) -> None:
    geometry = factory()

    assert geometry.dimension_records == (
        GeometryDimension(
            kind="lattice",
            value=expected_dimension,
            scope="infinite_family",
            method="translation_rank",
            exact=True,
        ),
    )
    assert "intrinsic_dimension" not in geometry.metadata


def test_embedding_and_lattice_dimensions_remain_independent() -> None:
    geometry = ring(5)

    assert geometry.embedding_dimension == 2
    assert geometry.dimension_records[0].value == 1.0


def test_irregular_reference_does_not_claim_dimension_from_embedding() -> None:
    geometry = irregular_cluster()

    assert geometry.embedding_dimension == 2
    assert geometry.dimension_records == ()
    assert "intrinsic_dimension" not in geometry.metadata


def test_dimension_records_are_retained_by_generator_provenance_wrapper() -> None:
    from toposc_lab.geometry import BUILTIN_GEOMETRY_GENERATORS

    geometry = BUILTIN_GEOMETRY_GENERATORS.generate(
        "square",
        parameters={"n_x": 2, "n_y": 3},
    )

    assert geometry.dimension_records[0].kind == "lattice"
    assert geometry.metadata["generation"]["generator_key"] == "square"
