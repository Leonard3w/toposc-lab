import numpy as np
import pytest

from toposc_lab.observables.results import StandardizedObservable
from toposc_lab.topology.pfaffian import (
    one_dimensional_pfaffian_invariant,
    pfaffian,
)
from toposc_lab.topology.symmetry import SymmetryClassification


def _skew_matrix(upper_triangle: np.ndarray) -> np.ndarray:
    result: np.ndarray = upper_triangle - upper_triangle.T
    return result


@pytest.mark.parametrize("dimension", [2, 4, 6, 8])
def test_pfaffian_square_equals_determinant(dimension: int) -> None:
    random = np.random.default_rng(100 + dimension)
    upper = np.triu(
        random.normal(size=(dimension, dimension))
        + 1.0j * random.normal(size=(dimension, dimension)),
        k=1,
    )
    matrix = _skew_matrix(upper)

    value = pfaffian(matrix, tolerance=0.0)

    assert value**2 == pytest.approx(np.linalg.det(matrix), rel=1.0e-10, abs=1.0e-10)


def test_pfaffian_has_correct_four_by_four_sign_convention() -> None:
    matrix = np.array(
        [
            [0.0, 2.0, 3.0, 5.0],
            [-2.0, 0.0, 7.0, 11.0],
            [-3.0, -7.0, 0.0, 13.0],
            [-5.0, -11.0, -13.0, 0.0],
        ]
    )

    assert pfaffian(matrix) == pytest.approx(2.0 * 13.0 - 3.0 * 11.0 + 5.0 * 7.0)


def _kitaev_endpoint_hamiltonians(
    *,
    chemical_potential: float,
    hopping: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    energy_zero = -chemical_potential - 2.0 * hopping
    energy_pi = -chemical_potential + 2.0 * hopping
    return (
        np.diag([energy_zero, -energy_zero]),
        np.diag([energy_pi, -energy_pi]),
    )


def _bdi_classification() -> SymmetryClassification:
    return SymmetryClassification.from_signature(
        time_reversal_square=1,
        particle_hole_square=1,
        chiral_symmetry=True,
    )


@pytest.mark.parametrize("chemical_potential", [-1.5, 0.0, 1.5])
def test_kitaev_chain_topological_regime_has_negative_invariant(
    chemical_potential: float,
) -> None:
    zero, pi = _kitaev_endpoint_hamiltonians(
        chemical_potential=chemical_potential,
    )

    result = one_dimensional_pfaffian_invariant(
        zero,
        pi,
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        _bdi_classification(),
    )

    assert result.invariant == -1
    assert result.is_topological
    assert result.pfaffian_product < 0.0


@pytest.mark.parametrize("chemical_potential", [-3.0, 3.0])
def test_kitaev_chain_trivial_regime_has_positive_invariant(
    chemical_potential: float,
) -> None:
    zero, pi = _kitaev_endpoint_hamiltonians(
        chemical_potential=chemical_potential,
    )

    result = one_dimensional_pfaffian_invariant(
        zero,
        pi,
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        _bdi_classification(),
    )

    assert result.invariant == 1
    assert not result.is_topological
    assert result.pfaffian_product > 0.0


def test_kitaev_transition_rejects_undefined_invariant() -> None:
    zero, pi = _kitaev_endpoint_hamiltonians(chemical_potential=2.0)

    with pytest.raises(ValueError, match="gap closing"):
        one_dimensional_pfaffian_invariant(
            zero,
            pi,
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            _bdi_classification(),
        )


def test_pfaffian_invariant_rejects_inapplicable_az_class() -> None:
    zero, pi = _kitaev_endpoint_hamiltonians(chemical_potential=0.0)
    class_aii = SymmetryClassification.from_signature(
        time_reversal_square=-1,
        particle_hole_square=None,
        chiral_symmetry=False,
    )

    with pytest.raises(ValueError, match="class D or BDI"):
        one_dimensional_pfaffian_invariant(
            zero,
            pi,
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            class_aii,
        )


def test_pfaffian_invariant_is_also_available_for_class_d() -> None:
    zero, pi = _kitaev_endpoint_hamiltonians(chemical_potential=0.0)
    class_d = SymmetryClassification.from_signature(
        time_reversal_square=None,
        particle_hole_square=1,
        chiral_symmetry=False,
    )

    result = one_dimensional_pfaffian_invariant(
        zero,
        pi,
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        class_d,
    )

    assert result.invariant == -1


def test_pfaffian_invariant_rejects_broken_particle_hole_relation() -> None:
    zero, pi = _kitaev_endpoint_hamiltonians(chemical_potential=0.0)
    pi = pi + 0.1 * np.eye(2)

    with pytest.raises(ValueError, match="violate particle-hole"):
        one_dimensional_pfaffian_invariant(
            zero,
            pi,
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            _bdi_classification(),
        )


def test_pfaffian_invariant_result_has_standardized_output() -> None:
    zero, pi = _kitaev_endpoint_hamiltonians(chemical_potential=0.0)
    result = one_dimensional_pfaffian_invariant(
        zero,
        pi,
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        _bdi_classification(),
    )

    assert isinstance(result, StandardizedObservable)
    record = result.to_observable_record()
    assert record.kind == "pfaffian_invariant_1d"
    assert record.scalars["invariant"] == -1
    assert record.metadata["method"] == "kitaev_majorana_pfaffian"


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(2),
        np.zeros((3, 3)),
        np.array([[0.0, np.nan], [-np.nan, 0.0]]),
    ],
)
def test_pfaffian_rejects_invalid_matrix(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError):
        pfaffian(matrix)
