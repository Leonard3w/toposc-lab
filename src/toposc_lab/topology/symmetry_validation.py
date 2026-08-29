"""Numerical validation of declared tenfold-way symmetries."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from toposc_lab.observables.results import ObservableRecord
from toposc_lab.observables.symmetries import SymmetryCheckResult
from toposc_lab.topology.symmetry import AZ_CLASS_ORDER, SymmetryClassification


@dataclass(frozen=True, slots=True)
class SymmetryOperators:
    """Numerical unitary parts of declared symmetry operators.

    Antiunitary symmetries are represented as ``T = U_T K`` and
    ``C = U_C K``. Chiral symmetry is the unitary operator ``S``.
    """

    time_reversal: np.ndarray | None = None
    particle_hole: np.ndarray | None = None
    chiral: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("time_reversal", "particle_hole", "chiral"):
            operator = getattr(self, name)
            if operator is not None:
                object.__setattr__(
                    self,
                    name,
                    _immutable_square_matrix(operator, name=name),
                )


@dataclass(frozen=True, slots=True)
class SymmetryValidationResult:
    """Numerical compatibility checks for one declared AZ classification.

    Present symmetries are tested explicitly. An absent declaration means that
    no operator is tested; it does not prove that no additional symmetry
    operator exists. The result therefore validates compatibility with the
    declared class, not uniqueness of that class.
    """

    classification: SymmetryClassification
    checks: tuple[SymmetryCheckResult, ...]
    satisfied: bool
    maximum_residual: float
    tolerance: float

    def to_observable_record(self) -> ObservableRecord:
        """Return per-check residuals and a stable AZ encoding."""
        labels = tuple(symmetry_class.value for symmetry_class in AZ_CLASS_ORDER)
        return ObservableRecord(
            kind="symmetry_validation",
            scalars={
                "satisfied": self.satisfied,
                "maximum_residual": self.maximum_residual,
                "tolerance": self.tolerance,
                "altland_zirnbauer_code": AZ_CLASS_ORDER.index(
                    self.classification.altland_zirnbauer_class
                ),
            },
            arrays={
                "check_satisfied": np.asarray(
                    [check.satisfied for check in self.checks],
                    dtype=bool,
                ),
                "check_residuals": np.asarray(
                    [check.residual for check in self.checks],
                    dtype=float,
                ),
            },
            metadata={
                "check_names": tuple(check.name for check in self.checks),
                "altland_zirnbauer_labels": labels,
            },
        )


def validate_symmetry_classification(
    hamiltonian: np.ndarray,
    classification: SymmetryClassification,
    operators: SymmetryOperators,
    *,
    tolerance: float = 1.0e-10,
) -> SymmetryValidationResult:
    """Validate supplied operators against a declared real-space AZ class.

    This finite-system relation does not implement the momentum reversal
    required when validating a Bloch Hamiltonian ``H(k)`` directly.
    """
    if not isinstance(classification, SymmetryClassification):
        raise TypeError("classification must be a SymmetryClassification")
    if not isinstance(operators, SymmetryOperators):
        raise TypeError("operators must be a SymmetryOperators instance")
    tolerance = _nonnegative_finite_real(tolerance, name="tolerance")
    matrix = _finite_square_matrix(hamiltonian, name="hamiltonian")
    _validate_operator_presence(classification, operators)

    identity = np.eye(matrix.shape[0], dtype=complex)
    checks = [
        _matrix_check(
            "hermiticity",
            matrix - matrix.conj().T,
            tolerance,
        )
    ]

    if classification.time_reversal is not None:
        time_reversal = _operator_for_matrix(
            operators.time_reversal,
            matrix,
            name="time_reversal",
        )
        checks.extend(
            (
                _matrix_check(
                    "time_reversal_unitarity",
                    time_reversal @ time_reversal.conj().T - identity,
                    tolerance,
                ),
                _matrix_check(
                    "time_reversal_square",
                    time_reversal @ time_reversal.conj()
                    - classification.time_reversal.square * identity,
                    tolerance,
                ),
                _matrix_check(
                    "time_reversal_relation",
                    time_reversal @ matrix.conj() @ time_reversal.conj().T
                    - matrix,
                    tolerance,
                ),
            )
        )

    if classification.particle_hole is not None:
        particle_hole = _operator_for_matrix(
            operators.particle_hole,
            matrix,
            name="particle_hole",
        )
        checks.extend(
            (
                _matrix_check(
                    "particle_hole_unitarity",
                    particle_hole @ particle_hole.conj().T - identity,
                    tolerance,
                ),
                _matrix_check(
                    "particle_hole_square",
                    particle_hole @ particle_hole.conj()
                    - classification.particle_hole.square * identity,
                    tolerance,
                ),
                _matrix_check(
                    "particle_hole_relation",
                    particle_hole @ matrix.conj() @ particle_hole.conj().T
                    + matrix,
                    tolerance,
                ),
            )
        )

    if classification.chiral_symmetry:
        chiral = _operator_for_matrix(
            operators.chiral,
            matrix,
            name="chiral",
        )
        checks.extend(
            (
                _matrix_check(
                    "chiral_unitarity",
                    chiral @ chiral.conj().T - identity,
                    tolerance,
                ),
                _matrix_check(
                    "chiral_square",
                    chiral @ chiral - identity,
                    tolerance,
                ),
                _matrix_check(
                    "chiral_relation",
                    chiral @ matrix @ chiral.conj().T + matrix,
                    tolerance,
                ),
            )
        )

    check_tuple = tuple(checks)
    return SymmetryValidationResult(
        classification=classification,
        checks=check_tuple,
        satisfied=all(check.satisfied for check in check_tuple),
        maximum_residual=max(check.residual for check in check_tuple),
        tolerance=tolerance,
    )


def _validate_operator_presence(
    classification: SymmetryClassification,
    operators: SymmetryOperators,
) -> None:
    expected = {
        "time_reversal": classification.time_reversal is not None,
        "particle_hole": classification.particle_hole is not None,
        "chiral": classification.chiral_symmetry,
    }
    for name, is_expected in expected.items():
        is_present = getattr(operators, name) is not None
        if is_expected and not is_present:
            raise ValueError(f"declared {name} symmetry requires an operator")
        if not is_expected and is_present:
            raise ValueError(f"operator supplied for absent {name} symmetry")


def _operator_for_matrix(
    operator: np.ndarray | None,
    matrix: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    if operator is None:  # Guarded by _validate_operator_presence.
        raise RuntimeError(f"missing validated {name} operator")
    if operator.shape != matrix.shape:
        raise ValueError(f"{name} operator must have shape {matrix.shape}")
    return operator


def _matrix_check(
    name: str,
    difference: np.ndarray,
    tolerance: float,
) -> SymmetryCheckResult:
    residual = float(np.max(np.abs(difference)))
    return SymmetryCheckResult(
        name=name,
        satisfied=residual <= tolerance,
        residual=residual,
        tolerance=tolerance,
    )


def _immutable_square_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    values = _finite_square_matrix(matrix, name=f"{name} operator").copy()
    values.setflags(write=False)
    return values


def _finite_square_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=complex)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.size == 0:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _nonnegative_finite_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value
