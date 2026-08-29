import numpy as np
import pytest

from toposc_lab.models.bhz_model import BHZModel, BHZModelParameters
from toposc_lab.models.kitaev_chain import KitaevChain, KitaevChainParameters
from toposc_lab.observables.results import StandardizedObservable
from toposc_lab.topology.symmetry import SymmetryClassification
from toposc_lab.topology.symmetry_validation import (
    SymmetryOperators,
    validate_symmetry_classification,
)


def test_bdi_validation_checks_relations_and_operator_squares() -> None:
    hamiltonian = np.diag([1.0, -1.0])
    identity = np.eye(2)
    particle_hole = np.array([[0.0, 1.0], [1.0, 0.0]])
    classification = SymmetryClassification.from_signature(
        time_reversal_square=1,
        particle_hole_square=1,
        chiral_symmetry=True,
    )

    result = validate_symmetry_classification(
        hamiltonian,
        classification,
        SymmetryOperators(
            time_reversal=identity,
            particle_hole=particle_hole,
            chiral=particle_hole,
        ),
    )

    assert result.satisfied
    assert result.maximum_residual == pytest.approx(0.0)
    assert tuple(check.name for check in result.checks) == (
        "hermiticity",
        "time_reversal_unitarity",
        "time_reversal_square",
        "time_reversal_relation",
        "particle_hole_unitarity",
        "particle_hole_square",
        "particle_hole_relation",
        "chiral_unitarity",
        "chiral_square",
        "chiral_relation",
    )


def test_wrong_antiunitary_square_fails_declared_class() -> None:
    zero_hamiltonian = np.zeros((2, 2))
    square_minus_one = np.array([[0.0, 1.0], [-1.0, 0.0]])
    particle_hole = np.array([[0.0, 1.0], [1.0, 0.0]])
    classification = SymmetryClassification.from_signature(
        time_reversal_square=1,
        particle_hole_square=1,
        chiral_symmetry=True,
    )

    result = validate_symmetry_classification(
        zero_hamiltonian,
        classification,
        SymmetryOperators(
            time_reversal=square_minus_one,
            particle_hole=particle_hole,
            chiral=np.eye(2),
        ),
    )

    checks = {check.name: check for check in result.checks}
    assert not result.satisfied
    assert not checks["time_reversal_square"].satisfied
    assert checks["time_reversal_relation"].satisfied


def test_nonunitary_operator_is_reported_as_failed_check() -> None:
    classification = SymmetryClassification.from_signature(
        time_reversal_square=1,
        particle_hole_square=None,
        chiral_symmetry=False,
    )

    result = validate_symmetry_classification(
        np.zeros((2, 2)),
        classification,
        SymmetryOperators(time_reversal=2.0 * np.eye(2)),
    )

    checks = {check.name: check for check in result.checks}
    assert not result.satisfied
    assert not checks["time_reversal_unitarity"].satisfied
    assert not checks["time_reversal_square"].satisfied


def test_nonhermitian_hamiltonian_is_reported_as_failed_check() -> None:
    classification = SymmetryClassification.from_signature(
        time_reversal_square=None,
        particle_hole_square=None,
        chiral_symmetry=False,
    )

    result = validate_symmetry_classification(
        np.array([[0.0, 1.0], [0.0, 0.0]]),
        classification,
        SymmetryOperators(),
    )

    assert not result.satisfied
    assert result.checks[0].name == "hermiticity"
    assert result.checks[0].residual == pytest.approx(1.0)


def test_class_a_requires_no_symmetry_operator() -> None:
    classification = SymmetryClassification.from_signature(
        time_reversal_square=None,
        particle_hole_square=None,
        chiral_symmetry=False,
    )

    result = validate_symmetry_classification(
        np.array([[1.0, 1.0j], [-1.0j, 2.0]]),
        classification,
        SymmetryOperators(),
    )

    assert result.satisfied
    assert tuple(check.name for check in result.checks) == ("hermiticity",)


def test_operator_presence_must_match_declaration() -> None:
    class_ai = SymmetryClassification.from_signature(
        time_reversal_square=1,
        particle_hole_square=None,
        chiral_symmetry=False,
    )
    class_a = SymmetryClassification.from_signature(
        time_reversal_square=None,
        particle_hole_square=None,
        chiral_symmetry=False,
    )

    with pytest.raises(ValueError, match="requires"):
        validate_symmetry_classification(np.eye(2), class_ai, SymmetryOperators())
    with pytest.raises(ValueError, match="absent"):
        validate_symmetry_classification(
            np.eye(2),
            class_a,
            SymmetryOperators(time_reversal=np.eye(2)),
        )


def test_symmetry_operators_are_defensive_read_only_copies() -> None:
    source = np.eye(2)
    operators = SymmetryOperators(time_reversal=source)
    source[0, 0] = 2.0

    assert operators.time_reversal is not None
    assert np.array_equal(operators.time_reversal, np.eye(2))
    assert not operators.time_reversal.flags.writeable


def test_kitaev_chain_validates_as_bdi_for_real_parameters() -> None:
    n_sites = 8
    model = KitaevChain(
        KitaevChainParameters(
            n_sites=n_sites,
            hopping=1.0,
            chemical_potential=0.3,
            pairing=0.5,
            boundary="open",
        )
    )
    identity = np.eye(2 * n_sites)
    sector_exchange = np.block(
        [
            [np.zeros((n_sites, n_sites)), np.eye(n_sites)],
            [np.eye(n_sites), np.zeros((n_sites, n_sites))],
        ]
    )
    classification = SymmetryClassification.from_signature(
        time_reversal_square=1,
        particle_hole_square=1,
        chiral_symmetry=True,
    )

    result = validate_symmetry_classification(
        model.hamiltonian(),
        classification,
        SymmetryOperators(
            time_reversal=identity,
            particle_hole=sector_exchange,
            chiral=sector_exchange,
        ),
    )

    assert result.satisfied


def test_bhz_model_validates_as_aii() -> None:
    n_x = 2
    n_y = 2
    n_sites = n_x * n_y
    model = BHZModel(
        BHZModelParameters(
            n_x=n_x,
            n_y=n_y,
            mass=-1.0,
            boundary_x="periodic",
            boundary_y="periodic",
        )
    )
    orbital_identity = np.eye(2)
    time_reversal_per_site = np.block(
        [
            [np.zeros((2, 2)), orbital_identity],
            [-orbital_identity, np.zeros((2, 2))],
        ]
    )
    classification = SymmetryClassification.from_signature(
        time_reversal_square=-1,
        particle_hole_square=None,
        chiral_symmetry=False,
    )

    result = validate_symmetry_classification(
        model.hamiltonian(),
        classification,
        SymmetryOperators(
            time_reversal=np.kron(np.eye(n_sites), time_reversal_per_site),
        ),
    )

    assert result.satisfied


def test_validation_result_has_standardized_output() -> None:
    classification = SymmetryClassification.from_signature(
        time_reversal_square=None,
        particle_hole_square=None,
        chiral_symmetry=False,
    )
    result = validate_symmetry_classification(
        np.eye(2),
        classification,
        SymmetryOperators(),
    )

    assert isinstance(result, StandardizedObservable)
    record = result.to_observable_record()
    assert record.kind == "symmetry_validation"
    assert record.scalars["satisfied"] is True
    assert np.array_equal(record.arrays["check_satisfied"], [True])
    assert record.metadata["check_names"] == ["hermiticity"]


@pytest.mark.parametrize("tolerance", [-1.0, np.nan, np.inf, True])
def test_validation_rejects_invalid_tolerance(tolerance: object) -> None:
    classification = SymmetryClassification.from_signature(
        time_reversal_square=None,
        particle_hole_square=None,
        chiral_symmetry=False,
    )

    with pytest.raises((TypeError, ValueError), match="tolerance"):
        validate_symmetry_classification(
            np.eye(2),
            classification,
            SymmetryOperators(),
            tolerance=tolerance,  # type: ignore[arg-type]
        )
