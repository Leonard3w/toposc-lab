import numpy as np
import pytest

from toposc_lab.models.ssh_chain import SSHChain, SSHChainParameters
from toposc_lab.observables.results import StandardizedObservable
from toposc_lab.topology.symmetry import SymmetryClassification
from toposc_lab.topology.winding import (
    RealSpaceWindingResult,
    real_space_winding_invariant,
)


def _ssh_winding(
    *,
    intracell_hopping: float,
    intercell_hopping: float,
    reverse_position: bool = False,
) -> RealSpaceWindingResult:
    n_cells = 20
    model = SSHChain(
        SSHChainParameters(
            n_cells=n_cells,
            v=intracell_hopping,
            w=intercell_hopping,
            boundary="open",
        )
    )
    positions = np.repeat(np.arange(n_cells, dtype=float), 2)
    if reverse_position:
        positions = -positions
    bulk_mask = np.zeros(n_cells, dtype=bool)
    bulk_mask[5:15] = True
    classification = SymmetryClassification.from_signature(
        time_reversal_square=None,
        particle_hole_square=None,
        chiral_symmetry=True,
    )
    return real_space_winding_invariant(
        model.hamiltonian(),
        np.diag([1.0, -1.0] * n_cells),
        positions,
        bulk_mask,
        classification,
    )


def test_topological_ssh_dimer_limit_has_unit_bulk_winding() -> None:
    result = _ssh_winding(intracell_hopping=0.0, intercell_hopping=1.0)

    assert result.is_quantized
    assert result.winding_number == 1
    assert result.winding_estimate == pytest.approx(1.0)
    assert np.allclose(result.local_marker[result.bulk_mask], 1.0)
    assert result.zero_mode_count == 2


def test_trivial_ssh_dimer_limit_has_zero_bulk_winding() -> None:
    result = _ssh_winding(intracell_hopping=1.0, intercell_hopping=0.0)

    assert result.is_quantized
    assert result.winding_number == 0
    assert result.winding_estimate == pytest.approx(0.0)
    assert result.zero_mode_count == 0


def test_reversing_spatial_orientation_reverses_winding_sign() -> None:
    result = _ssh_winding(
        intracell_hopping=0.0,
        intercell_hopping=1.0,
        reverse_position=True,
    )

    assert result.winding_number == -1
    assert result.winding_estimate == pytest.approx(-1.0)


def test_generic_topological_ssh_chain_is_quantized_in_bulk() -> None:
    result = _ssh_winding(intracell_hopping=0.4, intercell_hopping=1.0)

    assert result.winding_number == 1
    assert result.winding_estimate == pytest.approx(1.0, abs=2.0e-3)


def test_winding_rejects_broken_chiral_symmetry() -> None:
    n_cells = 4
    model = SSHChain(
        SSHChainParameters(
            n_cells=n_cells,
            v=0.4,
            w=1.0,
            boundary="open",
        )
    )
    hamiltonian = model.hamiltonian() + 0.1 * np.eye(2 * n_cells)
    classification = SymmetryClassification.from_signature(
        time_reversal_square=None,
        particle_hole_square=None,
        chiral_symmetry=True,
    )

    with pytest.raises(ValueError, match="violates chiral"):
        real_space_winding_invariant(
            hamiltonian,
            np.diag([1.0, -1.0] * n_cells),
            np.repeat(np.arange(n_cells), 2),
            np.ones(n_cells, dtype=bool),
            classification,
        )


def test_winding_rejects_nonchiral_az_class() -> None:
    class_d = SymmetryClassification.from_signature(
        time_reversal_square=None,
        particle_hole_square=1,
        chiral_symmetry=False,
    )

    with pytest.raises(ValueError, match="AIII, BDI, or CII"):
        real_space_winding_invariant(
            np.diag([1.0, -1.0]),
            np.diag([1.0, -1.0]),
            np.array([0.0, 0.0]),
            np.array([True]),
            class_d,
        )


def test_winding_rejects_chiral_operator_mixing_positions() -> None:
    classification = SymmetryClassification.from_signature(
        time_reversal_square=None,
        particle_hole_square=None,
        chiral_symmetry=True,
    )

    with pytest.raises(ValueError, match="distinct positions"):
        real_space_winding_invariant(
            np.diag([1.0, -1.0]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([0.0, 1.0]),
            np.array([True, True]),
            classification,
        )


def test_winding_result_has_standardized_output() -> None:
    result = _ssh_winding(intracell_hopping=0.0, intercell_hopping=1.0)

    assert isinstance(result, StandardizedObservable)
    record = result.to_observable_record()
    assert record.kind == "real_space_winding"
    assert record.scalars["winding_number"] == 1
    assert record.metadata["requires_explicit_bulk_mask"] is True


def test_bulk_mask_must_be_boolean_and_nonempty() -> None:
    classification = SymmetryClassification.from_signature(
        time_reversal_square=None,
        particle_hole_square=None,
        chiral_symmetry=True,
    )
    hamiltonian = np.array([[0.0, 1.0], [1.0, 0.0]])
    chiral = np.diag([1.0, -1.0])
    positions = np.array([0.0, 0.0])

    with pytest.raises(TypeError, match="boolean"):
        real_space_winding_invariant(
            hamiltonian,
            chiral,
            positions,
            np.array([1]),
            classification,
        )
    with pytest.raises(ValueError, match="select"):
        real_space_winding_invariant(
            hamiltonian,
            chiral,
            positions,
            np.array([False]),
            classification,
        )
