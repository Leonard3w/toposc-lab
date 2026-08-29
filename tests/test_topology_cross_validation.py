import numpy as np
import pytest

from toposc_lab.models.kitaev_chain import KitaevChain, KitaevChainParameters
from toposc_lab.models.qwz_model import QWZModel, QWZModelParameters
from toposc_lab.topology import (
    BottIndexResult,
    LocalChernMarkerResult,
    PfaffianInvariantResult,
    RealSpaceWindingResult,
    SpectralLocalizerResult,
    bott_index,
    local_chern_marker,
    one_dimensional_pfaffian_invariant,
    real_space_winding_invariant,
    spectral_localizer,
)
from toposc_lab.topology.symmetry import SymmetryClassification


def _class_a() -> SymmetryClassification:
    return SymmetryClassification.from_signature(
        time_reversal_square=None,
        particle_hole_square=None,
        chiral_symmetry=False,
    )


def _class_bdi() -> SymmetryClassification:
    return SymmetryClassification.from_signature(
        time_reversal_square=1,
        particle_hole_square=1,
        chiral_symmetry=True,
    )


def _qwz_diagnostics(
    mass: float,
) -> tuple[BottIndexResult, LocalChernMarkerResult, SpectralLocalizerResult]:
    size = 8
    model = QWZModel(
        QWZModelParameters(
            n_x=size,
            n_y=size,
            mass=mass,
            boundary_x="open",
            boundary_y="open",
        )
    )
    hamiltonian = model.hamiltonian()
    site_coordinates = model.lattice.coordinates.astype(float)
    basis_coordinates = np.repeat(site_coordinates, 2, axis=0)
    bulk_mask = np.all(
        (site_coordinates >= 3.0) & (site_coordinates < 5.0),
        axis=1,
    )
    classification = _class_a()
    return (
        bott_index(
            hamiltonian,
            basis_coordinates,
            np.array([size, size], dtype=float),
            classification,
        ),
        local_chern_marker(
            hamiltonian,
            basis_coordinates,
            1.0,
            bulk_mask,
            classification,
        ),
        spectral_localizer(
            hamiltonian,
            basis_coordinates,
            np.array([(size - 1.0) / 2.0, (size - 1.0) / 2.0]),
            classification,
            kappa=0.2,
        ),
    )


@pytest.mark.parametrize(
    ("mass", "expected_chern"),
    [(-3.0, 0), (-1.0, -1), (1.0, 1), (3.0, 0)],
)
def test_qwz_bott_local_marker_and_localizer_agree(
    mass: float,
    expected_chern: int,
) -> None:
    bott, marker, localizer = _qwz_diagnostics(mass)

    assert bott.is_quantized
    assert marker.is_quantized
    assert localizer.is_invertible
    assert bott.bott_index == expected_chern
    assert marker.chern_number == expected_chern
    assert localizer.local_chern_number == expected_chern
    assert localizer.signature == 2 * expected_chern


def _kitaev_endpoint_hamiltonians(
    chemical_potential: float,
    *,
    hopping: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    energy_zero = -chemical_potential - 2.0 * hopping
    energy_pi = -chemical_potential + 2.0 * hopping
    return (
        np.diag([energy_zero, -energy_zero]),
        np.diag([energy_pi, -energy_pi]),
    )


def _kitaev_diagnostics(
    chemical_potential: float,
) -> tuple[PfaffianInvariantResult, RealSpaceWindingResult]:
    n_sites = 32
    classification = _class_bdi()
    endpoint_zero, endpoint_pi = _kitaev_endpoint_hamiltonians(chemical_potential)
    pfaffian_result = one_dimensional_pfaffian_invariant(
        endpoint_zero,
        endpoint_pi,
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        classification,
    )

    model = KitaevChain(
        KitaevChainParameters(
            n_sites=n_sites,
            hopping=1.0,
            chemical_potential=chemical_potential,
            pairing=1.0,
            boundary="open",
        )
    )
    chiral_operator = np.block(
        [
            [np.zeros((n_sites, n_sites)), np.eye(n_sites)],
            [np.eye(n_sites), np.zeros((n_sites, n_sites))],
        ]
    )
    bulk_mask = np.zeros(n_sites, dtype=bool)
    bulk_mask[8:24] = True
    winding_result = real_space_winding_invariant(
        model.hamiltonian(),
        chiral_operator,
        np.tile(np.arange(n_sites, dtype=float), 2),
        bulk_mask,
        classification,
        zero_tolerance=1.0e-8,
    )
    return pfaffian_result, winding_result


@pytest.mark.parametrize(
    ("chemical_potential", "expected_topological"),
    [(-3.0, False), (-1.0, True), (0.0, True), (1.0, True), (3.0, False)],
)
def test_kitaev_pfaffian_parity_agrees_with_real_space_winding(
    chemical_potential: float,
    expected_topological: bool,
) -> None:
    pfaffian_result, winding_result = _kitaev_diagnostics(chemical_potential)

    assert winding_result.is_quantized
    assert winding_result.winding_number is not None
    winding_is_topological = abs(winding_result.winding_number) % 2 == 1
    assert pfaffian_result.is_topological is expected_topological
    assert winding_is_topological is expected_topological
    assert pfaffian_result.is_topological == winding_is_topological


@pytest.mark.parametrize("chemical_potential", [-2.0, 2.0])
def test_kitaev_transition_is_rejected_or_nonquantized(
    chemical_potential: float,
) -> None:
    endpoint_zero, endpoint_pi = _kitaev_endpoint_hamiltonians(chemical_potential)

    with pytest.raises(ValueError, match="gap closing"):
        one_dimensional_pfaffian_invariant(
            endpoint_zero,
            endpoint_pi,
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            _class_bdi(),
        )

    n_sites = 32
    model = KitaevChain(
        KitaevChainParameters(
            n_sites=n_sites,
            hopping=1.0,
            chemical_potential=chemical_potential,
            pairing=1.0,
            boundary="open",
        )
    )
    chiral_operator = np.block(
        [
            [np.zeros((n_sites, n_sites)), np.eye(n_sites)],
            [np.eye(n_sites), np.zeros((n_sites, n_sites))],
        ]
    )
    bulk_mask = np.zeros(n_sites, dtype=bool)
    bulk_mask[8:24] = True
    winding_result = real_space_winding_invariant(
        model.hamiltonian(),
        chiral_operator,
        np.tile(np.arange(n_sites, dtype=float), 2),
        bulk_mask,
        _class_bdi(),
    )

    assert not winding_result.is_quantized
    assert winding_result.winding_number is None


@pytest.mark.parametrize("mass", [-2.0, 2.0])
def test_qwz_projector_invariants_reject_exact_periodic_gap_closing(
    mass: float,
) -> None:
    size = 6
    model = QWZModel(
        QWZModelParameters(
            n_x=size,
            n_y=size,
            mass=mass,
            boundary_x="periodic",
            boundary_y="periodic",
        )
    )
    hamiltonian = model.hamiltonian()
    site_coordinates = model.lattice.coordinates.astype(float)
    basis_coordinates = np.repeat(site_coordinates, 2, axis=0)
    classification = _class_a()

    with pytest.raises(ValueError, match="Fermi-level"):
        bott_index(
            hamiltonian,
            basis_coordinates,
            np.array([size, size], dtype=float),
            classification,
        )
    with pytest.raises(ValueError, match="Fermi-level"):
        local_chern_marker(
            hamiltonian,
            basis_coordinates,
            1.0,
            np.ones(size * size, dtype=bool),
            classification,
        )
