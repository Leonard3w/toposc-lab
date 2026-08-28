from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from toposc_lab.models.geometry_kitaev_chain import GeometryKitaevChain
from toposc_lab.models.kitaev_chain import KitaevChain, KitaevChainParameters
from toposc_lab.solvers.exact_diagonalization import ExactDiagonalizationSolver


REGRESSION_CASES = (
    pytest.param(
        {
            "n_sites": 2,
            "hopping": 1.0,
            "chemical_potential": 0.3,
            "pairing": 0.7,
            "boundary": "open",
        },
        id="open-two-site",
    ),
    pytest.param(
        {
            "n_sites": 8,
            "hopping": 1.0,
            "chemical_potential": 0.0,
            "pairing": 1.0,
            "boundary": "open",
        },
        id="open-topological-point",
    ),
    pytest.param(
        {
            "n_sites": 7,
            "hopping": 1.0,
            "chemical_potential": 3.2,
            "pairing": 0.4,
            "boundary": "open",
        },
        id="open-trivial-regime",
    ),
    pytest.param(
        {
            "n_sites": 6,
            "hopping": -0.8,
            "chemical_potential": -0.6,
            "pairing": -0.3,
            "boundary": "open",
        },
        id="open-negative-couplings",
    ),
    pytest.param(
        {
            "n_sites": 2,
            "hopping": 1.0,
            "chemical_potential": 0.2,
            "pairing": 0.5,
            "boundary": "periodic",
        },
        id="periodic-two-site",
    ),
    pytest.param(
        {
            "n_sites": 3,
            "hopping": 0.9,
            "chemical_potential": -0.4,
            "pairing": 0.6,
            "boundary": "periodic",
        },
        id="periodic-three-site",
    ),
    pytest.param(
        {
            "n_sites": 8,
            "hopping": 1.2,
            "chemical_potential": 1.1,
            "pairing": 0.35,
            "boundary": "periodic",
        },
        id="periodic-clean",
    ),
    pytest.param(
        {
            "n_sites": 11,
            "hopping": 1.0,
            "chemical_potential": 0.4,
            "pairing": 0.8,
            "boundary": "open",
            "disorder_strength": 1.0,
            "disorder_seed": 42,
        },
        id="open-seeded-disorder",
    ),
    pytest.param(
        {
            "n_sites": 9,
            "hopping": 0.7,
            "chemical_potential": -0.2,
            "pairing": 0.45,
            "boundary": "periodic",
            "disorder_strength": 0.7,
            "disorder_seed": 7,
        },
        id="periodic-seeded-disorder",
    ),
)


@pytest.mark.parametrize("parameter_values", REGRESSION_CASES)
def test_geometry_and_legacy_kitaev_matrices_and_spectra_are_identical(
    parameter_values: Mapping[str, Any],
) -> None:
    params = KitaevChainParameters(**parameter_values)
    legacy_model = KitaevChain(params)
    geometry_model = GeometryKitaevChain(params)

    legacy_hamiltonian = legacy_model.hamiltonian()
    geometry_hamiltonian = geometry_model.hamiltonian()

    assert np.array_equal(geometry_model.disorder_profile, legacy_model.disorder_profile)
    assert geometry_model.basis_layout == legacy_model.basis_layout
    assert geometry_hamiltonian.dtype == legacy_hamiltonian.dtype
    assert np.array_equal(geometry_hamiltonian, legacy_hamiltonian)
    assert np.array_equal(
        np.linalg.eigvalsh(geometry_hamiltonian),
        np.linalg.eigvalsh(legacy_hamiltonian),
    )


@pytest.mark.parametrize("boundary", ["open", "periodic"])
def test_geometry_and_legacy_kitaev_mu_scans_are_identical(boundary: str) -> None:
    chemical_potentials = np.linspace(-3.0, 3.0, 13)
    solver = ExactDiagonalizationSolver()
    legacy_spectra: list[np.ndarray] = []
    geometry_spectra: list[np.ndarray] = []

    for chemical_potential in chemical_potentials:
        params = KitaevChainParameters(
            n_sites=10,
            hopping=1.0,
            chemical_potential=float(chemical_potential),
            pairing=0.65,
            boundary=boundary,
        )
        legacy_spectra.append(solver.solve(KitaevChain(params).hamiltonian()).eigenvalues)
        geometry_spectra.append(
            solver.solve(GeometryKitaevChain(params).hamiltonian()).eigenvalues
        )

    assert np.array_equal(np.asarray(geometry_spectra), np.asarray(legacy_spectra))
