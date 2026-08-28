from __future__ import annotations

import numpy as np
import pytest

from toposc_lab.geometry import chain, irregular_cluster
from toposc_lab.hamiltonians import (
    build_tight_binding_hamiltonian,
    uniform_edge_disorder,
    uniform_site_disorder,
)


def test_site_disorder_is_seeded_complete_and_bounded() -> None:
    geometry = irregular_cluster()

    first = uniform_site_disorder(geometry, width=0.8, seed=17)
    second = uniform_site_disorder(geometry, width=0.8, seed=17)

    assert first == second
    assert tuple(first) == geometry.site_indices
    assert all(-0.4 <= value <= 0.4 for value in first.values())


def test_edge_disorder_is_seeded_complete_and_bounded() -> None:
    geometry = irregular_cluster()

    first = uniform_edge_disorder(geometry, width=1.2, seed=29)
    second = uniform_edge_disorder(geometry, width=1.2, seed=29)

    assert first == second
    assert tuple(first) == geometry.edges
    assert all(-0.6 <= value <= 0.6 for value in first.values())


def test_disorder_maps_integrate_with_tight_binding_builder() -> None:
    geometry = chain(4)
    onsite = uniform_site_disorder(geometry, width=0.5, seed=3)
    hopping_offsets = uniform_edge_disorder(geometry, width=0.2, seed=5)
    hopping = {edge: -1.0 + offset for edge, offset in hopping_offsets.items()}

    hamiltonian = build_tight_binding_hamiltonian(
        geometry,
        onsite=onsite,
        hopping=hopping,
    )

    assert np.array_equal(np.diag(hamiltonian), np.asarray(tuple(onsite.values())))
    for edge in geometry.edges:
        assert hamiltonian[edge.source, edge.target] == hopping[edge]
        assert hamiltonian[edge.target, edge.source] == hopping[edge]
    assert np.array_equal(hamiltonian, hamiltonian.conj().T)


def test_zero_width_produces_exact_zero_maps() -> None:
    geometry = chain(3)

    assert uniform_site_disorder(geometry, width=0.0, seed=1) == {
        site: 0.0 for site in geometry.site_indices
    }
    assert uniform_edge_disorder(geometry, width=0.0, seed=1) == {
        edge: 0.0 for edge in geometry.edges
    }


@pytest.mark.parametrize("width", [-1.0, np.inf, np.nan])
def test_invalid_width_is_rejected(width: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        uniform_site_disorder(chain(2), width=width, seed=1)


@pytest.mark.parametrize("width", [True, "wide"])
def test_nonreal_width_is_rejected(width: object) -> None:
    with pytest.raises(TypeError, match="real number"):
        uniform_edge_disorder(chain(2), width=width, seed=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [True, 1.5])
def test_noninteger_seed_is_rejected(seed: object) -> None:
    with pytest.raises(TypeError, match="integer or None"):
        uniform_site_disorder(chain(2), width=1.0, seed=seed)  # type: ignore[arg-type]


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        uniform_edge_disorder(chain(2), width=1.0, seed=-1)
