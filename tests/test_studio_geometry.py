"""Parameterized geometry generation preserves the frozen default recipes."""

from dataclasses import asdict

import numpy as np
import pytest

from toposc_lab.geometry import geometry_to_bytes, validate_geometry
from toposc_lab.geometry.generators.hard_core_planar import (
    HardCorePlanarConfig,
    HardCorePlanarGenerationError,
    constrained_random_embedded_graph,
    hard_core_planar_graph,
    hard_core_planar_reference,
)


@pytest.mark.parametrize("generator", [hard_core_planar_graph, hard_core_planar_reference,
                                      constrained_random_embedded_graph])
def test_default_config_preserves_historical_bytes(generator):
    original = generator(seed=190001)
    explicit = generator(seed=190001, config=asdict(HardCorePlanarConfig()))
    assert geometry_to_bytes(original) == geometry_to_bytes(explicit)
    assert original.metadata["algorithm_version"] == 1
    assert "resolved_config" not in original.metadata


@pytest.mark.parametrize("generator", [hard_core_planar_graph, hard_core_planar_reference,
                                      constrained_random_embedded_graph])
def test_small_custom_generator_is_reproducible_and_respects_constraints(generator):
    config = HardCorePlanarConfig(
        n_sites=12, n_edges=16, box_maximum=3, minimum_separation=.1,
        maximum_edge_length=5, minimum_degree=1, maximum_degree=5,
        boundary_shell_thickness=.2, minimum_boundary_sites=1, maximum_boundary_sites=12,
        max_point_proposals=1000, max_complete_attempts=20,
    )
    geometry = generator(seed=7, config=config)
    assert geometry_to_bytes(geometry) == geometry_to_bytes(generator(seed=7, config=config))
    validate_geometry(geometry, require_connected=True).raise_for_errors()
    assert geometry.n_sites == 12 and geometry.n_edges == 16
    assert np.allclose(geometry.coordinates.min(axis=0), 0)
    assert np.allclose(geometry.coordinates.max(axis=0), 3)
    assert max(len(geometry.neighbors(i)) for i in geometry.site_indices) <= 5
    assert geometry.metadata["resolved_config"] == asdict(config)
    assert geometry.metadata["algorithm_version"] == 2


@pytest.mark.parametrize("changes", [
    {"n_sites": 2}, {"n_edges": 2}, {"n_edges": 1000}, {"minimum_degree": 5},
    {"minimum_separation": 9}, {"maximum_edge_length": .1},
    {"max_point_proposals": 1}, {"max_complete_attempts": 100001},
    {"require_connected": False}, {"forbid_crossings": "yes"}, {"boundary_shell_thickness": 5},
])
def test_impossible_constraints_rejected_before_sampling(changes):
    with pytest.raises(ValueError):
        HardCorePlanarConfig(**changes)


def test_declared_construction_budget_is_enforced():
    with pytest.raises(HardCorePlanarGenerationError, match="proposals|attempts"):
        hard_core_planar_graph(seed=190001, config={"max_point_proposals": 64, "max_complete_attempts": 1})
