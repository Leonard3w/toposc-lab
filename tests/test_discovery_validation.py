from pathlib import Path

import numpy as np
import pytest

from toposc_lab.active_learning.benchmark import source_provenance
from toposc_lab.data import validate_dataset_record
from toposc_lab.discovery import DiscoveryConfig
from toposc_lab.discovery.storage import decode_run, encode_run
from toposc_lab.discovery.validation import (
    disorder_evidence,
    integrate_validation,
    majorana_evidence,
)
from toposc_lab.generative.physics import ExactGeometryEvaluator, candidate
from toposc_lab.generative.space import GeometrySearchSpace


@pytest.mark.parametrize("mu,eligible", [(2.0, True), (8.0, False)])
def test_exact_reference_validation_and_codec(mu, eligible):
    space = GeometrySearchSpace()
    xy = space.coordinates
    grid = space.build(
        frozenset(e for e in space.edge_pool if np.linalg.norm(xy[e[0]] - xy[e[1]]) == 1)
    )
    evaluator = ExactGeometryEvaluator(source_provenance(Path.cwd()))
    record, run = evaluator.evaluate(candidate(grid, mu), 15001)
    recovered = decode_run(encode_run(run))
    np.testing.assert_array_equal(
        recovered.simulation_result.eigenvectors, run.simulation_result.eigenvectors
    )
    config = DiscoveryConfig(onsite_width=0.0)
    majorana = majorana_evidence(record, recovered)
    assert majorana["operator_phs_residual"] < 1e-10
    assert not majorana["majorana_claim"]
    assert all(abs(sum(s["site_probability"]) - 1) < 1e-10 for s in majorana["states"])
    members = [disorder_evidence(record, seed, config) for seed in (15002, 15003)]
    assert all(m["eligible"] == eligible for m in members)
    assert members[0]["quality"] == pytest.approx(
        record.observables[0].values["quality"], abs=1e-10
    )
    enriched = integrate_validation(record, majorana, members, config)
    validate_dataset_record(enriched).raise_for_errors()
    assert enriched.robustness[0].uncertainty["success_wilson_upper"] > 0
    assert enriched.observables[-1].values["status"] == "unavailable"
