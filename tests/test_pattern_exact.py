from pathlib import Path

import numpy as np
import pytest

from toposc_lab.active_learning.benchmark import source_provenance
from toposc_lab.data import validate_dataset_record
from toposc_lab.discovery import DiscoveryConfig
from toposc_lab.discovery.storage import decode_run, encode_run
from toposc_lab.discovery.validation import disorder_evidence, majorana_evidence
from toposc_lab.generative.physics import ExactGeometryEvaluator, candidate
from toposc_lab.generative.space import GeometrySearchSpace, embedded_edges
from toposc_lab.patterns.exact import evaluate_deletion
from toposc_lab.patterns.interventions import deletion_candidates


def test_deletion_adapter_against_independent_zero_disorder_evaluator():
    space = GeometrySearchSpace()
    g = space.build(
        frozenset(
            e
            for e in space.edge_pool
            if np.linalg.norm(space.coordinates[e[0]] - space.coordinates[e[1]]) == 1
        )
    )
    g = space.build(frozenset(embedded_edges(g)) - {deletion_candidates(g)[0]})
    provenance = source_provenance(Path.cwd())
    record, run = evaluate_deletion(g, 16399, provenance)
    independent = disorder_evidence(record, 16400, DiscoveryConfig(onsite_width=0))
    assert record.observables[0].values["quality"] == pytest.approx(
        independent["quality"], abs=1e-10
    )
    assert tuple(record.topology[0].parameters["indices"]) == tuple(independent["indices"])
    np.testing.assert_allclose(record.spectrum.eigenvalues, independent["spectrum"], atol=1e-10)
    assert record.provenance.seed == 16399
    assert record.provenance.git_commit == provenance.git_commit
    validate_dataset_record(record).raise_for_errors()
    evidence = majorana_evidence(record, decode_run(encode_run(run)))
    assert evidence["operator_phs_residual"] < 1e-10
    assert len(evidence["states"]) == 4
    assert len(record.spectrum.eigenvalues) == 72
    assert len(record.topology[0].parameters["signatures"]) == 3
    assert evidence["majorana_claim"] is False
    for state in evidence["states"]:
        assert sum(state["site_probability"]) == pytest.approx(1)
        assert len(state["polarization_real"]) == len(state["polarization_imag"]) == 36
    with pytest.raises(ValueError, match="inadmissible"):
        ExactGeometryEvaluator(provenance).evaluate(candidate(g), 16399)


def test_deletion_adapter_rejects_standard_discovery_parent_before_physics(monkeypatch):
    space = GeometrySearchSpace()
    grid = space.build(
        frozenset(
            e
            for e in space.edge_pool
            if np.linalg.norm(space.coordinates[e[0]] - space.coordinates[e[1]]) == 1
        )
    )

    def unexpected_physics(*args, **kwargs):
        pytest.fail("inadmissible simplification must fail before numerical evaluation")

    monkeypatch.setattr("toposc_lab.patterns.exact.evaluate_geometry", unexpected_physics)
    with pytest.raises(ValueError, match="inadmissible single-edge simplification"):
        evaluate_deletion(grid, 16399, source_provenance(Path.cwd()))
