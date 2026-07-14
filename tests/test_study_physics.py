"""Round-trip tests: stored studies must retain known physical signatures."""

from __future__ import annotations

import numpy as np

from toposc_lab.data.studies import (
    StudyMetadata,
    study_from_bytes,
    study_from_parameter_scan,
    study_to_bytes,
)
from toposc_lab.models.ssh_chain import SSHChain, SSHChainParameters
from toposc_lab.scans.analysis import analyze_parameter_scan
from toposc_lab.scans.haldane_scan import scan_haldane_mass
from toposc_lab.scans.model_scan import model_parameter_scan


def test_ssh_study_retains_edge_mode_signal_after_round_trip() -> None:
    """Open SSH chain: v < w has an edge zero mode, v > w does not."""
    scan = model_parameter_scan(
        "v",
        [0.2, 1.5],
        parameter_model=SSHChainParameters,
        model_factory=SSHChain,
        base_parameters={"n_cells": 16, "v": 0.2, "w": 1.0, "boundary": "open"},
    )
    analysis = analyze_parameter_scan(scan)
    study = study_from_parameter_scan(
        scan,
        StudyMetadata(study_name="SSH phase check", model_name="SSHChain"),
        observables=analysis.observable_arrays(),
    )
    loaded = study_from_bytes(study_to_bytes(study))

    # The topological point has an exponentially small finite-size splitting
    # and its lowest-energy state lives at the two open ends.
    assert loaded.arrays["minimum_abs_energy"][0] < 1.0e-8
    assert loaded.arrays["minimum_abs_energy"][1] > 0.1
    assert loaded.arrays["edge_weights"][0] > 0.8
    assert loaded.arrays["edge_weights"][1] < 0.3


def test_haldane_study_retains_chern_phase_after_round_trip() -> None:
    """At M=0 the standard Haldane model has |C|=1; outside it is trivial."""
    result = scan_haldane_mass(
        mass_values=np.array([-1.5, 0.0, 1.5]),
        n_k=21,
    )
    study = study_from_parameter_scan(
        result.scan,
        StudyMetadata(study_name="Haldane phase check", model_name="HaldaneModel"),
        observables={
            "bulk_gaps": result.bulk_gaps,
            "chern_numbers": result.chern_numbers,
        },
    )
    loaded = study_from_bytes(study_to_bytes(study))

    assert np.array_equal(loaded.arrays["chern_numbers"][[0, 2]], [0, 0])
    assert abs(loaded.arrays["chern_numbers"][1]) == 1
    assert np.all(loaded.arrays["bulk_gaps"] > 0.0)
