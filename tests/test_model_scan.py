from __future__ import annotations

import json

import numpy as np
import pytest

from toposc_lab.app.streamlit_app import _create_parameter_scan_study
from toposc_lab.models.ssh_chain import SSHChain, SSHChainParameters
from toposc_lab.scans.analysis import analyze_parameter_scan
from toposc_lab.scans.model_scan import model_parameter_scan


def test_generic_model_parameter_scan_updates_one_parameter() -> None:
    scan = model_parameter_scan(
        "v",
        np.array([0.4, 0.8, 1.2]),
        parameter_model=SSHChainParameters,
        model_factory=SSHChain,
        base_parameters={"n_cells": 4, "v": 0.5, "w": 1.0, "boundary": "open"},
    )

    assert scan.parameter_name == "v"
    assert np.array_equal(scan.parameter_values, [0.4, 0.8, 1.2])
    assert [result.parameters["v"] for result in scan.results] == [0.4, 0.8, 1.2]
    assert scan.metadata["base_parameters"] == {
        "n_cells": 4,
        "w": 1.0,
        "boundary": "open",
    }


def test_generic_model_parameter_scan_rejects_unknown_parameter() -> None:
    with pytest.raises(ValueError, match="Unknown parameter"):
        model_parameter_scan(
            "not_a_parameter",
            [0.0, 1.0],
            parameter_model=SSHChainParameters,
            model_factory=SSHChain,
            base_parameters={"n_cells": 4, "v": 0.5, "w": 1.0},
        )


def test_parameter_scan_analysis_has_one_value_per_scan_point() -> None:
    scan = model_parameter_scan(
        "v",
        [0.4, 1.4],
        parameter_model=SSHChainParameters,
        model_factory=SSHChain,
        base_parameters={"n_cells": 6, "v": 0.5, "w": 1.0, "boundary": "open"},
    )

    analysis = analyze_parameter_scan(scan)

    assert analysis.minimum_abs_energies.shape == (2,)
    assert analysis.bulk_gaps.shape == (2,)
    assert analysis.zero_mode_counts.shape == (2,)
    assert analysis.inverse_participation_ratios.shape == (2,)
    assert analysis.edge_weights.shape == (2,)
    assert set(analysis.observable_arrays()) == {
        "bulk_gaps",
        "zero_mode_counts",
        "inverse_participation_ratios",
        "edge_weights",
    }


def test_app_scan_builder_creates_downloadable_study() -> None:
    study = _create_parameter_scan_study(
        "ssh-chain",
        json.dumps(
            {"n_cells": 5, "v": 0.5, "w": 1.0, "boundary": "open"}
        ),
        "v",
        0.3,
        1.3,
        3,
    )

    assert study.metadata.model_name == "SSHChain"
    assert study.metadata.scan_parameters["parameter_name"] == "v"
    assert study.arrays["spectra"].shape == (3, 10)
    assert study.arrays["bulk_gaps"].shape == (3,)
    assert study.arrays["edge_weights"].shape == (3,)
