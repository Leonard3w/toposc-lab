from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from toposc_lab.data.studies import StudyData, StudyMetadata
from toposc_lab.visualization.study_plots import (
    plot_study_comparison,
    plot_study_observable,
    plot_study_spectrum,
)


def _study(offset: float = 0.0) -> StudyData:
    return StudyData(
        metadata=StudyMetadata(
            study_name="test scan",
            model_name="TestModel",
            scan_parameters={"parameter_name": "mass"},
            package_version="test",
        ),
        arrays={
            "parameter_values": np.array([-1.0, 0.0, 1.0]),
            "spectra": np.array(
                [
                    [-1.0, 1.0],
                    [-0.2, 0.2],
                    [-1.0, 1.0],
                ]
            ),
            "bulk_gaps": np.array([1.0 + offset, 0.2 + offset, 1.0 + offset]),
        },
    )


def test_plot_study_spectrum_uses_saved_arrays() -> None:
    figure, axes = plot_study_spectrum(_study(), show=False)

    assert axes.get_xlabel() == "mass"
    assert axes.get_ylabel() == "Energy"
    assert len(axes.lines) == 3  # Two bands and the zero-energy reference.

    plt.close(figure)


def test_plot_study_observable_uses_saved_arrays() -> None:
    figure, axes = plot_study_observable(
        _study(),
        "bulk_gaps",
        ylabel="Bulk gap",
        show=False,
    )

    assert axes.get_xlabel() == "mass"
    assert axes.get_ylabel() == "Bulk gap"
    assert len(axes.lines) == 1

    plt.close(figure)


def test_plot_study_comparison_adds_one_line_per_study() -> None:
    figure, axes = plot_study_comparison(
        {"clean": _study(), "disordered": _study(offset=0.1)},
        "bulk_gaps",
        show=False,
    )

    assert len(axes.lines) == 2
    assert {text.get_text() for text in axes.get_legend().get_texts()} == {
        "clean",
        "disordered",
    }

    plt.close(figure)
