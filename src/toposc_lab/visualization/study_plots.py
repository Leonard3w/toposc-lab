"""Wiederverwendbare Visualisierung gespeicherter Studien."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from toposc_lab.data.studies import StudyData
from toposc_lab.visualization.style import paper_style


def _study_array(study: StudyData, name: str) -> np.ndarray:
    """Lese ein benoetigtes Array mit einer hilfreichen Fehlermeldung aus."""
    try:
        return np.asarray(study.arrays[name])
    except KeyError as error:
        available = ", ".join(sorted(study.arrays))
        raise ValueError(
            f"Study does not contain array {name!r}. Available arrays: {available}"
        ) from error


def _parameter_label(study: StudyData) -> str:
    """Erzeuge eine lesbare Beschriftung aus den gespeicherten Scaninfos."""
    parameter_name = study.metadata.scan_parameters.get(
        "parameter_name", "Parameter"
    )
    return str(parameter_name).replace("_", " ")


def _validate_scan_axis(
    study: StudyData,
    values: np.ndarray,
) -> np.ndarray:
    parameter_values = _study_array(study, "parameter_values")

    if parameter_values.ndim != 1:
        raise ValueError("parameter_values must be one-dimensional")

    if values.shape[0] != parameter_values.size:
        raise ValueError(
            "First observable dimension must match parameter_values"
        )

    return parameter_values


def plot_study_spectrum(
    study: StudyData,
    *,
    axes: Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Plotte das gespeicherte Spektrum eines eindimensionalen Scans."""
    spectra = _study_array(study, "spectra")

    if spectra.ndim != 2:
        raise ValueError("spectra must have shape (n_parameters, n_states)")

    parameter_values = _validate_scan_axis(study, spectra)
    created_axes = axes is None

    with paper_style():
        if axes is None:
            figure, axes = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
        else:
            figure = axes.figure

        for band in spectra.T:
            axes.plot(
                parameter_values,
                band,
                linestyle="none",
                marker=".",
                markersize=2.5,
                color="tab:blue",
            )

        axes.axhline(0.0, color="0.35", linestyle="--", linewidth=0.9)
        axes.set_xlabel(xlabel or _parameter_label(study))
        axes.set_ylabel("Energy")
        axes.set_title(title or f"{study.metadata.study_name}: spectrum")

    if created_axes and show:
        plt.show()

    return figure, axes


def plot_study_observable(
    study: StudyData,
    observable_name: str,
    *,
    axes: Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    color: str = "tab:blue",
    drawstyle: str = "default",
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Plotte eine skalare Observable eines gespeicherten Scans.

    ``drawstyle='steps-mid'`` eignet sich etwa fuer ganzzahlige Invarianten
    wie die Chern-Zahl.
    """
    values = _study_array(study, observable_name)

    if values.ndim != 1:
        raise ValueError(
            f"Observable {observable_name!r} must be one-dimensional"
        )

    parameter_values = _validate_scan_axis(study, values)
    created_axes = axes is None

    with paper_style():
        if axes is None:
            figure, axes = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
        else:
            figure = axes.figure

        axes.plot(
            parameter_values,
            values,
            color=color,
            marker="o",
            markersize=3.2,
            drawstyle=drawstyle,
        )
        axes.set_xlabel(xlabel or _parameter_label(study))
        axes.set_ylabel(ylabel or observable_name.replace("_", " "))
        axes.set_title(title or f"{study.metadata.study_name}: {observable_name}")

    if created_axes and show:
        plt.show()

    return figure, axes


def plot_study_comparison(
    studies: Mapping[str, StudyData],
    observable_name: str,
    *,
    axes: Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    show: bool = True,
) -> tuple[Figure, Axes]:
    """Vergleiche dieselbe eindimensionale Observable mehrerer Studien."""
    if not studies:
        raise ValueError("studies must contain at least one StudyData object")

    created_axes = axes is None

    with paper_style():
        if axes is None:
            figure, axes = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
        else:
            figure = axes.figure

        for label, study in studies.items():
            values = _study_array(study, observable_name)

            if values.ndim != 1:
                raise ValueError(
                    f"Observable {observable_name!r} in {label!r} must be one-dimensional"
                )

            parameter_values = _validate_scan_axis(study, values)
            axes.plot(
                parameter_values,
                values,
                marker="o",
                markersize=3.0,
                label=label,
            )

        first_study = next(iter(studies.values()))
        axes.set_xlabel(xlabel or _parameter_label(first_study))
        axes.set_ylabel(ylabel or observable_name.replace("_", " "))
        axes.set_title(title or f"Comparison: {observable_name.replace('_', ' ')}")
        axes.legend(frameon=False)

    if created_axes and show:
        plt.show()

    return figure, axes
