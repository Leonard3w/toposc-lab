"""Speichere einen Haldane-Phasenscan reproduzierbar und lade ihn erneut."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toposc_lab.data.studies import (
    StudyMetadata,
    load_study,
    save_study,
    study_from_parameter_scan,
)
from toposc_lab.scans.haldane_scan import scan_haldane_mass
from toposc_lab.visualization.study_plots import plot_study_observable
from toposc_lab.visualization.style import add_panel_label, paper_style


def main() -> None:
    """Berechne einen Scan, speichere ihn und plotte die geladene Datei."""
    next_nearest_hopping = 0.2
    phase = float(np.pi / 2.0)
    result = scan_haldane_mass(
        mass_values=np.linspace(-2.0, 2.0, 81),
        next_nearest_neighbor_hopping=next_nearest_hopping,
        phase=phase,
        n_k=24,
    )

    metadata = StudyMetadata(
        study_name="Haldane mass phase diagram",
        model_name="HaldaneModel",
        model_parameters={
            "nearest_neighbor_hopping": 1.0,
            "next_nearest_neighbor_hopping": next_nearest_hopping,
            "phase": phase,
        },
        description=(
            "Bulk gap and Chern number across the Haldane mass transition."
        ),
        tags=("haldane", "chern", "phase-diagram"),
    )
    study = study_from_parameter_scan(
        result.scan,
        metadata,
        observables={
            "bulk_gaps": result.bulk_gaps,
            "chern_numbers": result.chern_numbers,
            "numerical_chern_numbers": result.numerical_chern_numbers,
        },
    )

    path = save_study("results/haldane_mass_phase_diagram", study)
    loaded = load_study(path)
    print(f"Saved and loaded: {path}")
    print(f"Created at: {loaded.metadata.created_at.isoformat()}")
    print(f"Package version: {loaded.metadata.package_version}")

    # Die Plotfunktionen erhalten nur die geladene Studie, keine Arrays.
    # Daher funktioniert dieselbe Visualisierung auch nach einem Neustart.
    with paper_style():
        _, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        plot_study_observable(
            loaded,
            "bulk_gaps",
            axes=axes[0],
            title="Loaded bulk-gap data",
            xlabel="Sublattice mass M",
            ylabel="Bulk band gap",
        )
        plot_study_observable(
            loaded,
            "chern_numbers",
            axes=axes[1],
            title="Loaded topological phase",
            xlabel="Sublattice mass M",
            ylabel="Chern number",
            color="tab:red",
            drawstyle="steps-mid",
        )
        add_panel_label(axes[0], "(a)")
        add_panel_label(axes[1], "(b)")
        plt.show()


if __name__ == "__main__":
    main()
