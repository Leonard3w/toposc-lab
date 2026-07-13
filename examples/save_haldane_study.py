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

    mass_values = loaded.arrays["parameter_values"]
    bulk_gaps = loaded.arrays["bulk_gaps"]
    chern_numbers = loaded.arrays["chern_numbers"]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    axes[0].plot(mass_values, bulk_gaps, "o-", markersize=3)
    axes[0].set_xlabel("Sublattice mass M")
    axes[0].set_ylabel("Bulk band gap")
    axes[0].set_title("Loaded bulk-gap data")

    axes[1].step(mass_values, chern_numbers, where="mid", color="tab:red")
    axes[1].set_xlabel("Sublattice mass M")
    axes[1].set_ylabel("Chern number")
    axes[1].set_title("Loaded topological phase")

    plt.show()


if __name__ == "__main__":
    main()
