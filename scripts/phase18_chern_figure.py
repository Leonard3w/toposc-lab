"""Render the saved pilot Chern markers with non-overlapping two-line titles.

Read-only on scientific records; no exact evaluations or changes to archived source.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toposc_lab.research.validation_reporting import load_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    snapshot = load_snapshot(args.study)
    candidates = snapshot["cohort"]["candidates"]
    fig, axes = plt.subplots(2, 5, figsize=(17, 8.4), constrained_layout=True)
    for c, ax in zip(candidates, axes.ravel(), strict=True):
        marker = snapshot["results"][c["id"] + ":clean"]["chern_marker"]
        title = c["id"]
        if marker["status"] == "available":
            xy = np.array(marker["positions"])
            artist = ax.scatter(
                xy[:, 0],
                xy[:, 1],
                c=marker["values"],
                cmap="coolwarm",
                vmin=-2,
                vmax=2,
                s=40,
                marker="s",
            )
            fig.colorbar(artist, ax=ax, shrink=0.7, label="marker (clipped +/-2)")
            title += f"\nbulk mean = {marker['bulk_mean']:.3f}"
        else:
            title += "\n" + marker["status"]
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    fig.suptitle("Clean local Chern marker | open finite BdG sample | bulk distance >=2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
