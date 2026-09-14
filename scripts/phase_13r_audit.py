"""Audit stored Phase-13R curves, prediction diagnostics and provenance without solves.

Run from the repository root with PYTHONPATH=src. Output is descriptive only;
neither these diagnostics nor plots alter the predeclared recovery gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toposc_lab.active_learning import Candidate
from toposc_lab.active_learning.campaign import read_snapshot
from toposc_lab.data import load_dataset, validate_dataset


def audit(directory: Path) -> dict:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    hashes = {}
    rows = []
    curves = {arm: [] for arm in ("active", "random", "evolutionary")}
    for trial in report["trials"]:
        path = directory / str(trial["seed"])
        datasets = {}
        for file in path.glob("*.json"):
            hashes[str(file.relative_to(directory))] = hashlib.sha256(file.read_bytes()).hexdigest()
            if "checkpoint" in file.name:
                continue
            dataset = load_dataset(file)
            validate_dataset(dataset).raise_for_errors()
            for record in dataset.records:
                if (
                    record.provenance.runtime["source_sha256"]
                    != report["provenance"]["runtime"]["source_sha256"]
                ):
                    raise ValueError("record source mismatch")
            datasets[file.stem] = dataset
        starts = [
            [Candidate(r.geometry, r.model).candidate_id for r in datasets[a].records[:8]]
            for a in curves
        ]
        if starts[0] != starts[1] or starts[0] != starts[2]:
            raise ValueError("arms do not share initial pool")
        heldout = {r.geometry.family_fingerprint for r in datasets["heldout"].records}
        lookup = {}
        for arm, arm_curves in curves.items():
            records = datasets[arm].records
            if len(records) != 24 or heldout & {r.geometry.family_fingerprint for r in records}:
                raise ValueError("budget or heldout leakage")
            ids = [Candidate(r.geometry, r.model).candidate_id for r in records]
            if arm != "evolutionary" and len(set(ids)) != 24:
                raise ValueError("duplicate acquired labels")
            lookup.update(zip(ids, records, strict=True))
            arm_curves.append(trial["arms"][arm]["best_so_far"])
        checkpoint = read_snapshot(path / "active-checkpoint.json")
        if (
            checkpoint["completed_cycles"] != 4
            or checkpoint["exact_attempts"] != 16
            or checkpoint["pending"]
        ):
            raise ValueError("incomplete or over-budget checkpoint")
        train_ids = {r.record_id for r in datasets["active"].records[:8]}
        train_candidates = set(starts[0])
        random_ids = {
            Candidate(r.geometry, r.model).candidate_id for r in datasets["random"].records
        }
        pool_union = set()
        for history in checkpoint["history"]:
            if history["prediction_failure"]:
                raise ValueError("prediction failure in benchmark")
            selected = {s["candidate_id"] for s in history["selected"]}
            predictions = {p["candidate_id"]: p for p in history["predictions"]}
            for prediction in predictions.values():
                if (
                    set(prediction["training_record_ids"]) != train_ids
                    or prediction["kind"] != "predicted"
                ):
                    raise ValueError("prediction provenance or future-label leakage")
            if train_candidates & selected:
                raise ValueError("duplicate selection")
            for selection in history["selected"]:
                if (
                    selection["strategy"].startswith("exploitation")
                    and predictions[selection["candidate_id"]]["is_ood"]
                ):
                    raise ValueError("OOD exploitation")
            train_candidates.update(selected)
            pool_union.update(history["pool_ids"])
            for verification in history["verifications"]:
                if verification["failure"]:
                    raise ValueError("exact simulation failure")
                train_ids.add(verification["record_id"])
            groups = {}
            # Random comparisons exclude all current training candidates before
            # acquisition. They are a pool intersection, not a fresh random test.
            for name, ids in (
                ("selected", selected),
                ("random_pool_overlap", random_ids - selected),
                ("ood", {i for i, p in predictions.items() if p["is_ood"]}),
            ):
                pairs = [
                    (predictions[i], lookup[i])
                    for i in sorted(ids)
                    if i in predictions and i in lookup
                ]
                errors = np.array(
                    [abs(p["value"] - r.observables[0].values["quality"]) for p, r in pairs]
                )
                sigma = np.array([p["uncertainty"] for p, r in pairs])
                groups[name] = {
                    "count": len(pairs),
                    "mae": float(errors.mean()) if len(pairs) else None,
                    "raw_gaussian_90_coverage_diagnostic": float(
                        np.mean(errors <= 1.644854 * sigma)
                    )
                    if len(pairs)
                    else None,
                    "mean_sigma": float(sigma.mean()) if len(pairs) else None,
                }
            sizes = sorted(lookup[i].geometry.to_geometry().n_sites for i in selected)
            rows.append(
                {
                    "seed": trial["seed"],
                    "cycle": history["cycle_index"],
                    "groups": groups,
                    "selected_sizes": sizes,
                    "minimum_batch_size_distance": int(min(np.diff(sizes))),
                    "pool_union_so_far": len(pool_union),
                }
            )
    diagnostic = {
        "passed": True,
        "new_exact_simulations": 0,
        "scope": "descriptive selected/random-pool-overlap errors; no calibration guarantee",
        "source_sha256": report["provenance"]["runtime"]["source_sha256"],
        "artifact_sha256": hashes,
        "rows": rows,
    }
    (directory / "recovery-diagnostics.json").write_text(
        json.dumps(diagnostic, indent=2), encoding="utf-8"
    )
    x = np.arange(1, 25)
    fig, axes = plt.subplots(4, 5, figsize=(15, 9), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        if i >= len(report["trials"]):
            ax.set_visible(False)
            continue
        for arm, values in curves.items():
            ax.plot(x, values[i], label=arm, lw=1.2)
        ax.axvline(8, color="grey", ls=":", lw=0.8)
        ax.set_title(str(report["trials"][i]["seed"]), fontsize=9)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.supxlabel("Exact search simulations (initial eight included)")
    fig.supylabel("Verified best score")
    fig.suptitle("Phase 13R: every held-out seed; shared warm start ends at 8")
    fig.tight_layout()
    fig.savefig(directory / "per-seed-curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(4, 5, figsize=(15, 9), sharex=True)
    for i, ax in enumerate(axes.flat):
        if i >= len(report["trials"]):
            ax.set_visible(False)
            continue
        for arm, values in curves.items():
            delta = np.asarray(values[i]) - values[i][7]
            ax.plot(x[7:], delta[7:], label=arm, lw=1.2)
        ax.set_title(str(report["trials"][i]["seed"]), fontsize=9)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.supxlabel("Exact search simulations (initial eight included)")
    fig.supylabel("Verified score gain from shared warm start; individual y scales")
    fig.suptitle("Phase 13R: acquisition progress for every held-out seed")
    fig.tight_layout()
    fig.savefig(directory / "per-seed-gain-curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, values in curves.items():
        matrix = np.array(values)
        line = axes[0].plot(x, matrix.mean(axis=0), label=arm)[0]
        axes[0].fill_between(
            x,
            np.quantile(matrix, 0.1, axis=0),
            np.quantile(matrix, 0.9, axis=0),
            color=line.get_color(),
            alpha=0.1,
        )
        delta = matrix - matrix[:, 7, None]
        axes[1].plot(x[7:], delta.mean(axis=0)[7:], label=arm)
    axes[0].set_title("Mean and 10–90% seed range (not CI)")
    axes[1].set_title("Mean gain from shared warm-start best")
    for ax in axes:
        ax.set_xlabel("Exact search simulations")
        ax.legend()
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Best verified score")
    axes[1].set_ylabel("Score gain")
    fig.tight_layout()
    fig.savefig(directory / "mean-learning-curves.png", dpi=160)
    plt.close(fig)
    return diagnostic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    result = audit(parser.parse_args().directory)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "artifacts": len(result["artifact_sha256"]),
                "cycles": len(result["rows"]),
            }
        )
    )
