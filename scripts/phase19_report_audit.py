"""Read-only final evidence audit and supplemental descriptive figures for Phase 19."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from toposc_lab.research.embedded import EmbeddedDomain
from toposc_lab.research.embedded_reporting import METRICS, paired_contrast, scalar_record
from toposc_lab.research.provenance import verify
from toposc_lab.research.storage import ResearchStore
from toposc_lab.research.validation_cohort import digest


def main() -> None:
    directory = Path("results/phase19-exploration-v2")
    store = ResearchStore(directory)
    verify(directory, store.get("manifest"))
    store.integrity_check()
    state = store.get("state")
    assert state["status"] == "COMPLETED", state
    cohort = store.get("validation_cohort")
    candidates = {c["id"]: c for c in cohort["candidates"]}
    domain = EmbeddedDomain(**cohort["domain"])
    rows, checksums = [], []
    with store.connect(readonly=True) as db:
        for saved in db.execute("SELECT * FROM objects WHERE kind='exact_result' ORDER BY id"):
            result = store.decode(saved)
            identity = saved["id"].split(":", 1)[0]
            assert result["status"] == "completed" and result["primary_valid"]
            rows.append(scalar_record(candidates[identity], result, domain))
            checksums.append([saved["id"], saved["checksum"]])
    assert len(rows) == 1510
    assert len({(r["candidate"], r["key"]) for r in rows}) == 1510
    families = list(dict.fromkeys(c["family"] for c in candidates.values()))
    tables = []
    for family in families:
        for width in (0.0, 3.0, 6.0, 9.0):
            group = [r for r in rows if r["family"] == family and r["width"] == width]
            record = {"family": family, "width": width, "n": len(group)}
            for metric in METRICS:
                values = [r[metric] for r in group if r[metric] is not None]
                record[metric] = {
                    "n": len(values),
                    "mean": float(np.mean(values)) if values else None,
                    "quantiles": np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).tolist()
                    if values
                    else None,
                }
            record["spatial_discordant_successes"] = sum(
                r["success"] == 1
                and r["interior_nonzero_fraction"] is not None
                and r["interior_nonzero_fraction"] < 1
                for r in group
            )
            tables.append(record)
    # Individual realizations are displayed, but shared seeds make them dependent.
    for width in (0.0, 3.0, 6.0, 9.0):
        fig, axes = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)
        for metric, ax in zip(METRICS, axes.flat, strict=True):
            for i, family in enumerate(families):
                values = sorted(
                    r[metric]
                    for r in rows
                    if r["family"] == family and r["width"] == width and r[metric] is not None
                )
                if values:
                    offsets = np.linspace(-0.18, 0.18, len(values))
                    ax.scatter(i + offsets, values, s=8, alpha=0.5)
            ax.set_xticks(range(len(families)), families, rotation=30, fontsize=7)
            ax.set_title(metric)
        fig.suptitle(
            f"W={width:g} | individual raw realizations | shared seeds: dependent observations"
        )
        fig.savefig(directory / "plots" / f"raw_diagnostics_W{width:g}.png", dpi=140)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    widths = (3.0, 6.0, 9.0)
    for family in families:
        identities = [c["id"] for c in candidates.values() if c["family"] == family]
        values = [
            [
                np.mean(
                    [
                        r["quality"]
                        for r in rows
                        if r["candidate"] == identity and r["width"] == width
                    ]
                )
                for identity in identities
            ]
            for width in widths
        ]
        quantiles = np.array([np.quantile(v, [0.25, 0.5, 0.75]) for v in values])
        axes[0].plot(widths, quantiles[:, 1], marker="o", label=family)
        axes[0].fill_between(widths, quantiles[:, 0], quantiles[:, 2], alpha=0.15)
        if family != "regular":
            differences = [
                [paired_contrast(rows, identity, width)["quality"] for identity in identities]
                for width in widths
            ]
            quantiles = np.array([np.quantile(v, [0.25, 0.5, 0.75]) for v in differences])
            axes[1].plot(widths, quantiles[:, 1], marker="o", label=family)
            axes[1].fill_between(widths, quantiles[:, 0], quantiles[:, 2], alpha=0.15)
    axes[0].set(xlabel="W", ylabel="Candidate mean Q: family median and IQR")
    axes[1].set(xlabel="W", ylabel="Paired mean Q difference to regular: median and IQR")
    axes[1].axhline(0, color="black", linestyle="--")
    axes[0].legend(fontsize=8)
    fig.suptitle("Descriptive geometry variation; bands are not confidence intervals")
    fig.savefig(directory / "plots" / "quality_baseline_curves.png", dpi=140)
    plt.close(fig)
    old = ResearchStore("results/phase18-confirmation")
    old.integrity_check()
    with old.connect(readonly=True) as db:
        old_checksums = [
            [r["id"], r["checksum"]]
            for r in db.execute(
                "SELECT id,checksum FROM objects WHERE kind='exact_result' ORDER BY id"
            )
        ]
    previous = json.loads(Path("docs/decisions/phase18_confirmation_summary.json").read_text())
    assert digest(old_checksums) == previous["numerical"]["result_checksum_digest"]
    audit = {
        "state": state,
        "attempts": dict(Counter(a["status"] for a in store.attempts())),
        "result_checksum_digest": digest(checksums),
        "records": len(rows),
        "raw_distributions": tables,
        "historical_2510_checksums_unchanged": True,
        "source_manifest": store.get("manifest"),
        "scope": "Descriptive only; no pooled independent-observation inference",
    }
    target = Path("docs/decisions/phase19_exploratory_audit.json")
    target.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in audit.items() if k not in ("raw_distributions", "source_manifest")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
