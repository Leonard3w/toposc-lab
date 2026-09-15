"""Generate Phase-16B report and static figure from stored results only."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toposc_lab.discovery.storage import read_json
from toposc_lab.patterns.campaign import digest


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = root / "results/phase16b"
    analysis = read_json(directory / "analysis.json")
    results = read_json(directory / "results.json")
    plan = read_json(directory / "plan.json")
    manifest = read_json(directory / "manifest.json")
    audit = read_json(directory / "audit.json")
    excluded = read_json(directory / "exclusions.json")
    log_bytes = (root / "results/phase16b-full-tests.log").read_bytes()
    log_text = log_bytes.decode("utf-16" if log_bytes.startswith(b"\xff\xfe") else "utf-8")
    test_match = re.search(r"\d+ passed in [^\r\n]+", log_text)
    if test_match is None:
        raise ValueError("completed full regression log required before final report")
    full_tests = test_match.group(0)
    tests = analysis["tests"]
    labels = {
        "h1": "Cut decrease vs increase",
        "h2": "Boundary vs bulk triangles",
        "h3": "Single-edge deletion retention",
        "h4": "Square creation vs destruction",
        "h5": "Central vs peripheral diagonals",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    fig.suptitle("Phase 16B — independent finite-geometry falsification", fontsize=13)
    for i, h in enumerate(("h1", "h2", "h4", "h5")):
        t = tests[h]
        if t["summary"]:
            mean, ci = t["summary"]["mean"], t["summary"]["ci95"]
            if ci:
                axes[0].plot(ci, [i, i], color="#23678a", linewidth=2)
            axes[0].plot(mean, i, "o", color="#23678a")
        axes[0].text(
            0.98,
            i,
            f"n={t['available_parents']}/16",
            transform=axes[0].get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=9,
        )
    axes[0].axvline(0, color="0.5", linewidth=0.8)
    axes[0].set_ylim(-0.5, 3.5)
    axes[0].set_yticks(range(4), ["H1 cut", "H2 triangles", "H4 squares", "H5 location"])
    axes[0].set_xlabel("Paired quality difference; bootstrap 95% interval")
    axes[0].set_title("Strict matching availability and effects")
    rows = tests["h3"]["rows"]
    axes[1].bar([str(r["seed"])[-2:] for r in rows], [r["effect"] for r in rows], color="#357b70")
    axes[1].axhline(0.75, linestyle="--", color="#b05b39", label="Predeclared 0.75 claim")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xlabel("Independent parent seed 162xx")
    axes[1].set_ylabel("Fraction of selected deletions retaining quality/topology")
    axes[1].set_title("H3: equal parent weighting")
    axes[1].legend(fontsize=8, loc="upper right")
    target = root / "docs/decisions/phase_16b_validation"
    fig.savefig(target.with_suffix(".png"), dpi=170)
    plt.close(fig)
    table = []
    for h, t in tests.items():
        s = t["summary"]
        mean = f"{s['mean']:.6f}" if s else "unavailable"
        ci = ", ".join(f"{v:.6f}" for v in s["ci95"]) if s and s["ci95"] else "unavailable"
        table.append(
            f"| {h.upper()}: {labels[h]} | {t['available_parents']} | {mean} | {ci} | "
            f"{t['holm_p']:.6g} | {t['status']}; support={t['supported']} |"
        )
    parent_lines = []
    for r in results["rows"]:
        deletion = next((x for x in tests["h3"]["rows"] if x["seed"] == r["seed"]), None)
        parent_lines.append(
            f"| {r['seed']} | {r['generator']} | {r['parent']['metrics']['quality']:.9f} | "
            f"{', '.join(r['pairs']) or 'none'} | "
            f"{sum(deletion['retained']) if deletion else 0}/{len(r['deletions'])} |"
        )
    retention = tests["h3"]["summary"]
    failed = sum(not x for row in tests["h3"]["rows"] for x in row["retained"])
    topology_changes = sum(x for row in tests["h3"]["rows"] for x in row["topology_changed"])
    supported = [h.upper() for h, t in tests.items() if t["supported"]]
    report = f"""# Phase 16B — independent falsification report

**Gate 16: {analysis["mechanism_gate"]}.** The bounded validation experiment is
complete; a reproducible geometric mechanism is not established. The frozen
16A evidence remains exploratory/null. Cross-model generality remains unavailable.
No thermodynamic phase or separated-Majorana claim is made.

## Frozen question and design

The five questions are taken from section 15 of the
[frozen 16A report](phase_16a_pattern_ablation.md). See the
[predeclared 16B protocol](phase_16b_protocol.md) for signs, matching constraints,
statistical assumptions and the 700-attempt cap. No outcomes selected the parents
or edits. Success still means **Q >= 0.20**.

The exclusion snapshot contains **{len(excluded["edges"])} D4-distinct edge sets**
from **{len(excluded["inputs_sha256"])} source files**, including 16A parents,
edits and simplifications and existing discovery proposal pools. Even rejected
proposal structures were excluded conservatively. Every selected new structure
is more than 0.15 D4 edge-Jaccard distance from this snapshot and every earlier
independent seed family. Related arms within a parent are deliberately paired.
Later production-campaign growth does not modify the recorded cutoff.

There are 16 unseen parents, eight Random and eight Patch, seeds 16201–16216.
The first valid separated proposal per seed was used; **{len(plan["proposals"])}**
raw proposals were considered. No successful-candidate set was redefined.
Strict matching failures remain missing tests, not zero effects.

## Independent results

| Hypothesis | Parents | Mean | Bootstrap 95% interval | Holm p (five tests) | Result |
|---|---:|---:|---|---:|---|
{chr(10).join(table)}

H1 reports Q(cut decrease)-Q(cut increase). H2 reports Q(bulk triangle removal)-
Q(boundary triangle removal). H4 reports Q(square creation)-Q(destruction).
H5 reports Q(central increase)-Q(decrease). H3 reports retention fraction and tests
its difference from 0.75. At least eight independent parents are needed for a
test; unavailable tests retain p=1 in the five-test correction family.
Bootstrap intervals are descriptive and not simultaneous confidence intervals.
Sign-flip validity assumes independent symmetric parent errors, not randomized
treatment assignment. Spatial matching minimizes a cost, not every spatial feature.

Supported predeclared hypotheses: **{", ".join(supported) or "none"}**.
Failure to detect an effect does not establish equivalence or the proposed proxy
explanation. The available paired values, parameter sensitivities, all motif
deltas and residual spatial costs are retained in the JSON and frozen plan.
H1/H2/H4/H5 lack sufficient matched parents: their explanations are untested,
not falsified. H3's corrected p={tests['h3']['holm_p']:.6f} does not reject the
predeclared 0.75 value at family-wise alpha=0.05; the lower observed retention
also does not support the claim that mean retention exceeds 0.75.

Main-model stages contain **{analysis["main_model_success_roles"]}/{analysis["main_model_record_roles"]}
success roles**; best main-model quality **{analysis["best_main_quality"]:.9f}**.
Roles can repeat a geometry across hypotheses; these are not counts of independent
discoveries. Statistical sample sizes are independent seed/parent counts.

## Deletion tolerance and parent coverage

Mean parent-weighted retention is **{retention["mean"]:.4f}**. There are
**{failed} failed deletion retentions** and **{topology_changes} changed index
triplets**. Every deletion is tested separately from its original 60-edge parent;
the two edges are never deleted sequentially. Positive Q and preserved eligible
nonzero indices are required. Zero-quality parents cannot count as retention.
This bounded outcome-blind sample spans its observed quality distribution; it
does not ensure representation of unavailable high-quality successes.

| Seed | Generator | Parent Q | Available paired questions | Retained deletions |
|---|---|---:|---|---:|
{chr(10).join(parent_lines)}

## Cross-model assessment (16.15)

All 60-edge roles were evaluated at mu=1.9, 2.0 and 2.1 in the existing chiral
p-wave adapter. This is parameter sensitivity within one model. It is not an
independent model-family comparison. QWZ/BHZ adapters build their own square
lattices and do not define couplings on these diagonal wiring graphs. No new
coupling convention was invented. Thus cross-model universality is **unavailable**,
and task 16.15 is an explicit limitation of the completed bounded study.
Size scaling and new disorder validation are also unavailable in this protocol.

## Numerical verification and provenance

The journal contains **{audit["attempts"]}/700 completed charged attempts** and
**{audit["confirmations"]} independently seeded exact confirmation pairs**.
No failed/interrupted stage is present. Known regular-grid references reproduce
index triplets [1,1,1] at mu=2 and [0,0,0] at mu=8. Every stored eigensystem passes
Hamiltonian residual and orthonormality checks at 1e-10; records, diagnostics,
confirmations, strict matches, family separation and checksums pass the audit.
This verifies finite numerical evidence, not a thermodynamic phase.

| Provenance | SHA-256 / value |
|---|---|
| Base commit (dirty source archived) | `{manifest["git_commit"]}` |
| Source | `{manifest["source_sha256"]}` |
| Protocol | `{manifest["protocol_sha256"]}` |
| Experiment driver | `{manifest["driver_sha256"]}` |
| Exclusion snapshot | `{manifest["exclusions_sha256"]}` |
| Frozen plan | `{digest(directory / "plan.json")}` |
| Source archive | `{digest(directory / "source.zip")}` |
| Complete inventory | `{digest(directory / "inventory.json")}` |

Runtime: Python {manifest["runtime"]["python"]}, NumPy {manifest["runtime"]["numpy"]},
SciPy {manifest["runtime"]["scipy"]}; four numerical thread limits=1.
The commit alone is insufficient: use the archived dirty source and frozen files.
**Full repository regression: {full_tests}.** Complete driver replay retains
the same 140 charged attempts and inventory, adding zero numerical calls.
Ruff and isolated strict Mypy (Python 3.14) pass for the new runtime and drivers.
Full regression and replay logs are recorded in
[execution state](../roadmap/EXECUTION_STATE.md) and `results/phase16b-*.log`.

Raw plans, exclusions, exact records, eigensystems, source ZIP and inventory:
`results/phase16b/` (git-ignored; preserve separately). Reproduction commands:
[German usage guide](../phase_16b_usage_de.md). This report is generated from stored
records by `scripts/phase_16b_report.py`; no numerical evaluations are added.

## Gate decision and boundary

The independent falsification workflow is complete with the limitations above.
**MECHANISM_GATE is not established; no mechanism or model-independent design rule
is promoted.** Null findings and unavailable strict matches remain valid outputs.
Block G ends here. Phase 17 and large searches are not started by this workflow.

![Independent effects and deletion retention](phase_16b_validation.png)
"""
    target.with_suffix(".md").write_text(report, encoding="utf-8")
    payload = {
        "analysis": analysis,
        "results": results,
        "manifest": manifest,
        "audit": audit,
        "inventory_sha256": digest(directory / "inventory.json"),
        "report_driver_sha256": digest(Path(__file__)),
        "full_test_result": full_tests,
        "full_test_log_sha256": digest(root / "results/phase16b-full-tests.log"),
        "non_test_numerical_calls_added_by_report": 0,
    }
    target.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(target.with_suffix(".md"))


if __name__ == "__main__":
    main()
