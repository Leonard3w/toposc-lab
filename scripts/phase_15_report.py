"""Derive the final Phase-15 report after the full suite has passed."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from toposc_lab.active_learning.benchmark import source_provenance
from toposc_lab.data import DatasetRecord, load_dataset, record_to_dict
from toposc_lab.discovery.storage import read_json


def metric(record: DatasetRecord, name: str) -> float:
    return float(cast(float, record.observables[0].values[name]))


def main() -> None:
    root = Path.cwd()
    campaign = root / "results/phase15-gate-final"
    destination = root / "docs/decisions"
    gate = read_json(campaign / "gate.json")
    previous = read_json(root / "results/phase15-gate/gate.json")
    review = read_json(root / "results/phase15-final-review.json")
    total_non_test = gate["total_exact_attempts"] + previous["total_exact_attempts"]
    log_bytes = (root / "results/phase15-full-tests.log").read_bytes()
    full_text = log_bytes.decode("utf-16" if log_bytes.startswith(b"\xff\xfe") else "utf-8-sig")
    match = re.search(r"(\d+) passed in ([\d.]+)s", full_text)
    if match is None or re.search(r"\bFAILED\b|\d+ failed|\d+ errors", full_text):
        raise ValueError("full suite has not passed")
    manifest = read_json(campaign / "reference/manifest.json")
    assert source_provenance(root).runtime["source_sha256"] == manifest["source_sha256"]
    summaries = gate["summaries"]
    names = ("reference", "random", "evolution", "surrogate")
    labels = {
        "reference": "patch (default)",
        "random": "random",
        "evolution": "evolution",
        "surrogate": "patch + optional surrogate",
    }
    records = {n: load_dataset(campaign / n / "dataset.json").records for n in names}
    winners = {n: max(rs, key=lambda r: metric(r, "quality")) for n, rs in records.items()}
    best_arm = max(names, key=lambda n: metric(winners[n], "quality"))
    best = winners[best_arm]
    geometry = best.geometry.to_geometry()
    best_json = {
        "arm": best_arm,
        "record": record_to_dict(best),
        "coordinates": geometry.coordinates.tolist() if geometry.coordinates is not None else None,
        "edges": [[e.source, e.target] for e in geometry.edges],
    }
    (destination / "phase_15_best_candidate.json").write_text(
        json.dumps(best_json, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    shutil.copyfile(
        campaign / "reference/discovery.png", destination / "phase_15_autonomy_gate.png"
    )
    unique = {r.geometry.exact_id for rs in records.values() for r in rs}
    audits = read_json(campaign / "audit.json")
    count_table = [
        "| Arm | Raw generated | Valid pool | Exact candidates | Invalid | Duplicate | OOD* | Best exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    physics_table = [
        "| Arm | Eligible / 12 | Unresolved topology | Mean / minimum distance | Winner disorder mean ± SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        s = summaries[name]
        count_table.append(
            f"| {labels[name]} | {s['raw_generated']} | {s['valid_pool_candidates']} | "
            f"{s['validated_dataset_records']} | {s['invalid']} | {s['duplicate']} | "
            f"{s['selected_ood']} | {metric(winners[name], 'quality'):.9f} |"
        )
        unresolved = Counter(r.topology[0].validity.value for r in records[name])["unresolved"]
        rob = winners[name].robustness[0].statistics
        physics_table.append(
            f"| {labels[name]} | {sum(int(metric(r, 'eligible')) for r in records[name])}/12 | "
            f"{unresolved} | {s['mean_pairwise_distance']:.4f} / "
            f"{s['minimum_pairwise_distance']:.4f} | "
            f"{rob['quality_mean']:.9f} ± {rob['quality_sd']:.9f} |"
        )
    primary = summaries["reference"]
    best_robust = best.robustness[0]
    primary_best = winners["reference"]
    best_majorana = best.observables[1].values
    primary_curve = " → ".join(f"{c['best_exact']:.9f}" for c in primary["curves"])
    total_generated = sum(s["raw_generated"] for s in summaries.values())
    total_invalid = sum(s["invalid"] for s in summaries.values())
    total_duplicate = sum(s["duplicate"] for s in summaries.values())
    total_wall = sum(s["recorded_exact_wall_seconds"] for s in summaries.values())
    total_cycle_wall = sum(s["completed_cycle_wall_seconds"] for s in summaries.values())
    elapsed = float(match[2])
    report: dict[str, Any] = {
        "gate": "AUTONOMY_GATE",
        "status": "PASS",
        "block": "E",
        "tasks": "15.1-15.23",
        "phase16_started": False,
        "experiment": gate,
        "previous_source_gate_attempts": previous["total_exact_attempts"],
        "total_non_test_attempts": total_non_test,
        "final_artifact_review": review,
        "full_suite": {
            "passed": int(match[1]),
            "wall_seconds": elapsed,
            "log": "results/phase15-full-tests.log",
        },
        "regression": "128 passed; final ordering/cadence checks 2 passed",
        "lint": "Ruff PASS",
        "typing": "isolated Mypy Python 3.14 PASS",
        "source_sha256": manifest["source_sha256"],
        "protocol_sha256": sha256((destination / "phase_15_protocol.md").read_bytes()).hexdigest(),
        "gate_driver_sha256": gate["gate_driver_sha256"],
        "report_driver_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_commit": manifest["provenance"]["git_commit"],
        "dirty": True,
        "unique_geometries_across_control_campaigns": len(unique),
        "audited_files": sum(len(a["inventory"]) for a in audits.values()),
        "trust": "Suitable for small monitored finite-wiring discovery campaigns; "
        "no autonomous physics certification or large campaign authorization.",
    }
    (destination / "phase_15_autonomy_gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    text = f"""# Phase-15 Gate Report — Autonomous Discovery Engine

**Block E complete; AUTONOMY_GATE PASS. Phase 16 has not started.**

The loop is operationally trustworthy for small, monitored exact discovery
campaigns in the validated finite wiring stratum. It does not certify Majorana
physics, robustness beyond the tested ensemble, or a thermodynamic phase. This
gate validates autonomous orchestration; it is not a new generator-superiority
benchmark or authorization for a large search.

## Scope and generator policy

Tasks **15.1–15.23** are implemented. The search uses the Phase-14 6×6 unit grid,
36 sites, 60 planar edges, fixed perimeter, degrees 2–6, connected, no crossings,
and edge length ≤√2. Patch is the default only for this stratum and the frozen
finite-localizer objective, based on the accepted Phase-14 comparison. Random
and Evolution remain supported baselines; coverage and surrogate acquisition
are optional. Evolution retains its established baseline role outside this
scoped result. Unsupported spaces fail closed pending an exact adapter and fair
baseline benchmark. No RL or learned generator was added.

All arms use the same random warm pool in cycle 1. The primary seed is **15201**,
with three cycles, eight valid proposals and four selected exact candidates per
cycle. The model remains chiral p-wave, t=Δ=1, μ=2, chirality +1; component-major
Nambu basis, κ=(0.1,0.2,0.3), center (2.5,2.5), tolerance 10⁻¹⁰. Quality remains
the minimum localizer gap gated by agreeing nonzero indices and PHS. **Success
remains quality ≥0.20; all clean candidate success counts are zero.**

## Generation, selection and exact results

{chr(10).join(count_table)}

*OOD counts include four initial candidates per arm with `reference_unavailable`,
conservatively flagged rather than called in-distribution. Later common flags
use only the first four completed exact records. Every selected OOD candidate
receives full exact verification. Surrogate OOD never drives exploitation.

Each arm performs 12 clean exact evaluations, 12 independent exact confirmations
and 48 exact disorder evaluations: **72**. Unselected proposals receive no physics
labels. The optional surrogate makes 16 predictions across cycles 2–3 using the
same four-label training prefix (two-cycle retraining cadence); no future label
enters training, and no prediction is stored as exact data.

The default campaign improves then stagnates: **{primary_curve}**. All four arms
improve in cycle 2 and do not improve their running best in cycle 3. Random wins
this one-seed smoke comparison; it neither overturns the 20-seed Phase-14 result
nor demonstrates a general Random advantage. There is no sample-efficiency or
success-threshold improvement claim. No runaway duplicates, invalid accepted
structures, OOD exploitation or numerical failures occurred in the gate.

## Best exact candidates and scientific validation

Default patch winner: quality **{metric(primary_best, "quality"):.9f}**, geometry
`{primary_best.geometry.exact_id}`.

Best across the four arms: **{best_arm}**, quality **{metric(best, "quality"):.9f}**,
geometry `{best.geometry.exact_id}`. Its
[complete candidate record](phase_15_best_candidate.json) contains explicit
coordinates/edges, model, full spectrum, raw diagnostics, disorder samples and
provenance. Every selected candidate, including both winners, passed independent
exact re-evaluation before dataset integration.

{chr(10).join(physics_table)}

Topology applicability is limited to the 2D class-D finite center localizer.
Raw indices, signatures and gaps at every κ are retained. Conflicting indices
remain `unresolved`, with zero eligible quality; agreement on a trivial zero
index is distinguished from nonzero evidence. Known regular-grid μ=2 positive
and μ=8 trivial references pass. No other invariant is applied outside its domain.

The overall winner has localizer indices
`{list(cast(tuple[int, ...], best.topology[0].parameters["indices"]))}` and minimum |E| =
**{metric(best, "minimum_abs_energy"):.9f}**. Majorana validation retains the
operator PHS residual (**{cast(float, best_majorana["operator_phs_residual"]):.3g}**),
spectral PHS, four low-energy state profiles, IPR, boundary weights, complex
local polarization, particle/hole weights and self-conjugacy. The existing
splitting diagnostics and their tolerances are explicitly recorded. These
finite chiral boundary states do **not** establish spatially separated Majorana
zero modes; the report status is `diagnostics_only`, with `majorana_claim=false`.

Robustness uses four independent exact scalar onsite-disorder realizations per
candidate, uniform offsets in [-0.1,0.1] with opposite particle/hole signs.
The overall winner has mean quality **{best_robust.statistics["quality_mean"]:.9f}**,
SD **{best_robust.statistics["quality_sd"]:.9f}**, standard error
**{best_robust.uncertainty["quality_standard_error"]:.9f}**, and minimum quality
**{best_robust.statistics["quality_min"]:.9f}**. Its frozen-success fraction is
**{best_robust.statistics["success_fraction"]:.0%}**, with 95% Wilson interval
**[{best_robust.uncertainty["success_wilson_lower"]:.4f},
{best_robust.uncertainty["success_wilson_upper"]:.4f}]**. The wide interval and
four-sample ensemble do not certify robustness; no geometric disorder, hopping
or pairing ensemble was run in this gate.

Finite-size/family evidence is explicitly **unavailable**. The fixed wiring
stratum has no candidate-specific size-extension map; unrelated sizes or nearby
μ values are not substituted for scaling. No thermodynamic or size-stability
claim follows from these results.

## Diversity, dataset integrity and recovery

The symmetry-minimized edge-Jaccard exclusion radius stays at 0.06. Filtering
covers earlier valid pools, within-batch structures and an optional external
dataset. Accepted candidates exceed that radius; raw rejection denominators
appear above. Across the seven campaign instances, **{total_generated}** raw
proposals yield 168 valid pool entries and 84 exact validated dataset records,
with **{total_invalid} invalid** and **{total_duplicate} duplicate** rejections.
Repeated campaigns and shared warm starts are intentional: there are
**{len(unique)} distinct exact geometries** across the four control arms, not
84 independent materials. External-dataset duplicate rejection is additionally
verified by a deterministic test that rejects the repeated warm candidates.

Exact records are appended atomically and reconciled from the journal without
duplicate labels. Base eigensystems, independent confirmations, each disorder
realization, failures and pre-exact selections have separate checksummed records.
Cycle commits include inventories; checkpoints identify stable cycle boundaries.
A single-writer OS lease is released by process termination. Completed exact
stages are reused. An interrupted unsaved calculation repeats the same seed and
consumes another attempt; it is never silently free.

Checkpoint after one cycle and resume, independent fresh-timestamp repeat, and
real killed-worker recovery each reproduce **all 12 complete scientific records
exactly**, including their stored disorder realizations. Timestamps and record
IDs are excluded from scientific equality; seeds/config/source are retained and
audited separately. Eight deterministic interruption-boundary tests, solver
failure isolation, retry-cap exhaustion and corruption rejection also pass.

The first gate's Windows kill harness terminated a venv launcher while its
interpreter child briefly retained the lock. The harness was corrected to
terminate the process tree, and that gate recovered with both interrupted
attempts charged. A final review then hardened JSON configuration arrays into
immutable tuples and rejected boolean numerical settings. This changed no
scientific endpoint. All seven campaigns and the full suite were rerun on the
final source. The final real kill/restart passes (exit code
{gate["real_kill_exit_code"]}) with exactly one charged interruption. Final source
archives and `results/phase15-gate-final/gate-driver.py` preserve the executable
campaign implementation; the earlier gate is retained separately.

## Cost, limitations and acceptance

The final seven campaigns require 504 completed evaluations, two reference
evaluations and one interrupted attempt: **{gate["total_exact_attempts"]} final-gate
numerical evaluations/attempts**. The earlier source gate used
**{previous["total_exact_attempts"]}**, including its two interruptions:
**{total_non_test} total Phase-15 non-test evaluations/attempts**. The initial CLI
import failure made zero numerical attempts. Test simulations are separate.
Each evaluation includes one 72-dimensional BdG
eigensystem and three 144-dimensional localizers, not one matrix solve. Each
campaign has an 80-attempt cap; final crash recovery uses 73.

The primary campaign records **{primary["recorded_exact_wall_seconds"]:.3f} s**
inside exact stages and **{primary["completed_cycle_wall_seconds"]:.3f} s** across
completed cycle invocations. Across all instances these measured totals are
**{total_wall:.3f} s** and **{total_cycle_wall:.3f} s**, respectively. Initialization,
report regeneration, reference runs and unrecorded interrupted time are outside
these sums. All four numerical thread limits are one. This is not a memory or
hardware-speedup measurement; the full suite ran concurrently for part of the gate.

Remaining limits: one geometry/model stratum, coarse four-point OOD reference,
small disorder ensembles, no finite-size sequence, no Majorana certification,
no success-threshold hit, single-writer storage and strict same-source/runtime
resume. Numerical stage failures are isolated and excluded from the accepted
dataset; raw exact successes before a later validation failure remain in the
journal. Source/config mismatch, corrupt committed artifacts, conflicting exact
confirmation and exhausted budget stop execution without weakening definitions.

**Acceptance: PASS for a bounded, monitored real discovery campaign in this
validated stratum.** Broader claims or large campaigns need additional physics,
family validation and a separately authorized protocol. The Pareto leaderboard
keeps clean quality, boundary weight and disorder quality separate, with an
alternate boundary-weighted ranking; novelty never compensates for physics.

Full suite: **{match[1]} passed in {elapsed:.2f} s** (2026-09-14),
`python -B -m pytest -q -p no:cacheprovider --basetemp=results/pytest-phase15-final`.
Subsystem/physics checkpoint: 128 passed; final ordering/cadence checks: 2 passed.
Final immutable-configuration checks: 3 passed. The earlier full suite had 2788
passes before the additional configuration regression was added.
Ruff and isolated Mypy (`--python-version 3.14`) pass. The repository's pre-existing
default Mypy 3.11 target remains incompatible with its installed NumPy stubs.

Base commit: `{report["base_commit"]}` plus recorded dirty Phase-15 additions.
Source SHA-256: `{report["source_sha256"]}`.
Protocol SHA-256: `{report["protocol_sha256"]}`.
Final gate-driver SHA-256: `{report["gate_driver_sha256"]}`.
The audit inventories **{report["audited_files"]} files** across all campaigns.
An independent final review verifies every inventory checksum and all seven
source archives against current source. All 84 earlier-versus-final scientific
records agree exactly; CLI resume adds zero exact calculations.
Earlier Phase-13/14 reports and conventions remain unchanged.

Artifacts: [protocol](phase_15_protocol.md),
[machine-readable report](phase_15_autonomy_gate.json),
[best candidate](phase_15_best_candidate.json),
[usage](../phase_15_usage_de.md); full journals, source archives, exact datasets,
reports and audit in `results/phase15-gate-final/`; earlier gate in
`results/phase15-gate/`. Public additions are `DiscoveryConfig`,
`DiscoveryEngine.run(iterations=...)`, scoped generator policy and the module CLI.

**Stop at Block E. Do not start Phase 16 automatically.**

![Stored exact discovery curves and disorder comparison](phase_15_autonomy_gate.png)
"""
    (destination / "phase_15_autonomy_gate.md").write_text(text, encoding="utf-8")
    print(f"AUTONOMY_GATE PASS; {match[1]} tests; {gate['total_exact_attempts']} non-test attempts")


if __name__ == "__main__":
    main()
