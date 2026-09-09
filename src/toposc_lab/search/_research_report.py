"""Frozen paired statistics and descriptive reporting from retained results only."""

from __future__ import annotations

from collections import Counter
from io import BytesIO
from math import comb, sqrt
from pathlib import Path
from typing import Any

from toposc_lab.evaluation.reproducibility import exact_geometry_id
from toposc_lab.search._research_storage import publish_derived
from toposc_lab.search.lexicographic_fitness import LexicographicFitness
from toposc_lab.search.phase_10_research import RESEARCH_PROTOCOL_ID, TRIAL_SEEDS
from toposc_lab.search.search_benchmark import SearchBenchmarkTrial


def exact_paired_pvalue(evolution_only: int, random_only: int) -> float:
    """Two-sided exact McNemar probability, including zero discordant pairs."""
    if any(type(n) is not int or n < 0 for n in (evolution_only, random_only)):
        raise ValueError("discordant counts must be nonnegative integers")
    n = evolution_only + random_only
    if n == 0:
        return 1.0
    return float(
        min(1.0, 2 * sum(comb(n, j) for j in range(min(evolution_only, random_only) + 1)) / 2**n)
    )


def wilson_interval(hits: int, count: int) -> tuple[float, float]:
    """Frozen two-sided 95% Wilson interval over independent trial outcomes."""
    if type(hits) is not int or type(count) is not int or not 0 <= hits <= count or count < 1:
        raise ValueError("Wilson interval requires integer 0 <= hits <= count and count > 0")
    z = 1.959963984540054
    p = hits / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def research_summary(
    trials: tuple[SearchBenchmarkTrial, ...],
    reference_proxy: float,
    *,
    preflight: bool,
) -> dict[str, Any]:
    """Separate the sole primary test from all descriptive sensitivity views."""
    if not trials:
        raise ValueError("summary requires completed pairs")
    table = Counter((any(t.evolution_arm.successes), any(t.random_arm.successes)) for t in trials)
    b, c = table[(True, False)], table[(False, True)]
    complete = not preflight and tuple(t.seed for t in trials) == TRIAL_SEEDS
    pvalue = exact_paired_pvalue(b, c) if complete else None
    decision = "preflight_only" if preflight else "incomplete_no_primary_inference"
    if complete:
        decision = "no_statistically_resolved_difference"
        if pvalue is not None and pvalue <= 0.05:
            decision = "evolution_search_advantage" if b > c else "evolution_search_disadvantage"
    arms: dict[str, Any] = {}
    selections: dict[str, list[dict[str, Any]]] = {}
    for arm_name in ("evolution", "random"):
        rows: list[dict[str, Any]] = []
        nominated = []
        curves = {str(factor): [0] * 32 for factor in (1.0, 1.05, 1.1, 1.2)}
        failure_counts: Counter[str] = Counter()
        for trial_index, trial in enumerate(trials):
            arm = getattr(trial, arm_name + "_arm")
            members = [m for history in arm.fitness_history for m in history.members]
            records: list[dict[str, Any]] = []
            best_history: list[list[float] | None] = []
            best: tuple[float, ...] | None = None
            for attempt, member in enumerate(members, 1):
                failure_counts[member.status.value] += 1
                if isinstance(member.fitness, LexicographicFitness):
                    values = member.fitness.values
                    best = values if best is None else max(best, values)
                    records.append(
                        {
                            "trial_index": trial_index,
                            "trial_seed": trial.seed,
                            "attempt_index": attempt - 1,
                            "geometry_id": exact_geometry_id(
                                member.population_member.genome.to_geometry()
                            ),
                            "values": list(values),
                        }
                    )
                best_history.append(None if best is None else list(best))
            eligible = [r for r in records if r["values"][0] == 1.0]
            ordered = sorted(records, key=lambda r: tuple(-v for v in r["values"]))
            strong = [
                r
                for r in ordered
                if r["values"][0] == 1.0 and r["values"][1] >= 1.10 * reference_proxy
            ]
            if strong:
                nominated.append(strong[0])
            pareto = [
                r
                for r in eligible
                if not any(
                    other["values"][1] >= r["values"][1]
                    and other["values"][2] >= r["values"][2]
                    and other["values"][1:] != r["values"][1:]
                    for other in eligible
                )
            ]
            for factor in (1.0, 1.05, 1.1, 1.2):
                hits = [
                    r["attempt_index"]
                    for r in eligible
                    if r["values"][1] >= factor * reference_proxy
                ]
                if hits:
                    for index in range(min(hits), 32):
                        curves[str(factor)][index] += 1
            ids = [exact_geometry_id(m.population_member.genome.to_geometry()) for m in members]
            rows.append(
                {
                    "seed": trial.seed,
                    "first_hit": arm.final_progress.first_success_attempt,
                    "success": any(arm.successes),
                    "available": len(records),
                    "best_so_far": best_history,
                    "best_eligible_proxy": max((r["values"][1] for r in eligible), default=None),
                    "duplicate_attempts": len(ids) - len(set(ids)),
                    "elite_counts_by_transition": (
                        [t.elitism.elite_count for t in trial.evolution.transitions]
                        if arm_name == "evolution"
                        else []
                    ),
                    "primary_ranking": [r["attempt_index"] for r in ordered],
                    "sensitivity_boundary_first": [
                        r["attempt_index"]
                        for r in sorted(
                            records,
                            key=lambda r: (-r["values"][0], -r["values"][2], -r["values"][1]),
                        )
                    ],
                    "eligible_pareto_attempts": [r["attempt_index"] for r in pareto],
                }
            )
        selected: list[dict[str, Any]] = []
        ids_seen = set()
        for item in sorted(
            nominated,
            key=lambda r: (
                tuple(-v for v in r["values"]),
                r["geometry_id"],
                r["trial_index"],
                r["attempt_index"],
            ),
        ):
            if item["geometry_id"] not in ids_seen:
                selected.append(item)
                ids_seen.add(item["geometry_id"])
            if len(selected) == 8:
                break
        selections[arm_name] = selected
        hit_count = sum(row["success"] for row in rows)
        arms[arm_name] = {
            "hit_count": hit_count,
            "trial_count": len(trials),
            "hit_fraction": hit_count / len(trials),
            "wilson_95": list(wilson_interval(hit_count, len(trials))),
            "trials": rows,
            "status_counts": dict(failure_counts),
            "sensitivity_hit_curves": curves,
        }
    differences = []
    for evolution, random in zip(
        arms["evolution"]["trials"], arms["random"]["trials"], strict=True
    ):
        left, right = evolution["best_eligible_proxy"], random["best_eligible_proxy"]
        a, b_values = evolution["best_so_far"][-1], random["best_so_far"][-1]
        differences.append(
            {
                "seed": evolution["seed"],
                "best_eligible_proxy_difference": (
                    None if left is None or right is None else left - right
                ),
                "lexicographic_comparison": None
                if a is None or b_values is None
                else ((a > b_values) - (a < b_values)),
                "evolution_available": a is not None,
                "random_available": b_values is not None,
            }
        )
    return {
        "protocol_id": RESEARCH_PROTOCOL_ID,
        "complete_primary_comparison": complete,
        "reference_proxy": reference_proxy,
        "threshold": 1.10 * reference_proxy,
        "paired_table": {
            "both": table[(True, True)],
            "evolution_only": b,
            "random_only": c,
            "neither": table[(False, False)],
        },
        "hit_fraction_difference": (b - c) / len(trials),
        "exact_mcnemar_pvalue": pvalue,
        "primary_decision": decision,
        "arms": arms,
        "paired_quality": differences,
        "selected_candidates": selections,
        "claim_boundary": "finite-size clean search efficiency; no disorder or causal geometry claim",
    }


def render_report(
    directory: Path,
    summary: dict[str, Any],
    trials: tuple[SearchBenchmarkTrial, ...],
) -> None:
    """Create static scientific figures and a readable report without reevaluating physics."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7, 4), layout="constrained")
    axis = figure.subplots()
    for name in ("evolution", "random"):
        data = summary["arms"][name]
        axis.step(
            range(1, 33),
            [n / len(trials) for n in data["sensitivity_hit_curves"]["1.1"]],
            where="post",
            label=name,
        )
    axis.set(
        xlabel="Evaluation attempts per arm",
        ylabel="Fraction of trials with a hit",
        ylim=(0, 1),
        title="Clean 64-site screening; threshold = 1.10 R",
    )
    axis.legend()
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=150)
    publish_derived(directory / "hit_curves.png", buffer.getvalue())
    lines = [
        "# Phase-10-Forschungslauf",
        "",
        f"Protokoll: `{RESEARCH_PROTOCOL_ID}`",
        "",
        f"Ergebnis: **{summary['primary_decision']}**",
        "",
        (
            "Dieser Versuch vergleicht die saubere Suchleistung bei 64 Knoten. "
            "Disorder-Robustheit und kausale Geometrievorteile wurden nicht untersucht."
        ),
        "",
        (
            f"Referenzproxy R: {summary['reference_proxy']:.10g}; "
            f"Trefferschwelle: {summary['threshold']:.10g}."
        ),
        "",
        "| Arm | Erfolgreiche Versuche | Trefferanteil | 95%-Wilson-Intervall |",
        "| --- | ---: | ---: | --- |",
    ]
    for name in ("evolution", "random"):
        data = summary["arms"][name]
        lines.append(
            f"| {name} | {data['hit_count']}/{len(trials)} | {data['hit_fraction']:.3f} | "
            f"{data['wilson_95'][0]:.3f}–{data['wilson_95'][1]:.3f} |"
        )
    lines += [
        "",
        f"Gepaarte Tabelle: `{summary['paired_table']}`.",
        "",
        (
            f"Primärer exakter p-Wert: `{summary['exact_mcnemar_pvalue']}`. "
            "`None` bedeutet: keine primäre Inferenz (z. B. Vorlauf)."
        ),
        "",
        "![Trefferkurven](hit_curves.png)",
        "",
        (
            "Rohwerte, Fehler, Sensitivitätsrankings und Kandidatenrollen stehen in `summary.json`. "
            "Wiederholte Bewertungen sind keine unabhängigen Replikationen."
        ),
        "",
    ]
    for name, selections in summary["selected_candidates"].items():
        for index, selection in enumerate(selections):
            trial = trials[selection["trial_index"]]
            arm = getattr(trial, name + "_arm")
            members = [m for h in arm.fitness_history for m in h.members]
            genome = members[selection["attempt_index"]].population_member.genome
            coords = genome.coordinates
            assert coords is not None
            plot = Figure(figsize=(5, 5), layout="constrained")
            ax = plot.subplots()
            for edge in genome.edges:
                points = coords[[edge.source, edge.target]]
                ax.plot(points[:, 0], points[:, 1], color="0.6", linewidth=0.8)
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=[
                    "tab:orange" if i in genome.boundary_sites else "tab:blue"
                    for i in range(genome.n_sites)
                ],
                s=16,
            )
            ax.set(aspect="equal", title=f"{name} candidate {index + 1}; clean screening only")
            buffer = BytesIO()
            plot.savefig(buffer, format="png", dpi=150)
            filename = f"candidate_{name}_{index:02d}.png"
            publish_derived(directory / filename, buffer.getvalue())
            lines += [f"![{name} Kandidat {index + 1}]({filename})", ""]
    publish_derived(directory / "report.md", "\n".join(lines).encode("utf-8"))
