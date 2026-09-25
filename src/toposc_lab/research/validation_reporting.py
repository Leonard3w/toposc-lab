"""Read-only evidence summaries, diagnostic figures and a gated confirmation recipe."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from toposc_lab.research.storage import ResearchStore, atomic_text, dumps
from toposc_lab.research.validation_cohort import digest
from toposc_lab.research.validation_statistics import (
    confirmation_grid,
    paired_interval,
    summarize,
    usable,
)


def load_snapshot(directory: Path) -> dict[str, Any]:
    store = ResearchStore(directory)
    with store.connect(readonly=True) as db:
        db.execute("BEGIN")

        def get(kind: str) -> Any:
            row = db.execute(
                "SELECT * FROM objects WHERE kind=? AND id='current'", (kind,)
            ).fetchone()
            return store.decode(row)

        results = {
            row["id"]: store.decode(row)
            for row in db.execute("SELECT * FROM objects WHERE kind='exact_result' ORDER BY id")
        }
        return {
            "settings": get("validation_settings"),
            "cohort": get("validation_cohort"),
            "manifest": get("manifest"),
            "state": get("state"),
            "results": results,
            "attempts": [dict(r) for r in db.execute("SELECT * FROM attempts ORDER BY number")],
        }


def analyze(snapshot: dict[str, Any]) -> dict[str, Any]:
    settings, cohort = snapshot["settings"], snapshot["cohort"]
    records = {}
    clean = {}
    # Split only at the first colon; keys are frozen candidate:stage.
    for key, result in snapshot["results"].items():
        identity, _ = key.split(":", 1)
        stage = result["stage"]
        if stage["kind"] == "clean":
            clean[identity] = result
        elif stage["kind"] == "disorder":
            records[identity, float(stage["width"]), int(stage["seed"])] = result
    identities = [c["id"] for c in cohort["candidates"]]
    summary = summarize(
        identities,
        settings["widths"],
        settings["seeds"],
        records,
        confirmation=settings["mode"] == "confirmation",
    )
    summary["clean"] = {
        key: {
            "primary_valid": usable(r),
            "metrics": r.get("metrics"),
            "boundary_window": r.get("boundary_window"),
            "chern_marker_status": r.get("chern_marker", {}).get("status"),
        }
        for key, r in clean.items()
    }
    matched_offsets = []
    for width in settings["widths"]:
        for seed in settings["seeds"]:
            available = [records.get((identity, width, seed)) for identity in identities]
            offsets = [r["onsite_offsets"] for r in available if r and "onsite_offsets" in r]
            # Offsets are recovered by subtraction from different matrices with the
            # same diagonal; tolerate only floating round-off, not RNG differences.
            equal = all(np.allclose(v, offsets[0], rtol=0, atol=1e-12) for v in offsets[1:])
            matched_offsets.append(
                {
                    "width": width,
                    "seed": seed,
                    "records": len(offsets),
                    "all_present": len(offsets) == len(identities),
                    "matched": equal,
                }
            )
    if not all(item["matched"] for item in matched_offsets):
        raise ValueError("Unpaired onsite fields in saved study")
    summary["paired_field_checks"] = matched_offsets
    summary["realization_records"] = len(snapshot["results"])
    summary["complete_schedule"] = len(snapshot["results"]) == len(identities) * (
        1 + len(settings["widths"]) * len(settings["seeds"])
    )
    summary["charged_attempts"] = len(snapshot["attempts"])
    summary["attempt_statuses"] = {
        status: sum(a["status"] == status for a in snapshot["attempts"])
        for status in sorted({a["status"] for a in snapshot["attempts"]})
    }
    summary["diagnostic_failures"] = {
        "spatial_invalid_entries": sum(
            not s["valid"] for r in snapshot["results"].values() for s in r.get("spatial", [])
        ),
        "chern_unavailable_or_error": sum(
            r.get("chern_marker", {}).get("status") != "available"
            for r in snapshot["results"].values()
        ),
        "empty_boundary_windows": sum(
            r.get("boundary_window", {}).get("status") == "empty_window"
            for r in snapshot["results"].values()
        ),
    }
    successful = [
        a for a in snapshot["attempts"] if a["status"] == "complete" and a["seconds"] is not None
    ]
    times = [a["seconds"] for a in successful]
    n_confirmation = len(identities) * (1 + 5 * 50)
    mean = float(np.mean(times)) if times else None
    report_rows = [r for r in snapshot["results"].values() if r.get("operations")]
    ops = {
        key: sum(r["operations"][key] for r in report_rows)
        for key in ("hamiltonian_diagonalizations", "localizer_diagonalizations")
    }
    size = sum(len(dumps(r).encode()) for r in snapshot["results"].values())
    summary["cost"] = {
        "exact_stage_seconds": sum(a["seconds"] or 0 for a in snapshot["attempts"]),
        "primary_seconds": sum(
            r.get("timing", {}).get("primary_seconds", 0) for r in snapshot["results"].values()
        ),
        "additional_seconds": sum(
            r.get("timing", {}).get("additional_seconds", 0) for r in snapshot["results"].values()
        ),
        "mean_stage_seconds": mean,
        "p95_stage_seconds": float(np.quantile(times, 0.95)) if times else None,
        "measured_operations": ops,
        "planned_confirmation_realizations": n_confirmation,
        "estimated_confirmation_seconds": mean * n_confirmation if mean else None,
        "conservative_confirmation_seconds_2x": mean * n_confirmation * 2 if mean else None,
        "suggested_exact_budget_with_50_retry_reserve": n_confirmation + 50,
        "raw_result_bytes": size,
        "estimated_confirmation_raw_bytes": size / len(snapshot["results"]) * n_confirmation
        if size
        else None,
        "scope": "One worker/BLAS thread on this machine; 2x allowance, not a guaranteed upper bound. "
        "SQLite WAL, exports and plots need additional disk; no external paid compute.",
    }
    summary["source_sha256"] = snapshot["manifest"]["source_sha256"]
    summary["cohort_sha256"] = cohort["sha256"]
    summary["mode"] = settings["mode"]
    return summary


def plot_results(directory: Path, snapshot: dict[str, Any], summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cohort, results, settings = snapshot["cohort"], snapshot["results"], snapshot["settings"]
    ids = [c["id"] for c in cohort["candidates"]]
    cols, rows = 5, max(1, (len(ids) + 4) // 5)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(ids)))
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    plots = directory / "plots"
    plots.mkdir(exist_ok=True)

    def grid() -> tuple[Any, Any]:
        fig, axes = plt.subplots(
            rows, cols, figsize=(17, 4.2 * rows), squeeze=False, constrained_layout=True
        )
        for ax in axes.ravel()[len(ids) :]:
            ax.set_visible(False)
        return fig, axes.ravel()

    def save(fig: Any, name: str) -> None:
        fig.savefig(plots / (name + ".png"), dpi=150)
        plt.close(fig)

    for quantity in ("success", "quality"):
        fig, axes = grid()
        fig.suptitle(
            "Phase 18 "
            + settings["mode"]
            + " | "
            + quantity
            + " | pointwise descriptive 95% intervals"
        )
        for identity, ax, color in zip(ids, axes, colors):
            entries = [r for r in summary["curves"] if r["candidate"] == identity]
            xs, ys, lo, hi = [], [], [], []
            for r in entries:
                if quantity == "success":
                    y, interval = r["success_fraction_valid"], r["wilson95_conditional_on_valid"]
                else:
                    ci = paired_interval(r["qualities"])
                    y, interval = ci["mean"], ci["interval"]
                    ax.scatter(
                        [r["width"]] * len(r["qualities"]),
                        r["qualities"],
                        color=color,
                        alpha=0.35,
                        s=16,
                    )
                if y is not None:
                    xs.append(r["width"])
                    ys.append(y)
                    lo.append(interval[0] if interval else y)
                    hi.append(interval[1] if interval else y)
            ax.plot(xs, ys, "o-", color=color)
            ax.fill_between(xs, lo, hi, alpha=0.18, color=color)
            ax.set(
                title=identity,
                xlabel="Onsite width W",
                ylabel="success / valid n" if quantity == "success" else "center Q",
            )
            if quantity == "success":
                ax.set_ylim(-0.05, 1.05)
                ax.axhline(0.5, ls=":", color="gray")
            ax.grid(alpha=0.2)
        save(fig, "robustness_" + quantity)

    fig, axes = grid()
    for c, ax in zip(cohort["candidates"], axes):
        coords = np.array(c["geometry"]["coordinates"])
        for a, b in c["geometry"]["edges"]:
            xy = coords[[a, b]]
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color="purple" if np.linalg.norm(xy[1] - xy[0]) > 2**0.5 + 1e-12 else "#486777",
                lw=0.65,
            )
        ax.scatter(coords[:, 0], coords[:, 1], s=7, color="black")
        ax.set(title=c["id"], aspect="equal", xlabel="x", ylabel="y")
    fig.suptitle("Frozen 100-site cohort | purple: length > sqrt(2); overpassed sites unconnected")
    save(fig, "cohort_geometries")

    for kappa in (0.1, 0.2, 0.3):
        fig, axes = grid()
        for identity, ax in zip(ids, axes):
            r = results.get(identity + ":clean", {})
            probes = [
                p for p in r.get("spatial", []) if p["kappa"] == kappa and p["gap"] is not None
            ]
            if probes:
                xy = np.array([p["point"] for p in probes])
                artist = ax.scatter(
                    xy[:, 0],
                    xy[:, 1],
                    c=[p["gap"] for p in probes],
                    vmin=0,
                    vmax=0.6,
                    s=150,
                    cmap="viridis",
                )
                for p in probes:
                    ax.annotate(
                        str(p["index"]) if p["valid"] else "?",
                        p["point"],
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white",
                    )
                fig.colorbar(artist, ax=ax, shrink=0.7, label="localizer gap")
            ax.set(
                title=identity,
                aspect="equal",
                xlabel="x",
                ylabel="y",
                xlim=(-0.7, 9.7),
                ylim=(-0.7, 9.7),
            )
        fig.suptitle(f"Clean localizer probes | kappa={kappa} | labels=index; no interpolation")
        save(fig, "localizer_clean_k" + str(kappa).replace(".", ""))

    fig, axes = grid()
    for identity, ax in zip(ids, axes):
        marker = results.get(identity + ":clean", {}).get("chern_marker", {})
        if marker.get("status") == "available":
            xy = np.array(marker["positions"])
            artist = ax.scatter(
                xy[:, 0],
                xy[:, 1],
                c=marker["values"],
                vmin=-2,
                vmax=2,
                cmap="coolwarm",
                s=40,
                marker="s",
            )
            fig.colorbar(artist, ax=ax, shrink=0.7, label="Chern marker (clipped +/-2)")
            ax.text(
                0.02,
                1.01,
                f"bulk mean={marker['bulk_mean']:.3f}",
                transform=ax.transAxes,
                fontsize=8,
            )
        ax.set(title=identity, aspect="equal", xlabel="x", ylabel="y")
    fig.suptitle(
        "Clean local Chern marker | finite open sample; boundary compensation; bulk distance >=2"
    )
    save(fig, "chern_clean")

    fig, axes = grid()
    width_colors = plt.get_cmap("plasma")(np.linspace(0.05, 0.85, len(settings["widths"]) + 1))
    for identity, ax in zip(ids, axes):
        for width, color in zip([0.0] + settings["widths"], width_colors):
            raw = [
                r
                for key, r in results.items()
                if key.startswith(identity + ":") and r["stage"]["width"] == width
            ]
            windows = [r.get("boundary_window", {}).get("window") for r in raw]
            windows = [w for w in windows if w is not None]
            if windows:
                profiles = np.array([w["shell_weights"] for w in windows])
                ax.plot(
                    range(profiles.shape[1]),
                    profiles.mean(axis=0),
                    "o-",
                    color=color,
                    label=f"W={width:g}, n={len(windows)}",
                )
        ax.set(title=identity, xlabel="distance shell from nearest edge", ylabel="window weight")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6)
    fig.suptitle(
        "Fixed |E|<=0.5 window | mean projector weight | profiles are not exponential fits"
    )
    save(fig, "boundary_profiles")

    fig, axes = grid()
    for identity, ax in zip(ids, axes):
        for width, color in zip([0.0] + settings["widths"], width_colors):
            raw = [
                r
                for key, r in results.items()
                if key.startswith(identity + ":") and r["stage"]["width"] == width
            ]
            for r in raw:
                points = [
                    p for p in r.get("spatial", []) if p["kappa"] == 0.2 and p["point"][1] == 4.5
                ]
                points.sort(key=lambda p: p["point"][0])
                ax.plot(
                    [p["point"][0] for p in points],
                    [p["gap"] if p["valid"] else np.nan for p in points],
                    "o-",
                    color=color,
                    alpha=0.65,
                    markersize=3,
                    label=f"W={width:g}" if r is raw[0] else None,
                )
        ax.set(title=identity, xlabel="x at y=4.5", ylabel="localizer gap, kappa=0.2")
        ax.legend(fontsize=6)
    fig.suptitle(
        "Spatial center-to-boundary cuts | individual realizations | boundary is not a pass criterion"
    )
    save(fig, "localizer_disorder_cuts")

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for identity, color in zip(ids, colors):
        xs, ys = [], []
        for rkey, r in results.items():
            window = r.get("boundary_window", {}).get("window")
            if rkey.startswith(identity + ":") and usable(r) and window is not None:
                xs.append(r["metrics"]["quality"])
                ys.append(window["strip_weights"]["1"])
        ax.scatter(xs, ys, color=color, label=identity, alpha=0.65, s=25)
    ax.set(
        xlabel="center localizer Q",
        ylabel="|E|<=0.5 projector weight in strip d<1",
        title="Quality vs boundary localization | all clean/disorder realizations | no combined score",
    )
    ax.legend(fontsize=8, bbox_to_anchor=(1, 1), loc="upper left")
    ax.grid(alpha=0.2)
    save(fig, "quality_boundary_tradeoff")


def export_results(directory: Path) -> str:
    directory = Path(directory)
    snapshot = load_snapshot(directory)
    summary = analyze(snapshot)
    settings = snapshot["settings"]
    cohort = snapshot["cohort"]
    atomic_text(
        directory / "reports" / "summary.json", json.dumps(summary, indent=2, allow_nan=False)
    )
    atomic_text(
        directory / "reports" / "realizations.jsonl",
        "\n".join(dumps({"record_id": key, **r}) for key, r in snapshot["results"].items()) + "\n",
    )
    stream = io.StringIO(newline="")
    fields = [
        "candidate",
        "width",
        "planned",
        "present",
        "valid",
        "invalid",
        "missing",
        "successes",
        "success_fraction_valid",
        "quality_mean",
    ]
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    writer.writerows({key: row[key] for key in fields} for row in summary["curves"])
    atomic_text(directory / "reports" / "curves.csv", stream.getvalue())
    plot_results(directory, snapshot, summary)
    if settings["mode"] == "pilot" and summary["complete_schedule"]:
        grid = confirmation_grid(summary, settings["widths"])
        proposal = {
            **settings,
            "mode": "confirmation",
            "widths": grid["widths"],
            "seeds": list(range(181001, 181051)),
            "clean_seed": 181000,
            "exact_budget": None,
            "wall_seconds": None,
            "pilot_summary_sha256": digest(summary),
            "grid_selection": grid,
            "authorization": "NOT STARTED: explicit user compute budget required",
        }
        atomic_text(directory / "confirmation_proposal.json", json.dumps(proposal, indent=2))
    cost = summary["cost"]
    lines = [
        "# Phase 18: " + settings["mode"] + " - Forschungsbericht",
        "",
        (
            f"Geplante Realisierungen: {snapshot['state']['planned_realizations']}; "
            f"gespeichert: {summary['realization_records']}; verbuchte Versuche: {summary['charged_attempts']}."
        ),
        f"Vollständiger Ablauf: {summary['complete_schedule']}. Modus: {summary['scope']}.",
        "",
        "## Kohorte (vor neuer Physik eingefroren)",
        "",
        "| Kandidat | Quelle | Langbindungen | Bulk-Gradvarianz | min. Achsenschnitt-Leitwert |",
        "|---|---|---:|---:|---:|",
    ]
    for c in cohort["candidates"]:
        d = c["descriptors"]
        lines.append(
            f"| {c['id']} | {c['source']['kind']} | {d['long_bond_fraction']:.4f} | "
            f"{d['bulk_degree_variance']:.4f} | {d['balanced_cut_min_conductance']:.4f} |"
        )
    lines += [
        "",
        (
            "Kontrollpaare verändern mehrere Deskriptoren gleichzeitig; siehe cohort.json "
            "für sämtliche Confounder und reversible Kanten-Edits. Kein isolierter kausaler Effekt."
        ),
        "",
        "## Zentrumskriterium und Unsicherheit",
        "",
        (
            "Erfolg: unverändertes Phase-17-Kriterium Q>=0.20. Wilson-Intervalle beziehen sich "
            "auf numerisch gültige Realisierungen. Fehlende und ungültige Werte sind separat "
            "ausgewiesen; Worst/Best-Grenzen stehen in summary.json."
        ),
        "",
        "| Kandidat | W | Erfolg/gültig | ungültig/fehlend | Q-Mittel |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary["curves"]:
        q = "fehlend" if r["quality_mean"] is None else f"{r['quality_mean']:.6f}"
        lines.append(
            f"| {r['candidate']} | {r['width']:g} | {r['successes']}/{r['valid']} | "
            f"{r['invalid']}/{r['missing']} | {q} |"
        )
    lines += [
        "",
        "## Gepaarte Qualitätsunterschiede über das ganze W-Raster",
        "",
        (
            "Je Seed erst über Breiten mitteln, dann Seed-Blöcke resampeln. Drei historische "
            "Kontraste verwenden Bonferroni-98.333%-Intervalle; Kontrollen deskriptive 95%. "
            "Der Pilot mit zwei Seeds liefert keine belastbare Bestätigung."
        ),
        "",
    ]
    for r in summary["primary_seed_block_contrasts"]:
        lines.append(
            f"- {r['candidate']}: n={r['pairs']}, Differenz={r['mean']}, Intervall={r['interval']}."
        )
    lines += [
        "",
        "## Räumliche Topologie und Randphysik",
        "",
        (
            "17 Probeorte × 3 Skalen je 100-Orte-Realisierung; Zentrum wird wiederverwendet. "
            "Alle Indizes, Lücken und Gültigkeiten sind in den Einzelresultaten erhalten. "
            "Ränder sind diagnostisch und keine zusätzlichen Bestehensbedingungen."
        ),
        "",
        (
            "Randfenster |E|<=0.5 mit vollständig aufgenommenen nahen Entartungsgruppen. "
            "Randstreifen d<1,<2,<3, Abstandsschalen und mittlere Tiefe; "
            "Zustandsanzahl und leere Fenster separat. Qualität und Randgewicht bleiben getrennt."
        ),
        "",
        f"Diagnostikstatus: `{dumps(summary['diagnostic_failures'])}`.",
        "",
        (
            "Der lokale Chern-Marker ist ein deskriptiver endlicher BdG-Projektormarker "
            "mit explizitem Bulk-Bereich. Weder Mobilitätslücke noch Bulk-Spektrallücke folgen daraus. "
            "Randlokalisierung zertifiziert weder chirale Majorana-Moden noch isolierte Nullmoden."
        ),
        "",
        "## Gemessener Aufwand und Bestätigungsvorschlag",
        "",
        (
            f"Exakte Stufen insgesamt: {cost['exact_stage_seconds']:.3f} s; "
            f"Zentrum: {cost['primary_seconds']:.3f} s; Zusatzdiagnostik: {cost['additional_seconds']:.3f} s."
        ),
        f"Mittlere Stufe: {cost['mean_stage_seconds']} s; P95: {cost['p95_stage_seconds']} s.",
        f"Operationen: `{dumps(cost['measured_operations'])}`.",
        (
            f"Bestätigung: {cost['planned_confirmation_realizations']} geplante Realisierungen; "
            f"Vorschlag mit Reserve: {cost['suggested_exact_budget_with_50_retry_reserve']} Versuche."
        ),
        (
            f"Projektion: {cost['estimated_confirmation_seconds']} s; "
            f"2-facher Zeitansatz: {cost['conservative_confirmation_seconds_2x']} s."
        ),
        (
            f"Geschätzte rohe JSON-Ergebnisse: {cost['estimated_confirmation_raw_bytes']} Bytes; "
            "zusätzlicher Platz für WAL, Exporte und Bilder erforderlich."
        ),
        "",
        (
            "Der große Lauf wurde nicht gestartet. Rasterregel und getrennte Seeds stehen "
            "in confirmation_proposal.json, sofern der Pilot vollständig ist. Die Datei enthält "
            "absichtlich noch kein autorisiertes Rechenbudget."
        ),
        "",
        "## Offene Fragen und zurückgestellte Arbeiten",
        "",
        "- Reproduziert sich der Vorteil auf 50 neuen Seeds, auch bei informativem W?",
        "- Wie groß ist der konsistente Innenbereich über die diskreten Proben hinaus?",
        "- Gehen Qualitätsunterschiede mit anderen Randprofilen/Zustandszahlen einher?",
        "- Welche strukturellen Effekte bleiben nach Kontrolle der anderen Deskriptoren bestehen?",
        "- Größenfamilien, Nachbarparameter, entfernungsabhängige Kopplungen und Transport bleiben vertagt.",
        (
            "- Hopping-Unordnung/Kantenausfälle: allgemeine Bibliothek vorhanden, im eingefrorenen "
            "Phase-17-Adapter nicht auswählbar; benötigen getrennte versionierte Protokolle."
        ),
        "",
        "## Reproduzierbarkeit und Einstieg",
        "",
        f"Quellhash: `{summary['source_sha256']}`. Kohortenhash: `{cohort['sha256']}`.",
        (
            "SQLite mit Payload-Prüfsummen ist maßgeblich; JSONL/CSV/PNG sind reproduzierbare Exporte. "
            "source.zip, protocol.md, study.json und cohort.json bewahren die Definitionen. "
            "results/ ist Git-ignoriert und separat zu sichern."
        ),
        "",
        "```powershell",
        f".venv/Scripts/python.exe -B -m toposc_lab.research.validation report {directory.as_posix()}",
        "```",
        "",
        (
            "Methodische Referenzen: [Cerjan/Loring](https://arxiv.org/abs/2411.03515), "
            "[Bianco/Resta](https://arxiv.org/abs/1111.5697)."
        ),
        "",
    ]
    text = "\n".join(lines)
    atomic_text(directory / "final_report.md", text)
    return text
