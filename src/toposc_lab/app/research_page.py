"""Streamlit entry to the same durable autonomous research service as TOPOSC Live."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)


def _exact_score(candidate: dict[str, Any]) -> float | None:
    if candidate.get("origin") == "exact":
        value = candidate.get("score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            import math

            return float(value) if math.isfinite(value) else None
    return None


def _candidate_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_id", candidate.get("id", "")))


def _geometry_plot(candidate: dict[str, Any]) -> Any:
    import plotly.graph_objects as go

    geometry = candidate.get("geometry", {})
    points = geometry.get("coordinates", geometry.get("positions", []))
    edges = geometry.get("edges", [])
    figure = go.Figure()
    if points:
        xs: list[float | None] = []
        ys: list[float | None] = []
        for a, b in edges:
            xs.extend((points[a][0], points[b][0], None))
            ys.extend((points[a][1], points[b][1], None))
        figure.add_trace(
            go.Scatter(x=xs, y=ys, mode="lines", line={"color": "#64748b"}, hoverinfo="skip")
        )
        figure.add_trace(
            go.Scatter(
                x=[p[0] for p in points],
                y=[p[1] for p in points],
                mode="markers",
                marker={"color": "#0d9488", "size": 7},
                text=list(range(len(points))),
            )
        )
    figure.update_layout(
        title=f"Stored geometry · {_candidate_id(candidate)}",
        showlegend=False,
        xaxis={"scaleanchor": "y"},
        height=350,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return figure


def _candidate_view(st: Any, candidate: dict[str, Any], key: str) -> None:
    st.subheader(_candidate_id(candidate))
    st.write("Validation state:", candidate.get("validation_state", "PROPOSED"))
    st.plotly_chart(_geometry_plot(candidate), key=key, width="stretch")
    st.markdown("**Exact numerical results**")
    st.metric("Exact objective", _exact_score(candidate))
    st.json(
        {
            "raw_metrics": candidate.get("raw_metrics", {}),
            "exact_results": candidate.get("exact_results", {}),
            "validation_results": candidate.get("validation_results", {}),
        },
        expanded=False,
    )
    st.markdown("**Surrogate predictions — selection guidance only**")
    st.json(candidate.get("prediction", {}))
    with st.expander("Descriptors, lineage, acquisition and reproducibility metadata"):
        st.json(candidate)


def render_research_page(st: Any, service: Any = None) -> None:
    """Render configuration and live pages without owning the scientific process."""
    from toposc_lab.research.config import ExperimentConfig
    from toposc_lab.research.service import ResearchService
    from toposc_lab.research.strategies import STRATEGY_REGISTRY

    backend = service or ResearchService()
    st.title("Autonomous Research")
    st.caption(
        "Exact numerical physics is ground truth. Surrogate predictions guide selection and are labeled separately."
    )
    root = Path(st.text_input("Experiment root directory", "results/research", key="research_root"))
    records = backend.list_experiments(root)
    paths = [str(record["directory"]) for record in records]
    if paths:
        selected = st.selectbox(
            "Persisted experiments · inspect / resume", paths, key="research_persisted_selection"
        )
        if st.button("Inspect selected experiment"):
            st.session_state["research_directory"] = selected
    directory_text = st.text_input(
        "Open experiment directory",
        st.session_state.get("research_directory", ""),
        key="research_open_directory",
    )
    if st.button("Open experiment") and directory_text:
        st.session_state["research_directory"] = directory_text
    config_tab, live_tab = st.tabs(("New Experiment", "Run / Analysis"))
    with config_tab:
        if "research_pending_clone" in st.session_state:
            st.session_state["research_config_json"] = st.session_state.pop(
                "research_pending_clone"
            )
        if "research_config_json" not in st.session_state:
            defaults = ExperimentConfig().to_dict()
            defaults["output_directory"] = str(root / "new-experiment")
            st.session_state["research_config_json"] = _json(defaults)
        current_text = st.session_state["research_config_json"]
        try:
            current = json.loads(current_text)
        except (ValueError, TypeError):
            current = ExperimentConfig().to_dict()
        st.caption(
            "The complete JSON below includes geometry constraints, mutation rates, descriptors, "
            "disorder seeds and widths, validation stages, surrogate settings and compute limits."
        )
        with st.expander("Quick configuration", expanded=True), st.form("research_quick_config"):
            name = st.text_input("Experiment name", current.get("name", "Experiment"))
            question = st.text_area("Research question", current.get("question", ""))
            algorithms = list(STRATEGY_REGISTRY)
            algorithm = st.selectbox(
                "Search algorithm",
                algorithms,
                index=algorithms.index(current["algorithm"])
                if current.get("algorithm") in algorithms
                else 0,
            )
            a, b, c = st.columns(3)
            seed = a.number_input("Random seed", min_value=0, value=int(current.get("seed", 17001)))
            side = b.number_input(
                "Grid side (sites = side²)",
                min_value=2,
                value=int(current.get("space", {}).get("side", 10)),
            )
            budget = c.number_input(
                "Exact physics attempt budget",
                min_value=1,
                value=int(current.get("exact_budget", 280)),
            )
            cycles = a.number_input(
                "Search cycles", min_value=1, value=int(current.get("cycles", 10))
            )
            candidate_budget = b.number_input(
                "Candidate proposal budget",
                min_value=1,
                value=int(current.get("candidate_budget", 10000)),
            )
            checkpoint_every = c.number_input(
                "Checkpoint every exact attempts",
                min_value=1,
                value=int(current.get("checkpoint_every", 250)),
            )
            allocation = current.get("search", {}).get("allocation", {})
            exploitation = a.number_input(
                "Exploitation %",
                min_value=0.0,
                max_value=100.0,
                value=float(allocation.get("exploitation", 0.6)) * 100,
            )
            uncertainty = b.number_input(
                "Uncertainty %",
                min_value=0.0,
                max_value=100.0,
                value=float(allocation.get("uncertainty", 0.2)) * 100,
            )
            novelty = c.number_input(
                "Novelty %",
                min_value=0.0,
                max_value=100.0,
                value=float(allocation.get("novelty", 0.2)) * 100,
            )
            output = st.text_input(
                "New output directory",
                current.get("output_directory", str(root / "new-experiment")),
            )
            if st.form_submit_button("Apply quick configuration"):
                current.update(
                    name=name,
                    question=question,
                    algorithm=algorithm,
                    seed=seed,
                    exact_budget=budget,
                    candidate_budget=candidate_budget,
                    cycles=cycles,
                    checkpoint_every=checkpoint_every,
                    output_directory=output,
                )
                current.setdefault("space", {})["side"] = side
                current.setdefault("search", {})["allocation"] = {
                    "exploitation": exploitation / 100,
                    "uncertainty": uncertainty / 100,
                    "novelty": novelty / 100,
                }
                st.session_state["research_config_json"] = _json(current)
        upload = st.file_uploader("Load configuration", type="json", key="research_config_upload")
        if upload and st.button("Apply uploaded config"):
            try:
                loaded = ExperimentConfig.from_dict(json.loads(upload.getvalue())).to_dict()
                st.session_state["research_config_json"] = _json(loaded)
            except (ValueError, TypeError) as error:
                st.error(str(error))
        text = st.text_area(
            "Complete experiment configuration (JSON)", key="research_config_json", height=440
        )
        try:
            config = ExperimentConfig.from_dict(json.loads(text))
            config.validate_plugins()
            st.download_button(
                "Save config", _json(config.to_dict()) + "\n", "experiment.json", "application/json"
            )
            if st.button("Start run", type="primary"):
                target = backend.create(config.to_dict(), Path(config.output_directory))
                pid = backend.launch(target)
                st.session_state["research_directory"] = str(target)
                st.success(
                    f"Independent worker started (PID {pid}). Open Run / Analysis to monitor."
                )
        except (OSError, ValueError, TypeError, KeyError) as error:
            st.error(f"Configuration / launch rejected: {error}")
    with live_tab:
        directory = st.session_state.get("research_directory")
        if not directory:
            st.info("Open a persisted experiment or start one from New Experiment.")
            return

        def live() -> None:
            try:
                snapshot = backend.snapshot(Path(directory))
                _render_snapshot(st, backend, Path(directory), snapshot)
            except (OSError, ValueError, TypeError, KeyError) as error:
                st.error(f"Snapshot unavailable: {error}")

        st.fragment(run_every="3s")(live)()


def _render_snapshot(st: Any, backend: Any, directory: Path, snapshot: dict[str, Any]) -> None:
    import plotly.graph_objects as go

    config, state = snapshot["config"], snapshot["state"]
    candidates = snapshot.get("candidates", [])
    lookup = {_candidate_id(candidate): candidate for candidate in candidates}
    st.subheader(f"{config.get('name', 'Experiment')} · {state.get('status', 'unknown').upper()}")
    controls = st.columns(6)
    for column, (label, action) in zip(
        controls,
        (
            ("Pause", "pause"),
            ("Resume", "resume"),
            ("Stop safely", "stop"),
            ("Checkpoint now", "checkpoint"),
            ("Archive", "archive"),
        ),
        strict=False,
    ):
        if column.button(label, key=f"research_control_{action}"):
            backend.control(directory, action)
            st.info(f"{label} requested at the next safe evaluation boundary.")
    if controls[5].button("Duplicate experiment"):
        clone = json.loads(_json(config))
        clone.pop("experiment_id", None)
        clone["name"] += " (copy)"
        clone["output_directory"] = str(directory.with_name(directory.name + "-copy"))
        st.session_state["research_pending_clone"] = _json(clone)
        st.rerun()
    tabs = st.tabs(
        (
            "Dashboard",
            "MAP-Elites",
            "Search Progress",
            "Candidate Explorer",
            "Baselines",
            "Checkpoints / Log",
            "Final Report",
        )
    )
    with tabs[0]:
        columns = st.columns(4)
        for index, (label, value) in enumerate(
            (
                ("Exact attempts", state.get("exact_evaluations", 0)),
                ("Remaining exact budget", state.get("remaining_exact_budget")),
                ("Current cycle", state.get("cycle", 0)),
                ("Generated proposals", state.get("generated", 0)),
                ("Best exact score", state.get("best_score")),
                ("Archive coverage", state.get("archive_coverage")),
                ("Elapsed seconds", state.get("elapsed_seconds")),
                ("Surrogate error", state.get("surrogate_error")),
            )
        ):
            columns[index % 4].metric(label, value)
        for warning in snapshot.get("warnings", state.get("warnings", [])):
            st.warning(str(warning))
        st.caption(
            "Missing finite-size or Majorana validation never implies a positive scientific claim."
        )
        with st.expander("All run diagnostics and acquisition allocation"):
            st.json({**state, "allocation": config.get("search", {}).get("allocation")})
    with tabs[1]:
        metric = st.selectbox(
            "Archive color",
            ("Exact score", "Surrogate uncertainty (prediction)", "Generation"),
            key="research_archive_metric",
        )
        archive = snapshot.get("archive", [])
        if isinstance(archive, dict):
            archive = list(archive.values())
        cells = [
            (row.get("cell", row.get("archive_cell")), lookup.get(_candidate_id(row), row))
            for row in archive
        ]
        cells = [
            (cell, row) for cell, row in cells if isinstance(cell, (tuple, list)) and len(cell) >= 2
        ]
        descriptors = config.get("search", {}).get(
            "behavior_descriptors", ["descriptor x", "descriptor y"]
        )
        if cells:
            values = [
                _exact_score(row)
                if metric == "Exact score"
                else row.get("prediction", {}).get("uncertainty")
                if metric.startswith("Surrogate")
                else row.get("generation")
                for _, row in cells
            ]
            fig = go.Figure(
                go.Scatter(
                    x=[cell[0] for cell, _ in cells],
                    y=[cell[1] for cell, _ in cells],
                    mode="markers",
                    marker={
                        "symbol": "square",
                        "size": 27,
                        "color": [v if v is not None else 0 for v in values],
                        "colorscale": "Viridis",
                        "showscale": True,
                    },
                    customdata=[_candidate_id(row) for _, row in cells],
                    text=[
                        f"{_candidate_id(row)}<br>{metric}: {value}<br>Validation: {row.get('validation_state')}"
                        for (_, row), value in zip(cells, values, strict=True)
                    ],
                    hovertemplate="%{text}<extra></extra>",
                )
            )
            fig.update_layout(
                xaxis_title=f"{descriptors[0]} bin", yaxis_title=f"{descriptors[1]} bin", height=460
            )
            selection = st.plotly_chart(
                fig, key="research_archive", on_select="rerun", width="stretch"
            )
            selected_points = selection.get("selection", {}).get("points", [])
            if selected_points:
                st.session_state["research_selected_candidate"] = selected_points[0]["customdata"]
        else:
            st.info("No occupied exact-evaluated archive cells yet.")
    with tabs[2]:
        metric = st.selectbox(
            "Progress metric",
            (
                "best_score",
                "best_validated_score",
                "archive_coverage",
                "diversity",
                "surrogate_error",
                "duplicate_rate",
                "invalid_rate",
            ),
        )
        history = snapshot.get("history", [])
        history_values = [row for row in history if isinstance(row.get(metric), (float, int))]
        if history_values:
            st.plotly_chart(
                go.Figure(
                    go.Scatter(
                        x=[row.get("exact_evaluations") for row in history_values],
                        y=[row[metric] for row in history_values],
                        mode="lines+markers",
                    )
                ).update_layout(
                    xaxis_title="Exact physics attempts (including failures)",
                    yaxis_title=metric.replace("_", " "),
                ),
                key="research_progress",
                width="stretch",
            )
        else:
            st.info("No measured values for this diagnostic yet.")
    with tabs[3]:
        if candidates:
            default = st.session_state.get("research_selected_candidate", next(iter(lookup)))
            selected = st.multiselect(
                "Candidates to compare",
                list(lookup),
                default=[default] if default in lookup else [],
                max_selections=3,
            )
            for column, candidate_id in zip(
                st.columns(max(1, len(selected))), selected, strict=False
            ):
                with column:
                    _candidate_view(st, lookup[candidate_id], f"research_candidate_{candidate_id}")
        else:
            st.info("No persisted candidates yet.")
    with tabs[4]:
        baseline_rows = [row for row in candidates if row.get("baseline") or row.get("is_baseline")]
        st.caption(
            "Matched references share sites and exact protocols. Compare search methods at equal expensive exact-evaluation budgets."
        )
        st.dataframe(
            [
                {
                    "Candidate": _candidate_id(row),
                    "Family": row.get("family"),
                    "Exact score": _exact_score(row),
                    "Validation": row.get("validation_state"),
                }
                for row in baseline_rows
            ],
            width="stretch",
        )
        if baseline_rows:
            chosen = st.selectbox(
                "Inspect reference geometry and exact evidence",
                [_candidate_id(row) for row in baseline_rows],
            )
            _candidate_view(st, lookup[chosen], "research_baseline_geometry")
    with tabs[5]:
        checkpoints = snapshot.get("checkpoints", [])
        if checkpoints:
            chosen_index = st.selectbox(
                "Checkpoint report",
                range(len(checkpoints)),
                format_func=lambda i: (
                    f"{i + 1} · exact attempts {checkpoints[i].get('exact_evaluations', '?')}"
                ),
            )
            st.json(checkpoints[chosen_index], expanded=True)
        st.markdown("**Chronological research log**")
        st.json(snapshot.get("events", []), expanded=False)
    with tabs[6]:
        report = snapshot.get("report")
        if report:
            st.markdown(report)
            st.download_button("Save final report", report, "final_report.md", "text/markdown")
        else:
            st.info("The worker generates the structured scientific report at completion.")
