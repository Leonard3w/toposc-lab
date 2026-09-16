"""Autonomous research workbench: durable snapshots, detached workers, exact evidence.

The widgets never evaluate physics. The service owns all experiment changes and
the independent runner; closing a page only stops its read-only refresh thread.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from toposc_live.models import GeometrySnapshot, display, mapping, number
from toposc_live.pages import panel, scroll_page, text_box
from toposc_live.refresh import Task
from toposc_live.widgets import GeometryView


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)


def identifier(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_id", candidate.get("id", "")))


def exact(candidate: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], mapping(candidate.get("exact_results", candidate.get("exact"))))


def prediction(candidate: dict[str, Any]) -> dict[str, Any]:
    return cast(
        dict[str, Any], mapping(candidate.get("prediction", candidate.get("surrogate_predictions")))
    )


def table_item(table: QTableWidget, row: int, column: int) -> QTableWidgetItem:
    item = table.item(row, column)
    if item is None:
        raise IndexError("Research table cell has no record")
    return cast(QTableWidgetItem, item)


def score(candidate: dict[str, Any]) -> float | None:
    """A prediction never supplies the exact-score column or curve."""
    return number(candidate.get("score")) if candidate.get("origin") == "exact" else None


def geometry(candidate: dict[str, Any]) -> GeometrySnapshot:
    value = mapping(candidate.get("geometry"))
    if not value:
        return GeometrySnapshot(message="Select a candidate to preview its geometry")
    positions = value.get("coordinates", value.get("positions", ()))
    edges = value.get("edges", ())
    try:
        points = tuple(tuple(float(v) for v in point) for point in positions)
        bonds = tuple((int(edge[0]), int(edge[1])) for edge in edges)
        if any(len(point) < 2 or not all(math.isfinite(v) for v in point) for point in points):
            raise ValueError("invalid coordinates")
        if any(a < 0 or b < 0 or a >= len(points) or b >= len(points) for a, b in bonds):
            raise ValueError("invalid bond indices")
        return GeometrySnapshot(
            points,
            bonds,
            tuple(value.get("boundary_sites", value.get("boundary", ()))),
            "Stored geometry",
        )
    except (TypeError, ValueError, IndexError):
        return GeometrySnapshot(message="Stored geometry could not be displayed")


def candidate_details(candidate: dict[str, Any]) -> str:
    """Render every scientific record without interpreting absent validation."""
    preview = {
        **candidate,
        "geometry": {
            key: value
            for key, value in mapping(candidate.get("geometry")).items()
            if key != "archive"
        },
    }
    return "\n\n".join(
        (
            (
                f"CANDIDATE {identifier(candidate)}\n"
                f"Validation state: {candidate.get('validation_state', candidate.get('status', 'PROPOSED'))}"
            ),
            "EXACT OBJECTIVE / RAW METRICS\n"
            + json_text({"score": score(candidate), "metrics": candidate.get("raw_metrics", {})}),
            "EXACT NUMERICAL RESULTS\n" + json_text(exact(candidate)),
            "SURROGATE PREDICTIONS — not scientific validation\n"
            + json_text(prediction(candidate)),
            "VALIDATION / TOPOLOGY / MAJORANA / DISORDER / FINITE SIZE\n"
            + json_text(candidate.get("validation_results", candidate.get("validation", {}))),
            "DESCRIPTORS\n" + json_text(candidate.get("descriptors", {})),
            "LINEAGE, REPRODUCIBILITY AND RAW RECORD\n" + json_text(preview),
        )
    )


class ResearchPlot(QWidget):
    """A labeled plot shared by evaluation-budget and diagnostic views."""

    def __init__(self) -> None:
        super().__init__()
        self.curves: dict[str, list[tuple[float, float]]] = {}
        self.x_label, self.y_label = "Exact physics attempts", "Exact score"
        self.setMinimumSize(350, 290)

    def set_data(
        self,
        curves: dict[str, list[tuple[float, float]]],
        y_label: str,
        x_label: str = "Exact physics attempts (including failures)",
    ) -> None:
        self.curves, self.y_label, self.x_label = curves, y_label, x_label
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101d2b"))
        painter.setPen(QColor("#c4ceda"))
        points = [point for curve in self.curves.values() for point in curve]
        if not points:
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "No recorded values for this metric yet"
            )
            return
        area = QRectF(72, 48, max(1, self.width() - 105), max(1, self.height() - 115))
        xmax = max(1.0, max(p[0] for p in points))
        ymin = min(0.0, min(p[1] for p in points))
        ymax = max(p[1] for p in points)
        span = (ymax - ymin) or 1.0

        def position(x: float, y: float) -> QPointF:
            return QPointF(
                area.left() + x / xmax * area.width(),
                area.bottom() - (y - ymin) / span * area.height(),
            )

        painter.drawLine(area.bottomLeft(), area.bottomRight())
        painter.drawLine(area.bottomLeft(), area.topLeft())
        painter.drawText(10, 22, self.y_label)
        for step in range(5):
            fraction = step / 4
            painter.drawText(
                5, int(area.bottom() - fraction * area.height()), f"{ymin + fraction * span:.3g}"
            )
            painter.drawText(
                int(area.left() + fraction * area.width()),
                int(area.bottom()) + 20,
                f"{fraction * xmax:g}",
            )
        painter.drawText(72, self.height() - 10, self.x_label)
        for index, (name, curve) in enumerate(self.curves.items()):
            painter.setPen(QPen(QColor(("#62d0c9", "#e5ae62", "#90aaff", "#e399c6")[index % 4]), 2))
            previous = None
            for x, y in curve:
                point = position(x, y)
                if previous is not None:
                    painter.drawLine(previous, point)
                painter.drawEllipse(point, 3, 3)
                previous = point
            painter.drawText(75 + index * 230, self.height() - 35, name)


class ArchiveView(QWidget):
    """Clickable two-descriptor grid, colored only by the selected metric."""

    candidate_selected = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.cells: list[tuple[tuple[int, int], dict[str, Any]]] = []
        self.bins = (10, 10)
        self.axes = ("coordination_variance", "clustering_coefficient")
        self.bounds: list[Any] = []
        self.color_metric = "Exact score"
        self.hit_boxes: list[tuple[QRectF, str]] = []
        self.setMinimumSize(430, 400)
        self.setMouseTracking(True)

    def metric_value(self, candidate: dict[str, Any]) -> float | None:
        if self.color_metric == "Exact score":
            return score(candidate)
        if self.color_metric == "Surrogate uncertainty (prediction)":
            pred = prediction(candidate)
            return cast(float | None, number(pred.get("uncertainty", pred.get("std"))))
        return cast(float | None, number(candidate.get("generation", candidate.get("cycle"))))

    def set_data(
        self, candidates: list[dict[str, Any]], archive: Any, config: dict[str, Any]
    ) -> None:
        search = mapping(config.get("search"))
        descriptors = tuple(search.get("behavior_descriptors", self.axes))
        self.axes = (descriptors + ("single descriptor",))[:2]
        bins = search.get("archive_bins", (10, 10))
        self.bins = (
            (bins, bins if len(descriptors) > 1 else 1)
            if isinstance(bins, int)
            else (tuple(bins) + (1,))[:2]
        )
        self.bounds = search.get("descriptor_bounds", [])
        lookup = {identifier(candidate): candidate for candidate in candidates}
        rows = list(archive.values()) if isinstance(archive, dict) else (archive or [])
        self.cells = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            cell = row.get("cell", row.get("archive_cell", row.get("bin", row.get("coordinates"))))
            candidate = lookup.get(str(row.get("candidate_id", row.get("id", ""))), row)
            if isinstance(cell, (list, tuple)) and cell:
                self.cells.append(((int(cell[0]), int(cell[1]) if len(cell) > 1 else 0), candidate))
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101d2b"))
        painter.setPen(QColor("#c4ceda"))
        painter.drawText(14, 24, self.color_metric + " · click an occupied cell to inspect")
        area = QRectF(65, 45, max(1, self.width() - 90), max(1, self.height() - 115))
        nx, ny = self.bins
        if nx <= 0 or ny <= 0:
            return
        cw, ch = area.width() / nx, area.height() / ny
        painter.setPen(QPen(QColor("#34485d"), 1))
        for x in range(nx + 1):
            painter.drawLine(
                QPointF(area.left() + x * cw, area.top()),
                QPointF(area.left() + x * cw, area.bottom()),
            )
        for y in range(ny + 1):
            painter.drawLine(
                QPointF(area.left(), area.top() + y * ch),
                QPointF(area.right(), area.top() + y * ch),
            )
        values = [self.metric_value(candidate) for _, candidate in self.cells]
        finite = [v for v in values if v is not None]
        vmin, vmax = (min(finite), max(finite)) if finite else (0.0, 0.0)
        self.hit_boxes = []
        for ((x, y), candidate), value in zip(self.cells, values):
            if not 0 <= x < nx or not 0 <= y < ny:
                continue
            box = QRectF(
                area.left() + x * cw + 1,
                area.bottom() - (y + 1) * ch + 1,
                max(1, cw - 2),
                max(1, ch - 2),
            )
            fraction = (value - vmin) / (vmax - vmin or 1) if value is not None else 0
            color = (
                QColor.fromHsvF(0.48, 0.7, 0.4 + 0.6 * fraction)
                if value is not None
                else QColor("#697788")
            )
            painter.fillRect(box, color)
            self.hit_boxes.append((box, identifier(candidate)))
            if cw > 45 and ch > 28:
                painter.setPen(QColor("#ffffff"))
                painter.drawText(
                    box,
                    Qt.AlignmentFlag.AlignCenter,
                    f"{value:.3g}" if value is not None else "n/a",
                )
        painter.setPen(QColor("#c4ceda"))
        painter.drawText(65, self.height() - 44, f"x: {self.axes[0]} · y: {self.axes[1]}")
        painter.drawText(
            65,
            self.height() - 22,
            f"{len(self.cells)}/{nx * ny} cells · range {vmin:.3g} … {vmax:.3g} · gray = unavailable",
        )
        if not self.cells:
            painter.drawText(
                area, Qt.AlignmentFlag.AlignCenter, "No exact-evaluated elites recorded yet"
            )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        for box, candidate_id in self.hit_boxes:
            if box.contains(event.position()):
                self.candidate_selected.emit(candidate_id)
                break

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        for box, candidate_id in self.hit_boxes:
            if box.contains(event.position()):
                candidate = next(c for _, c in self.cells if identifier(c) == candidate_id)
                self.setToolTip(
                    f"{candidate_id}\nExact score: {display(score(candidate))}\n"
                    f"Validation: {candidate.get('validation_state', 'unavailable')}\n"
                    f"Prediction uncertainty: {display(prediction(candidate).get('uncertainty'))}\n"
                    f"Generation: {display(candidate.get('generation'))}"
                )
                return
        self.setToolTip("")


class ResearchPage(QWidget):
    """The native integrated workbench, with an injectable backend for UI tests."""

    def __init__(
        self, output_root: Path, service: Any = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        from toposc_lab.research.config import ExperimentConfig
        from toposc_lab.research.service import ResearchService
        from toposc_lab.research.strategies import STRATEGY_REGISTRY

        self.service = service or ResearchService()
        self.config_type = ExperimentConfig
        self.root = output_root
        self.directory: Path | None = None
        self.snapshot: dict[str, Any] = {}
        self.task: Task | None = None
        self.operation: Task | None = None
        self.closed = False
        self.last_scan = 0.0
        self._syncing = False
        self._checkpoint_records: list[dict[str, Any]] = []
        self.fields: dict[str, QWidget] = {}
        layout = QVBoxLayout(self)
        heading = QLabel("AUTONOMOUS RESEARCH")
        heading.setStyleSheet("font-size: 21px; font-weight: 650;")
        layout.addWidget(heading)
        self.message = QLabel(
            "Exact numerical physics is ground truth. Predictions guide selection only."
        )
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        toolbar = QHBoxLayout()
        self.experiments = QComboBox()
        self.experiments.setMinimumWidth(280)
        toolbar.addWidget(self.experiments, 1)
        for label, callback in (
            ("Inspect selected", self.inspect_selected),
            ("Open experiment…", self.open_directory),
            ("Refresh", self.refresh),
            ("Duplicate experiment", self.clone_experiment),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            toolbar.addWidget(button)
        layout.addLayout(toolbar)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self.config_page = QWidget()
        self.tabs.addTab(scroll_page(self.config_page), "New Experiment")
        defaults = ExperimentConfig().to_dict()
        defaults["output_directory"] = str(output_root / "new-experiment")
        self._make_config(defaults, STRATEGY_REGISTRY)
        self._make_dashboard()
        self._make_archive()
        self._make_progress()
        self._make_candidates()
        self._make_baselines()
        self._make_log()
        self.report = QTextBrowser()
        self.report.setOpenExternalLinks(False)
        self.tabs.addTab(self.report, "Final Report")
        self.timer = QTimer(self)
        self.timer.setInterval(2500)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def _make_config(self, defaults: dict[str, Any], registry: Any) -> None:
        layout = QVBoxLayout(self.config_page)
        hint = QLabel(
            "Configure Experiment 001. Fixed sites and physical protocol; only connectivity is searched. "
            "Advanced JSON exposes every geometry, mutation, validation, surrogate and compute setting."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        layout.addLayout(form)
        names = tuple(registry.keys()) if hasattr(registry, "keys") else tuple(registry.names())
        specifications = (
            ("name", "Experiment name", "text"),
            ("question", "Research question", "text"),
            ("algorithm", "Search algorithm", "algorithm"),
            ("seed", "Random seed", "int"),
            ("space.side", "Grid side (sites = side²)", "int"),
            ("exact_budget", "Exact physics attempt budget", "int"),
            ("candidate_budget", "Generated candidate budget", "int"),
            ("cycles", "Search cycles", "int"),
            ("pool_size", "Proposal pool per cycle", "int"),
            ("batch_size", "Exact selections per cycle", "int"),
            ("checkpoint_every", "Checkpoint interval (exact attempts)", "int"),
            ("retrain_every", "Surrogate retraining interval", "int"),
            ("search.allocation.exploitation", "Exploitation %", "float"),
            ("search.allocation.uncertainty", "Uncertainty %", "float"),
            ("search.allocation.novelty", "Novelty %", "float"),
            ("concurrency", "Exact worker concurrency", "int"),
            ("output_directory", "New output directory", "text"),
        )
        for path, label, kind in specifications:
            if kind == "algorithm":
                widget: Any = QComboBox()
                widget.addItems(names)
                widget.currentTextChanged.connect(lambda _, p=path: self._form_changed(p))
            elif kind == "int":
                widget = QSpinBox()
                widget.setRange(0, 2147483647)
                widget.valueChanged.connect(lambda _, p=path: self._form_changed(p))
            elif kind == "float":
                widget = QDoubleSpinBox()
                widget.setRange(0, 100)
                widget.setDecimals(1)
                widget.setSingleStep(5)
                widget.valueChanged.connect(lambda _, p=path: self._form_changed(p))
            else:
                widget = QLineEdit()
                widget.textChanged.connect(lambda _, p=path: self._form_changed(p))
            self.fields[path] = widget
            form.addRow(label, widget)
        self.editor = QPlainTextEdit()
        self.editor.setMinimumHeight(380)
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(panel("Complete configuration · human-readable JSON", self.editor))
        self.editor.textChanged.connect(self._configuration_changed)
        buttons = QHBoxLayout()
        for label, callback in (
            ("Apply JSON / validate", self.validate_config),
            ("Save config…", self.save_config),
            ("Load config…", self.load_config),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        self.start = QPushButton("Start run")
        self.start.clicked.connect(self.start_run)
        buttons.addWidget(self.start)
        layout.addLayout(buttons)
        self.config_status = QLabel()
        self.config_status.setWordWrap(True)
        layout.addWidget(self.config_status)
        self.set_config(defaults)

    def _configuration_changed(self) -> None:
        if hasattr(self, "config_status"):
            self.config_status.setText(
                "Configuration edited. Start validates the full JSON before creating a run."
            )

    def _form_changed(self, path: str) -> None:
        if self._syncing or not hasattr(self, "editor"):
            return
        try:
            data = json.loads(self.editor.toPlainText())
            node = data
            parts = path.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            widget = self.fields[path]
            value = (
                widget.value()
                if isinstance(widget, (QSpinBox, QDoubleSpinBox))
                else (widget.currentText() if isinstance(widget, QComboBox) else widget.text())
            )
            node[parts[-1]] = float(value) / 100 if path.startswith("search.allocation.") else value
            self.editor.setPlainText(json_text(data))
        except (ValueError, TypeError, AttributeError) as error:
            self.config_status.setText(f"Fix the JSON before editing form fields: {error}")

    def set_config(self, data: dict[str, Any]) -> None:
        self._syncing = True
        try:
            self.editor.setPlainText(json_text(data))
            for path, widget in self.fields.items():
                value: Any = data
                for part in path.split("."):
                    value = mapping(value).get(part)
                if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    if path.startswith("search.allocation.") and isinstance(value, (int, float)):
                        value *= 100
                    numeric = value if isinstance(value, (int, float)) else 0
                    if isinstance(widget, QSpinBox):
                        widget.setValue(int(numeric))
                    else:
                        widget.setValue(float(numeric))
                elif isinstance(widget, QComboBox):
                    widget.setCurrentText(str(value or ""))
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value or ""))
                    widget.setCursorPosition(0)
        finally:
            self._syncing = False

    def configuration(self) -> dict[str, Any]:
        raw = json.loads(self.editor.toPlainText())
        if not isinstance(raw, dict):
            raise TypeError("Configuration must be a JSON object")
        config = self.config_type.from_dict(raw)
        config.validate_plugins()
        return cast(dict[str, Any], config.to_dict())

    def validate_config(self) -> None:
        try:
            data = self.configuration()
            self.set_config(data)
            self.config_status.setText(
                "Valid configuration. Exact results retain the configured frozen validation criteria."
            )
        except (ValueError, TypeError, KeyError) as error:
            self.config_status.setText(f"Configuration rejected: {error}")

    def save_config(self, path: Path | None = None) -> None:
        try:
            data = self.configuration()
            if not isinstance(path, Path):
                name, _ = QFileDialog.getSaveFileName(
                    self, "Save experiment configuration", "experiment.json", "JSON (*.json)"
                )
                if not name:
                    return
                path = Path(name)
            path.write_text(json_text(data) + "\n", encoding="utf-8")
            self.config_status.setText(f"Saved configuration: {path}")
        except (OSError, ValueError, TypeError) as error:
            self.config_status.setText(f"Save failed: {error}")

    def load_config(self, path: Path | None = None) -> None:
        try:
            if not isinstance(path, Path):
                name, _ = QFileDialog.getOpenFileName(
                    self, "Load experiment configuration", "", "JSON (*.json)"
                )
                if not name:
                    return
                path = Path(name)
            data = self.config_type.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            ).to_dict()
            self.set_config(data)
            self.config_status.setText(f"Loaded configuration: {path}")
        except (OSError, ValueError, TypeError) as error:
            self.config_status.setText(f"Load rejected: {error}")

    def _make_dashboard(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.run_title = QLabel("Open or start an experiment")
        self.run_title.setWordWrap(True)
        layout.addWidget(self.run_title)
        controls = QHBoxLayout()
        self.controls: dict[str, QPushButton] = {}
        for action, label in (
            ("pause", "Pause"),
            ("resume", "Resume"),
            ("stop", "Stop safely"),
            ("checkpoint", "Checkpoint now"),
            ("archive", "Archive experiment"),
        ):
            button = QPushButton(label)
            button.setEnabled(False)
            button.clicked.connect(lambda _, a=action: self.request_control(a))
            self.controls[action] = button
            controls.addWidget(button)
        layout.addLayout(controls)
        self.dashboard_values: dict[str, QLabel] = {}
        grid = QGridLayout()
        grid.setSpacing(8)
        metric_definitions = (
            ("status", "Run status"),
            ("cycle", "Search cycle"),
            ("exact_budget", "Exact physics attempts"),
            ("elapsed", "Elapsed time"),
            ("generated", "Generated candidates"),
            ("best_score", "Best exact score"),
            ("best_validated", "Best finite validated"),
            ("coverage", "Archive coverage"),
            ("candidate_throughput", "Candidate throughput"),
            ("exact_throughput", "Exact simulation throughput"),
            ("surrogate_status", "Surrogate training"),
            ("surrogate_error", "Prequential prediction error"),
            ("rejection_rates", "Duplicate / invalid rate"),
            ("allocation", "Selection allocation"),
            ("checkpoint", "Latest checkpoint"),
            ("diversity", "Family diversity"),
        )
        for index, (key, title) in enumerate(metric_definitions):
            value = QLabel("Unavailable")
            value.setWordWrap(True)
            value.setStyleSheet("font-size: 16px; font-weight: 600; padding: 2px;")
            self.dashboard_values[key] = value
            card = panel(title, value)
            card.setMinimumHeight(82)
            grid.addWidget(card, index // 4, index % 4)
        for column in range(4):
            grid.setColumnStretch(column, 1)
        layout.addLayout(grid)
        self.dashboard_values["best_score"].setStyleSheet(
            "font-size: 19px; font-weight: 600; color: #62d0c9; padding: 2px;"
        )
        self.dashboard_values["surrogate_error"].setStyleSheet(
            "font-size: 19px; font-weight: 600; color: #e5ae62; padding: 2px;"
        )
        self.dashboard_values["surrogate_error"].setToolTip(
            "Mean absolute error of saved predictions compared with exact labels obtained later. "
            "This is prediction calibration, not scientific validation."
        )
        self.dashboard_values["diversity"].setToolTip(
            "Shannon entropy of exact-evaluated candidate-family frequencies."
        )
        self.metrics = text_box()
        self.raw_state_toggle = QPushButton("Show detailed run state")
        self.raw_state_toggle.setCheckable(True)
        self.raw_state_panel = panel("Detailed persisted state", self.metrics)
        self.raw_state_panel.setVisible(False)
        self.raw_state_panel.setMaximumHeight(180)
        self.raw_state_toggle.toggled.connect(self.raw_state_panel.setVisible)
        self.raw_state_toggle.toggled.connect(
            lambda checked: self.raw_state_toggle.setText(
                "Hide detailed run state" if checked else "Show detailed run state"
            )
        )
        self.health = text_box()
        layout.addWidget(panel("Scientific warnings / numerical failures", self.health), 1)
        layout.addWidget(self.raw_state_toggle)
        layout.addWidget(self.raw_state_panel)
        self.tabs.addTab(page, "Live Dashboard")

    def _update_dashboard_metrics(self, config: dict[str, Any], state: dict[str, Any]) -> None:
        def numeric(value: Any, suffix: str = "") -> str:
            parsed = number(value)
            return f"{parsed:.3g}{suffix}" if parsed is not None else "Unavailable"

        def percentage(value: Any) -> str:
            parsed = number(value)
            return f"{parsed * 100:.1f}%" if parsed is not None else "Unavailable"

        used = state.get("exact_evaluations", 0)
        elapsed = number(state.get("elapsed_seconds"))
        elapsed_text = "Unavailable"
        if elapsed is not None:
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            elapsed_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        allocated = mapping(mapping(config.get("search")).get("allocation"))
        checkpoint_records = self.snapshot.get("checkpoints", [])
        latest = checkpoint_records[-1] if checkpoint_records else {}
        checkpoint_count = mapping(latest.get("state")).get(
            "exact_evaluations", latest.get("exact_evaluations")
        )
        best_validated_id = state.get("best_validated_candidate")
        validated_candidate = (
            self.find_candidate(str(best_validated_id)) if best_validated_id else {}
        )
        validated_score = score(validated_candidate) if validated_candidate else None
        values = {
            "status": str(state.get("status", "Unknown")).replace("_", " ").title(),
            "cycle": f"{state.get('cycle', 0)} / {config.get('cycles', '—')}",
            "exact_budget": f"{used} used · {max(0, config.get('exact_budget', 0) - used)} left",
            "elapsed": elapsed_text,
            "generated": display(state.get("generated", 0)),
            "best_score": numeric(state.get("best_score")),
            "best_validated": numeric(validated_score)
            if validated_score is not None
            else "None passed",
            "coverage": percentage(state.get("archive_coverage")),
            "candidate_throughput": numeric(state.get("candidate_throughput"), " / s"),
            "exact_throughput": numeric(state.get("simulation_throughput"), " / s"),
            "surrogate_status": str(state.get("surrogate_status", "Awaiting exact labels"))
            .replace("_", " ")
            .capitalize(),
            "surrogate_error": numeric(state.get("surrogate_error"), " MAE"),
            "rejection_rates": f"{percentage(state.get('duplicate_rate'))} / {percentage(state.get('invalid_rate'))}",
            "allocation": f"{percentage(allocated.get('exploitation'))} exploit · {percentage(allocated.get('uncertainty'))} uncertain\n{percentage(allocated.get('novelty'))} novel",
            "checkpoint": f"#{latest.get('id', len(checkpoint_records))} · {checkpoint_count} attempts"
            if latest
            else "No checkpoint yet",
            "diversity": numeric(state.get("diversity")),
        }
        for key, value in values.items():
            self.dashboard_values[key].setText(value)
        self.dashboard_values["best_validated"].setToolTip(
            f"Candidate: {best_validated_id}"
            if best_validated_id
            else "No candidate has passed the configured exact finite-system validation gates. "
            "No Majorana or thermodynamic-phase claim is implied."
        )
        self.dashboard_values["checkpoint"].setToolTip(
            str(mapping(latest.get("report")).get("created", ""))
        )

    def _make_archive(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.archive_metric = QComboBox()
        self.archive_metric.addItems(
            ("Exact score", "Surrogate uncertainty (prediction)", "Generation")
        )
        self.archive_metric.currentTextChanged.connect(self._archive_metric_changed)
        layout.addWidget(self.archive_metric)
        self.archive = ArchiveView()
        self.archive.candidate_selected.connect(self.select_candidate)
        layout.addWidget(self.archive, 1)
        self.tabs.addTab(page, "MAP-Elites")

    def _archive_metric_changed(self, metric: str) -> None:
        self.archive.color_metric = metric
        self.archive.update()

    def _make_progress(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.progress_metric = QComboBox()
        self.progress_metric.addItems(
            (
                "Best exact / best validated score",
                "Archive coverage",
                "Diversity",
                "Surrogate error",
                "Duplicate rate",
                "Invalid rate",
                "Candidate-family distribution",
                "Exact robustness distribution",
            )
        )
        self.progress_metric.currentTextChanged.connect(self.update_progress)
        layout.addWidget(self.progress_metric)
        self.progress = ResearchPlot()
        layout.addWidget(self.progress, 1)
        self.distribution = text_box()
        self.distribution.setMaximumHeight(180)
        layout.addWidget(self.distribution)
        self.tabs.addTab(page, "Search Progress")

    @staticmethod
    def _table(headers: tuple[str, ...]) -> QTableWidget:
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSortingEnabled(True)
        return table

    def _make_candidates(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter candidate ID, family, mutation or validation state")
        self.filter.textChanged.connect(self.filter_candidates)
        layout.addWidget(self.filter)
        self.candidates = self._table(
            (
                "Candidate ID",
                "Exact score",
                "Validation state",
                "Family",
                "Generation",
                "Predicted score (not exact)",
                "Prediction uncertainty",
            )
        )
        self.candidates.itemSelectionChanged.connect(self.show_selection)
        layout.addWidget(self.candidates, 1)
        hint = QLabel(
            "Select two rows (Ctrl-click) for side-by-side geometry and raw-evidence comparison."
        )
        layout.addWidget(hint)
        comparison = QSplitter(Qt.Orientation.Horizontal)
        self.geometry_views: list[GeometryView] = []
        self.candidate_texts: list[QPlainTextEdit] = []
        for _ in range(2):
            column = QWidget()
            column_layout = QVBoxLayout(column)
            view, details = GeometryView(), text_box()
            column_layout.addWidget(view, 1)
            column_layout.addWidget(details, 1)
            comparison.addWidget(column)
            self.geometry_views.append(view)
            self.candidate_texts.append(details)
        layout.addWidget(comparison, 3)
        self.tabs.addTab(page, "Candidate Explorer")

    def _make_baselines(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.baseline_note = QLabel(
            "Matched references share fixed sites, physical protocol and exact evaluation accounting. "
            "Cheap proposal counts do not establish algorithm superiority."
        )
        self.baseline_note.setWordWrap(True)
        layout.addWidget(self.baseline_note)
        self.baseline_table = self._table(
            (
                "Candidate ID",
                "Reference / comparison role",
                "Exact score",
                "Validation state",
                "Exact attempts",
            )
        )
        self.baseline_table.itemSelectionChanged.connect(self.inspect_baseline)
        layout.addWidget(self.baseline_table, 1)
        self.baseline_details = text_box()
        self.baseline_geometry = GeometryView()
        preview = QSplitter(Qt.Orientation.Horizontal)
        preview.addWidget(self.baseline_geometry)
        preview.addWidget(self.baseline_details)
        layout.addWidget(preview, 2)
        self.baseline_table.cellDoubleClicked.connect(
            lambda row, _: self.select_candidate(table_item(self.baseline_table, row, 0).text())
        )
        self.tabs.addTab(page, "Baselines")

    def _make_log(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.checkpoints = QListWidget()
        self.checkpoints.currentRowChanged.connect(self.show_checkpoint)
        self.checkpoint_text = text_box()
        splitter.addWidget(self.checkpoints)
        splitter.addWidget(self.checkpoint_text)
        splitter.setSizes([220, 700])
        layout.addWidget(splitter, 1)
        self.log = text_box()
        layout.addWidget(panel("Chronological research log", self.log), 1)
        self.tabs.addTab(page, "Checkpoints / Research Log")

    def _operate(self, callback: Callable[[], Any], done: Callable[[Any], None]) -> None:
        if self.operation is not None and self.operation.isRunning():
            self.message.setText("An experiment operation is already in progress.")
            return
        if self.operation is not None:
            self.operation.deleteLater()
        self.operation = Task(callback, self)
        self.operation.result.connect(done)
        self.operation.error.connect(
            lambda error: self.message.setText(f"Operation failed: {error}")
        )
        self.operation.start()

    def start_run(self) -> None:
        try:
            config = self.configuration()
            target = str(config.get("output_directory", "")).strip()
            if not target:
                raise ValueError("Choose a new output directory")
            directory = Path(target).resolve()
        except (ValueError, TypeError, KeyError) as error:
            self.config_status.setText(f"Start rejected: {error}")
            return

        def start() -> Path:
            created = self.service.create(config, directory)
            self.service.launch(created)
            return Path(created)

        self.message.setText(
            "Creating reproducible experiment and starting its independent worker…"
        )
        self._operate(start, self.monitor)

    def request_control(self, action: str) -> None:
        if self.directory is None:
            return
        directory = self.directory
        self._operate(
            lambda: self.service.control(directory, action),
            lambda _: self._control_received(action),
        )

    def _control_received(self, action: str) -> None:
        self.message.setText(
            f"{action.capitalize()} requested. The worker applies safe controls between exact evaluations."
        )
        self.refresh()

    def open_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Open autonomous experiment", str(self.root)
        )
        if directory:
            self.monitor(Path(directory))

    def inspect_selected(self) -> None:
        path = self.experiments.currentData()
        if path:
            self.monitor(Path(path))

    def monitor(self, directory: Path) -> None:
        self.directory = Path(directory).resolve()
        self.tabs.setCurrentIndex(1)
        self.message.setText(
            f"Monitoring {self.directory}. Closing the window leaves its worker running."
        )
        self.refresh()

    def clone_experiment(self) -> None:
        try:
            source = self.snapshot.get("config") or self.configuration()
            config = json.loads(json_text(source))
            config["name"] = str(config.get("name", "Experiment")) + " (copy)"
            config["output_directory"] = str(
                self.root / (Path(str(config.get("output_directory", "experiment"))).name + "-copy")
            )
            config.pop("experiment_id", None)
            self.set_config(config)
            self.tabs.setCurrentIndex(0)
            self.config_status.setText(
                "Duplicated configuration for a new experiment. Existing data is retained."
            )
        except (ValueError, TypeError) as error:
            self.message.setText(f"Clone rejected: {error}")

    def refresh(self) -> None:
        if self.closed or (self.task is not None and self.task.isRunning()):
            return
        directory = self.directory
        scan = time.monotonic() - self.last_scan >= 30
        if scan:
            self.last_scan = time.monotonic()

        def read() -> tuple[Any, Any, Any]:
            return (
                directory,
                self.service.list_experiments(self.root) if scan else None,
                (self.service.snapshot(directory) if directory is not None else None),
            )

        if self.task is not None:
            self.task.deleteLater()
        self.task = Task(read, self)
        self.task.result.connect(self._received)
        self.task.error.connect(
            lambda error: self.message.setText(
                f"Refresh failed; displayed data may be stale: {error}"
            )
        )
        self.task.start()

    def _received(self, payload: tuple[Any, Any, Any]) -> None:
        directory, experiments, snapshot = payload
        if experiments is not None:
            chosen = self.experiments.currentData()
            self.experiments.clear()
            for experiment in experiments:
                path = experiment.get("directory", experiment.get("output_directory"))
                if path:
                    name = mapping(experiment.get("config")).get(
                        "name", experiment.get("name", Path(path).name)
                    )
                    self.experiments.addItem(
                        f"{name} · {experiment.get('status', 'unknown')}", str(path)
                    )
            if chosen:
                self.experiments.setCurrentIndex(self.experiments.findData(chosen))
        if snapshot is not None and directory == self.directory:
            self.update_snapshot(snapshot)

    def update_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.snapshot = snapshot
        config, state = mapping(snapshot.get("config")), mapping(snapshot.get("state"))
        status = str(state.get("status", "unknown"))
        self.run_title.setText(
            f"{config.get('name', 'Experiment')} · {status.upper()} · {state.get('experiment_id', '')}"
        )
        running = status.lower() in ("running", "starting", "initialized", "created")
        for action, button in self.controls.items():
            allowed = (
                running
                if action == "pause"
                else status.lower() != "archived"
                if action == "checkpoint"
                else status.lower() in ("paused", "interrupted", "failed", "created", "initialized")
                if action == "resume"
                else not running
                if action == "archive"
                else status.lower() not in ("completed", "archived")
            )
            button.setEnabled(self.directory is not None and allowed)
        used = state.get("exact_evaluations", 0)
        remaining = max(0, config.get("exact_budget", 0) - used)
        metrics = {
            **state,
            "remaining_exact_budget": remaining,
            "selection_allocation": mapping(config.get("search")).get("allocation", {}),
            "exact_evidence_policy": "Exact calculations only; predictions shown separately",
        }
        self.metrics.setPlainText(
            "\n".join(
                f"{key.replace('_', ' ').capitalize()}: {display(value)}"
                for key, value in metrics.items()
            )
        )
        self._update_dashboard_metrics(config, state)
        warnings = snapshot.get("warnings", state.get("warnings", []))
        failures = snapshot.get("errors", [])
        self.health.setPlainText(
            "\n".join(
                [f"• {warning}" for warning in warnings]
                + [f"• Numerical failure: {display(failure)}" for failure in failures]
                + ["Missing finite-size or Majorana validation never implies a positive result."]
            )
        )
        candidates = snapshot.get("candidates", [])
        self.archive.set_data(candidates, snapshot.get("archive", []), config)
        selected = [
            table_item(self.candidates, index.row(), 0).text()
            for index in self.candidates.selectionModel().selectedRows()
        ]
        rows = []
        for candidate in candidates:
            pred = prediction(candidate)
            rows.append(
                (
                    identifier(candidate),
                    score(candidate),
                    candidate.get("validation_state", candidate.get("status")),
                    candidate.get("family"),
                    candidate.get("generation", candidate.get("cycle")),
                    pred.get("score", pred.get("mean")),
                    pred.get("uncertainty", pred.get("std")),
                )
            )
        self._fill_table(self.candidates, rows)
        for row in range(self.candidates.rowCount()):
            table_item(self.candidates, row, 1).setForeground(QColor("#62d0c9"))
            for column in (5, 6):
                table_item(self.candidates, row, column).setForeground(QColor("#e5ae62"))
        self.candidates.blockSignals(True)
        for row in range(self.candidates.rowCount()):
            if table_item(self.candidates, row, 0).text() in selected:
                for column in range(self.candidates.columnCount()):
                    table_item(self.candidates, row, column).setSelected(True)
        self.candidates.blockSignals(False)
        self.filter_candidates()
        self.show_selection()
        self.update_progress()
        self.update_baselines()
        previous = self.checkpoints.currentRow()
        self._checkpoint_records = snapshot.get("checkpoints", [])
        self.checkpoints.clear()
        for record in self._checkpoint_records:
            report = mapping(record.get("report"))
            state = mapping(record.get("state"))
            self.checkpoints.addItem(
                f"{report.get('created', record.get('created_at', record.get('timestamp', 'Checkpoint')))}\n"
                f"exact attempts: {state.get('exact_evaluations', record.get('exact_evaluations', '?'))}"
            )
        if self._checkpoint_records:
            self.checkpoints.setCurrentRow(
                previous
                if 0 <= previous < len(self._checkpoint_records)
                else len(self._checkpoint_records) - 1
            )
        self.log.setPlainText("\n\n".join(json_text(event) for event in snapshot.get("events", [])))
        report = snapshot.get("report")
        self.report.setMarkdown(
            report
            if isinstance(report, str)
            else (
                "# Final report\n\nThe worker writes a structured scientific report at completion. "
                "Checkpoint reports are available during the run."
            )
        )

    @staticmethod
    def _fill_table(table: QTableWidget, rows: list[tuple[Any, ...]]) -> None:
        table.blockSignals(True)
        table.setSortingEnabled(False)
        table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem()
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    item.setData(Qt.ItemDataRole.DisplayRole, value)
                else:
                    item.setText(display(value))
                table.setItem(row, column, item)
        table.setSortingEnabled(True)
        table.resizeColumnsToContents()
        table.blockSignals(False)

    def filter_candidates(self) -> None:
        query = self.filter.text().casefold()
        for row in range(self.candidates.rowCount()):
            candidate_id = table_item(self.candidates, row, 0).text()
            candidate = self.find_candidate(candidate_id)
            self.candidates.setRowHidden(row, query not in json_text(candidate).casefold())

    def find_candidate(self, candidate_id: str) -> dict[str, Any]:
        return next(
            (
                candidate
                for candidate in self.snapshot.get("candidates", [])
                if identifier(candidate) == candidate_id
            ),
            {},
        )

    def select_candidate(self, candidate_id: str) -> None:
        self.filter.clear()
        for row in range(self.candidates.rowCount()):
            if table_item(self.candidates, row, 0).text() == candidate_id:
                self.candidates.selectRow(row)
                self.tabs.setCurrentIndex(4)
                self.candidates.scrollToItem(table_item(self.candidates, row, 0))
                return

    def show_selection(self) -> None:
        rows = self.candidates.selectionModel().selectedRows()
        for index, (view, details) in enumerate(zip(self.geometry_views, self.candidate_texts)):
            candidate = (
                self.find_candidate(table_item(self.candidates, rows[index].row(), 0).text())
                if index < len(rows)
                else {}
            )
            view.set_geometry(geometry(candidate))
            details.setPlainText(
                candidate_details(candidate)
                if candidate
                else "Select a candidate to inspect its stored evidence."
            )
        if hasattr(self, "baseline_table"):
            self.update_baselines()

    def update_progress(self) -> None:
        metric = self.progress_metric.currentText()
        history = self.snapshot.get("history", [])
        keys = {
            "Best exact / best validated score": ("best_score", "best_validated_score"),
            "Archive coverage": ("archive_coverage",),
            "Diversity": ("diversity",),
            "Surrogate error": ("surrogate_error",),
            "Duplicate rate": ("duplicate_rate",),
            "Invalid rate": ("invalid_rate",),
        }.get(metric, ())
        curves: dict[str, list[tuple[float, float]]] = {}
        for key in keys:
            points = []
            for step, row in enumerate(history):
                y = number(row.get(key))
                x = number(
                    row.get(
                        "training_step" if metric == "Surrogate error" else "exact_evaluations",
                        step,
                    )
                )
                if x is not None and y is not None:
                    points.append((x, y))
            curves[key.replace("_", " ")] = points
        if metric == "Surrogate error":
            points = []
            for version in self.snapshot.get("model_versions", []):
                x, y = number(version.get("model_version")), number(version.get("mae"))
                if x is not None and y is not None:
                    points.append((x, y))
            curves = {"Prequential mean absolute error": sorted(points)}
        distribution: dict[str, Any] = {}
        if metric == "Candidate-family distribution":
            for candidate in self.snapshot.get("candidates", []):
                family = str(candidate.get("family", "unspecified"))
                distribution[family] = distribution.get(family, 0) + 1
            curves = {
                "candidate count": [
                    (float(i), float(count)) for i, count in enumerate(distribution.values(), 1)
                ]
            }
        elif metric == "Exact robustness distribution":
            values = []
            for candidate in self.snapshot.get("candidates", []):
                result = mapping(candidate.get("raw_metrics"))
                value = (
                    number(result.get("robustness_success_fraction"))
                    if candidate.get("origin") == "exact"
                    else None
                )
                if value is not None:
                    values.append(value)
            curves = {
                "observed exact robustness": [
                    (float(i), value) for i, value in enumerate(sorted(values), 1)
                ]
            }
            distribution = {
                "count": len(values),
                "interpretation": "Stored exact robustness metric; absent values are excluded.",
            }
        self.distribution.setPlainText(
            json_text(distribution)
            if distribution
            else "Empty curves mean that the selected metric has not been measured. "
            "Best validated score remains absent until its exact gates pass."
        )
        x_label = (
            "Surrogate training step"
            if metric == "Surrogate error"
            else "Family index (see counts below)"
            if metric == "Candidate-family distribution"
            else "Candidate rank"
            if metric == "Exact robustness distribution"
            else "Exact physics attempts (including failures)"
        )
        self.progress.set_data(curves, metric, x_label)

    def update_baselines(self) -> None:
        candidates = self.snapshot.get("candidates", [])
        rows = []

        def attempts(candidate: dict[str, Any]) -> int | None:
            ledger = self.snapshot.get("attempts")
            return (
                sum(attempt.get("candidate") == identifier(candidate) for attempt in ledger)
                if isinstance(ledger, list)
                else None
            )

        for candidate in candidates:
            if candidate.get("baseline") or candidate.get("is_baseline"):
                rows.append(
                    (
                        identifier(candidate),
                        candidate.get("family", "baseline"),
                        score(candidate),
                        candidate.get("validation_state"),
                        attempts(candidate),
                    )
                )
        evaluated = [candidate for candidate in candidates if score(candidate) is not None]
        if evaluated:
            best = max(evaluated, key=lambda candidate: score(candidate) or 0.0)
            rows.append(
                (
                    identifier(best),
                    "Current best exact",
                    score(best),
                    best.get("validation_state"),
                    attempts(best),
                )
            )
        best_validated_id = mapping(self.snapshot.get("state")).get("best_validated_candidate")
        best_validated = self.find_candidate(str(best_validated_id))
        if best_validated:
            rows.append(
                (
                    identifier(best_validated),
                    "Best validated",
                    score(best_validated),
                    best_validated.get("validation_state"),
                    attempts(best_validated),
                )
            )
        for index in self.candidates.selectionModel().selectedRows()[:2]:
            candidate = self.find_candidate(table_item(self.candidates, index.row(), 0).text())
            rows.append(
                (
                    identifier(candidate),
                    "Selected comparison",
                    score(candidate),
                    candidate.get("validation_state"),
                    attempts(candidate),
                )
            )
        self._fill_table(self.baseline_table, rows)

    def inspect_baseline(self) -> None:
        rows = self.baseline_table.selectionModel().selectedRows()
        candidate = (
            self.find_candidate(table_item(self.baseline_table, rows[0].row(), 0).text())
            if rows
            else {}
        )
        self.baseline_geometry.set_geometry(geometry(candidate))
        self.baseline_details.setPlainText(
            "\n\n".join(
                candidate_details(
                    self.find_candidate(table_item(self.baseline_table, row.row(), 0).text())
                )
                for row in rows[:2]
            )
        )

    def show_checkpoint(self, row: int) -> None:
        if 0 <= row < len(self._checkpoint_records):
            record = self._checkpoint_records[row]
            report = record.get("report", {})
            self.checkpoint_text.setPlainText(
                (report if isinstance(report, str) else json_text(report))
                + "\n\nCHECKPOINT STATE\n"
                + json_text(
                    {
                        key: value
                        for key, value in record.items()
                        if key not in ("report", "strategy")
                    }
                )
            )

    def shutdown(self) -> None:
        self.closed = True
        self.timer.stop()
        for task in (self.task, self.operation):
            if task is not None:
                task.wait()  # Bounded I/O/launch only; independent physics worker is untouched.

    def closeEvent(self, event: Any) -> None:
        self.shutdown()
        event.accept()
