"""Focused v1 pages consuming application services and display contracts."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from toposc_lab.discovery.config import DiscoveryConfig
from toposc_live.configuration import ConfigurationAdapter, LaunchPlan
from toposc_live.models import (
    CampaignSnapshot,
    CandidateSnapshot,
    GeometrySnapshot,
    display,
    mapping,
)
from toposc_live.widgets import GeometryView, QualityPlot


def text_box() -> QPlainTextEdit:
    box = QPlainTextEdit()
    box.setReadOnly(True)
    return box


def panel(title: str, widget: QWidget) -> QGroupBox:
    group = QGroupBox(title)
    layout = QVBoxLayout(group)
    layout.addWidget(widget)
    return group


class LauncherPage(QWidget):
    launch_requested = Signal(object)

    def __init__(self, output_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.adapter = ConfigurationAdapter()
        self.reviewed: tuple[LaunchPlan, ...] = ()
        defaults = self.adapter.defaults()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Configure a bounded discovery campaign"))
        form = QFormLayout()
        self.name = QLineEdit("Research campaign")
        self.output = QLineEdit(str(output_root / "new-campaign"))
        self.generator = QLineEdit(defaults.generator)
        self.generator.setPlaceholderText("Engine generator identifier")
        self.seed, self.seeds, self.cycles = QSpinBox(), QSpinBox(), QSpinBox()
        self.batch, self.pool = QSpinBox(), QSpinBox()
        for widget, minimum, maximum, value in (
            (self.seed, 0, 2147483647, defaults.seed),
            (self.seeds, 1, 100, 1),
            (self.cycles, 1, 100000, defaults.cycles),
            (self.batch, 4, 10000, defaults.batch_size),
            (self.pool, 4, 100000, defaults.pool_size),
        ):
            widget.setRange(minimum, maximum)
            widget.setValue(value)
            widget.valueChanged.connect(self.invalidate)
        for text_widget in (self.name, self.output, self.generator):
            text_widget.textChanged.connect(self.invalidate)
        form.addRow("Campaign name", self.name)
        form.addRow("Supported protocol", QLabel(defaults.stratum))
        form.addRow("Generator identifier", self.generator)
        hint = QLabel(
            "Current engine: auto, patch, random, evolution, coverage. Validation is authoritative."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        form.addRow("First seed", self.seed)
        form.addRow("Independent seeds", self.seeds)
        form.addRow("Cycles per seed", self.cycles)
        form.addRow("Exact candidates per cycle", self.batch)
        form.addRow("Pool size", self.pool)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output)
        browse = QPushButton("Choose parent…")
        browse.clicked.connect(self.choose_parent)
        output_row.addWidget(browse)
        form.addRow("New output location", output_row)
        self.one_cycle = QCheckBox(
            "Run one cycle, then checkpoint and exit (resume remaining cycles later)"
        )
        self.one_cycle.toggled.connect(self.invalidate)
        form.addRow(self.one_cycle)
        self.budget = QLabel()
        form.addRow("Exact attempt cap", self.budget)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.validate = QPushButton("Validate and review configuration")
        self.validate.clicked.connect(self.review)
        self.start = QPushButton("Start Campaign")
        self.start.setEnabled(False)
        self.start.clicked.connect(lambda: self.launch_requested.emit(self.reviewed))
        buttons.addWidget(self.validate)
        buttons.addWidget(self.start)
        layout.addLayout(buttons)
        self.summary = text_box()
        layout.addWidget(self.summary, 1)
        self.invalidate()

    def choose_parent(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose output parent")
        if selected:
            self.output.setText(str(Path(selected) / "new-campaign"))

    def invalidate(self) -> None:
        self.reviewed = ()
        if hasattr(self, "start"):
            self.start.setEnabled(False)
        if hasattr(self, "budget"):
            try:
                config = self.configuration()
                self.budget.setText(
                    f"{config.exact_attempt_cap} per seed / {config.exact_attempt_cap * self.seeds.value()} total, including reserve"
                )
            except (ValueError, TypeError) as error:
                self.budget.setText(str(error))
        if hasattr(self, "summary"):
            self.summary.setPlainText(
                "Review required before launch. Scientific model, thresholds and tolerances use the official frozen configuration."
            )

    def configuration(self) -> DiscoveryConfig:
        return self.adapter.build(
            generator=self.generator.text().strip(),
            seed=self.seed.value(),
            cycles=self.cycles.value(),
            batch_size=self.batch.value(),
            pool_size=self.pool.value(),
        )

    def review(self) -> None:
        try:
            if not self.output.text().strip():
                raise ValueError("Enter a new campaign output location.")
            self.reviewed = self.adapter.plans(
                self.name.text(),
                Path(self.output.text()),
                self.configuration(),
                self.seeds.value(),
                self.one_cycle.isChecked(),
            )
            self.summary.setPlainText(self.adapter.summary(self.reviewed))
            self.start.setEnabled(True)
        except (ValueError, TypeError, OSError) as error:
            self.reviewed = ()
            self.start.setEnabled(False)
            self.summary.setPlainText(f"Configuration rejected: {error}")


class LivePage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.title = QLabel("Open a campaign to inspect persisted results")
        self.title.setWordWrap(True)
        layout.addWidget(self.title)
        upper = QSplitter(Qt.Orientation.Horizontal)
        self.geometry_view, self.plot = GeometryView(), QualityPlot()
        upper.addWidget(panel("Best exact geometry", self.geometry_view))
        upper.addWidget(panel("Best exact quality · frozen threshold in amber", self.plot))
        layout.addWidget(upper, 2)
        lower = QSplitter(Qt.Orientation.Horizontal)
        self.progress, self.health, self.events = text_box(), text_box(), text_box()
        lower.addWidget(panel("Campaign progress / generator results", self.progress))
        lower.addWidget(panel("Campaign health / scientific scope", self.health))
        lower.addWidget(panel("Recent persisted events / log tail", self.events))
        layout.addWidget(lower, 2)

    def update_snapshot(self, snapshot: CampaignSnapshot) -> None:
        best = snapshot.best
        self.title.setText(
            f"{snapshot.name}  •  {snapshot.status.upper()}\n"
            f"Best exact quality: {display(best.quality if best else None)}   |   "
            f"Threshold: {display(snapshot.threshold)}   |   {snapshot.threshold_label}\n"
            f"Started: {display(snapshot.started)}   |   Elapsed seconds: {display(snapshot.elapsed)}"
        )
        self.geometry_view.set_geometry(best.geometry if best else GeometrySnapshot())
        self.plot.set_data(snapshot.curves, snapshot.threshold)
        generators: dict[str, list[CandidateSnapshot]] = {}
        for candidate in snapshot.candidates:
            generators.setdefault(candidate.generator, []).append(candidate)
        comparison = "\n\nStored exact candidates by generator (descriptive):\n"
        comparison += (
            "\n".join(
                f"{key}: {len(rows)} candidates; best {display(rows[0].quality)}"
                for key, rows in generators.items()
            )
            or "unavailable"
        )
        if len(generators) < 2:
            comparison += "\nCross-generator comparison unavailable in this campaign."
        self.progress.setPlainText(
            f"Campaign: {snapshot.directory}\nProtocol: {display(snapshot.config.get('stratum'))}\n"
            + "\n".join(f"{key}: {display(value)}" for key, value in snapshot.progress.items())
            + comparison
        )
        self.health.setPlainText(
            "\n".join(f"{key}: {display(value)}" for key, value in snapshot.health.items())
            + "\n\n"
            + (best.majorana if best else "Majorana: unavailable; no claim inferred")
            + "\n\nWarnings:\n"
            + ("\n".join(snapshot.warnings) or "None from read artifacts")
        )
        self.events.setPlainText(
            "\n".join(snapshot.events) + "\n\nProcess log tail:\n" + snapshot.log
        )


class LeaderboardPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.candidates: dict[str, CandidateSnapshot] = {}
        layout = QVBoxLayout(self)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter by generator, geometry ID, topology, seed or cycle")
        self.filter.textChanged.connect(self.apply_filter)
        layout.addWidget(self.filter)
        self.table = QTableWidget(0, 13)
        self.table.setHorizontalHeaderLabels(
            [
                "Rank",
                "Exact quality",
                "Generator",
                "Campaign Seed",
                "Exact Eval Seed",
                "Cycle",
                "Topology",
                "Robustness",
                "OOD",
                "Geometry ID",
                "Confirmation",
                "Localizer indices",
                "Minimum |E|",
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSortIndicator(0, Qt.SortOrder.AscendingOrder)
        self.table.itemSelectionChanged.connect(self.select_candidate)
        layout.addWidget(self.table, 2)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.geometry_view, self.details = GeometryView(), text_box()
        splitter.addWidget(panel("Selected candidate geometry", self.geometry_view))
        splitter.addWidget(panel("Stored exact diagnostics", self.details))
        layout.addWidget(splitter, 2)

    def update_snapshot(self, snapshot: CampaignSnapshot) -> None:
        selected = self.table.currentItem()
        selected_id = selected.data(Qt.ItemDataRole.UserRole) if selected else None
        self.candidates = {c.selection_key: c for c in snapshot.candidates}
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(snapshot.candidates))
        selection = 0
        for row, candidate in enumerate(snapshot.candidates):
            values = [
                row + 1,
                candidate.quality,
                candidate.generator,
                candidate.campaign_seed,
                candidate.seed,
                candidate.cycle,
                candidate.topology,
                "stored" if candidate.robustness is not None else None,
                mapping(candidate.ood).get("is_ood")
                if isinstance(candidate.ood, dict)
                else candidate.ood,
                candidate.geometry_id,
                candidate.confirmation,
                candidate.indices,
                candidate.minimum_abs_energy,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem()
                item.setData(
                    Qt.ItemDataRole.DisplayRole,
                    value if type(value) in (int, float) else display(value),
                )
                item.setData(Qt.ItemDataRole.UserRole, candidate.selection_key)
                if column == 8:
                    item.setToolTip(display(candidate.ood))
                elif column == 9:
                    item.setToolTip(candidate.geometry_id)
                self.table.setItem(row, column, item)
            if candidate.selection_key == selected_id:
                selection = row
        self.table.selectRow(selection)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(9, 230)
        self.apply_filter()
        if not snapshot.candidates:
            self.details.setPlainText("Exact candidates unavailable")
            self.geometry_view.set_geometry(GeometrySnapshot())

    def select_candidate(self) -> None:
        item = self.table.currentItem()
        candidate = self.candidates.get(item.data(Qt.ItemDataRole.UserRole)) if item else None
        if candidate:
            self.details.setPlainText(candidate.details())
            self.geometry_view.set_geometry(candidate.geometry)

    def apply_filter(self) -> None:
        query = self.filter.text().casefold()
        for row in range(self.table.rowCount()):
            self.table.setRowHidden(
                row,
                not any(
                    (item := self.table.item(row, column)) is not None
                    and query in item.text().casefold()
                    for column in range(self.table.columnCount())
                ),
            )


def scroll_page(widget: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(widget)
    return scroll
