"""Native group selection, descriptive summaries and explicit user exports."""

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from toposc_live.groups import GroupDefinition, GroupMember, GroupSnapshot, export_group
from toposc_live.models import CampaignSnapshot, display
from toposc_live.pages import LeaderboardPage, text_box
from toposc_live.widgets import QualityPlot


class GroupDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Create Campaign Group — read-only analysis")
        self.resize(1000, 550)
        layout = QVBoxLayout(self)
        self.name = QLineEdit("Campaign group")
        layout.addWidget(self.name)
        layout.addWidget(
            QLabel("Add individual campaigns or a parent folder. Exclusions require a reason.")
        )
        controls = QHBoxLayout()
        for label, callback in (
            ("Add campaign…", self.add_campaign),
            ("Add seed folder…", self.add_folder),
            ("Remove selected", self.remove_selected),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            controls.addWidget(button)
        layout.addLayout(controls)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Campaign directory", "Role", "Exclusion reason"])
        self.table.setColumnWidth(0, 510)
        self.table.setColumnWidth(1, 160)
        self.table.setColumnWidth(2, 250)
        layout.addWidget(self.table)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Validate and open group")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def add_path(self, path: Path) -> None:
        value = str(path.resolve())
        if any(
            (item := self.table.item(r, 0)) is not None and item.text() == value
            for r in range(self.table.rowCount())
        ):
            return
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(value))
        role = QComboBox()
        role.addItems(["included", "technical_dropout", "excluded"])
        self.table.setCellWidget(row, 1, role)
        self.table.setItem(row, 2, QTableWidgetItem(""))

    def add_campaign(self) -> None:
        value = QFileDialog.getExistingDirectory(self, "Choose campaign directory")
        if value:
            self.add_path(Path(value))

    def add_folder(self) -> None:
        value = QFileDialog.getExistingDirectory(self, "Choose parent of seed directories")
        if value:
            try:
                for path in sorted(Path(value).iterdir()):
                    if (
                        path.is_dir()
                        and not path.is_symlink()
                        and (path / "manifest.json").is_file()
                    ):
                        self.add_path(path)
            except OSError as error:
                QMessageBox.warning(self, "Folder unavailable", str(error))

    def remove_selected(self) -> None:
        for row in sorted({item.row() for item in self.table.selectedItems()}, reverse=True):
            self.table.removeRow(row)

    def definition(self) -> GroupDefinition:
        members = []
        for row in range(self.table.rowCount()):
            role = self.table.cellWidget(row, 1)
            assert isinstance(role, QComboBox)
            directory, reason = self.table.item(row, 0), self.table.item(row, 2)
            assert directory is not None and reason is not None
            members.append(
                GroupMember(
                    Path(directory.text()).resolve(),
                    role.currentText(),
                    reason.text(),
                )
            )
        return GroupDefinition(self.name.text(), tuple(members))


class GroupPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.snapshot: GroupSnapshot | None = None
        layout = QVBoxLayout(self)
        self.title = QLabel("Campaign Group — open a saved group or create one")
        self.title.setWordWrap(True)
        layout.addWidget(self.title)
        self.warning = QLabel()
        self.warning.setWordWrap(True)
        layout.addWidget(self.warning)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Top N"))
        self.top_n = QSpinBox()
        self.top_n.setRange(1, 10000)
        self.top_n.setValue(20)
        self.top_n.valueChanged.connect(self.refresh_candidates)
        controls.addWidget(self.top_n)
        for label, kind in (
            ("Export JSON…", "json"),
            ("Export CSV…", "csv"),
            ("Save group definition…", "definition"),
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, k=kind: self.export(k))
            controls.addWidget(button)
        controls.addStretch()
        layout.addLayout(controls)
        tabs = QTabWidget()
        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.summary = text_box()
        self.plot = QualityPlot()
        self.plot.axis_label = "Completed cycles (equal batch/pool configuration)"
        self.plot.setMinimumHeight(330)
        split.addWidget(self.summary)
        split.addWidget(self.plot)
        split.setSizes([380, 750])
        overview_layout.addWidget(split, 3)
        self.members = QTableWidget(0, 7)
        self.members.setHorizontalHeaderLabels(
            [
                "Campaign Seed",
                "Role",
                "Status",
                "Best exact Q",
                "Last improvement cycle",
                "Last cycle gain",
                "Reason / directory",
            ]
        )
        from PySide6.QtWidgets import QAbstractItemView

        self.members.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        overview_layout.addWidget(self.members, 2)
        note = QLabel(
            "Observed finite-model quality only. White curve: equal-seed mean through the common completed horizon. No extrapolation. Last gains are descriptive; no saturation, Majorana or thermodynamic claim."
        )
        note.setWordWrap(True)
        overview_layout.addWidget(note)
        self.leaderboard = LeaderboardPage()
        tabs.addTab(overview, "Summary / seed progress")
        tabs.addTab(self.leaderboard, "Group top candidates")
        layout.addWidget(tabs)

    def loading(self, definition: GroupDefinition) -> None:
        self.snapshot = None
        self.title.setText(f"Validating group: {definition.name}")
        self.warning.clear()
        self.summary.clear()
        self.members.setRowCount(0)
        self.plot.set_data({}, None)
        self.leaderboard.update_snapshot(CampaignSnapshot(Path(), "Loading"))

    def update_snapshot(self, snapshot: GroupSnapshot) -> None:
        self.snapshot = snapshot
        self.title.setText(snapshot.definition.name)
        self.warning.setText("\n".join(snapshot.warnings))
        labels = "\n".join(
            f"{k.replace('_', ' ')}: {display(v)}" for k, v in snapshot.statistics.items()
        )
        self.summary.setPlainText(
            labels
            + "\n\nAll totals exclude excluded/dropout seeds. Valid pool counts accepted proposals; duplicate proposals are reported separately. Sample SD uses n−1.\n\n"
            + "\n".join(snapshot.warnings)
        )
        self.plot.set_data(snapshot.curves, snapshot.statistics["frozen_threshold"])
        self.members.setRowCount(len(snapshot.members))
        for row, member in enumerate(snapshot.members):
            values = [
                member["Campaign Seed"],
                member["role"],
                member["status"],
                member["best_exact_quality"],
                member.get("last_improvement_completed_cycle"),
                member.get("last_cycle_gain"),
                member["reason"] or member["directory"],
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(display(value))
                item.setToolTip(member["directory"])
                self.members.setItem(row, column, item)
        self.members.resizeColumnsToContents()
        self.refresh_candidates()

    def refresh_candidates(self) -> None:
        if self.snapshot:
            self.leaderboard.update_snapshot(
                CampaignSnapshot(
                    Path(),
                    self.snapshot.definition.name,
                    config=self.snapshot.config,
                    candidates=self.snapshot.candidates[: self.top_n.value()],
                )
            )

    def export(self, kind: str) -> None:
        if self.snapshot is None:
            return
        suffix = "json" if kind == "definition" else kind
        value, _ = QFileDialog.getSaveFileName(
            self,
            "Export outside campaign directories (new file)",
            f"campaign-group.{suffix}",
            f"{suffix.upper()} (*.{suffix})",
        )
        if not value:
            return
        try:
            path = Path(value).resolve()
            if kind == "definition":
                if any(
                    path.is_relative_to(m.directory.resolve())
                    for m in self.snapshot.definition.members
                ):
                    raise ValueError("Save definitions outside campaign directories")
                with path.open("x", encoding="utf-8") as stream:
                    json.dump(self.snapshot.definition.payload(), stream, indent=2)
            else:
                export_group(self.snapshot, path, self.top_n.value())
        except (ValueError, OSError) as error:
            QMessageBox.warning(self, "Export rejected", str(error))
