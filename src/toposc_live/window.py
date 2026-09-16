"""Main application shell. Campaign processes are not owned by Qt widgets."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from toposc_live.configuration import ConfigurationAdapter, LaunchPlan
from toposc_live.group_page import GroupDialog, GroupPage
from toposc_live.groups import GroupDefinition, GroupSnapshot
from toposc_live.models import CampaignSnapshot, sequence
from toposc_live.pages import LauncherPage, LeaderboardPage, LivePage, scroll_page, text_box
from toposc_live.processes import ProcessService, read_metadata, write_metadata
from toposc_live.reader import CampaignReader
from toposc_live.refresh import RefreshService, Task
from toposc_live.research_page import ResearchPage


class MainWindow(QMainWindow):
    def __init__(
        self,
        roots: tuple[Path, ...],
        state_directory: Path | None = None,
        initial: Path | None = None,
        initial_group: Path | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("TOPOSC LIVE — Research campaigns")
        self.resize(1440, 940)
        self.processes = ProcessService(state_directory)
        self.preferences = self.processes.state_directory / "preferences.json"
        saved = read_metadata(self.preferences)
        recent = tuple(Path(p) for p in sequence(saved.get("recent")) if isinstance(p, str))
        self.refresh_service = RefreshService(CampaignReader(self.processes), roots, recent, self)
        self.operation: Task | None = None
        self.snapshot: CampaignSnapshot | None = None
        self.current: Path | None = None
        shell = QWidget()
        self.setCentralWidget(shell)
        layout = QVBoxLayout(shell)
        title = QLabel("TOPOSC LIVE")
        title.setStyleSheet("font-size: 24px; font-weight: 650; padding: 6px;")
        title.setFixedHeight(48)
        layout.addWidget(title)
        toolbar = QHBoxLayout()
        self.open_button = QPushButton("Open / Monitor…")
        self.open_button.clicked.connect(self.open_campaign)
        self.scan_button = QPushButton("Rescan campaigns")
        self.scan_button.clicked.connect(self.refresh_service.rescan)
        self.resume_button = QPushButton("Review Resume…")
        self.resume_button.clicked.connect(self.review_resume)
        for button in (self.open_button, self.scan_button, self.resume_button):
            toolbar.addWidget(button)
        for label, callback in (("New Group…", self.new_group), ("Open Group…", self.open_group)):
            button = QPushButton(label)
            button.clicked.connect(callback)
            toolbar.addWidget(button)
        for label in ("Legacy stop unavailable", "Legacy pause unavailable"):
            button = QPushButton(label)
            button.setEnabled(False)
            button.setToolTip(
                "The current engine has no safe mid-run stop/pause request interface."
            )
            toolbar.addWidget(button)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        sidebar = QWidget()
        side_layout = QVBoxLayout(sidebar)
        side_layout.addWidget(QLabel("Campaigns · persisted records"))
        self.campaign_list = QListWidget()
        self.campaign_list.itemActivated.connect(self.select_item)
        self.campaign_list.itemClicked.connect(self.select_item)
        side_layout.addWidget(self.campaign_list)
        hint = QLabel(
            "Refresh: 2 s\nDiscovery scan: 30 s\nClosing this window leaves campaigns running."
        )
        hint.setWordWrap(True)
        side_layout.addWidget(hint)
        splitter.addWidget(sidebar)
        self.tabs = QTabWidget()
        self.launcher = LauncherPage(roots[0] if roots else Path.cwd() / "results")
        self.live, self.leaderboard = LivePage(), LeaderboardPage()
        self.tabs.addTab(scroll_page(self.launcher), "Campaign Launcher")
        self.tabs.addTab(self.live, "Live Campaign")
        self.tabs.addTab(self.leaderboard, "Candidate Leaderboard / Best Candidate")
        self.group_page = GroupPage()
        self.tabs.addTab(self.group_page, "Campaign Group")
        self.research_page = ResearchPage((roots[0] if roots else Path.cwd() / "results") / "research")
        self.tabs.addTab(self.research_page, "Autonomous Research")
        self.launcher.launch_requested.connect(self.launch)
        splitter.addWidget(self.tabs)
        splitter.setSizes([265, 1175])
        layout.addWidget(splitter, 1)
        self.refresh_service.snapshot.connect(self.receive_snapshot)
        self.refresh_service.group_snapshot.connect(self.receive_group)
        self.refresh_service.campaigns.connect(self.receive_campaigns)
        self.refresh_service.error.connect(self.show_status_error)
        self.statusBar().showMessage(
            "Read-only monitoring. Launch requires a validated configuration review."
        )
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #182532; color: #e2e8ef; }
            QGroupBox { border: 1px solid #415163; border-radius: 4px; margin-top: 12px; padding-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
            QLineEdit, QSpinBox, QPlainTextEdit, QListWidget, QTableWidget { background: #101d2b; border: 1px solid #415163; selection-background-color: #285e72; selection-color: #ffffff; padding: 5px; }
            QPushButton { background: #29475a; padding: 8px 12px; border: 1px solid #527186; border-radius: 4px; }
            QPushButton:hover { background: #355a70; }
            QPushButton:disabled { color: #81909c; background: #24313e; }
            QTabBar::tab { padding: 9px 12px; }
            QTabBar::tab:selected { background: #29475a; }
            QHeaderView::section { background: #29475a; padding: 5px; }
        """)
        if initial_group:
            self.open_group_path(initial_group)
        elif initial:
            self.monitor(initial)
        elif recent and recent[0].is_dir():
            self.monitor(recent[0])
        self.refresh_service.refresh()

    def show_status_error(self, message: str) -> None:
        if self.refresh_service.group is not None:
            self.group_page.title.setText(f"Group unavailable / stale: {message}")
            self.group_page.snapshot = None  # Retain visible last data; prevent stale export.
        self.statusBar().showMessage(
            f"Refresh unavailable; last displayed snapshot may be stale: {message}"
        )

    def open_campaign(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Open campaign directory")
        if directory:
            self.monitor(Path(directory))

    def new_group(self) -> None:
        dialog = GroupDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.monitor_group(dialog.definition())

    def open_group(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open group definition", "", "JSON (*.json)")
        if path:
            self.open_group_path(Path(path))

    def open_group_path(self, path: Path) -> None:
        try:
            self.monitor_group(GroupDefinition.read(path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.show_status_error(f"Group definition rejected: {error}")

    def monitor_group(self, definition: GroupDefinition) -> None:
        self.current = None
        self.snapshot = None
        self.group_page.loading(definition)
        self.tabs.setCurrentIndex(3)
        self.refresh_service.select_group(definition)
        self.statusBar().showMessage("Validating group compatibility and stored evidence…")

    def receive_group(self, snapshot: GroupSnapshot) -> None:
        self.group_page.update_snapshot(snapshot)
        self.statusBar().showMessage(
            f"Read-only group: {snapshot.definition.name}; {len(snapshot.warnings)} warning(s)"
        )

    def monitor(self, directory: Path) -> None:
        self.current = directory.resolve()
        self.snapshot = None
        empty = CampaignSnapshot(self.current, self.current.name, "Loading")
        self.live.update_snapshot(empty)
        self.leaderboard.update_snapshot(empty)
        self.refresh_service.select(self.current)
        self.tabs.setCurrentIndex(1)
        self.statusBar().showMessage(f"Loading {self.current}")
        self.live.title.setText(f"Loading {self.current}…")
        try:
            write_metadata(
                self.preferences, {"recent": [str(p) for p in self.refresh_service.recent]}
            )
        except OSError as error:
            self.statusBar().showMessage(
                f"Monitoring; recent-path preferences could not be saved: {error}"
            )

    def select_item(self, item: QListWidgetItem) -> None:
        self.monitor(Path(item.data(Qt.ItemDataRole.UserRole)))

    def receive_campaigns(self, campaigns: list[tuple[Path, str]]) -> None:
        self.campaign_list.clear()
        for path, status in campaigns:
            item = QListWidgetItem(f"{path.name}\n{status}")
            item.setToolTip(str(path))
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.campaign_list.addItem(item)
            if path == self.current:
                self.campaign_list.setCurrentItem(item)

    def receive_snapshot(self, snapshot: CampaignSnapshot) -> None:
        self.snapshot = snapshot
        self.live.update_snapshot(snapshot)
        self.leaderboard.update_snapshot(snapshot)
        self.statusBar().showMessage(
            f"{snapshot.status} · {snapshot.directory} · {len(snapshot.warnings)} reader warning(s)"
        )

    def run_operation(self, operation: Callable[[], Any], callback: Callable[[Any], None]) -> None:
        if self.operation is not None and self.operation.isRunning():
            return
        if self.operation:
            self.operation.deleteLater()
        self.launcher.start.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.operation = Task(operation, self)
        self.operation.result.connect(callback)
        self.operation.error.connect(
            lambda message: QMessageBox.warning(self, "Operation rejected", message)
        )
        self.operation.finished.connect(lambda: self.resume_button.setEnabled(True))
        self.operation.start()

    def launch(self, plans: tuple[LaunchPlan, ...]) -> None:
        if not plans:
            return

        def execute() -> list[Path]:
            launched = []
            for plan in plans:
                try:
                    self.processes.launch(plan)
                    launched.append(plan.directory)
                except Exception as error:
                    raise RuntimeError(
                        f"Launch failed at {plan.directory}: {error}. "
                        f"Already launched (still independent): {launched}"
                    ) from error
            return launched

        def ready(paths: list[Path]) -> None:
            self.launcher.invalidate()
            self.refresh_service.recent = tuple(
                dict.fromkeys((*paths, *self.refresh_service.recent))
            )
            self.monitor(paths[0])
            self.refresh_service.rescan()

        self.run_operation(execute, ready)

    def review_resume(self) -> None:
        if self.current is None:
            QMessageBox.information(self, "Resume", "Open an existing campaign first.")
            return
        directory = self.current

        def validate() -> LaunchPlan:
            if self.processes.inspect(directory).get("status") in (
                "Running",
                "Starting",
                "Unknown",
            ):
                raise ValueError("An existing process may be active; resume is unavailable.")
            return ConfigurationAdapter().resume_plan(directory)

        def review(plan: LaunchPlan) -> None:
            dialog = QDialog(self)
            dialog.setWindowTitle("Review compatible campaign resume")
            dialog.resize(780, 640)
            layout = QVBoxLayout(dialog)
            content = text_box()
            content.setPlainText(ConfigurationAdapter().summary((plan,)))
            layout.addWidget(content)
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Resume Campaign")
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                # Queue after the validation thread's finished signal is handled.
                from PySide6.QtCore import QTimer

                QTimer.singleShot(0, lambda: self.launch((plan,)))

        self.run_operation(validate, review)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.refresh_service.shutdown()
        self.research_page.shutdown()
        if self.operation:
            self.operation.wait()  # Configuration/launch I/O only; never process.wait().
        event.accept()
