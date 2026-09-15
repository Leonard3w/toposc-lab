"""Background I/O with one outstanding refresh and queued Qt signal delivery."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from toposc_live.groups import GroupDefinition, load_group
from toposc_live.reader import CampaignReader, discover_campaigns


class Task(QThread):
    result = Signal(object)
    error = Signal(str)

    def __init__(self, operation: Callable[[], Any], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.operation = operation

    def run(self) -> None:
        try:
            self.result.emit(self.operation())
        except Exception as error:  # noqa: BLE001 -- I/O errors must not take down the GUI
            self.error.emit(f"{type(error).__name__}: {error}")


class RefreshService(QObject):
    snapshot = Signal(object)
    group_snapshot = Signal(object)
    campaigns = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        reader: CampaignReader,
        roots: tuple[Path, ...],
        recent: tuple[Path, ...],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.reader, self.roots, self.recent = reader, roots, recent
        self.directory: Path | None = None
        self.group: GroupDefinition | None = None
        self.task: Task | None = None
        self.discover_next = True
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.discovery_timer = QTimer(self)
        self.discovery_timer.setInterval(30000)
        self.discovery_timer.timeout.connect(self.rescan)
        self.discovery_timer.start()

    def rescan(self) -> None:
        self.discover_next = True
        self.refresh()

    def select(self, directory: Path) -> None:
        self.group = None
        self.directory = directory
        self.recent = tuple(dict.fromkeys((directory, *self.recent)))[:50]
        self.refresh()

    def select_group(self, definition: GroupDefinition) -> None:
        self.directory = None
        self.group = definition
        self.refresh()

    def refresh(self) -> None:
        if self.task is not None and self.task.isRunning():
            return
        directory, scan = self.directory, self.discover_next
        group = self.group
        roots, recent = self.roots, self.recent
        self.discover_next = False

        def load() -> tuple[Any, Any, Any]:
            campaigns = None
            if scan:
                jobs = tuple(Path(j["directory"]) for j in self.reader.processes.jobs())
                paths = discover_campaigns(roots, (*recent, *jobs))
                campaigns = [(path, self.reader.load(path).status) for path in paths]
            return (
                campaigns,
                self.reader.load(directory) if directory else None,
                load_group(self.reader, group) if group else None,
            )

        if self.task:
            self.task.deleteLater()
        self.task = Task(load, self)
        self.task.result.connect(self._received)
        self.task.error.connect(self.error)
        self.task.start()

    def _received(self, result: tuple[Any, Any, Any]) -> None:
        campaigns, snapshot, group = result
        if campaigns is not None:
            self.campaigns.emit(campaigns)
        if snapshot is not None and snapshot.directory == self.directory:
            self.snapshot.emit(snapshot)
        if group is not None and group.definition == self.group:
            self.group_snapshot.emit(group)

    def shutdown(self) -> None:
        self.timer.stop()
        self.discovery_timer.stop()
        if self.task:
            self.task.wait()  # Wait only for read-only I/O, never for a campaign process.
