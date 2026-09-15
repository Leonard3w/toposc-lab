"""Optional offscreen Qt tests; mock campaign launching and use stored fixtures."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from toposc_live.models import CampaignSnapshot, CandidateSnapshot, GeometrySnapshot
from toposc_live.pages import LauncherPage, LeaderboardPage
from toposc_live.window import MainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_launcher_review_and_invalidation(app, tmp_path):
    page = LauncherPage(tmp_path)
    assert not page.start.isEnabled()
    page.review()
    assert page.start.isEnabled()
    assert "0.2" in page.summary.toPlainText()
    page.generator.setText("unsupported")
    assert not page.start.isEnabled()
    page.review()
    assert "rejected" in page.summary.toPlainText()
    assert not page.start.isEnabled()
    page.close()


def test_leaderboard_sort_filter_selection(app, tmp_path):
    page = LeaderboardPage()
    snapshot = CampaignSnapshot(
        tmp_path,
        "fixture",
        candidates=(
            CandidateSnapshot("high", "geometry-a", "unknown-generator", quality=0.12),
            CandidateSnapshot("low", "geometry-b", "other-generator", quality=0.02),
        ),
    )
    page.update_snapshot(snapshot)
    page.table.sortItems(1, Qt.SortOrder.AscendingOrder)
    page.table.selectRow(0)
    assert "geometry-b" in page.details.toPlainText()
    page.filter.setText("unknown-generator")
    assert sum(not page.table.isRowHidden(r) for r in range(2)) == 1
    page.update_snapshot(snapshot)
    assert "geometry-b" in page.details.toPlainText()
    page.close()


def test_native_window_refresh_close_no_process_control(app, tmp_path, monkeypatch):
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "UI must not launch or terminate scientific processes during inspection"
        )

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    window = MainWindow((tmp_path / "campaigns",), tmp_path / "app-state")
    geometry = GeometrySnapshot(
        ((0, 0), (1, 0), (0, 1)), ((0, 1), (1, 2)), (0, 1), "Stored 2D embedding"
    )
    candidate = CandidateSnapshot(
        "r",
        "g",
        "future-generator",
        quality=0.11,
        geometry=geometry,
        majorana="diagnostics_only; majorana_claim=false",
    )
    snapshot = CampaignSnapshot(
        tmp_path,
        "Fixture only",
        "Running",
        {"success_threshold": 0.2},
        candidates=(candidate,),
        curves={"future-generator": ((1, 0.1), (3, 0.11))},
    )
    window.show()
    window.receive_snapshot(snapshot)
    app.processEvents()
    assert "RUNNING" in window.live.title.text()
    assert "majorana_claim=false" in window.live.health.toPlainText()
    assert not window.grab().isNull()
    window.close()
    assert not window.refresh_service.timer.isActive()


def test_background_refresh_delivers_snapshot(app, tmp_path):
    window = MainWindow((tmp_path,), tmp_path / "ui-state", tmp_path / "campaign")
    deadline = time.monotonic() + 5
    while window.snapshot is None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    try:
        assert window.snapshot is not None
        assert window.snapshot.directory == (tmp_path / "campaign").resolve()
    finally:
        window.close()


def test_group_page_top_n_distinct_seed_selection(app, tmp_path):
    from test_live_groups import aggregate, snapshot

    from toposc_live.group_page import GroupDialog, GroupPage

    group = aggregate(snapshot(tmp_path / "a", 10), snapshot(tmp_path / "b", 11, (0.3, 0.4)))
    page = GroupPage()
    page.update_snapshot(group)
    page.show()
    app.processEvents()
    assert page.members.rowCount() == 2
    assert "global best exact quality: 0.4" in page.summary.toPlainText()
    assert page.plot.curves["Group mean"][-1][1] == pytest.approx(0.3)
    board = page.leaderboard
    assert board.table.horizontalHeaderItem(3).text() == "Campaign Seed"
    assert board.table.horizontalHeaderItem(4).text() == "Exact Eval Seed"
    assert board.table.item(0, 3).text() == "11"
    assert board.table.item(0, 4).text() == "1101"
    board.table.selectRow(0)
    assert "Campaign Seed: 11" in board.details.toPlainText()
    page.top_n.setValue(1)
    assert board.table.rowCount() == 1
    assert not page.grab().isNull()
    page.close()
    dialog = GroupDialog()
    dialog.add_path(tmp_path / "a")
    dialog.add_path(tmp_path / "a")
    assert len(dialog.definition().members) == 1
    dialog.close()


def test_group_background_refresh_and_single_campaign_switch(app, tmp_path, monkeypatch):
    from test_live_groups import aggregate, snapshot

    from toposc_live import refresh

    group = aggregate(snapshot(tmp_path / "a", 10), snapshot(tmp_path / "b", 11))
    monkeypatch.setattr(refresh, "load_group", lambda reader, definition: group)
    window = MainWindow((tmp_path / "empty",), tmp_path / "ui-state")
    window.monitor_group(group.definition)
    deadline = time.monotonic() + 5
    while window.group_page.snapshot is None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    try:
        assert window.group_page.snapshot == group
        assert window.current is None
        window.show_status_error("incompatible fixture")
        assert window.group_page.snapshot is None
        assert "stale" in window.group_page.title.text()
        window.monitor(tmp_path / "single")
        assert window.refresh_service.group is None
    finally:
        window.close()
