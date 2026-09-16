"""Offscreen research workflow checks; UI tests never run scientific workers."""

import json
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QItemSelectionModel, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from toposc_lab.research.config import ExperimentConfig
from toposc_lab.research.strategies import STRATEGY_REGISTRY
from toposc_live.research_page import ResearchPage, candidate_details, geometry, score


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class FakeService:
    def __init__(self):
        self.calls = []

    def create(self, config, directory):
        self.calls.append(("create", config, directory))
        return directory

    def launch(self, directory):
        self.calls.append(("launch", directory))
        return 123

    def list_experiments(self, root):
        return []

    def snapshot(self, directory):
        return fixture_snapshot(directory)

    def control(self, directory, action):
        self.calls.append((action, directory))


def fixture_snapshot(directory):
    config = ExperimentConfig().to_dict()
    config["output_directory"] = str(directory)
    candidate = {
        "id": "exact-one",
        "origin": "exact",
        "score": 0.15,
        "validation_state": "EXACT_EVALUATED",
        "family": "regular",
        "baseline": True,
        "generation": 1,
        "archive_cell": [2, 3],
        "geometry": {
            "coordinates": [[0, 0], [1, 0], [0, 1]],
            "edges": [[0, 1], [0, 2]],
            "boundary_sites": [0, 1, 2],
        },
        "prediction": {"score": 0.9, "uncertainty": 0.25, "ood": True},
        "exact_results": {"clean": {"quality": 0.15}},
        "validation_results": {"majorana": {"passed": False, "majorana_claim": False}},
        "raw_metrics": {"quality": 0.15, "robustness_success_fraction": 0.0},
    }
    predicted = {
        **candidate,
        "id": "predicted-only",
        "origin": "surrogate",
        "score": 99,
        "validation_state": "SURROGATE_SCREENED",
        "baseline": False,
        "exact_results": {},
        "raw_metrics": {},
    }
    return {
        "config": config,
        "state": {
            "status": "PAUSED",
            "exact_evaluations": 3,
            "best_score": 0.15,
            "elapsed_seconds": 12,
        },
        "candidates": [candidate, predicted],
        "archive": [candidate],
        "history": [
            {
                "exact_evaluations": 3,
                "best_score": 0.15,
                "best_validated_score": None,
                "archive_coverage": 1 / 144,
            }
        ],
        "checkpoints": [{"id": "cp1", "exact_evaluations": 3, "report": "No Majorana claim"}],
        "events": [{"kind": "paused", "timestamp": "2026-09-15"}],
        "report": "# Test report\n\nInconclusive: exact gates have not passed.",
    }


def wait_ui(app, predicate):
    end = time.monotonic() + 5
    while not predicate() and time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)
    assert predicate()


@pytest.fixture
def page(app, tmp_path):
    backend = FakeService()
    widget = ResearchPage(tmp_path, backend)
    yield widget
    widget.close()
    app.processEvents()


def test_config_roundtrip_dynamic_registry_and_validation(page, tmp_path):
    assert page.fields["space.side"].value() == 10
    assert page.fields["search.allocation.exploitation"].value() == 60
    page.fields["search.allocation.exploitation"].setValue(50)
    assert json.loads(page.editor.toPlainText())["search"]["allocation"]["exploitation"] == 0.5
    assert {
        page.fields["algorithm"].itemText(i) for i in range(page.fields["algorithm"].count())
    } == set(STRATEGY_REGISTRY)
    page.fields["name"].setText("UI configured experiment")
    page.fields["seed"].setValue(17040)
    target = tmp_path / "experiment.json"
    page.save_config(target)
    saved = json.loads(target.read_text())
    assert saved["name"] == "UI configured experiment"
    assert saved["seed"] == 17040
    page.fields["seed"].setValue(2)
    page.load_config(target)
    assert page.fields["seed"].value() == 17040
    page.editor.setPlainText('{"seed": false}')
    page.validate_config()
    assert "rejected" in page.config_status.text()
    page.start_run()
    assert page.service.calls == []


def test_exact_and_predictions_never_share_scores(page, tmp_path):
    snapshot = fixture_snapshot(tmp_path)
    exact_candidate, prediction_candidate = snapshot["candidates"]
    assert score(exact_candidate) == 0.15
    assert score(prediction_candidate) is None
    assert "SURROGATE PREDICTIONS" in candidate_details(prediction_candidate)
    assert "majorana_claim" in candidate_details(exact_candidate)
    page.update_snapshot(snapshot)
    assert page.progress.curves["best score"] == [(3.0, 0.15)]
    assert page.progress.curves["best validated score"] == []
    predicted_row = next(
        i for i in range(2) if page.candidates.item(i, 0).text() == "predicted-only"
    )
    assert page.candidates.item(predicted_row, 1).text() == "unavailable"
    assert page.candidates.item(predicted_row, 5).text() == "0.9"


def test_baseline_attempt_counts_use_complete_ledger_including_failed_stages(page, tmp_path):
    snapshot = fixture_snapshot(tmp_path)
    snapshot["attempts"] = [
        {"candidate": "exact-one", "status": status}
        for status in ("complete", "failed", "interrupted", "complete")
    ]
    page.update_snapshot(snapshot)
    for row in range(page.baseline_table.rowCount()):
        assert page.baseline_table.item(row, 4).text() == "4"
    snapshot.pop("attempts")
    page.update_snapshot(snapshot)
    for row in range(page.baseline_table.rowCount()):
        assert page.baseline_table.item(row, 4).text() == "unavailable"


def test_dashboard_metric_hierarchy_and_completed_checkpoint(page, app, tmp_path):
    snapshot = fixture_snapshot(tmp_path)
    snapshot["state"].update(
        status="COMPLETED",
        cycle=2,
        generated=40,
        archive_coverage=0.25,
        duplicate_rate=0.1,
        invalid_rate=0.2,
        surrogate_status="ready",
        surrogate_error=0.03,
        pending_kind="internal-state-only",
    )
    page.directory = tmp_path
    page.update_snapshot(snapshot)
    assert len(page.dashboard_values) == 16
    assert page.dashboard_values["status"].text() == "Completed"
    assert page.dashboard_values["exact_budget"].text() == "3 used · 277 left"
    assert page.dashboard_values["elapsed"].text() == "00:00:12"
    assert page.dashboard_values["best_validated"].text() == "None passed"
    assert page.dashboard_values["coverage"].text() == "25.0%"
    assert page.dashboard_values["rejection_rates"].text() == "10.0% / 20.0%"
    assert "60.0% exploit" in page.dashboard_values["allocation"].text()
    assert not any("pending_kind" in label.text() for label in page.dashboard_values.values())
    assert page.raw_state_panel.isHidden()
    assert page.controls["checkpoint"].isEnabled()
    assert not page.controls["resume"].isEnabled()
    page.raw_state_toggle.click()
    assert not page.raw_state_panel.isHidden()


def test_archive_click_comparison_filters_and_checkpoint(page, app, tmp_path):
    page.directory = tmp_path
    page.update_snapshot(fixture_snapshot(tmp_path))
    page.resize(1250, 950)
    page.show()
    page.tabs.setCurrentIndex(2)
    app.processEvents()
    assert len(page.archive.hit_boxes) == 1
    center = page.archive.hit_boxes[0][0].center()
    QTest.mouseClick(
        page.archive, Qt.MouseButton.LeftButton, pos=QPoint(int(center.x()), int(center.y()))
    )
    assert page.tabs.currentIndex() == 4
    assert "exact-one" in page.candidate_texts[0].toPlainText()
    assert len(page.geometry_views[0].geometry_data.edges) == 2
    model = page.candidates.selectionModel()
    for row in range(2):
        model.select(
            page.candidates.model().index(row, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )
    assert all("CANDIDATE" in box.toPlainText() for box in page.candidate_texts)
    page.filter.setText("predicted-only")
    assert sum(not page.candidates.isRowHidden(i) for i in range(2)) == 1
    page.checkpoints.setCurrentRow(0)
    assert "No Majorana claim" in page.checkpoint_text.toPlainText()
    assert "Inconclusive" in page.report.toPlainText()
    assert page.baseline_table.rowCount() >= 2
    assert not page.grab().isNull()


def test_start_controls_and_cloning_use_shared_service(page, app, tmp_path):
    page.start_run()
    wait_ui(app, lambda: page.directory is not None)
    wait_ui(app, lambda: page.operation is not None and not page.operation.isRunning())
    assert [call[0] for call in page.service.calls] == ["create", "launch"]
    page.request_control("resume")
    wait_ui(app, lambda: any(call[0] == "resume" for call in page.service.calls))
    wait_ui(app, lambda: page.operation is not None and not page.operation.isRunning())
    page.update_snapshot(fixture_snapshot(tmp_path))
    assert page.controls["resume"].isEnabled()
    assert not page.controls["pause"].isEnabled()
    old_config = json.loads(page.editor.toPlainText())
    page.clone_experiment()
    clone = json.loads(page.editor.toPlainText())
    assert clone["name"].endswith("(copy)")
    assert clone["output_directory"] != old_config["output_directory"]
    assert page.tabs.currentIndex() == 0
    assert len(page.service.calls) == 3  # Clone edits config, creating no worker.


def test_bad_geometry_is_inspectable_without_paint_failure():
    snapshot = geometry({"geometry": {"coordinates": [[0, 0]], "edges": [[0, 5]]}})
    assert not snapshot.edges
    assert "could not" in snapshot.message


def test_real_service_checkpoint_snapshot_survives_ui_restart(app, tmp_path, monkeypatch):
    from toposc_lab.research.service import ResearchService

    def forbidden(*args, **kwargs):
        raise AssertionError("Read-only UI must not launch an exact worker")

    service = ResearchService()
    config = ExperimentConfig(space={"side": 3}).to_dict()
    directory = service.create(config, tmp_path / "persisted")
    monkeypatch.setattr("subprocess.Popen", forbidden)
    service.control(directory, "pause")
    expected = service.snapshot(directory)
    assert expected["state"]["exact_evaluations"] == 0
    page = ResearchPage(tmp_path)
    try:
        page.monitor(directory)
        wait_ui(app, lambda: bool(page.snapshot))
        assert "PAUSED" in page.run_title.text()
        assert "exact attempts: 0" in page.checkpoints.item(0).text()
        assert page.controls["resume"].isEnabled()
    finally:
        page.close()
    restored = ResearchPage(tmp_path)
    try:
        restored.monitor(directory)
        wait_ui(app, lambda: bool(restored.snapshot))
        assert restored.snapshot["state"]["experiment_id"] == expected["state"]["experiment_id"]
        assert restored.snapshot["state"]["exact_evaluations"] == 0
    finally:
        restored.close()


def test_streamlit_research_area_config_without_launch(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    test = AppTest.from_string(
        "from toposc_lab.app.research_page import render_research_page\n"
        "import streamlit as st\n"
        "render_research_page(st)\n"
    ).run(timeout=30)
    assert not test.exception
    assert any(item.value == "Autonomous Research" for item in test.title)
    assert any(button.label == "Start run" for button in test.button)
    assert not list(tmp_path.rglob("research.sqlite3"))


def test_streamlit_snapshot_renders_exact_archive_and_prediction_separately(tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    snapshot = fixture_snapshot(tmp_path)
    script = (
        "import json\nfrom pathlib import Path\nimport streamlit as st\n"
        "from toposc_lab.app.research_page import _render_snapshot\n"
        "class Backend:\n"
        "    def control(self, *args): raise AssertionError('No control without user input')\n"
        f"snapshot = json.loads({json.dumps(snapshot)!r})\n"
        "_render_snapshot(st, Backend(), Path('.'), snapshot)\n"
    )
    test = AppTest.from_string(script).run(timeout=30)
    assert not test.exception
    assert len(test.tabs) == 7
    assert any("EXACT" in str(metric.label).upper() for metric in test.metric)
    assert any("Inconclusive" in str(markdown.value) for markdown in test.markdown)


def test_ui_reads_real_exact_archive_and_matched_attempt_counts(app, tmp_path):
    from toposc_lab.research.engine import ResearchEngine
    from toposc_lab.research.service import ResearchService

    config = ExperimentConfig(
        space={"side": 4},
        exact_budget=8,
        candidate_budget=20,
        pool_size=2,
        batch_size=1,
        cycles=1,
        checkpoint_every=4,
        physics={"disorder_widths": [0.2], "disorder_seeds": [171, 172]},
    )
    directory = ResearchService.create(config, tmp_path / "real-ui-smoke")
    ResearchEngine(directory).run()
    snapshot = ResearchService.snapshot(directory)
    assert snapshot["state"]["status"] == "COMPLETED"
    assert snapshot["state"]["exact_evaluations"] == 8
    assert len(snapshot["candidates"]) == 2
    page = ResearchPage(tmp_path, FakeService())
    try:
        page.update_snapshot(snapshot)
        assert len(page.archive.cells) >= 1
        assert page.candidates.rowCount() == 2
        for row in range(page.baseline_table.rowCount()):
            assert page.baseline_table.item(row, 4).text() == "4"
        assert (
            "Inconclusive" in page.report.toPlainText()
            or "inconclusive" in page.report.toPlainText()
        )
        assert "exact attempts: 8" in page.checkpoints.item(page.checkpoints.count() - 1).text()
        selected = snapshot["candidates"][0]
        page.select_candidate(selected["id"])
        assert len(page.geometry_views[0].geometry_data.coordinates) == 16
        assert "SURROGATE PREDICTIONS" in page.candidate_texts[0].toPlainText()
    finally:
        page.close()
