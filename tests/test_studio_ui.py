"""Phase-20 UI contracts: mocked services, stored evidence, no scientific runs."""
import copy
import json
import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialogButtonBox

from toposc_lab.research.studio_config import PRESETS, preset, preview_config
from toposc_live.research_page import ResearchPage
from toposc_live.studio_editor import CATEGORIES, StudioConfigEditor


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class MockService:
    def __init__(self):
        self.calls = []

    def list_experiments(self, root):
        return []

    def create(self, config, directory):
        self.calls.append(("create", copy.deepcopy(config), directory))
        return directory

    def launch(self, directory):
        self.calls.append(("launch", directory))
        return 123

    def snapshot(self, directory):
        return {"config": preset("quick_test"), "state": {"status": "CREATED"}, "candidates": []}

    def control(self, directory, action):
        self.calls.append((action, directory))

    def follow_up(self, directory, identities, **kwargs):
        self.calls.append(("follow_up", directory, identities, kwargs))
        result = preset("quick_test")
        result["physics"]["disorder_widths"] = kwargs["widths"]
        result["physics"]["disorder_seeds"] = kwargs["seeds"]
        result["output_directory"] = kwargs["output_directory"]
        return result

    def candidate(self, directory, identity):
        from toposc_lab.research.space import FixedConnectivitySpace, geometry_to_payload

        self.calls.append(("candidate", directory, identity))
        return {
            "id": identity, "origin": "exact",
            "geometry": geometry_to_payload(FixedConnectivitySpace(side=4).reference()),
            "exact_results": {"clean": {"spectrum": [-1, -0.1, 0.1, 1],
                "boundary": {"site_probability": [1 / 16] * 16},
                "low_state": {"site_probability": [0.1] * 8 + [0.025] * 8}}},
        }


@pytest.fixture
def page(app, tmp_path):
    widget = ResearchPage(tmp_path, MockService())
    widget.timer.stop()
    yield widget
    if widget.preview_dialog is not None:
        widget.preview_dialog.reject()
    widget.close()
    app.processEvents()


def wait(app, condition):
    deadline = time.monotonic() + 5
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert condition()


def test_standard_expert_only_changes_visibility_and_preset_is_editable(app):
    editor = StudioConfigEditor(preset("quick_test"))
    before = editor.raw()
    assert tuple(editor.sections.tabText(i) for i in range(10)) == CATEGORIES
    assert tuple(editor.preset.itemText(i) for i in range(editor.preset.count())) == tuple(PRESETS)
    editor.expert.setChecked(True)
    editor.expert.setChecked(False)
    assert editor.raw() == before
    editor.fields["name"].setText("My independent experiment")
    assert editor.raw()["name"] == "My independent experiment"
    editor.fields["studio.variables.connectivity"].setCurrentText("locked")
    assert editor.raw()["studio"]["variables"]["connectivity"] == "locked"
    editor.close()


def test_scientific_tolerance_and_large_seed_survive_ui_roundtrip(app):
    from PySide6.QtWidgets import QLineEdit

    data = preset("quick_test")
    data["physics"]["tolerance"] = 1e-14
    data["seed"] = 2**40
    editor = StudioConfigEditor(data)
    tolerance = editor.fields["physics.tolerance"]
    seed = editor.fields["seed"]
    assert isinstance(tolerance, QLineEdit)
    assert isinstance(seed, QLineEdit)
    assert float(tolerance.text()) == 1e-14
    assert int(seed.text()) == 2**40
    tolerance.setText("1e-13")
    tolerance.editingFinished.emit()
    seed.setText(str(2**41))
    seed.editingFinished.emit()
    assert editor.raw()["physics"]["tolerance"] == 1e-13
    assert editor.raw()["seed"] == 2**41
    assert not preview_config(editor.raw())["errors"]
    for invalid in ("not-a-number", "NaN"):
        tolerance.setText(invalid)
        tolerance.editingFinished.emit()
        assert editor.raw()["physics"]["tolerance"] == invalid
        assert preview_config(editor.raw())["errors"]
    tolerance.setText("1e-13")
    tolerance.editingFinished.emit()
    assert not preview_config(editor.raw())["errors"]
    editor.close()


def test_family_change_preserves_locks_and_keeps_draft_controls_available(app):
    editor = StudioConfigEditor(preset("quick_test"))
    locks = copy.deepcopy(editor.raw()["studio"]["variables"])
    families = editor.fields["space.families"]
    for i in range(families.count()):
        if families.item(i).text() == "amorphous_planar":
            families.item(i).setCheckState(Qt.CheckState.Checked)
    app.processEvents()
    assert editor.raw()["studio"]["variables"] == locks
    assert "space.generator.n_sites" in editor.fields
    assert "studio.variables.coordinates" in editor.fields
    editor.fields["studio.variables.coordinates"].setCurrentText("variable")
    assert editor.raw()["studio"]["variables"]["coordinates"] == "variable"
    editor.close()


def test_preview_is_nonmutating_then_start_passes_exact_reviewed_config(page, app):
    page.start_run()
    assert page.service.calls == []
    assert page.reviewed_preview["errors"] == []
    reviewed = copy.deepcopy(page.reviewed_preview)
    page.preview_dialog.accept()
    wait(app, lambda: len(page.service.calls) >= 2)
    assert page.service.calls[0][0] == "create"
    assert page.service.calls[0][1] == reviewed["config"]
    assert preview_config(page.service.calls[0][1], check_output=False)["config_sha256"] == reviewed["config_sha256"]
    assert page.service.calls[1][0] == "launch"


def test_changed_config_invalidates_preview_and_invalid_config_cannot_start(page):
    page.start_run()
    page.fields["name"].setText("Changed after review")
    page.preview_dialog.accept()
    assert page.service.calls == []
    assert "changed after review" in page.config_status.text()
    data = page.studio_editor.raw()
    data["physics"]["disorder_seeds"] = [4, 4]
    page.editor.setPlainText(json.dumps(data))
    page.start_run()
    assert page.service.calls == []
    if page.reviewed_preview.get("errors"):
        ok = page.preview_dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok)
        assert not ok.isEnabled()


def test_export_load_and_followup_only_prepare_config(page, tmp_path):
    destination = tmp_path / "saved-preset.json"
    original = page.configuration()
    page.save_config(destination)
    page.fields["name"].setText("temporary")
    page.load_config(destination)
    assert page.configuration() == original
    page.directory = tmp_path / "old-run"
    page.prepare_follow_up(["candidate-1"], "[3, 6]", "[9001, 9002]", str(tmp_path / "follow-up"))
    assert [call[0] for call in page.service.calls] == ["follow_up"]
    assert page.studio_editor.raw()["physics"]["disorder_widths"] == [3, 6]
    assert page.tabs.currentIndex() == 0


def test_historical_controls_disabled_and_numeric_filter_ignores_predictions(page, tmp_path):
    page.directory = tmp_path
    snapshot = page.service.snapshot(tmp_path)
    snapshot["state"].update(status="PAUSED", historical_read_only=True)
    snapshot["candidates"] = [
        {"id": "a", "origin": "exact", "raw_metrics": {"quality": 0.1}, "prediction": {"quality": 0.99}},
        {"id": "b", "origin": "exact", "raw_metrics": {"quality": 0.5}},
    ]
    page.update_snapshot(snapshot)
    assert all(not button.isEnabled() for button in page.controls.values())
    page.request_control("resume")
    assert page.service.calls == []
    page.metric_filter.setCurrentText("Q / quality")
    page.metric_min.setText("0.3")
    visible = [page.candidates.item(i, 0).text() for i in range(2) if not page.candidates.isRowHidden(i)]
    assert visible == ["b"]


def test_metric_filters_use_named_observables_not_unrelated_indices():
    stage = {
        "status": "completed", "primary_valid": True,
        "stage": {"kind": "disorder", "width": 3, "seed": 12},
        "metrics": {"quality": .3, "boundary_weight": .99},
        "indices": [1, 1, 1], "localizer_gaps": [.2, .3, .4],
        "domain": {"bounds": [0, 7, 0, 7]},
        "spatial": [{"point": [3.5, 3.5], "valid": True, "gap": .08}],
        "chern_marker": {"status": "available", "bulk_mean": .44},
        "boundary_window": {"window": {"strip_weights": {"1": .7}}},
    }
    candidate = {"exact_results": {"disorder": stage}, "prediction": {"chern_bulk_mean": 1}}
    assert ResearchPage.metric_values(candidate, "Chern") == [.44]
    assert ResearchPage.metric_values(candidate, "Interior Localizer gap") == [.08]
    assert ResearchPage.metric_values(candidate, "Edge weight") == [.7]
    stage["primary_valid"] = False
    assert ResearchPage.metric_values(candidate, "Q / quality") == []
    assert ResearchPage.metric_values({"exact_results": {"x": {"indices": [1]}}}, "Chern") == []
    # Historical read adapters store reduced scalar metrics, without raw profiles.
    reduced = {"exact_results": {"disorder": {
        "status": "completed", "primary_valid": True,
        "metrics": {"chern_bulk_mean": .44, "interior_localizer_gap": .08, "edge_weight": .7},
    }}}
    assert ResearchPage.metric_values(reduced, "Chern") == [.44]
    assert ResearchPage.metric_values(reduced, "Interior Localizer gap") == [.08]


def test_selected_observables_control_inspection_filters_and_baseline_scope(page, tmp_path):
    snapshot = page.service.snapshot(tmp_path)
    snapshot["config"]["studio"]["observables"] = ["chern", "spectrum"]
    page.update_snapshot(snapshot)
    labels = [page.metric_filter.itemText(i) for i in range(page.metric_filter.count())]
    assert "Chern" in labels and "Q / quality" not in labels
    assert "fixed sites" not in page.baseline_note.text()


def test_candidate_evidence_reads_full_record_off_ui_thread(page, app, tmp_path):
    page.directory = tmp_path
    snapshot = page.service.snapshot(tmp_path)
    snapshot["candidates"] = [{"id": "stored", "origin": "exact"}]
    page.update_snapshot(snapshot)
    page.candidates.selectRow(0)
    page.show_candidate_evidence()
    wait(app, lambda: hasattr(page, "evidence_dialog"))
    assert page.service.calls == [("candidate", tmp_path, "stored")]
    assert "plot unavailable" not in page.evidence_dialog.evidence_status.text()
    axes = page.evidence_dialog.evidence_canvas.figure.axes
    assert len(axes[0].collections) > 0  # Spectrum points, not an empty/error figure.
    assert len(axes[1].collections) > 0  # Geometry and stored density.
    assert page.evidence_dialog.evidence_profiles.count() == 2
    page.evidence_dialog.evidence_profiles.setCurrentIndex(1)
    assert "plot unavailable" not in page.evidence_dialog.evidence_status.text()
    assert not page.evidence_dialog.grab().isNull()
    page.evidence_dialog.close()


def test_launcher_alias_and_labs_are_lazy(app, monkeypatch):
    import subprocess

    import toposc_live.__main__ as native
    from toposc_lab.app import streamlit_app
    from toposc_live.labs import LabsPage

    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Labs launched eagerly"))
    labs = LabsPage()
    assert labs.process is None
    labs.shutdown()
    monkeypatch.setattr(native, "main", lambda: calls.append("native") or 0)
    with pytest.raises(SystemExit) as result:
        streamlit_app.main()
    assert result.value.code == 0
    assert calls == ["native"]
    project = Path(__file__).resolve().parents[1]
    assert 'toposc-ui = "toposc_live.__main__:main"' in (project / "pyproject.toml").read_text()
    launcher = (project / "start_toposc.bat").read_text()
    assert "-m toposc_live" in launcher and "pause" in launcher
    assert "pip install" not in launcher and "venv create" not in launcher
