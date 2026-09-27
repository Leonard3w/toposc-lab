"""Schema-driven presentation of the canonical research configuration.

Widgets edit one JSON document. Standard mode changes visibility only; validation,
defaults, scientific bounds and capability choices belong to studio_config.
"""
from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from toposc_live.pages import scroll_page

CATEGORIES = (
    "Experiment", "Geometry", "Physics", "Disorder / Ensemble", "Search Space",
    "Search Method", "Search Settings", "Objectives", "Numerical Settings", "Output / Storage",
)


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)


def get_path(data: dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_path(data: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


class StudioConfigEditor(QWidget):
    changed = Signal()
    fields_changed = Signal()

    def __init__(self, data: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from toposc_lab.research.studio_config import PRESETS

        self.fields: dict[str, QWidget] = {}
        self._rows: list[tuple[QWidget, QWidget, bool]] = []
        self._syncing = False
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        self.preset = QComboBox()
        self.preset.addItems(list(PRESETS))
        load = QPushButton("Load preset")
        load.clicked.connect(self.load_preset)
        self.expert = QCheckBox("Expert Mode")
        self.expert.toggled.connect(self._visibility)
        toolbar.addWidget(QLabel("Preset"))
        toolbar.addWidget(self.preset)
        toolbar.addWidget(load)
        toolbar.addStretch()
        toolbar.addWidget(self.expert)
        layout.addLayout(toolbar)
        note = QLabel(
            "Standard Mode hides advanced controls only. All resolved values remain in the "
            "configuration. Search Space explicitly records locked and variable quantities."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.sections = QTabWidget()
        layout.addWidget(self.sections)
        self.json_label = QLabel("Complete canonical configuration · JSON import/export below")
        layout.addWidget(self.json_label)
        self.editor = QPlainTextEdit()
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.editor.setMinimumHeight(230)
        self.editor.textChanged.connect(self.changed)
        layout.addWidget(self.editor)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.set_config(data)

    def raw(self) -> dict[str, Any]:
        value = json.loads(self.editor.toPlainText())
        if not isinstance(value, dict):
            raise TypeError("Configuration must be a JSON object")
        return value

    def load_preset(self) -> None:
        from toposc_lab.research.studio_config import preset

        try:
            output = self.raw().get("output_directory")
            data = preset(self.preset.currentText())
            if output:
                data["output_directory"] = output
            self.set_config(data)
            self.status.setText("Preset copied into this editable configuration; nothing started.")
        except (ValueError, TypeError, KeyError) as error:
            self.status.setText(str(error))

    def set_config(self, data: dict[str, Any]) -> None:
        self._syncing = True
        try:
            self.editor.setPlainText(json_text(data))
            self._build_fields(data)
        finally:
            self._syncing = False
        self.changed.emit()

    def _build_fields(self, data: dict[str, Any]) -> None:
        from toposc_lab.research.studio_config import field_specs

        specifications = field_specs(data)
        selected = self.sections.currentIndex()
        while self.sections.count():
            widget = self.sections.widget(0)
            self.sections.removeTab(0)
            widget.deleteLater()
        self.fields.clear()
        self._rows.clear()
        forms: dict[str, QFormLayout] = {}
        for category in CATEGORIES:
            content = QWidget()
            form = QFormLayout(content)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            forms[category] = form
            self.sections.addTab(scroll_page(content), category)
        for spec in specifications:
            path = spec["path"]
            value = get_path(data, path, spec.get("default"))
            widget = self._widget(spec, value)
            self.fields[path] = widget
            label = QLabel(str(spec.get("label", path)))
            label.setWordWrap(True)
            tooltip = str(spec.get("description", "")) + f"\nConfig: {path}\nDefault: {spec.get('default')}"
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            widget.setEnabled(spec.get("editable", True))
            forms.get(spec.get("category", "Experiment"), forms["Experiment"]).addRow(label, widget)
            self._rows.append((label, widget, bool(spec.get("expert", False))))
        if selected >= 0:
            self.sections.setCurrentIndex(min(selected, self.sections.count() - 1))
        self._visibility()
        self.fields_changed.emit()

    def _widget(self, spec: dict[str, Any], value: Any) -> QWidget:
        path, kind = spec["path"], spec.get("type", "str")
        choices = spec.get("choices")
        if path == "space.families":
            from toposc_lab.research.studio_config import FAMILIES

            widget = QListWidget()
            widget.setMaximumHeight(130)
            for family in FAMILIES:
                item = QListWidgetItem(family, widget)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if family in value else Qt.CheckState.Unchecked)
            widget.itemChanged.connect(lambda _: self._edit(path, [
                widget.item(i).text() for i in range(widget.count())
                if widget.item(i).checkState() == Qt.CheckState.Checked
            ]))
        elif choices:
            widget = QComboBox()
            widget.addItems([str(item) for item in choices])
            if str(value) not in [widget.itemText(i) for i in range(widget.count())]:
                widget.addItem(str(value))
            widget.setCurrentText(str(value))
            widget.currentTextChanged.connect(lambda text: self._edit(path, text))
        elif kind == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(value))
            widget.toggled.connect(lambda checked: self._edit(path, checked))
        elif kind == "float" or (
            kind == "int" and value is not None and not -2147483647 <= value <= 2147483647
        ):
            # Qt's decimal spin box rounds tiny tolerances; QSpinBox only stores
            # signed 32-bit integers. Preserve the serialized numerical value.
            widget = QLineEdit(json.dumps(value))
            widget.editingFinished.connect(lambda: self._edit_text(path, widget.text(), kind))
        elif kind == "int" and value is not None:
            widget = QSpinBox()
            low = spec.get("min")
            high = spec.get("max")
            low = -2147483647 if low is None else low
            high = 2147483647 if high is None else high
            widget.setRange(min(low, value), max(high, value))
            widget.setValue(value)
            widget.valueChanged.connect(lambda number: self._edit(path, number))
        else:
            widget = QLineEdit(json.dumps(value) if kind in ("list", "dict") or value is None else str(value))
            if kind == "str":
                widget.textChanged.connect(lambda text: self._edit(path, text))
            else:
                widget.editingFinished.connect(lambda: self._edit_text(path, widget.text(), kind))
        return widget

    def _edit_text(self, path: str, text: str, kind: str) -> None:
        try:
            value = json.loads(text) if kind in ("list", "dict", "int", "float", "bool") else text
            json.dumps(value, allow_nan=False)
            self._edit(path, value)
        except ValueError as error:
            # Keep an invalid draft invalid. Otherwise Preview could silently
            # use the previous valid value while the user sees different text.
            self._edit(path, text)
            self.status.setText(f"{path}: invalid JSON value ({error})")

    def _edit(self, path: str, value: Any) -> None:
        if self._syncing:
            return
        try:
            data = self.raw()
            if path == "geometry_space":
                from toposc_lab.research.studio_config import switch_geometry_space

                data = switch_geometry_space(data, str(value))
                self.status.setText("Geometry space changed. Its applicable search settings are shown; review locks and constraints.")
            else:
                set_path(data, path, value)
            self.editor.setPlainText(json_text(data))
            self.status.setText("Configuration edited. Preview validates all constraints before Start.")
            if path in ("algorithm", "geometry_space", "space.families", "studio.mode"):
                QTimer.singleShot(0, self.refresh_fields)
        except (TypeError, ValueError, KeyError) as error:
            self.status.setText(f"Fix the JSON before editing controls: {error}")

    def refresh_fields(self) -> None:
        try:
            self._syncing = True
            self._build_fields(self.raw())
        except (ValueError, TypeError, KeyError) as error:
            self.status.setText(f"Apply JSON / validate to resolve dependent settings: {error}")
        finally:
            self._syncing = False

    def _visibility(self) -> None:
        expert = self.expert.isChecked()
        for label, widget, advanced in self._rows:
            label.setVisible(expert or not advanced)
            widget.setVisible(expert or not advanced)
        self.editor.setVisible(expert)
        self.json_label.setVisible(expert)
