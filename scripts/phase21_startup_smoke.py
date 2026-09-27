"""Launch the real native entry point with a bounded Qt event-loop lifetime."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from toposc_live.__main__ import main as native_main
    from toposc_live.window import MainWindow

    observed = []
    original = MainWindow.show

    def show(window):
        original(window)
        def finish():
            observed.append({"title": window.windowTitle(), "visible": window.isVisible(),
                             "screenshot_saved": window.grab().save(str(args.output / "startup.png"))})
            window.close()
            QApplication.instance().quit()
        QTimer.singleShot(500, finish)

    # Only bound test lifetime; construction, entry point and widgets are real.
    MainWindow.show = show
    sys.argv = ["toposc-live", "--root", str(args.output / "empty-results"),
                "--state-dir", str(args.output / "isolated-user-state")]
    code = native_main()
    if code != 0 or not observed or not observed[0]["visible"]:
        raise RuntimeError("Native startup smoke failed")
    report = {"status": "PASS", "entry_point": "toposc_live.__main__.main", "window": observed[0],
              "experiment_started": False, "mode": os.environ["QT_QPA_PLATFORM"]}
    (args.output / "startup.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
