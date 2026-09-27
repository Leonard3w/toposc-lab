"""Start the native desktop frontend; optional Qt imports occur after argument parsing."""

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TOPOSC Research Studio: unified local experiment control"
    )
    parser.add_argument(
        "--root", action="append", type=Path, help="Campaign discovery root; repeatable"
    )
    parser.add_argument("--open", dest="campaign", type=Path, help="Open a persisted campaign")
    parser.add_argument("--group", type=Path, help="Open a read-only campaign group definition")
    parser.add_argument(
        "--state-dir", type=Path, help="Application preferences and process metadata directory"
    )
    args = parser.parse_args()
    try:
        from PySide6.QtCore import QCoreApplication, Qt
        from PySide6.QtWidgets import QApplication

        from toposc_live.window import MainWindow
    except ImportError as error:
        print(
            f"Desktop dependencies unavailable ({error}). Install with: python -m pip install -e '.[live]'",
            file=sys.stderr,
        )
        return 2
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("TOPOSC Research Studio")
    app.setOrganizationName("toposc-lab")
    app.setStyle("Fusion")
    window = MainWindow(
        tuple(args.root or [Path.cwd() / "results"]), args.state_dir, args.campaign, args.group
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
