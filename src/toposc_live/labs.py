"""Preserve the established interactive teaching labs inside the native studio.

The localhost helper starts only after an explicit Open action. It never owns
research workers and is terminated when the native application closes.
"""
from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class LabsPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.process: subprocess.Popen | None = None
        self.log = None
        self.browser = None
        self.layout = QVBoxLayout(self)
        self.message = QLabel(
            "Existing model simulations, parameter scans, study comparison, quantum gases, "
            "ensembles and Quantum Hall learning labs are available here. Opening the labs "
            "starts their local display service; experiments still use Research Studio."
        )
        self.message.setWordWrap(True)
        self.layout.addWidget(self.message)
        self.open_button = QPushButton("Open learning and model labs")
        self.open_button.clicked.connect(self.start)
        self.layout.addWidget(self.open_button)
        self.timer = QTimer(self)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.poll)

    def start(self) -> None:
        if self.process is not None:
            return
        try:
            if importlib.util.find_spec("streamlit") is None:
                raise ImportError("Streamlit is unavailable; install the documented .[live,app] extras")
            from PySide6.QtWebEngineWidgets import QWebEngineView

            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            self.url = f"http://127.0.0.1:{port}"
            from toposc_lab.app import streamlit_app

            command = [
                sys.executable, "-B", "-m", "streamlit", "run", str(Path(streamlit_app.__file__).resolve()),
                "--server.address=127.0.0.1", f"--server.port={port}", "--server.headless=true",
                "--browser.gatherUsageStats=false", "--server.fileWatcherType=none",
            ]
            environment = os.environ.copy()
            environment["TOPOSC_INTERNAL_LABS"] = "1"
            self.log = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115 -- closed with the widget service
            options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
            self.process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                                            stdout=self.log, stderr=self.log, **options)
            self.browser = QWebEngineView()
            self.layout.addWidget(self.browser, 1)
            self.started = time.monotonic()
            self.open_button.setEnabled(False)
            self.message.setText("Starting the existing local learning labs…")
            self.timer.start()
        except (ImportError, OSError, RuntimeError) as error:
            self.shutdown()
            self.message.setText(f"Learning labs unavailable: {error}. No packages were installed.")

    def poll(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is not None:
            details = ""
            if self.log is not None:
                self.log.seek(0)
                details = self.log.read().decode("utf-8", errors="replace")[-1800:]
            self.shutdown()
            self.message.setText("The local labs service stopped. " + details)
            return
        try:
            with urlopen(self.url + "/_stcore/health", timeout=0.1) as response:
                ready = response.status == 200
        except (URLError, OSError):
            ready = False
        if ready:
            self.timer.stop()
            self.browser.setUrl(QUrl(self.url))
            self.message.setText("Learning and model labs · use Research Studio for persistent experiments.")
        elif time.monotonic() - self.started > 60:
            self.shutdown()
            self.message.setText("The local labs service did not become ready. Check the optional .[app] environment.")

    def shutdown(self) -> None:
        self.timer.stop()
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
        self.open_button.setEnabled(True)
