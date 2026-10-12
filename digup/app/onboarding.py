"""First run: get llama.cpp and the model (once, about 1 GB with CUDA), and pick the folders to search, with how long
the first pass will take for each."""

from __future__ import annotations

import os
import threading
import time

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

from .. import downloads, paths
from ..crawler import crawl
from ..model import FileKind
from ..settings import Settings
from . import theme

# Rough seconds per file on a mid-range GPU (measured on the testbed with CUDA; the CPU is several times slower).
SECONDS = {FileKind.IMAGE: 0.05, FileKind.SCREENSHOT: 0.12, FileKind.PDF: 0.6, FileKind.DOC: 0.1,
           FileKind.AUDIO: 2.0, FileKind.VIDEO: 6.0}


class Bridge(QObject):
    progress = Signal(str, float, str)  # what, fraction, label
    finished = Signal(str)  # "" or an error
    estimate = Signal(str, str)  # folder, text


def describe(seconds: float) -> str:
    if seconds < 90:
        return "about a minute"
    if seconds < 3600:
        return f"about {round(seconds / 60)} minutes"
    return f"about {seconds / 3600:.1f} hours"


class Onboarding(QWidget):
    done = Signal()

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.setObjectName("Window")
        self.setWindowTitle("Welcome to DigUp")
        self.setWindowIcon(theme.app_icon())
        self.resize(620, 600)
        self.bridge = Bridge()
        self.bridge.progress.connect(self._progress)
        self.bridge.finished.connect(self._downloaded)
        self.bridge.estimate.connect(self._estimated)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 22, 28, 20)
        layout.setSpacing(10)
        title = QLabel("Describe it. Dig it up.")
        title.setStyleSheet("font-size: 20pt; font-weight: 600;")
        layout.addWidget(title)
        intro = QLabel("DigUp finds the files on your PC by what's in them, with Google DeepMind's EmbeddingGemma 2: "
                       "photos by what they show, recordings and videos by what's said and shown, PDFs and documents "
                       "by what they say, in 100+ languages. Everything stays on this PC.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.backend = downloads.best_backend()
        need = downloads.runtime_bytes(self.backend) if not downloads.runtime_ready(self.backend) else 0
        need += sum(f.bytes for f in downloads.missing_models())
        self.download_label = QLabel(
            f"Downloading the model and llama.cpp for {self.backend.upper()} ({need / 1e6:,.0f} MB, once)…"
            if need else "The model is ready.")
        layout.addWidget(self.download_label)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(1000 if not need else 0)
        layout.addWidget(self.bar)

        layout.addSpacing(8)
        layout.addWidget(QLabel("<b>Which folders should DigUp search?</b>"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        self.rows = QVBoxLayout(holder)
        self.rows.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(holder)
        layout.addWidget(scroll, 1)
        self.checks: dict[str, tuple[QCheckBox, QLabel]] = {}
        for folder in settings.folders or Settings.default_folders():
            self._add_row(folder, checked=True)
        add = QPushButton("Add another folder…")
        add.clicked.connect(self._pick)
        layout.addWidget(add, 0, Qt.AlignmentFlag.AlignLeft)
        hotkey = QLabel(f"Then press <b>{settings.hotkey}</b> in any app and describe what you're looking for.")
        layout.addWidget(hotkey)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.start = QPushButton("Start")
        self.start.setObjectName("Primary")
        self.start.clicked.connect(self._start)
        buttons.addWidget(self.start)
        layout.addLayout(buttons)
        self._stop = False
        if need:
            threading.Thread(target=self._download, daemon=True).start()

    def showEvent(self, event) -> None:
        theme.decorate(self, theme.dark_mode())
        super().showEvent(event)

    def closeEvent(self, event) -> None:
        super().closeEvent(event)

    def _add_row(self, folder: str, checked: bool) -> None:
        row = QHBoxLayout()
        check = QCheckBox(folder.replace(os.path.expanduser("~"), "~"))
        check.setChecked(checked)
        check.setToolTip(folder)
        note = QLabel("counting…")
        note.setStyleSheet("color: gray;")
        row.addWidget(check, 1)
        row.addWidget(note)
        self.rows.addLayout(row)
        self.checks[folder] = (check, note)
        threading.Thread(target=self._estimate, args=(folder,), daemon=True).start()

    def _pick(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder", os.path.expanduser("~"))
        if folder:
            folder = paths.normalize(folder)
            if folder not in self.checks:
                self._add_row(folder, True)

    def _estimate(self, folder: str) -> None:
        try:
            report = crawl([folder], self.settings.options(), lambda: self._stop)
        except Exception:
            self.bridge.estimate.emit(folder, "")
            return
        files = report.candidates
        seconds = sum(SECONDS.get(f.kind, 0.2) for f in files)
        if self.backend == "cpu":
            seconds *= 6
        text = f"{len(files):,} files, {describe(seconds)}" if files else "nothing to read"
        self.bridge.estimate.emit(folder, text)

    def _estimated(self, folder: str, text: str) -> None:
        if folder in self.checks:
            self.checks[folder][1].setText(text)

    def _download(self) -> None:
        try:
            if not downloads.runtime_ready(self.backend):
                downloads.download_runtime(self.backend, lambda n, t: self.bridge.progress.emit(
                    "runtime", n / t if t else 0, f"llama.cpp ({self.backend.upper()})"), lambda: self._stop)
            if downloads.missing_models():
                downloads.download_models(lambda n, t: self.bridge.progress.emit(
                    "model", n / t if t else 0, "EmbeddingGemma 2"), lambda: self._stop)
            self.bridge.finished.emit("")
        except Exception as error:  # noqa: BLE001
            self.bridge.finished.emit(str(error))

    def _progress(self, what: str, fraction: float, label: str) -> None:
        # The runtime is the first fifth of the bar when it's needed.
        value = fraction * 0.3 if what == "runtime" else 0.3 + fraction * 0.7
        self.bar.setValue(int(value * 1000))
        self.download_label.setText(f"Downloading {label}… {fraction * 100:.0f}%")

    def _downloaded(self, error: str) -> None:
        if error:
            self.download_label.setText(f"The download stopped: {error}. DigUp will try again when it starts next.")
        else:
            self.bar.setValue(1000)
            self.download_label.setText("The model is ready.")

    def _start(self) -> None:
        self.settings.folders = [f for f, (check, _) in self.checks.items() if check.isChecked()]
        self.settings.onboarded = True
        self.settings.save()
        self.done.emit()
        self.hide()
