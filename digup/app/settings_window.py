"""Settings: the folders to search, code search, the hotkey and apps results open in, and the index itself."""

from __future__ import annotations

import os
import subprocess
import sys
import winreg

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QKeySequenceEdit, QLabel,
                               QLineEdit, QListWidget, QMessageBox, QPushButton, QTabWidget, QVBoxLayout, QWidget)

from .. import downloads, paths
from ..settings import Settings
from ..store import CODE_FILE, MAIN_FILE, IndexStore
from . import opener, theme

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def launch_command() -> str:
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    return f'"{pythonw if os.path.exists(pythonw) else sys.executable}" -m digup.app.main'


def set_launch_at_login(on: bool) -> None:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        if on:
            winreg.SetValueEx(key, "DigUp", 0, winreg.REG_SZ, launch_command() + " --background")
        else:
            try:
                winreg.DeleteValue(key, "DigUp")
            except FileNotFoundError:
                pass


class FolderList(QWidget):
    changed = Signal()

    def __init__(self, folders: list[str], add_label: str = "Add folder…"):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.addItems(folders)
        layout.addWidget(self.list)
        row = QHBoxLayout()
        add = QPushButton(add_label)
        remove = QPushButton("Remove")
        add.clicked.connect(self._add)
        remove.clicked.connect(self._remove)
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch(1)
        layout.addLayout(row)

    def folders(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]

    def _add(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder", os.path.expanduser("~"))
        if folder:
            folder = paths.normalize(folder)
            if folder.lower() not in (f.lower() for f in self.folders()):
                self.list.addItem(folder)
                self.changed.emit()

    def _remove(self) -> None:
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))
        self.changed.emit()


class SettingsWindow(QWidget):
    saved = Signal(bool, bool)  # folders changed, code changed

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.setObjectName("Window")
        self.setWindowTitle("DigUp Settings")
        self.setWindowIcon(theme.app_icon())
        self.resize(640, 540)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        tabs = QTabWidget()
        layout.addWidget(tabs, 1)

        # Folders
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(QLabel("DigUp reads the pictures, PDFs, documents, recordings and videos in these folders. It "
                           "never changes, moves or uploads them."))
        self.folders = FolderList(settings.folders)
        v.addWidget(self.folders, 2)
        v.addWidget(QLabel("Skip these folders:"))
        self.excluded = FolderList(settings.excluded_folders, "Skip a folder…")
        v.addWidget(self.excluded, 1)
        form = QFormLayout()
        self.extensions = QLineEdit(settings.excluded_extensions)
        self.extensions.setPlaceholderText("e.g. heic, wav, *.mkv")
        form.addRow("Skip these file types:", self.extensions)
        v.addLayout(form)
        tabs.addTab(page, "Folders")

        # Code
        page = QWidget()
        v = QVBoxLayout(page)
        self.code_on = QCheckBox("Search code in these folders (start a search with “code:”)")
        self.code_on.setChecked(settings.code_search)
        v.addWidget(self.code_on)
        v.addWidget(QLabel("DigUp reads source files, notebooks and a repo's docs; it skips what git ignores, vendored "
                           "and generated code, and anything that looks like a key or a password."))
        self.code_folders = FolderList(settings.code_folders)
        v.addWidget(self.code_folders, 1)
        form = QFormLayout()
        self.editor = QComboBox()
        self.editor.addItem("Automatic", "auto")
        for key, name in opener.EDITOR_NAMES.items():
            installed = key == "notepad" or opener.EDITORS[key]() is not None
            self.editor.addItem(name + ("" if installed else " (not found)"), key)
        self.editor.setCurrentIndex(max(0, self.editor.findData(settings.editor)))
        form.addRow("Open code in:", self.editor)
        v.addLayout(form)
        tabs.addTab(page, "Code")

        # General
        page = QWidget()
        form = QFormLayout(page)
        self.hotkey = QKeySequenceEdit(QKeySequence(settings.hotkey))
        form.addRow("Search hotkey:", self.hotkey)
        self.login = QCheckBox("Start DigUp when I sign in to Windows")
        self.login.setChecked(settings.launch_at_login)
        form.addRow("", self.login)
        self.battery = QCheckBox("Index on battery (the first pass can be heavy)")
        self.battery.setChecked(settings.index_on_battery)
        form.addRow("", self.battery)
        self.saver = QCheckBox("Pause while battery saver is on")
        self.saver.setChecked(settings.pause_on_battery_saver)
        form.addRow("", self.saver)
        self.player = QComboBox()
        for label, key in (("Automatic (VLC, mpv, MPC-HC, PotPlayer)", "auto"), ("VLC", "vlc"), ("mpv", "mpv"),
                           ("MPC-HC", "mpc-hc"), ("PotPlayer", "potplayer"), ("The default app", "default")):
            self.player.addItem(label, key)
        self.player.setCurrentIndex(max(0, self.player.findData(settings.media_player)))
        form.addRow("Play videos and recordings in:", self.player)
        self.pdf = QComboBox()
        for label, key in (("Automatic (SumatraPDF, Edge)", "auto"), ("Microsoft Edge", "edge"),
                           ("SumatraPDF", "sumatra"), ("The default app", "default")):
            self.pdf.addItem(label, key)
        self.pdf.setCurrentIndex(max(0, self.pdf.findData(settings.pdf_viewer)))
        form.addRow("Open PDFs in:", self.pdf)
        self.backend = QComboBox()
        best = downloads.best_backend()
        for label, key in ((f"Automatic ({best.upper()})", "auto"), ("NVIDIA (CUDA)", "cuda"),
                           ("Any GPU (Vulkan)", "vulkan"), ("CPU only", "cpu")):
            self.backend.addItem(label, key)
        self.backend.setCurrentIndex(max(0, self.backend.findData(settings.backend)))
        form.addRow("Run the model on:", self.backend)
        tabs.addTab(page, "General")

        # Index
        page = QWidget()
        v = QVBoxLayout(page)
        self.summary = QLabel()
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.summary.setWordWrap(True)
        v.addWidget(self.summary, 1)
        row = QHBoxLayout()
        refresh = QPushButton("Refresh")
        retry = QPushButton("Try failed files again")
        rebuild = QPushButton("Read everything again")
        logs = QPushButton("Open logs")
        refresh.clicked.connect(self.refresh_summary)
        retry.clicked.connect(self._retry)
        rebuild.clicked.connect(self._rebuild)
        logs.clicked.connect(lambda: os.startfile(str(paths.logs())) if paths.logs().exists() else None)
        for b in (refresh, retry, rebuild, logs):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        tabs.addTab(page, "Index")
        tabs.currentChanged.connect(lambda i: i == 3 and self.refresh_summary())

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        save = QPushButton("Save")
        save.setObjectName("Primary")
        cancel.clicked.connect(self.close)
        save.clicked.connect(self._save)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        layout.addLayout(buttons)
        self.engine = None

    def showEvent(self, event) -> None:
        theme.decorate(self, theme.dark_mode())
        super().showEvent(event)

    def refresh_summary(self) -> None:
        lines = [f"Data folder: {paths.home()}",
                 f"Model runs on: llama.cpp {downloads.LLAMA_TAG}, {self.settings.backend_choice or downloads.best_backend()}",
                 ""]
        for name, label in ((MAIN_FILE, "Files"), (CODE_FILE, "Code")):
            file = paths.index() / name
            if not file.exists():
                continue
            store = IndexStore(paths.index(), name)
            counts = store.counts()
            lines.append(f"{label}: {counts['vectors']:,} vectors, {IndexStore.size_of(file) / 1e6:.0f} MB")
            for kind, states in sorted(counts["by_kind_and_state"].items()):
                lines.append(f"    {kind}: " + ", ".join(f"{n:,} {s}" for s, n in sorted(states.items())))
            failed = store.db.execute("SELECT path, detail FROM files WHERE state = 'failed' LIMIT 8").fetchall()
            if failed:
                lines.append("    Couldn't read:")
                lines += [f"      {os.path.basename(p)}: {d}" for p, d in failed]
            store.close()
            lines.append("")
        self.summary.setText("\n".join(lines))

    def _retry(self) -> None:
        for name in (MAIN_FILE, CODE_FILE):
            if (paths.index() / name).exists():
                store = IndexStore(paths.index(), name)
                store.retry_failed()
                store.close()
        if self.engine:
            self.engine.resync()
        self.refresh_summary()

    def _rebuild(self) -> None:
        if QMessageBox.question(self, "Read everything again?",
                                "DigUp will read and embed every file again. That takes as long as the first pass.") \
                != QMessageBox.StandardButton.Yes:
            return
        for name in (MAIN_FILE, CODE_FILE):
            if (paths.index() / name).exists():
                store = IndexStore(paths.index(), name)
                store.reset_all()
                store.close()
        if self.engine:
            self.engine.resync()
        self.refresh_summary()

    def _save(self) -> None:
        s = self.settings
        folders, code_folders = self.folders.folders(), self.code_folders.folders()
        excluded = self.excluded.folders()
        extensions = self.extensions.text().strip()
        folders_changed = (folders != s.folders or excluded != s.excluded_folders or extensions != s.excluded_extensions)
        code_changed = code_folders != s.code_folders or self.code_on.isChecked() != s.code_search
        s.folders, s.code_folders, s.excluded_folders, s.excluded_extensions = folders, code_folders, excluded, extensions
        s.code_search = self.code_on.isChecked()
        s.editor = self.editor.currentData()
        hotkey = self.hotkey.keySequence().toString(QKeySequence.SequenceFormat.PortableText)
        if hotkey:
            s.hotkey = hotkey.replace("Meta", "Win")
        s.launch_at_login = self.login.isChecked()
        s.index_on_battery = self.battery.isChecked()
        s.pause_on_battery_saver = self.saver.isChecked()
        s.media_player = self.player.currentData()
        s.pdf_viewer = self.pdf.currentData()
        backend_changed = self.backend.currentData() != s.backend
        s.backend = self.backend.currentData()
        s.save()
        try:
            set_launch_at_login(s.launch_at_login)
        except OSError:
            pass
        if not s.code_search and (paths.index() / CODE_FILE).exists() and code_changed:
            pass  # the code index stays until it's turned on again; nothing reads it while off
        self.saved.emit(folders_changed or backend_changed, code_changed)
        self.close()
