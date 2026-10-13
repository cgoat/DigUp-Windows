"""Settings → People: turning face search on (InsightFace's models, downloaded only after reading their licence),
and the people DigUp found, to name, merge (one name for two groups) or hide."""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QHBoxLayout, QInputDialog, QLabel, QListView,
                               QListWidget, QListWidgetItem, QMenu, QMessageBox, QProgressBar, QPushButton,
                               QStackedWidget, QTextEdit, QVBoxLayout, QWidget)

from .. import faces, paths, people
from ..settings import Settings
from ..store import MAIN_FILE, IndexStore

TILE = 96


class _Bridge(QObject):
    progress = Signal(float)
    downloaded = Signal(str)  # "" or an error
    loaded = Signal(object)  # (people, counts)
    face = Signal(int, object)  # person id, QImage


class PeoplePage(QWidget):
    show_person = Signal(int, str, object)  # group id, name, face picture: open the search window on their photos

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.engine = None
        self.bridge = _Bridge()
        self.bridge.progress.connect(lambda f: self.bar.setValue(int(f * 1000)))
        self.bridge.downloaded.connect(self._downloaded)
        self.bridge.loaded.connect(self._show_people)
        self.bridge.face.connect(self._set_face)
        self._items: dict[int, QListWidgetItem] = {}
        self._loading = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 0)
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        # Not set up yet: what it does, the licence, and the download.
        intro = QWidget()
        v = QVBoxLayout(intro)
        title = QLabel("Find people in your photos")
        title.setStyleSheet("font-size: 14pt; font-weight: 600;")
        v.addWidget(title)
        about = QLabel(
            "Searching for a person by name, or for everyone who looks like someone, needs a model that recognises "
            "faces: EmbeddingGemma knows what a photo shows, not who is in it. DigUp can use InsightFace's buffalo_l "
            "models (SCRFD finds faces, ArcFace tells people apart), about 290 MB, run on this PC with your GPU. "
            "DigUp then groups faces into people; you name them once and search with the name "
            "(“Srinivas”, “Srinivas at the beach”), or right-click a photo and choose Find this person.")
        about.setWordWrap(True)
        v.addWidget(about)
        licence = QTextEdit()
        licence.setReadOnly(True)
        licence.setPlainText("Licence\n\n" + faces.LICENSE)
        licence.setMaximumHeight(110)
        v.addWidget(licence)
        self.agree = QCheckBox("I've read this: I'll use these models for non-commercial purposes (my own photos)")
        v.addWidget(self.agree)
        row = QHBoxLayout()
        self.download_button = QPushButton("Download face models (290 MB)")
        self.download_button.setObjectName("Primary")
        self.download_button.setEnabled(False)
        self.agree.toggled.connect(self.download_button.setEnabled)
        self.download_button.clicked.connect(self._download)
        row.addWidget(self.download_button)
        row.addStretch(1)
        v.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.hide()
        v.addWidget(self.bar)
        self.download_note = QLabel("")
        self.download_note.setObjectName("Muted")
        v.addWidget(self.download_note)
        v.addStretch(1)
        self.stack.addWidget(intro)

        # Set up: the switch, the people, and what to do with them.
        main = QWidget()
        v = QVBoxLayout(main)
        top = QHBoxLayout()
        self.enabled = QCheckBox("Find people in my photos")
        self.enabled.setChecked(settings.faces)
        self.enabled.toggled.connect(self._toggle)
        self.summary = QLabel("")
        self.summary.setObjectName("Muted")
        top.addWidget(self.enabled)
        top.addSpacing(16)
        top.addWidget(self.summary, 1)
        v.addLayout(top)
        hint = QLabel("Double-click someone to name them. Giving two groups the same name merges them (one person "
                      "can start out as two groups). Right-click for more.")
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.grid = QListWidget()
        self.grid.setViewMode(QListView.ViewMode.IconMode)
        self.grid.setIconSize(QSize(TILE, TILE))
        self.grid.setGridSize(QSize(TILE + 40, TILE + 52))
        self.grid.setResizeMode(QListView.ResizeMode.Adjust)
        self.grid.setMovement(QListView.Movement.Static)
        self.grid.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.grid.setWordWrap(True)
        self.grid.itemDoubleClicked.connect(self._rename)
        self.grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.grid.customContextMenuRequested.connect(self._menu)
        v.addWidget(self.grid, 1)
        row = QHBoxLayout()
        self.show_hidden = QCheckBox("Show hidden")
        self.show_hidden.toggled.connect(lambda _: self.refresh())
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        remove = QPushButton("Remove face models…")
        remove.clicked.connect(self._remove)
        row.addWidget(self.show_hidden)
        row.addStretch(1)
        row.addWidget(refresh)
        row.addWidget(remove)
        v.addLayout(row)
        self.stack.addWidget(main)
        self._timer = QTimer(self)
        self._timer.setInterval(20_000)
        self._timer.timeout.connect(lambda: self.isVisible() and self.refresh())
        self._update_stack()

    def set_engine(self, engine) -> None:
        self.engine = engine
        engine.people_changed.connect(lambda: self.isVisible() and self.refresh())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._update_stack()
        self.refresh()
        self._timer.start()

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    def _update_stack(self) -> None:
        self.stack.setCurrentIndex(1 if faces.models_ready() else 0)

    # MARK: Download

    def _download(self) -> None:
        self.download_button.setEnabled(False)
        self.agree.setEnabled(False)
        self.bar.show()
        self.download_note.setText("Downloading InsightFace buffalo_l…")

        def run():
            try:
                faces.download(lambda n, t: self.bridge.progress.emit(n / t if t else 0))
                self.bridge.downloaded.emit("")
            except Exception as error:  # noqa: BLE001
                self.bridge.downloaded.emit(str(error))

        threading.Thread(target=run, daemon=True).start()

    def _downloaded(self, error: str) -> None:
        self.bar.hide()
        if error:
            self.download_note.setText(f"The download stopped: {error}")
            self.download_button.setEnabled(True)
            self.agree.setEnabled(True)
            return
        self.settings.faces = True
        self.settings.save()
        self.enabled.setChecked(True)
        self._update_stack()
        self.summary.setText("Looking for faces in your photos: they'll appear here as groups are found.")
        if self.engine:
            self.engine.resync(False)

    def _toggle(self, on: bool) -> None:
        self.settings.faces = on
        self.settings.save()
        if on and self.engine:
            self.engine.resync(False)

    def _remove(self) -> None:
        if QMessageBox.question(self, "Remove the face models?",
                                "The face models are deleted and face search turns off. The faces found and the "
                                "names you gave are kept, should you turn it on again.") \
                != QMessageBox.StandardButton.Yes:
            return
        faces.remove()
        self.settings.faces = False
        self.settings.save()
        self.agree.setEnabled(True)
        self.agree.setChecked(False)
        self._update_stack()

    # MARK: The people

    def refresh(self) -> None:
        if self._loading or not faces.models_ready():
            return
        self._loading = True
        include_hidden = self.show_hidden.isChecked()

        def load():
            try:
                store = IndexStore(paths.index(), MAIN_FILE)
                found = people.people(store, include_hidden)
                counts = people.counts(store)
                store.close()
            except Exception:  # noqa: BLE001 - shown as nothing yet
                found, counts = [], {}
            self.bridge.loaded.emit((found, counts))

        threading.Thread(target=load, daemon=True).start()

    def _show_people(self, data) -> None:
        self._loading = False
        found, counts = data
        if counts:
            text = (f"{counts['faces']:,} faces · {len(found):,} people found · {counts['named']:,} named")
            if counts["to_scan"]:
                text += f" · {counts['to_scan']:,} photos still to look at"
            self.summary.setText(text)
        selected = self.grid.currentItem().data(Qt.ItemDataRole.UserRole) if self.grid.currentItem() else None
        old = {pid: item.icon() for pid, item in self._items.items()}
        self.grid.clear()
        self._items = {}
        wanted = []
        for person in found:
            label = (person.name or "Unnamed") + f"\n{person.photos:,} photo" + ("" if person.photos == 1 else "s")
            if person.hidden:
                label += " (hidden)"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, person.id)
            item.setData(Qt.ItemDataRole.UserRole + 1, person.name or "")
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
            if person.id in old and not old[person.id].isNull():
                item.setIcon(old[person.id])
            else:
                wanted.append(person)
            self.grid.addItem(item)
            self._items[person.id] = item
            if person.id == selected:
                self.grid.setCurrentItem(item)
        if wanted:
            threading.Thread(target=self._load_faces, args=(wanted[:300],), daemon=True).start()

    def _load_faces(self, wanted) -> None:
        for person in wanted:
            if not person.cover_file:
                continue
            crop = faces.crop(person.cover_file, person.cover_box, TILE)
            if crop is None:
                continue
            image = QImage(crop.data, crop.shape[1], crop.shape[0], crop.shape[1] * 3, QImage.Format.Format_RGB888)
            try:
                self.bridge.face.emit(person.id, image.copy())
            except RuntimeError:
                return  # the window closed

    def _set_face(self, pid: int, image: QImage) -> None:
        item = self._items.get(pid)
        if item is not None:
            item.setIcon(QIcon(QPixmap.fromImage(image)))
            item.setData(Qt.ItemDataRole.UserRole + 2, image)

    def _rename(self, item: QListWidgetItem) -> None:
        pid = item.data(Qt.ItemDataRole.UserRole)
        current = item.data(Qt.ItemDataRole.UserRole + 1)
        name, ok = QInputDialog.getText(self, "Name this person", "Name (a name another group has merges the two):",
                                        text=current)
        if not ok:
            return
        store = IndexStore(paths.index(), MAIN_FILE)
        people.rename(store, pid, name)
        store.close()
        self.refresh()

    def _menu(self, pos) -> None:
        item = self.grid.itemAt(pos)
        if item is None:
            return
        pid = item.data(Qt.ItemDataRole.UserRole)
        name = item.data(Qt.ItemDataRole.UserRole + 1)
        hidden = "(hidden)" in item.text()
        menu = QMenu(self)
        menu.addAction("Show their photos", lambda: self.show_person.emit(
            pid, name or "this person", item.data(Qt.ItemDataRole.UserRole + 2)))
        menu.addAction("Name…" if not name else "Rename…", lambda: self._rename(item))
        menu.addAction("Unhide" if hidden else "Hide (not someone I'll look for)", lambda: self._hide(pid, not hidden))
        menu.exec(self.grid.viewport().mapToGlobal(pos))

    def _hide(self, pid: int, hidden: bool) -> None:
        store = IndexStore(paths.index(), MAIN_FILE)
        people.hide(store, pid, hidden)
        store.close()
        self.refresh()
