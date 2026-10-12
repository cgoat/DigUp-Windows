"""The search panel: a box that opens over any app (Ctrl+Shift+Space), finds as you type, and opens a result where it
matched. Results on the left, a preview on the right: the picture, the PDF page, the video at its moment, or the
passage with your words marked."""

from __future__ import annotations

import html
import os
import time

from PySide6.QtCore import (QAbstractListModel, QEvent, QModelIndex, QPoint, QRect, QSize, Qt, QTimer, QUrl,
                            Signal)
from PySide6.QtGui import QColor, QCursor, QFont, QFontMetrics, QGuiApplication, QImage, QKeySequence, QPainter, QPainterPath, QPixmap, QShortcut, QTextDocument
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QFrame, QHBoxLayout, QLabel, QLineEdit, QListView,
                               QPushButton, QSizePolicy, QStackedWidget, QStyle, QStyledItemDelegate, QTextBrowser,
                               QVBoxLayout, QWidget)

from .. import searchtext
from ..model import FileKind
from ..searcher import SearchHit
from . import opener, theme
from .query import QueryService, Results
from .thumbs import Thumbnails

KIND_LABEL = {FileKind.SCREENSHOT: "Screenshot", FileKind.IMAGE: "Picture", FileKind.PDF: "PDF",
              FileKind.DOC: "Document", FileKind.AUDIO: "Audio", FileKind.VIDEO: "Video", FileKind.CODE: "Code"}
KIND_GLYPH = {FileKind.SCREENSHOT: "", FileKind.IMAGE: "", FileKind.PDF: "", FileKind.DOC: "",
              FileKind.AUDIO: "", FileKind.VIDEO: "", FileKind.CODE: ""}  # Segoe Fluent Icons
FILTERS = [("All", None), ("Pictures", {FileKind.IMAGE}), ("Screenshots", {FileKind.SCREENSHOT}),
           ("PDFs", {FileKind.PDF}), ("Documents", {FileKind.DOC}), ("Audio", {FileKind.AUDIO}),
           ("Video", {FileKind.VIDEO})]
ROW_HEIGHT = 64
THUMB = 48


def where(hit: SearchHit) -> str:
    if hit.loc is None:
        return ""
    if hit.kind in (FileKind.AUDIO, FileKind.VIDEO):
        return f"at {opener.format_time(hit.loc)}"
    if hit.kind == FileKind.PDF:
        return f"page {int(hit.loc)}"
    if hit.kind == FileKind.CODE:
        return f"line {int(hit.focus or hit.loc)}"
    return ""


def short_folder(path: str) -> str:
    folder = os.path.dirname(path)
    home = os.path.expanduser("~")
    if folder.lower().startswith(home.lower()):
        folder = "~" + folder[len(home):]
    return folder


def marked_html(text: str, color: str) -> str:
    """« » marks (from keyword search) as bold, colored spans."""
    out = html.escape(text)
    return out.replace("«", f"<b style='color:{color}'>").replace("»", "</b>")


class ResultModel(QAbstractListModel):
    def __init__(self):
        super().__init__()
        self.hits: list[SearchHit] = []
        self.strong = 0

    def set(self, hits: list[SearchHit], strong: int) -> None:
        self.beginResetModel()
        self.hits, self.strong = hits, strong
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self.hits)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        hit = self.hits[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return os.path.basename(hit.path)
        if role == Qt.ItemDataRole.UserRole:
            return hit
        if role == Qt.ItemDataRole.ToolTipRole:
            return hit.path
        return None


class ResultDelegate(QStyledItemDelegate):
    def __init__(self, panel: "SearchPanel"):
        super().__init__(panel)
        self.panel = panel

    def sizeHint(self, option, index) -> QSize:
        extra = 22 if index.row() == self.panel.model.strong and 0 < index.row() else 0
        return QSize(option.rect.width(), ROW_HEIGHT + extra)

    def paint(self, painter: QPainter, option, index) -> None:
        p = self.panel.colors
        hit: SearchHit = index.data(Qt.ItemDataRole.UserRole)
        rect = QRect(option.rect)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if index.row() == self.panel.model.strong and index.row() > 0:
            painter.setPen(QColor(p.muted))
            font = QFont(option.font)
            font.setPointSizeF(8.5)
            painter.setFont(font)
            painter.drawText(rect.adjusted(14, 4, -8, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                             "Weaker matches")
            rect.setTop(rect.top() + 22)
        if option.state & QStyle.StateFlag.State_Selected:
            path = QPainterPath()
            path.addRoundedRect(rect.adjusted(6, 2, -6, -2), 8, 8)
            painter.fillPath(path, QColor(p.selection))
        # The picture, or the kind's glyph.
        thumb_rect = QRect(rect.left() + 14, rect.top() + (rect.height() - THUMB) // 2, THUMB, THUMB)
        image = self.panel.thumbs.get(hit, 160)
        clip = QPainterPath()
        clip.addRoundedRect(thumb_rect, 6, 6)
        if image is not None and not image.isNull():
            painter.setClipPath(clip)
            pix = QPixmap.fromImage(image).scaled(THUMB, THUMB, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                                  Qt.TransformationMode.SmoothTransformation)
            painter.drawPixmap(thumb_rect.left() - (pix.width() - THUMB) // 2,
                               thumb_rect.top() - (pix.height() - THUMB) // 2, pix)
            painter.setClipping(False)
        else:
            painter.fillPath(clip, QColor(p.raised))
            painter.setPen(QColor(p.muted))
            painter.setFont(QFont("Segoe Fluent Icons, Segoe MDL2 Assets", 16))
            painter.drawText(thumb_rect, Qt.AlignmentFlag.AlignCenter, KIND_GLYPH.get(hit.kind, ""))
        # Name, then kind, place and folder, then the passage.
        left = thumb_rect.right() + 12
        width = rect.right() - left - 12
        title_font = QFont(option.font)
        title_font.setPointSizeF(10.5)
        title_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(title_font)
        painter.setPen(QColor(p.text))
        fm = QFontMetrics(title_font)
        painter.drawText(QRect(left, rect.top() + 8, width, fm.height()), Qt.AlignmentFlag.AlignLeft,
                         fm.elidedText(os.path.basename(hit.path), Qt.TextElideMode.ElideMiddle, width))
        small = QFont(option.font)
        small.setPointSizeF(8.8)
        painter.setFont(small)
        sm = QFontMetrics(small)
        meta = " · ".join(x for x in (KIND_LABEL.get(hit.kind, ""), where(hit), short_folder(hit.path)) if x)
        painter.setPen(QColor(p.muted))
        painter.drawText(QRect(left, rect.top() + 8 + fm.height() + 1, width, sm.height()), Qt.AlignmentFlag.AlignLeft,
                         sm.elidedText(meta, Qt.TextElideMode.ElideMiddle, width))
        if hit.excerpt:
            doc = QTextDocument()
            doc.setDefaultFont(small)
            excerpt = searchtext.excerpt(hit.excerpt, 220)
            doc.setHtml(f"<span style='color:{p.muted}'>{marked_html(excerpt, p.mark)}</span>")
            doc.setTextWidth(10_000)
            painter.translate(left - 4, rect.top() + 8 + fm.height() + sm.height() - 1)
            painter.setClipRect(QRect(0, 0, width + 4, sm.height() + 6))
            doc.drawContents(painter)
        painter.restore()


class Chip(QPushButton):
    def __init__(self, text: str):
        super().__init__(text)
        self.setObjectName("Chip")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)


class Preview(QFrame):
    """The selected result: its picture, the video or recording at its moment, or its text."""

    def __init__(self, panel: "SearchPanel"):
        super().__init__()
        self.panel = panel
        self.setObjectName("Preview")
        self.setMinimumWidth(320)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(8)
        self.stack = QStackedWidget()
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.text = QTextBrowser()
        self.text.setObjectName("PreviewText")
        self.text.setOpenLinks(False)
        self.stack.addWidget(self.image)
        self.stack.addWidget(self.text)
        self.player = None
        self.video = None
        layout.addWidget(self.stack, 1)
        self.title = QLabel()
        self.title.setObjectName("PreviewTitle")
        self.title.setWordWrap(True)
        self.meta = QLabel()
        self.meta.setObjectName("PreviewMeta")
        self.meta.setWordWrap(True)
        layout.addWidget(self.title)
        layout.addWidget(self.meta)
        self.hit: SearchHit | None = None
        panel.thumbs.ready.connect(self._thumb_ready)

    def _ensure_player(self) -> bool:
        if self.player is not None:
            return True
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
            from PySide6.QtMultimediaWidgets import QVideoWidget
        except ImportError:
            return False
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.audio.setVolume(0.8)
        self.player.setAudioOutput(self.audio)
        self.video = QVideoWidget()
        self.player.setVideoOutput(self.video)
        self.stack.addWidget(self.video)
        self._seek_to = None
        self.player.mediaStatusChanged.connect(self._media_status)
        return True

    def _media_status(self, status) -> None:
        from PySide6.QtMultimedia import QMediaPlayer

        if status in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia) and self._seek_to is not None:
            self.player.setPosition(int(self._seek_to * 1000))
            self._seek_to = None

    def stop(self) -> None:
        if self.player is not None:
            self.player.stop()
            self.player.setSource(QUrl())

    def show_hit(self, hit: SearchHit | None) -> None:
        self.stop()
        self.hit = hit
        if hit is None:
            self.title.setText("")
            self.meta.setText("")
            self.image.clear()
            self.stack.setCurrentWidget(self.image)
            return
        self.title.setText(os.path.basename(hit.path))
        info = hit.info or {}
        parts = [KIND_LABEL.get(hit.kind, ""), where(hit)]
        if info.get("width"):
            parts.append(f"{info['width']}×{info['height']}")
        if info.get("pages"):
            parts.append(f"{info['pages']} pages")
        if info.get("duration"):
            parts.append(opener.format_time(float(info["duration"])))
        parts.append(time.strftime("%d %b %Y", time.localtime(hit.modified)))
        self.meta.setText(" · ".join(p for p in parts if p) + "\n" + short_folder(hit.path))
        # A video shows its frame at the moment until it's played (Ctrl+Space).
        if hit.kind in (FileKind.IMAGE, FileKind.SCREENSHOT, FileKind.PDF, FileKind.VIDEO):
            image = self.panel.thumbs.get(hit, 900)
            if image is not None:
                self._show_image(image)
                return
            self.image.setText("…")
            self.stack.setCurrentWidget(self.image)
            return
        self._show_text(hit)

    def toggle_play(self) -> None:
        """Plays the video or recording from its moment, or pauses it."""
        hit = self.hit
        if hit is None or hit.kind not in (FileKind.VIDEO, FileKind.AUDIO) or not self._ensure_player():
            return
        from PySide6.QtMultimedia import QMediaPlayer

        if self.player.source().isEmpty():
            self._seek_to = hit.loc or 0
            self.player.setSource(QUrl.fromLocalFile(hit.path))
            if hit.kind == FileKind.VIDEO:
                self.stack.setCurrentWidget(self.video)
            self.player.play()
        elif self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _show_image(self, image: QImage) -> None:
        pix = QPixmap.fromImage(image)
        size = self.stack.size()
        self.image.setPixmap(pix.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation))
        self.stack.setCurrentWidget(self.image)

    def _show_text(self, hit: SearchHit) -> None:
        p = self.panel.colors
        body = None
        if hit.kind in (FileKind.DOC, FileKind.CODE, FileKind.PDF):
            body = self.panel.service_text(hit)
        if hit.kind in (FileKind.AUDIO,):
            body = None
        words = self.panel.words
        if body:
            text = body[:6000]
            spans = sorted(m for w in words for m in searchtext.matches(w, text))
            out, cursor = [], 0
            for a, b in spans:
                if a < cursor:
                    continue
                out.append(html.escape(text[cursor:a]) + f"<b style='color:{p.mark}'>" + html.escape(text[a:b]) + "</b>")
                cursor = b
            out.append(html.escape(text[cursor:]))
            mono = "font-family: Cascadia Mono, Consolas, monospace; font-size: 9pt;" if hit.kind == FileKind.CODE else ""
            self.text.setHtml(f"<div style='white-space: pre-wrap; {mono}'>{''.join(out)}</div>")
        elif hit.excerpt:
            self.text.setHtml(marked_html(hit.excerpt, p.mark))
        elif hit.kind == FileKind.AUDIO:
            self.text.setHtml(f"<p style='color:{p.muted}'>Press Ctrl+Space to play from "
                              f"{opener.format_time(hit.loc or 0)}.</p>")
        else:
            self.text.setHtml("")
        self.stack.setCurrentWidget(self.text)

    def _thumb_ready(self, key, image) -> None:
        if self.hit is not None and image is not None and key == Thumbnails.key(self.hit, 900):
            self._show_image(image)


class SearchPanel(QWidget):
    open_settings = Signal()

    def __init__(self, service: QueryService, settings, engine_status=None):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.setObjectName("Panel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.service = service
        self.settings = settings
        self.colors = theme.palette()
        self.thumbs = Thumbnails()
        self.thumbs.ready.connect(lambda *_: self.list.viewport().update())
        self.words: list[str] = []
        self.kinds = None
        self.last_query = ""
        self.resize(980, 560)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        top = QHBoxLayout()
        top.setContentsMargins(18, 10, 18, 4)
        icon = QLabel("")
        icon.setFont(QFont("Segoe Fluent Icons, Segoe MDL2 Assets", 15))
        icon.setStyleSheet(f"color: {self.colors.muted}")
        self.search = QLineEdit()
        self.search.setObjectName("Search")
        self.search.setPlaceholderText("Describe what you're looking for…")
        self.search.installEventFilter(self)
        top.addWidget(icon)
        top.addSpacing(6)
        top.addWidget(self.search, 1)
        outer.addLayout(top)
        chips = QHBoxLayout()
        chips.setContentsMargins(18, 2, 18, 10)
        chips.setSpacing(6)
        self.chip_group = QButtonGroup(self)
        self.chip_group.setExclusive(True)
        for i, (label, kinds) in enumerate(FILTERS):
            chip = Chip(label)
            chip.setChecked(i == 0)
            self.chip_group.addButton(chip, i)
            chips.addWidget(chip)
        self.chip_group.idClicked.connect(self._filter)
        chips.addStretch(1)
        outer.addLayout(chips)
        divider = QFrame()
        divider.setObjectName("Divider")
        outer.addWidget(divider)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        self.model = ResultModel()
        self.list = QListView()
        self.list.setModel(self.model)
        self.list.setItemDelegate(ResultDelegate(self))
        self.list.setUniformItemSizes(False)
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setMouseTracking(True)
        self.list.doubleClicked.connect(lambda _: self.open_selected())
        self.list.selectionModel().currentChanged.connect(self._current_changed)
        self.list.installEventFilter(self)
        self.empty = QLabel()
        self.empty.setObjectName("Empty")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setWordWrap(True)
        left = QStackedWidget()
        left.addWidget(self.list)
        left.addWidget(self.empty)
        self.left = left
        self.preview = Preview(self)
        body.addWidget(left, 11)
        body.addWidget(self.preview, 9)
        outer.addLayout(body, 1)
        divider2 = QFrame()
        divider2.setObjectName("Divider")
        outer.addWidget(divider2)
        self.status = QLabel()
        self.status.setObjectName("Status")
        self.status.setContentsMargins(18, 6, 18, 8)
        outer.addWidget(self.status)
        self.hints = "Enter open · Ctrl+Enter show in folder · Ctrl+C copy file · Ctrl+Space play · Tab filter · Esc close"
        self.engine_line = ""
        self._set_status()

        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(140)
        self.debounce.timeout.connect(self._search_now)
        self.search.textChanged.connect(lambda _: self.debounce.start())
        service.results.connect(self._show_results)
        for keys, action in (("Ctrl+Return", self.reveal_selected), ("Ctrl+Enter", self.reveal_selected),
                             ("Ctrl+,", self._settings), ("Escape", self.hide_panel)):
            QShortcut(QKeySequence(keys), self, action)
        self._show_empty("Search your pictures, PDFs, documents, recordings and videos by what's in them.\n\n"
                         "Try “a dog on the beach”, “the clause about pets in the lease”, or “code: retry with backoff”.")

    # MARK: Showing and hiding

    def toggle(self) -> None:
        if self.isVisible() and self.isActiveWindow():
            self.hide_panel()
        else:
            self.show_panel()

    def show_panel(self) -> None:
        self.service.warm_up()
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry()
        width = min(1040, int(area.width() * 0.72))
        height = min(600, int(area.height() * 0.66))
        self.resize(width, height)
        self.move(area.left() + (area.width() - width) // 2, area.top() + int(area.height() * 0.16))
        self.show()
        theme.decorate(self, theme.dark_mode(), frameless=True)
        self.raise_()
        self.activateWindow()
        self._force_foreground()
        self.search.setFocus()
        self.search.selectAll()

    def _force_foreground(self) -> None:
        """Windows only lets the foreground app take focus; the hotkey makes us it, but be sure."""
        try:
            import ctypes

            ctypes.windll.user32.SetForegroundWindow(int(self.winId()))
        except Exception:
            pass

    def hide_panel(self) -> None:
        self.preview.stop()
        self.hide()

    def changeEvent(self, event) -> None:
        if event.type() == QEvent.Type.ActivationChange and not self.isActiveWindow() and self.isVisible():
            QTimer.singleShot(120, lambda: (not self.isActiveWindow()) and self.hide_panel())
        super().changeEvent(event)

    def _settings(self) -> None:
        self.hide_panel()
        self.open_settings.emit()

    # MARK: Searching

    def _filter(self, index: int) -> None:
        self.kinds = FILTERS[index][1]
        self._search_now()

    def _search_now(self) -> None:
        text = self.search.text().strip()
        self.last_query = text
        if not text:
            self.model.set([], 0)
            self._show_empty("Search your pictures, PDFs, documents, recordings and videos by what's in them.")
            self.preview.show_hit(None)
            return
        self.service.search(text, self.kinds)

    def _show_results(self, results: Results) -> None:
        if results.query.strip() != self.last_query:
            return
        self.words = results.words
        previous = self.current_hit()
        self.model.set(results.hits, results.strong)
        if not results.hits:
            self._show_empty("Looking…" if not results.final else "Nothing found.\nTry describing it another way.")
            self.preview.show_hit(None)
            return
        self.left.setCurrentWidget(self.list)
        row = 0
        if previous is not None:
            for i, h in enumerate(results.hits):
                if h.path == previous.path:
                    row = i
                    break
        self.list.setCurrentIndex(self.model.index(row))
        found = f"{len(results.hits)} result" + ("" if len(results.hits) == 1 else "s") + ("" if results.final else " (matching words; meaning on its way)")
        self._set_status(found)

    def _show_empty(self, text: str) -> None:
        self.empty.setText(text)
        self.left.setCurrentWidget(self.empty)

    def set_engine_line(self, text: str) -> None:
        self.engine_line = text
        self._set_status()

    def _set_status(self, left: str = "") -> None:
        bits = [b for b in (left, self.engine_line, self.hints) if b]
        self.status.setText("   ·   ".join(bits))

    def current_hit(self) -> SearchHit | None:
        index = self.list.currentIndex()
        return index.data(Qt.ItemDataRole.UserRole) if index.isValid() else None

    def _current_changed(self, current, _previous) -> None:
        self.preview.show_hit(current.data(Qt.ItemDataRole.UserRole) if current.isValid() else None)

    def service_text(self, hit: SearchHit) -> str | None:
        searcher = self.service._searchers.get("code.sqlite" if hit.kind == FileKind.CODE else "index.sqlite")
        if searcher is None:
            return None
        try:
            return searcher.text_of_segment(hit.segment_id)
        except Exception:
            return None

    # MARK: Acting on a result

    def open_selected(self) -> None:
        hit = self.current_hit()
        if hit is None:
            return
        self.preview.stop()
        message = opener.open_hit(hit, self.settings, self.words)
        self._set_status(message)
        self.hide_panel()

    def reveal_selected(self) -> None:
        hit = self.current_hit()
        if hit:
            opener.reveal(hit.path)
            self.hide_panel()

    def copy_selected(self) -> None:
        hit = self.current_hit()
        if hit:
            opener.copy_file(hit.path)
            self._set_status(f"copied {os.path.basename(hit.path)}: paste it anywhere")

    def _move(self, step: int) -> None:
        count = self.model.rowCount()
        if not count:
            return
        row = max(0, min(count - 1, self.list.currentIndex().row() + step))
        self.list.setCurrentIndex(self.model.index(row))

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.KeyPress:
            key, mods = event.key(), event.modifiers()
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                self._move(1 if key == Qt.Key.Key_Down else -1)
                return True
            if key in (Qt.Key.Key_PageDown, Qt.Key.Key_PageUp):
                self._move(6 if key == Qt.Key.Key_PageDown else -6)
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if mods & Qt.KeyboardModifier.ControlModifier:
                    self.reveal_selected()
                else:
                    self.open_selected()
                return True
            if key == Qt.Key.Key_C and mods & Qt.KeyboardModifier.ControlModifier and not self.search.hasSelectedText():
                self.copy_selected()
                return True
            if key == Qt.Key.Key_Space and (obj is self.list or mods & Qt.KeyboardModifier.ControlModifier):
                self.preview.toggle_play()
                return True
            if key == Qt.Key.Key_Tab:
                step = -1 if mods & Qt.KeyboardModifier.ShiftModifier else 1
                index = (self.chip_group.checkedId() + step) % len(FILTERS)
                self.chip_group.button(index).setChecked(True)
                self._filter(index)
                return True
        return super().eventFilter(obj, event)
