"""The search panel: a box that opens over any app (Ctrl+Shift+Space), finds as you type, and opens a result where it
matched. Results on the left, a preview on the right: the picture, the PDF page, the video at its moment, or the
passage with your words marked."""

from __future__ import annotations

import html
import os
import time

from PySide6.QtCore import (QAbstractListModel, QEvent, QModelIndex, QRect, QSize, Qt, QTimer, QUrl,
                            Signal)
from PySide6.QtGui import QColor, QCursor, QFont, QFontMetrics, QGuiApplication, QImage, QKeySequence, QPainter, QPainterPath, QPixmap, QShortcut, QTextDocument
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QListView, QMenu, QPushButton, QSizePolicy, QStackedWidget, QStyle, QStyledItemDelegate, QTextBrowser,
                               QVBoxLayout, QWidget)

from .. import rules, searchtext
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
        super().__init__(None)  # an ordinary window: a title bar, resizable, in the taskbar, stays when you click away
        self.setObjectName("Panel")
        self.setWindowTitle("DigUp")
        self.setWindowIcon(theme.app_icon())
        self.setMinimumSize(720, 420)
        self._placed = False
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
        picture = QPushButton("\uEB9F")
        picture.setObjectName("Gear")
        picture.setFont(QFont("Segoe Fluent Icons, Segoe MDL2 Assets", 13))
        picture.setToolTip("Search with a picture: pick one, drop one on this window, or paste one (Ctrl+V)")
        picture.setCursor(Qt.CursorShape.PointingHandCursor)
        picture.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        picture.clicked.connect(self.pick_image)
        top.addWidget(picture)
        gear = QPushButton("")
        gear.setObjectName("Gear")
        gear.setFont(QFont("Segoe Fluent Icons, Segoe MDL2 Assets", 13))
        gear.setToolTip("Settings, folders and statistics (Ctrl+,)")
        gear.setCursor(Qt.CursorShape.PointingHandCursor)
        gear.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        gear.clicked.connect(self._settings)
        top.addWidget(gear)
        outer.addLayout(top)
        # Searching with a picture: which one, and a way back to words.
        self.query_bar = QFrame()
        self.query_bar.setObjectName("QueryBar")
        bar = QHBoxLayout(self.query_bar)
        bar.setContentsMargins(18, 0, 18, 6)
        self.query_thumb = QLabel()
        self.query_thumb.setFixedSize(44, 44)
        self.query_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.query_label = QLabel()
        clear = QPushButton("\uE711")
        clear.setObjectName("Gear")
        clear.setFont(QFont("Segoe Fluent Icons, Segoe MDL2 Assets", 10))
        clear.setToolTip("Back to searching with words (Esc)")
        clear.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        clear.clicked.connect(self.clear_image_query)
        bar.addWidget(self.query_thumb)
        bar.addSpacing(8)
        bar.addWidget(self.query_label, 1)
        bar.addWidget(clear)
        self.query_bar.hide()
        outer.addWidget(self.query_bar)
        self.image_query = None  # ("image", path, name) or ("similar", hit, name)
        self.setAcceptDrops(True)
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
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._context_menu)
        self.empty = QLabel()
        self.empty.setObjectName("Empty")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setWordWrap(True)
        self.more = QPushButton("Show more")
        self.more.setObjectName("More")
        self.more.setCursor(Qt.CursorShape.PointingHandCursor)
        self.more.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.more.clicked.connect(self.show_more)
        self.more.hide()
        self.results_box = QWidget()
        box = QVBoxLayout(self.results_box)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)
        box.addWidget(self.list, 1)
        more_row = QHBoxLayout()
        more_row.setContentsMargins(14, 6, 14, 10)
        more_row.addWidget(self.more)
        more_row.addStretch(1)
        box.addLayout(more_row)
        self.limit = QueryService.PAGE  # results asked for; "Show more" adds a page
        self._loading_more = 0  # how many results were shown before "Show more" (0: not loading more)
        left = QStackedWidget()
        left.addWidget(self.results_box)
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
        self.hints = "Enter open · Ctrl+Enter show in folder · Ctrl+C copy file · Ctrl+Shift+F find similar · Ctrl+Space play · Tab filter · Esc clear"
        self.engine_line = ""
        self._set_status()

        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(140)
        self.debounce.timeout.connect(self._search_now)
        self.search.textChanged.connect(self._typed)
        service.results.connect(self._show_results)
        for keys, action in (("Ctrl+Return", self.reveal_selected), ("Ctrl+Enter", self.reveal_selected),
                             ("Ctrl+,", self._settings), ("Escape", self._escape),
                             ("Ctrl+Shift+F", self.find_similar)):
            QShortcut(QKeySequence(keys), self, action)
        self._show_empty("Search your pictures, PDFs, documents, recordings and videos by what's in them.\n\n"
                         "Try “a dog on the beach”, “the clause about pets in the lease”, or “code: retry with backoff”.")

    # MARK: Showing and hiding

    def toggle(self) -> None:
        """The hotkey: brings the window up, or minimizes it when it's already in front."""
        if self.isVisible() and self.isActiveWindow() and not self.isMinimized():
            self.preview.stop()
            self.showMinimized()
        else:
            self.show_panel()

    def show_panel(self) -> None:
        self.service.warm_up()
        if not self._placed:
            # The first time: centered on the screen with the mouse. After that it stays where you put it.
            screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
            area = screen.availableGeometry()
            width = min(1100, int(area.width() * 0.72))
            height = min(680, int(area.height() * 0.72))
            self.resize(width, height)
            self.move(area.left() + (area.width() - width) // 2, area.top() + int(area.height() * 0.12))
            self._placed = True
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()
        theme.decorate(self, theme.dark_mode())
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

    def closeEvent(self, event) -> None:
        """Closing the window keeps DigUp running in the notification area (Quit is in its menu)."""
        self.preview.stop()
        event.ignore()
        self.hide()

    def _escape(self) -> None:
        if self.image_query:
            self.clear_image_query()
        elif self.search.text():
            self.search.clear()
        self.search.setFocus()

    def _settings(self) -> None:
        self.open_settings.emit()

    # MARK: Searching

    def _filter(self, index: int) -> None:
        self.kinds = FILTERS[index][1]
        self.limit = QueryService.PAGE
        if self.image_query:
            self._run_image_query()
        else:
            self._search_now()

    def _typed(self, _text: str) -> None:
        if self.image_query and self.search.text().strip():
            self._end_image_query()  # typing goes back to words
        self.debounce.start()

    def _search_now(self) -> None:
        if self.image_query:
            return
        text = self.search.text().strip()
        if text != self.last_query:
            self.limit = QueryService.PAGE
        self.last_query = text
        if not text:
            self.model.set([], 0)
            self._show_empty("Search your pictures, PDFs, documents, recordings and videos by what's in them.")
            self.preview.show_hit(None)
            return
        self.service.search(text, self.kinds, self.limit)

    def _show_results(self, results: Results) -> None:
        if results.query.strip() != self.last_query:
            return
        self.words = results.words
        previous = self.current_hit()
        shown_before = self._loading_more
        if shown_before and not results.final:
            return  # a words-only pass would shrink the list while more are loading
        self._loading_more = 0
        self.more.setEnabled(True)
        self.more.setText(f"Show {QueryService.PAGE} more")
        self.more.setVisible(results.final and results.more)
        self.model.set(results.hits, results.strong)
        if not results.hits and self.image_query:
            self._show_empty("Nothing in your folders looks like that.")
            self.preview.show_hit(None)
            return
        if not results.hits:
            self._show_empty("Looking…" if not results.final else "Nothing found.\nTry describing it another way.")
            self.preview.show_hit(None)
            return
        self.left.setCurrentWidget(self.results_box)
        row = 0
        if shown_before and shown_before < len(results.hits):
            row = shown_before  # the first of the new ones
        elif previous is not None:
            for i, h in enumerate(results.hits):
                if h.path == previous.path:
                    row = i
                    break
        self.list.setCurrentIndex(self.model.index(row))
        self.list.scrollTo(self.model.index(row), QAbstractItemView.ScrollHint.PositionAtTop if shown_before
                           else QAbstractItemView.ScrollHint.EnsureVisible)
        n = len(results.hits)
        found = f"{n:,} result" + ("" if n == 1 else "s")
        if not results.final:
            found += " (matching words; meaning on its way)"
        elif results.more:
            found += ", more below"
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

    def reveal_selected(self) -> None:
        hit = self.current_hit()
        if hit:
            opener.reveal(hit.path)

    def copy_selected(self) -> None:
        hit = self.current_hit()
        if hit:
            opener.copy_file(hit.path)
            self._set_status(f"copied {os.path.basename(hit.path)}: paste it anywhere")

    # MARK: Searching with a picture

    def pick_image(self) -> None:
        start = os.path.join(os.path.expanduser("~"), "Pictures")
        exts = " ".join(f"*.{e}" for e in sorted(rules.IMAGE_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(self, "Search with a picture", start, f"Pictures ({exts});;All files (*)")
        if path:
            self.search_with_image(os.path.normpath(path))

    def search_with_image(self, path: str, name: str | None = None) -> None:
        """Results that look like the picture at `path` (indexed or not)."""
        self._start_image_query(("image", path, name or os.path.basename(path)), QImage(path))

    def find_similar(self) -> None:
        """Results like the selected one, from the vector the index holds for it."""
        hit = self.current_hit()
        if hit is not None:
            self._start_image_query(("similar", hit, os.path.basename(hit.path)), self.thumbs.get(hit, 160))

    def _start_image_query(self, query, image: QImage | None) -> None:
        self.image_query = query
        self.limit = QueryService.PAGE
        self.search.blockSignals(True)
        self.search.clear()
        self.search.blockSignals(False)
        self.search.setPlaceholderText("Type to search with words again…")
        kind, target, name = query
        what = ("pictures, screenshots, video moments and pages that look like it" if kind == "image"
                or getattr(target, "kind", None) not in (FileKind.DOC, FileKind.CODE, FileKind.AUDIO)
                else "files about the same thing")
        self.query_label.setText(f"<b>Looking for things like</b> {html.escape(name)}"
                                 f"<br><span style='color:{self.colors.muted}'>{what}</span>")
        self.query_thumb.clear()
        if image is not None and not image.isNull():
            self.query_thumb.setPixmap(QPixmap.fromImage(image).scaled(
                44, 44, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.query_thumb.setFont(QFont("Segoe Fluent Icons, Segoe MDL2 Assets", 16))
            self.query_thumb.setText(KIND_GLYPH.get(getattr(target, "kind", None), "\uEB9F"))
        self.query_bar.show()
        self._run_image_query()

    def _run_image_query(self) -> None:
        kind, target, _ = self.image_query
        key = f"{kind}:{target if kind == 'image' else target.path}"
        self.last_query = key
        self.words = []
        if not self._loading_more:
            self.model.set([], 0)
            self.preview.show_hit(None)
            self._show_empty("Looking at the picture…" if kind == "image" else "Looking…")
        if kind == "image":
            self.service.search_image(target, self.kinds, key, self.limit)
        else:
            self.service.search_similar(target, self.kinds, key, self.limit)

    def show_more(self) -> None:
        """The next page of results, below the ones shown (the query's vector is cached: no model work)."""
        if self._loading_more or not self.model.hits:
            return
        self._loading_more = len(self.model.hits)
        self.limit += QueryService.PAGE
        self.more.setText("Loading…")
        self.more.setEnabled(False)
        if self.image_query:
            self._run_image_query()
        else:
            self.service.search(self.last_query, self.kinds, self.limit)

    def _end_image_query(self) -> None:
        self.image_query = None
        self.query_bar.hide()
        self.search.setPlaceholderText("Describe what you're looking for…")

    def clear_image_query(self) -> None:
        self._end_image_query()
        self.last_query = ""
        self.model.set([], 0)
        self.preview.show_hit(None)
        self._show_empty("Search your pictures, PDFs, documents, recordings and videos by what's in them.")
        self.search.setFocus()

    @staticmethod
    def _image_paths(mime) -> list[str]:
        return [u.toLocalFile() for u in mime.urls()
                if u.isLocalFile() and rules.ext_of(u.toLocalFile()) in rules.IMAGE_EXTENSIONS]

    def _image_from_mime(self, mime) -> bool:
        """A picture dropped or pasted: a file, or image data (a browser, the Snipping Tool)."""
        files = self._image_paths(mime)
        if files:
            self.search_with_image(os.path.normpath(files[0]))
            return True
        if mime.hasImage():
            image = QImage(mime.imageData())
            if image.isNull():
                return False
            import tempfile

            folder = os.path.join(tempfile.gettempdir(), "DigUp-query")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"pasted-{int(time.time() * 1000)}.png")
            image.save(path)
            self.search_with_image(path, "the pasted picture")
            return True
        return False

    def dragEnterEvent(self, event) -> None:
        mime = event.mimeData()
        if self._image_paths(mime) or mime.hasImage():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        if self._image_from_mime(event.mimeData()):
            event.acceptProposedAction()
            self.show_panel()

    def _context_menu(self, pos) -> None:
        index = self.list.indexAt(pos)
        if not index.isValid():
            return
        self.list.setCurrentIndex(index)
        menu = QMenu(self)
        menu.addAction("Open", self.open_selected)
        menu.addAction("Find similar\tCtrl+Shift+F", self.find_similar)
        menu.addAction("Show in folder\tCtrl+Enter", self.reveal_selected)
        menu.addAction("Copy file\tCtrl+C", self.copy_selected)
        menu.exec(self.list.viewport().mapToGlobal(pos))

    def _move(self, step: int) -> None:
        count = self.model.rowCount()
        if not count:
            return
        current = self.list.currentIndex().row()
        if step > 0 and current == count - 1 and self.more.isVisible():
            self.show_more()
            return
        row = max(0, min(count - 1, current + step))
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
            if key == Qt.Key.Key_V and mods & Qt.KeyboardModifier.ControlModifier and obj is self.search:
                mime = QGuiApplication.clipboard().mimeData()
                if mime is not None and (mime.hasImage() or self._image_paths(mime)) and self._image_from_mime(mime):
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
