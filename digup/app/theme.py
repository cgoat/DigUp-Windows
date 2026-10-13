"""Looks: follows Windows' light or dark mode, with Windows 11's rounded corners, shadow and Mica where available."""

from __future__ import annotations

import ctypes
import winreg
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

RESOURCES = Path(__file__).parent / "resources"


def dark_mode() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as key:
            return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 0
    except OSError:
        return False


def taskbar_dark() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as key:
            return winreg.QueryValueEx(key, "SystemUsesLightTheme")[0] == 0
    except OSError:
        return True


@dataclass
class Palette:
    window: str
    surface: str
    raised: str
    border: str
    text: str
    muted: str
    accent: str
    accent_text: str
    selection: str
    mark: str


DARK = Palette(window="#1f2421", surface="#262c28", raised="#2f3631", border="#3a423c", text="#eef2ee",
               muted="#9aa79e", accent="#f2a22a", accent_text="#1b1405", selection="#34433a", mark="#ffd166")
LIGHT = Palette(window="#fbfbf9", surface="#ffffff", raised="#f1f3ef", border="#dde2dc", text="#1b211d",
                muted="#5f6b63", accent="#c77a0a", accent_text="#ffffff", selection="#e4efe6", mark="#b26a00")


def palette() -> Palette:
    return DARK if dark_mode() else LIGHT


def stylesheet(p: Palette) -> str:
    return f"""
    QWidget {{ color: {p.text}; font-family: 'Segoe UI Variable Text', 'Segoe UI'; font-size: 10pt; }}
    #Panel, #Window {{ background: {p.window}; }}
    #Search {{ background: transparent; border: none; font-size: 17pt; padding: 6px 2px;
               selection-background-color: {p.accent}; selection-color: {p.accent_text}; }}
    #Gear {{ background: transparent; border: none; color: {p.muted}; padding: 4px 8px; border-radius: 6px; }}
    #Gear:hover {{ background: {p.raised}; color: {p.text}; }}
    #More {{ background: transparent; border: 1px solid {p.border}; border-radius: 6px; padding: 5px 14px;
              color: {p.text}; }}
    #More:hover {{ border-color: {p.accent}; color: {p.accent}; }}
    #QueryBar {{ background: transparent; }}
    #Card {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: 8px; }}
    #CardValue {{ font-size: 16pt; font-weight: 600; }}
    #CardLabel {{ color: {p.muted}; font-size: 8.5pt; }}
    #Muted {{ color: {p.muted}; }}
    QHeaderView::section {{ background: {p.raised}; color: {p.muted}; border: none; border-bottom: 1px solid {p.border};
                            padding: 5px 8px; }}
    #Divider {{ background: {p.border}; max-height: 1px; min-height: 1px; }}
    #Chip {{ background: {p.raised}; border: 1px solid {p.border}; border-radius: 12px; padding: 3px 11px;
             color: {p.muted}; }}
    #Chip:checked {{ background: {p.accent}; color: {p.accent_text}; border-color: {p.accent}; }}
    #Chip:hover:!checked {{ color: {p.text}; }}
    QListView {{ background: transparent; border: none; outline: none; }}
    #Status {{ color: {p.muted}; font-size: 8.5pt; }}
    #Preview {{ background: {p.surface}; border-left: 1px solid {p.border}; }}
    #PreviewTitle {{ font-size: 11pt; font-weight: 600; }}
    #PreviewMeta {{ color: {p.muted}; font-size: 9pt; }}
    #PreviewText {{ background: transparent; border: none; font-size: 10pt; }}
    #Empty {{ color: {p.muted}; font-size: 11pt; }}
    QPushButton {{ background: {p.raised}; border: 1px solid {p.border}; border-radius: 6px; padding: 6px 14px; }}
    QPushButton:hover {{ border-color: {p.accent}; }}
    QPushButton#Primary {{ background: {p.accent}; color: {p.accent_text}; border-color: {p.accent}; font-weight: 600; }}
    QPushButton:disabled {{ color: {p.muted}; }}
    QLineEdit, QComboBox, QSpinBox {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: 6px;
                                      padding: 5px 8px; }}
    QLineEdit:focus, QComboBox:focus {{ border-color: {p.accent}; }}
    QComboBox QAbstractItemView {{ background: {p.surface}; selection-background-color: {p.selection}; }}
    QListWidget, QTreeWidget, QTextEdit {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: 6px; }}
    QListWidget::item, QTreeWidget::item {{ padding: 4px; }}
    QListWidget::item:selected, QTreeWidget::item:selected {{ background: {p.selection}; color: {p.text}; }}
    QTabWidget::pane {{ border: none; }}
    QTabBar::tab {{ background: transparent; padding: 8px 16px; color: {p.muted}; border-bottom: 2px solid transparent; }}
    QTabBar::tab:selected {{ color: {p.text}; border-bottom-color: {p.accent}; }}
    QProgressBar {{ background: {p.raised}; border: none; border-radius: 4px; height: 8px; text-align: center;
                    color: transparent; }}
    QProgressBar::chunk {{ background: {p.accent}; border-radius: 4px; }}
    QCheckBox::indicator {{ width: 16px; height: 16px; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {p.border}; border-radius: 4px; min-height: 30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QToolTip {{ background: {p.raised}; color: {p.text}; border: 1px solid {p.border}; }}
    QMenu {{ background: {p.surface}; border: 1px solid {p.border}; padding: 4px; }}
    QMenu::item {{ padding: 6px 24px 6px 12px; border-radius: 4px; }}
    QMenu::item:selected {{ background: {p.selection}; }}
    QMenu::item:disabled {{ color: {p.muted}; }}
    QMenu::separator {{ height: 1px; background: {p.border}; margin: 4px 6px; }}
    """


def app_icon() -> QIcon:
    return QIcon(str(RESOURCES / "digup.ico"))


def tray_icon(progress: float | None = None, paused: bool = False) -> QIcon:
    """The boxed D, drawn in the taskbar's color; while indexing, an amber arc shows how far it got."""
    color = "#ffffff" if taskbar_dark() else "#000000"
    svg = (RESOURCES / "menubar.svg").read_text().replace("#000", color)
    icon = QIcon()
    for size in (16, 20, 24, 32, 40, 48, 64):
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if paused:
            painter.setOpacity(0.5)
        QSvgRenderer(svg.encode()).render(painter)
        if progress is not None:
            from PySide6.QtCore import QRectF, Qt
            from PySide6.QtGui import QPen

            pen = QPen(QColor("#f2a22a"))
            pen.setWidthF(max(1.5, size / 9))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            inset = pen.widthF() / 2 + 0.5
            rect = QRectF(inset, inset, size - 2 * inset, size - 2 * inset)
            painter.drawArc(rect, 90 * 16, -int(360 * 16 * max(0.04, min(progress, 1.0))))
        painter.end()
        icon.addPixmap(pixmap)
    return icon


# MARK: Windows 11 window decoration

DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWA_SYSTEMBACKDROP_TYPE = 38
DWMWCP_ROUND = 2


def _dwm(hwnd: int, attribute: int, value: int) -> None:
    try:
        v = ctypes.c_int(value)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(ctypes.c_void_p(hwnd), attribute, ctypes.byref(v), ctypes.sizeof(v))
    except Exception:
        pass


class MARGINS(ctypes.Structure):
    _fields_ = [("left", ctypes.c_int), ("right", ctypes.c_int), ("top", ctypes.c_int), ("bottom", ctypes.c_int)]


def decorate(widget, dark: bool, frameless: bool = False) -> None:
    """Rounded corners and a shadow for a frameless window; dark title bars for framed ones (Windows 11; ignored
    on Windows 10)."""
    hwnd = int(widget.winId())
    _dwm(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, 1 if dark else 0)
    _dwm(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND)
    if frameless:
        try:
            margins = MARGINS(1, 1, 1, 1)
            ctypes.windll.dwmapi.DwmExtendFrameIntoClientArea(ctypes.c_void_p(hwnd), ctypes.byref(margins))
        except Exception:
            pass


def apply(app) -> Palette:
    """The stylesheet, and the palette's accent (checkboxes, focus rings, selections) in DigUp's amber."""
    from PySide6.QtGui import QPalette

    p = palette()
    app.setStyleSheet(stylesheet(p))
    qp = app.palette()
    for role in (QPalette.ColorRole.Highlight, QPalette.ColorRole.Accent):
        qp.setColor(role, QColor(p.accent))
    qp.setColor(QPalette.ColorRole.HighlightedText, QColor(p.accent_text))
    app.setPalette(qp)
    return p
