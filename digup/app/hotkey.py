"""A system-wide hotkey (RegisterHotKey) that opens the search panel from any app."""

from __future__ import annotations

import ctypes
from ctypes import wintypes

from PySide6.QtCore import QAbstractNativeEventFilter, QObject, Signal
from PySide6.QtWidgets import QApplication, QWidget

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]

WM_HOTKEY = 0x0312
MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x8, 0x4000
MODIFIERS = {"ctrl": MOD_CONTROL, "control": MOD_CONTROL, "shift": MOD_SHIFT, "alt": MOD_ALT, "win": MOD_WIN,
             "meta": MOD_WIN}
KEYS = {"space": 0x20, "enter": 0x0D, "return": 0x0D, "tab": 0x09, "`": 0xC0, "/": 0xBF, ".": 0xBE, ",": 0xBC,
        ";": 0xBA, "-": 0xBD, "=": 0xBB}
KEYS.update({f"f{i}": 0x70 + i - 1 for i in range(1, 25)})


def parse(text: str) -> tuple[int, int] | None:
    """"Ctrl+Shift+Space" -> (modifiers, virtual key)."""
    mods, key = 0, None
    for part in text.replace(" ", "").split("+"):
        low = part.lower()
        if low in MODIFIERS:
            mods |= MODIFIERS[low]
        elif low in KEYS:
            key = KEYS[low]
        elif len(part) == 1 and part.isalnum():
            key = ord(part.upper())
        else:
            return None
    return (mods, key) if key is not None and mods else None


class _Filter(QAbstractNativeEventFilter):
    def __init__(self, hotkey: "Hotkey"):
        super().__init__()
        self.hotkey = hotkey

    def nativeEventFilter(self, event_type, message):
        if event_type == b"windows_generic_MSG":
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == WM_HOTKEY and msg.wParam == self.hotkey.ID:
                self.hotkey.pressed.emit()
                return True, 0
        return False, 0


class Hotkey(QObject):
    pressed = Signal()
    ID = 0xD16

    def __init__(self):
        super().__init__()
        self._window = QWidget()  # a window of our own to receive WM_HOTKEY
        self._window.setWindowTitle("DigUp hotkey")
        self._hwnd = int(self._window.winId())
        self._filter = _Filter(self)
        QApplication.instance().installNativeEventFilter(self._filter)
        self.registered: str | None = None

    def register(self, text: str) -> bool:
        """Takes `text` as the hotkey; False when it can't be parsed or another app holds it."""
        self.unregister()
        parsed = parse(text)
        if not parsed:
            return False
        mods, key = parsed
        if not user32.RegisterHotKey(self._hwnd, self.ID, mods | MOD_NOREPEAT, key):
            return False
        self.registered = text
        return True

    def unregister(self) -> None:
        if self.registered:
            user32.UnregisterHotKey(self._hwnd, self.ID)
            self.registered = None
