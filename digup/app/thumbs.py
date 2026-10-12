"""Pictures for results: the image itself, the PDF page that matched, the video frame at the moment. Made off the UI
thread and kept in a small cache."""

from __future__ import annotations

import subprocess
from collections import OrderedDict

from PIL import Image
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtGui import QImage

from .. import extractors
from ..model import FileKind
from ..searcher import SearchHit

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _qimage(image: Image.Image) -> QImage:
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    q = QImage(data, rgba.width, rgba.height, rgba.width * 4, QImage.Format.Format_RGBA8888)
    return q.copy()


def render(hit: SearchHit, side: int) -> Image.Image | None:
    """A picture of the hit at most `side` pixels on its longer side, or None (text, sound)."""
    try:
        if hit.kind in (FileKind.IMAGE, FileKind.SCREENSHOT):
            image = extractors.to_rgb(extractors.open_image(hit.path))
            image.thumbnail((side, side), Image.Resampling.LANCZOS)
            return image
        if hit.kind == FileKind.PDF:
            import pypdfium2 as pdfium

            document = pdfium.PdfDocument(hit.path)
            try:
                page = document[max(0, int(hit.loc or 1) - 1)]
                width, height = page.get_size()
                image = page.render(scale=side / max(width, height, 1)).to_pil()
                return extractors.to_rgb(image)
            finally:
                document.close()
        if hit.kind == FileKind.VIDEO:
            from ..media import ffmpeg

            at = hit.loc or 0.0
            out = subprocess.run(
                [ffmpeg(), "-hide_banner", "-loglevel", "error", "-ss", f"{at:.2f}", "-i", hit.path, "-frames:v", "1",
                 "-vf", f"scale='if(gt(iw,ih),{side},-2)':'if(gt(iw,ih),-2,{side})'", "-f", "image2pipe",
                 "-vcodec", "png", "-"],
                capture_output=True, timeout=20, creationflags=NO_WINDOW)
            if out.returncode == 0 and out.stdout:
                import io

                return Image.open(io.BytesIO(out.stdout)).convert("RGB")
    except Exception:  # noqa: BLE001 - no picture, then
        return None
    return None


class _Job(QRunnable):
    def __init__(self, key, hit: SearchHit, side: int, sink: "Thumbnails"):
        super().__init__()
        self.key, self.hit, self.side, self.sink = key, hit, side, sink

    def run(self) -> None:
        image = render(self.hit, self.side)
        self.sink._done(self.key, _qimage(image) if image else None)


class Thumbnails(QObject):
    ready = Signal(object, object)  # key, QImage | None

    def __init__(self, capacity: int = 300):
        super().__init__()
        self.cache: OrderedDict = OrderedDict()
        self.capacity = capacity
        self.pending: set = set()
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(3)

    @staticmethod
    def key(hit: SearchHit, side: int):
        return hit.path, hit.loc if hit.kind in (FileKind.PDF, FileKind.VIDEO) else None, side

    def get(self, hit: SearchHit, side: int) -> QImage | None:
        """The picture if it's ready; otherwise starts it and `ready` fires later."""
        key = self.key(hit, side)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if hit.kind in (FileKind.IMAGE, FileKind.SCREENSHOT, FileKind.PDF, FileKind.VIDEO) and key not in self.pending:
            self.pending.add(key)
            self.pool.start(_Job(key, hit, side, self))
        return None

    def _done(self, key, image) -> None:
        self.pending.discard(key)
        self.cache[key] = image
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        self.ready.emit(key, image)
