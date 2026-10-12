"""Readers for pictures, screenshots (Windows OCR), PDFs (PDFium) and documents (txt, md, html, rtf, docx, odt)."""

from __future__ import annotations

import asyncio
import html as htmllib
import re
import threading
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from . import searchtext

try:  # iPhone photos
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - optional
    pass

Image.MAX_IMAGE_PIXELS = 400_000_000


class ExtractError(Exception):
    pass


class TooSmall(ExtractError):
    pass


class Locked(ExtractError):
    def __str__(self) -> str:
        return "password-protected"


# MARK: Images


@dataclass
class PreparedImage:
    path: str
    width: int
    height: int
    taken: str | None


def open_image(path: str) -> Image.Image:
    try:
        image = Image.open(path)
        image.draft("RGB", (4096, 4096))  # JPEGs decode at a smaller scale straight away
        image = ImageOps.exif_transpose(image)
    except Exception as error:
        raise ExtractError(f"unreadable: {error}") from error
    return image


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA", "P", "PA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def prepare_image(path: str, max_pixel: int, min_side: int, folder: Path, png: bool) -> tuple[PreparedImage, Image.Image]:
    """A downscaled copy for the model, and the full picture (for OCR)."""
    with Image.open(path) as probe:
        width, height = probe.size
        exif = probe.getexif() if hasattr(probe, "getexif") else {}
    if max(width, height) < min_side:
        raise TooSmall(f"small image {width}×{height}")
    taken = None
    try:
        taken = exif.get_ifd(0x8769).get(0x9003) if exif else None  # DateTimeOriginal
    except Exception:
        pass
    full = to_rgb(open_image(path))
    small = full.copy()
    small.thumbnail((max_pixel, max_pixel), Image.Resampling.LANCZOS)
    out = folder / (uuid.uuid4().hex + (".png" if png else ".jpg"))
    small.save(out, "PNG" if png else "JPEG", quality=90)
    return PreparedImage(str(out), width, height, str(taken) if taken else None), full


# MARK: OCR (Windows.Media.Ocr: built into Windows 10 and 11, the user's display languages)

_ocr_lock = threading.Lock()
_ocr_engine = None
_ocr_failed = False


def _engine():
    global _ocr_engine, _ocr_failed
    if _ocr_engine is None and not _ocr_failed:
        try:
            from winrt.windows.media.ocr import OcrEngine

            _ocr_engine = OcrEngine.try_create_from_user_profile_languages()
            _ocr_failed = _ocr_engine is None
        except Exception:
            _ocr_failed = True
    return _ocr_engine


def ocr_available() -> bool:
    return _engine() is not None


def ocr(image: Image.Image) -> str:
    """The text in a picture, line by line. Empty when Windows OCR isn't there."""
    engine = _engine()
    if engine is None:
        return ""
    try:
        from winrt.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
        from winrt.windows.media.ocr import OcrEngine

        limit = OcrEngine.max_image_dimension
        picture = image
        if max(picture.size) > limit:
            picture = picture.copy()
            picture.thumbnail((limit, limit))
        rgba = picture.convert("RGBA")
        bitmap = SoftwareBitmap.create_copy_from_buffer(rgba.tobytes(), BitmapPixelFormat.RGBA8, rgba.width, rgba.height)

        async def run():
            result = await engine.recognize_async(bitmap)
            return "\n".join(line.text for line in result.lines)

        with _ocr_lock:
            return asyncio.run(run())
    except Exception:
        return ""


# MARK: PDFs


@dataclass
class Page:
    number: int  # from 1
    text: str
    image_path: str | None = None  # rendered because the page had too little text (scans, slides)
    ocr: bool = False
    garbled: str | None = None  # text in glyph order: only for a meaning vector


def pdf_pages(path: str, wanted, min_text: int, render_pixels: int, folder: Path) -> tuple[list[Page], int, str | None]:
    """Reads the pages in `wanted(total)` (from 0). Pages with real text become text; scans and slides are rendered
    for the vision encoder (and read by OCR for keyword search)."""
    import pypdfium2 as pdfium

    try:
        document = pdfium.PdfDocument(path)
    except pdfium.PdfiumError as error:
        if "password" in str(error).lower():
            raise Locked() from error
        raise ExtractError(f"can't open PDF: {error}") from error
    try:
        total = len(document)
        read = wanted(total)
        pages: list[Page] = []
        for index in range(max(read.start, 0), min(read.stop, total)):
            page = document[index]
            try:
                text_page = page.get_textpage()
                text = text_page.get_text_bounded().replace("\r\n", "\n").strip()
                text_page.close()
            except Exception:
                text = ""
            garbled = None
            problem = searchtext.page_text_problem(text)
            if problem == "glyph_order":
                text, garbled = "", text
            elif len(text) >= min_text:
                pages.append(Page(index + 1, text))
                page.close()
                continue
            width, height = page.get_size()
            scale = render_pixels / max(width, height, 1)
            image = page.render(scale=scale).to_pil()
            page.close()
            image = to_rgb(image)
            if not text and problem is None:
                text = ocr(image).strip()
            out = folder / (uuid.uuid4().hex + ".png")
            image.save(out, "PNG")
            pages.append(Page(index + 1, text, str(out), ocr=bool(text) and problem is None, garbled=garbled))
        title = None
        try:
            title = (document.get_metadata_dict() or {}).get("Title") or None
        except Exception:
            pass
        return pages, total, title
    finally:
        document.close()


def pdf_page_texts(path: str, up_to: int) -> list[str]:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(path)
    try:
        out = []
        for index in range(min(up_to, len(document))):
            tp = document[index].get_textpage()
            out.append(tp.get_text_bounded())
        return out
    finally:
        document.close()


# MARK: Documents


def read_text_file(path: str) -> str:
    data = Path(path).read_bytes()
    for bom, codec in ((b"\xef\xbb\xbf", "utf-8-sig"), (b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16")):
        if data.startswith(bom):
            return data.decode(codec, "replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", "replace")


def doc_text(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    if ext == "docx":
        return _xml_text(path, "word/document.xml", paragraph=r"</w:p>", tab=r"<w:tab/>")
    if ext == "odt":
        return _xml_text(path, "content.xml", paragraph=r"</text:p>|</text:h>", tab=r"<text:tab/>")
    if ext == "rtf":
        return rtf_to_text(read_text_file(path))
    if ext in ("html", "htm"):
        return plain_text_from_html(read_text_file(path))
    return read_text_file(path)


def _xml_text(path: str, member: str, paragraph: str, tab: str) -> str:
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read(member).decode("utf-8", "replace")
    except (zipfile.BadZipFile, KeyError, OSError) as error:
        raise ExtractError(f"unreadable: {error}") from error
    xml = re.sub(tab, "\t", xml)
    xml = re.sub(paragraph, "\n", xml)
    xml = re.sub(r"<w:br/>|<text:line-break/>", "\n", xml)
    return htmllib.unescape(re.sub(r"<[^>]+>", "", xml)).strip()


def rtf_to_text(rtf: str) -> str:
    """Good enough for search: drops groups that hold no text (fonts, colors, pictures), decodes escapes."""
    out, stack, ignorable = [], [], False
    skip_destinations = {"fonttbl", "colortbl", "stylesheet", "info", "pict", "object", "header", "footer",
                         "datastore", "themedata", "listtable", "listoverridetable", "rsidtbl", "xmlnstbl",
                         "generator", "latentstyles"}
    token = re.compile(r"\\([a-z]{1,32})(-?\d{1,10})? ?|\\'([0-9a-f]{2})|\\([^a-z])|([{}])|[\r\n]+|(.)", re.I | re.S)
    skip = 0  # characters after \uN that stand in for it in old readers (\uc1: one)
    for match in token.finditer(rtf):
        word, arg, hexcode, char, brace, plain = match.groups()
        if skip and (plain or hexcode):
            skip -= 1
            continue
        skip = 0
        if brace == "{":
            stack.append(ignorable)
        elif brace == "}":
            ignorable = stack.pop() if stack else False
        elif char:
            if char == "*":
                ignorable = True
            elif not ignorable:
                out.append({"~": " ", "-": "", "_": "-"}.get(char, char))
        elif word:
            if word in skip_destinations:
                ignorable = True
            elif ignorable:
                continue
            elif word in ("par", "line", "sect", "page"):
                out.append("\n")
            elif word == "tab":
                out.append("\t")
            elif word == "u" and arg:
                out.append(chr(int(arg) % 65536))
                skip = 1
        elif hexcode and not ignorable:
            out.append(bytes([int(hexcode, 16)]).decode("cp1252", "replace"))
        elif plain and not ignorable:
            out.append(plain)
    return "".join(out).strip()


def plain_text_from_html(html: str) -> str:
    text = html
    for pattern in (r"(?is)<script\b.*?</script\s*>", r"(?is)<style\b.*?</style\s*>", r"(?s)<!--.*?-->",
                    r"(?is)<head\b.*?</head\s*>"):
        text = re.sub(pattern, " ", text)
    text = re.sub(r"(?i)<(br|/p|/div|/li|/h[1-6]|/tr|/section|/article)\b[^>]*>", "\n", text)
    text = htmllib.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\s*\n\s*", "\n", text).strip()


def chunks(text: str, size: int = 1800, overlap: int = 200, limit: int = 5001) -> list[str]:
    """~`size`-character pieces (~450 tokens) that overlap a little, breaking at whitespace."""
    clean = re.sub(r"[ \t\u00a0]+", " ", text)
    clean = re.sub(r"\n\s*\n\s*\n+", "\n\n", clean).strip()
    out: list[str] = []
    start, n = 0, len(clean)
    while start < n and len(out) < limit:
        end = min(start + size, n)
        if end < n:
            space = max(clean.rfind(" ", start, end), clean.rfind("\n", start, end))
            if space - start > size // 2:
                end = space
        piece = clean[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        nxt = max(end - overlap, start)
        if nxt <= start:
            nxt = end
        while nxt < end and not clean[nxt].isspace():
            nxt += 1
        start = nxt if nxt < end else end
    return out
