"""Builds a folder of test files to try DigUp on: pictures, a screenshot, PDFs (text and scanned), documents, spoken
audio (Windows text-to-speech) and a video with a soundtrack. Needs ffmpeg (imageio-ffmpeg's is used).

    python scripts/make_testbed.py C:\\path\\to\\testbed
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent


def font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def picture(path: Path, draw) -> None:
    image = Image.new("RGB", (1024, 768), (135, 190, 235))
    draw(ImageDraw.Draw(image))
    image.save(path, quality=92)


def beach(d):
    d.rectangle((0, 450, 1024, 768), fill=(236, 214, 160))
    d.rectangle((0, 380, 1024, 450), fill=(40, 110, 190))
    d.ellipse((800, 60, 920, 180), fill=(255, 220, 90))
    # a dog: body, head, legs, tail
    d.ellipse((380, 470, 600, 580), fill=(150, 90, 40))
    d.ellipse((560, 420, 660, 510), fill=(150, 90, 40))
    d.polygon([(600, 425), (620, 380), (640, 430)], fill=(110, 60, 25))
    for x in (400, 440, 540, 575):
        d.rectangle((x, 560, x + 22, 640), fill=(150, 90, 40))
    d.line((385, 500, 330, 450), fill=(150, 90, 40), width=14)
    d.ellipse((625, 450, 637, 462), fill=(0, 0, 0))


def city(d):
    d.rectangle((0, 0, 1024, 768), fill=(20, 24, 60))
    import random

    random.seed(3)
    x = 0
    while x < 1024:
        w, h = random.randint(60, 140), random.randint(200, 600)
        d.rectangle((x, 768 - h, x + w, 768), fill=(40, 40, 50))
        for wy in range(768 - h + 15, 760, 30):
            for wx in range(x + 10, x + w - 10, 24):
                if random.random() < 0.6:
                    d.rectangle((wx, wy, wx + 10, wy + 14), fill=(255, 210, 120))
        x += w + 8
    d.ellipse((860, 60, 940, 140), fill=(240, 240, 220))


def screenshot(path: Path) -> None:
    image = Image.new("RGB", (1440, 900), (245, 245, 245))
    d = ImageDraw.Draw(image)
    d.rectangle((0, 0, 1440, 60), fill=(32, 33, 36))
    d.text((24, 16), "Checkout - Order #4815162342", fill="white", font=font(24))
    d.rounded_rectangle((420, 260, 1020, 640), radius=18, fill="white", outline=(220, 53, 69), width=4)
    d.text((470, 300), "Payment declined", fill=(220, 53, 69), font=font(44))
    d.text((470, 380), "Your card was declined by the issuer.", fill=(60, 60, 60), font=font(26))
    d.text((470, 430), "Error code: K7Q2LM", fill=(60, 60, 60), font=font(26))
    d.text((470, 520), "Try another card or contact your bank.", fill=(60, 60, 60), font=font(24))
    image.save(path)


def pdf_with_text(path: Path) -> None:
    pages = []
    texts = [
        ("Residential Lease Agreement", "This lease is made between the landlord and the tenant for the apartment at "
         "14 Elm Street. The term is twelve months starting on the first of March. Rent is due on the first day of "
         "each month and late payments incur a fee of fifty dollars after a five day grace period. "),
        ("Section 7: Pets", "No pets are allowed in the unit without the prior written consent of the landlord. If "
         "consent is given, the tenant must pay a pet deposit of three hundred dollars and is responsible for any "
         "damage caused by the animal. Dogs over forty pounds are not permitted. Cats must be kept indoors. "),
        ("Section 9: Utilities", "The tenant pays for electricity, internet and gas. The landlord pays for water, "
         "sewer and trash collection. Thermostat settings must stay above fifty five degrees in winter. "),
    ]
    for title, body in texts:
        page = Image.new("RGB", (1240, 1754), "white")
        d = ImageDraw.Draw(page)
        d.text((100, 100), title, fill="black", font=font(40))
        y = 200
        words = (body * 3).split()
        line = ""
        for w in words:
            if len(line) + len(w) > 70:
                d.text((100, y), line, fill="black", font=font(26))
                y += 40
                line = ""
            line += w + " "
        d.text((100, y), line, fill="black", font=font(26))
        pages.append(page)
    # Pillow writes image-only PDFs: a "scanned" lease. Use reportlab-free text PDF via a tiny hand-written PDF below.
    pages[0].save(path.with_name("scanned lease.pdf"), save_all=True, append_images=pages[1:])
    write_text_pdf(path, texts)


def write_text_pdf(path: Path, pages: list[tuple[str, str]]) -> None:
    """A minimal PDF with a real text layer (Helvetica), one page per (title, body)."""
    objects: list[bytes] = []

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    page_ids = []
    font_id = 3
    contents = []
    for title, body in pages:
        lines = []
        line = ""
        for w in (body * 2).split():
            if len(line) + len(w) > 85:
                lines.append(line)
                line = ""
            line += w + " "
        lines.append(line)
        stream = f"BT /F1 20 Tf 72 760 Td ({esc(title)}) Tj ET\n"
        y = 720
        for l in lines:
            stream += f"BT /F1 11 Tf 72 {y} Td ({esc(l)}) Tj ET\n"
            y -= 16
        contents.append(stream.encode("latin-1"))
    n = len(pages)
    # 1 catalog, 2 pages, 3 font, then (page, content) pairs
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i in range(n):
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {font_id} 0 R "
                       f">> >> /Contents {5 + 2 * i} 0 R >>".encode())
        objects.append(b"<< /Length %d >>\nstream\n" % len(contents[i]) + contents[i] + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects):
        offsets.append(len(out))
        out += f"{i + 1} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def docx(path: Path, paragraphs: list[str]) -> None:
    body = "".join(f"<w:p><w:r><w:t xml:space=\"preserve\">{p}</w:t></w:r></w:p>" for p in paragraphs)
    document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                f"<w:body>{body}</w:body></w:document>")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/'
                   'content-types"><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/'
                   'document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.'
                   'main+xml"/></Types>')
        z.writestr("word/document.xml", document)


def speech(path: Path, text: str) -> bool:
    """Windows text-to-speech (System.Speech) to a WAV."""
    script = (
        "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{path}'); $s.Speak(\"{text}\"); $s.Dispose()"
    )
    result = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True)
    return result.returncode == 0 and path.exists()


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "testbed").resolve()
    (root / "Pictures").mkdir(parents=True, exist_ok=True)
    (root / "Pictures" / "Screenshots").mkdir(exist_ok=True)
    (root / "Documents").mkdir(exist_ok=True)
    (root / "Music").mkdir(exist_ok=True)
    (root / "Videos").mkdir(exist_ok=True)

    picture(root / "Pictures" / "IMG_2041.jpg", beach)
    picture(root / "Pictures" / "IMG_2077.jpg", city)
    Image.open(HERE.parent / "tests" / "reference" / "photo.jpg").save(root / "Pictures" / "IMG_1999.jpg")
    screenshot(root / "Pictures" / "Screenshots" / "Screenshot 2026-10-11 093107.png")
    pdf_with_text(root / "Documents" / "lease.pdf")
    docx(root / "Documents" / "Tokyo trip.docx", [
        "Day 1: Flight to Tokyo Haneda on November 3.",
        "Day 2: Train to Kamakura to see the Great Buddha, and back for dinner in Shibuya.",
        "Day 3: Tsukiji outer market for breakfast, then teamLab Planets.",
    ])
    (root / "Documents" / "recipe.txt").write_text(
        "Grandma's banana bread\n\nMash three ripe bananas, mix with melted butter, sugar, one egg and vanilla. "
        "Fold in flour and baking soda. Bake at 175 C for an hour until a skewer comes out clean.", "utf-8")
    (root / "Documents" / "ইলিশ.txt").write_text(
        "সরষে ইলিশ রান্নার পদ্ধতি: ইলিশ মাছ ধুয়ে লবণ ও হলুদ মাখিয়ে রাখুন। সরষে বাটা, কাঁচা লঙ্কা আর সরষের তেল দিয়ে "
        "মাছ ভাপে রান্না করুন।", "utf-8")
    (root / "Documents" / "notes.html").write_text(
        "<html><head><title>Meeting notes</title></head><body><h1>Q3 planning</h1><p>We agreed to migrate the "
        "billing service to Postgres 17 before the holiday freeze.</p></body></html>", "utf-8")
    (root / "Documents" / "letter.rtf").write_text(
        r"{\rtf1\ansi{\fonttbl\f0\fswiss Helvetica;}\f0\pard Dear neighbour,\par the community garden opens on "
        r"Saturday. Bring gloves and seedlings; tomatoes and basil grow best along the south fence.\par}", "latin-1")

    talk = root / "Music" / "podcast episode 12.wav"
    ok = speech(talk, "Welcome back to the show. Today we talk about sleep. Most adults need seven to nine hours of "
                      "sleep a night, and keeping the bedroom cool and dark helps a lot. Caffeine after two in the "
                      "afternoon can keep you awake. Next week we will discuss running a first marathon.")
    print("speech:", "ok" if ok else "unavailable")

    video = root / "Videos" / "beach day.mp4"
    frames = root / "Videos" / "_frames"
    frames.mkdir(exist_ok=True)
    picture(frames / "a.png", city)
    picture(frames / "b.png", beach)
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error",
                    "-loop", "1", "-t", "8", "-i", str(frames / "a.png"),
                    "-loop", "1", "-t", "8", "-i", str(frames / "b.png"),
                    "-f", "lavfi", "-t", "16", "-i", "sine=frequency=330:sample_rate=16000",
                    "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p[v]",
                    "-map", "[v]", "-map", "2:a", "-shortest", "-r", "10", str(video)], check=True)
    for f in frames.iterdir():
        f.unlink()
    frames.rmdir()
    print("testbed:", root)


if __name__ == "__main__":
    main()
