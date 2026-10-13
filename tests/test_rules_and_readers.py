"""What gets indexed, and how documents are read: rules, the crawler, chunks, HTML, RTF and DOCX, image sizing."""

import zipfile

from digup import extractors, rules
from digup.crawler import crawl
from digup.llama import target_size
from digup.model import FileKind, IndexOptions


def test_classify(tmp_path):
    o = IndexOptions()
    ts = tmp_path / "app.ts"
    ts.write_text("export const x = 1;\n" * 40)
    assert rules.classify(str(ts), o) == ("skip", "code or data file")
    video = tmp_path / "clip.ts"
    video.write_bytes(b"".join(b"\x47" + b"\0" * 187 for _ in range(4)))
    assert rules.classify(str(video), o) == ("index", FileKind.VIDEO)
    assert rules.classify(str(tmp_path / "a.py"), IndexOptions(include_code=True)) == ("index", FileKind.DOC)
    assert rules.classify(str(tmp_path / "IMG_1.HEIC"), o) == ("index", FileKind.IMAGE)
    assert rules.classify(str(tmp_path / "Screenshot 2026-10-11 093107.png"), o) == ("index", FileKind.SCREENSHOT)
    assert rules.classify(str(tmp_path / "Screenshots" / "x.png"), o) == ("index", FileKind.SCREENSHOT)
    assert rules.classify(str(tmp_path / "id_rsa"), o) == ("skip", "key or password file")
    assert rules.classify(str(tmp_path / "cert.pfx"), o) == ("skip", "key or password file")
    assert rules.classify(str(tmp_path / "talk.m4a"), o) == ("index", FileKind.AUDIO)
    assert rules.classify(str(tmp_path / "x.exe"), o) == ("ignore", None)
    assert rules.classify(str(tmp_path / "a.HEIC"), IndexOptions(excluded_extensions={"heic"}))[0] == "skip"


def test_classify_code(tmp_path):
    o = IndexOptions(code=True)
    kind = lambda name: rules.classify_code(str(tmp_path / name), o)
    for name in ("retry.py", "App.swift", "README.md", "settings.yaml", "explore.ipynb", "Makefile",
                 "Dockerfile.dev", "build.ps1", "Program.cs"):
        assert kind(name) == ("index", FileKind.CODE), name
    assert kind("package.json") == ("skip", "data file")
    assert kind("bundle.min.js") == ("skip", "minified")
    assert kind("LICENSE.txt") == ("skip", "license")
    assert kind(".env.local") == ("skip", "key or password file")


def test_machine_names_and_datasets():
    assert rules.looks_machine_named("clip_000123")
    assert rules.looks_machine_named("frame-0042")
    assert rules.looks_machine_named("3f9a1c0e-12ab-4cde")
    assert rules.looks_machine_named("000123")
    assert not rules.looks_machine_named("IMG_1234")
    assert not rules.looks_machine_named("PXL_20260101_123456789")
    assert not rules.looks_machine_named("Screenshot 2026-10-01 093107")
    assert not rules.looks_machine_named("Tokyo itinerary")
    files = [("C:\\x\\clips", f"clip_{i:06d}", FileKind.VIDEO, 5_000_000) for i in range(400)]
    files += [("C:\\x\\holiday", f"IMG_{i}", FileKind.IMAGE, 3_000_000) for i in range(400)]
    # 50x50 dataset tiles whose names don't look machine-made; a few small graphics; a folder only partly tiny.
    files += [("C:\\x\\tiles", f"9383_idx5_x{i}_y751_class1", FileKind.IMAGE, 6_000) for i in range(150)]
    files += [("C:\\x\\icons", f"logo{i}", FileKind.IMAGE, 5_000) for i in range(40)]
    files += [("C:\\x\\mixed", f"IMG_{i}", FileKind.IMAGE, 6_000 if i < 60 else 2_000_000) for i in range(150)]
    assert rules.dataset_folders(files, roots={"C:\\x"}) == {"C:\\x\\clips": 400, "C:\\x\\tiles": 150}


def test_crawler(tmp_path):
    (tmp_path / "Pictures").mkdir()
    (tmp_path / "Pictures" / "IMG_1.jpg").write_bytes(b"x")
    (tmp_path / "notes.md").write_text("hello")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "readme.md").write_text("skip me")
    project = tmp_path / "proj"
    project.mkdir()
    (project / "package.json").write_text("{}")
    (project / "logo.png").write_bytes(b"x")
    (project / "Screenshot 1.png").write_bytes(b"x")
    report = crawl([str(tmp_path)], IndexOptions())
    names = sorted(c.path.split("\\")[-1] for c in report.candidates)
    assert names == ["IMG_1.jpg", "Screenshot 1.png", "notes.md"]
    assert report.skipped["inside a code project"] == 1
    assert all(c.inode for c in report.candidates)  # NTFS file ids: moves are noticed


def test_chunks():
    text = " ".join(f"word{i}" for i in range(1, 601))
    chunks = extractors.chunks(text, size=500, overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 500 for c in chunks)
    assert all(c.startswith("word") for c in chunks)  # never starts mid-word
    assert chunks[-1].endswith("word600")


def test_html_rtf_docx(tmp_path):
    html = ("<html><head><title>x</title><style>p{}</style></head><body><h1>Trip &amp; plans</h1>"
            "<p>Fly to Tokyo on Friday.</p><script>var a=1;</script><p>Caf&eacute; &mdash; ramen</p></body></html>")
    assert extractors.plain_text_from_html(html) == "Trip & plans\nFly to Tokyo on Friday.\nCafé — ramen"
    rtf = r"{\rtf1\ansi{\fonttbl\f0\fswiss Helvetica;}\f0\pard Caf\'e9 opens\par on Saturday \u8212? soon\par}"
    assert extractors.rtf_to_text(rtf) == "Café opens\non Saturday — soon"
    docx = tmp_path / "a.docx"
    with zipfile.ZipFile(docx, "w") as z:
        z.writestr("word/document.xml", '<w:document xmlns:w="w"><w:body><w:p><w:r><w:t>Day 1: Haneda</w:t></w:r></w:p>'
                                        "<w:p><w:r><w:t>Day 2: Kamakura</w:t></w:r></w:p></w:body></w:document>")
    assert extractors.doc_text(str(docx)) == "Day 1: Haneda\nDay 2: Kamakura"


def test_target_size_fills_the_budget():
    # The macOS repo's ImageSizingTests: transformers' resize target.
    assert target_size(900, 1440, 280) == (624, 1008)
    assert target_size(900, 1440, 140) == (432, 672)
    assert target_size(900, 1440, 70) == (288, 480)
    assert target_size(768, 768, 280) == (768, 768)
    assert target_size(768, 768, 140) == (528, 528)
    assert target_size(300, 300, 140) == (528, 528)
    assert target_size(10, 4000, 140) == (48, 6720)
