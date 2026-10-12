"""The shared vocabulary: kinds of files and segments, indexing options, EmbeddingGemma 2's prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

DIM = 768


class FileKind(str, Enum):
    SCREENSHOT = "screenshot"
    IMAGE = "image"
    PDF = "pdf"
    DOC = "doc"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"  # only in the code index


# The order of the first indexing pass: what people look for most comes first.
INDEXING_ORDER_SQL = (
    "CASE kind WHEN 'screenshot' THEN 0 WHEN 'image' THEN 1 WHEN 'pdf' THEN 2 WHEN 'doc' THEN 3 "
    "WHEN 'audio' THEN 4 WHEN 'video' THEN 5 ELSE 6 END"
)


class SegmentKind(str, Enum):
    NAME = "name"  # file name + folder names (keyword search only)
    IMAGE = "image"
    OCR = "ocr"  # text Windows OCR read from a screenshot (keyword search only)
    PAGE = "page"  # a PDF page's text
    PAGE_IMAGE = "page_image"  # a PDF page rendered as a picture (scans, slides)
    CHUNK = "chunk"  # a piece of a document's text
    AUDIO = "audio"  # a window of audio (a file or a video's soundtrack)
    FRAME = "frame"  # a video keyframe
    LINES = "lines"  # a stretch of a code file: loc..loc_end are its first and last lines
    CELLS = "cells"  # a stretch of a notebook: loc..loc_end are its first and last cells


class Modality(str, Enum):
    """What produced a segment's vector. Scores are normalized per modality (text vectors sit higher)."""

    NONE = "none"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"


@dataclass
class IndexOptions:
    """Indexing settings. Anything that changes vectors is part of `fingerprint`; an index never mixes settings."""

    code: bool = False
    code_stretch: int = 1800
    max_code_bytes: int = 1 << 20
    max_notebook_bytes: int = 20 << 20
    include_code: bool = False
    include_datasets: bool = False
    min_image_side: int = 256
    big_pdf_pages: int = 50
    pdf_step: int = 30  # pages
    doc_step: int = 40  # chunks, ~64,000 characters
    max_pages: int = 5_000
    max_chunks: int = 5_000
    min_page_text: int = 200  # fewer characters: embed the rendered page instead
    photo_budget: int = 140  # vision soft tokens: 70 / 140 / 280 / 560 / 1120
    screenshot_budget: int = 280
    page_budget: int = 280
    frame_budget: int = 70
    frame_interval: float = 3.0
    audio_window: float = 30.0  # the model ignores audio past 30 s of one input
    audio_hop: float = 25.0
    excluded_folders: list[str] = field(default_factory=list)
    excluded_extensions: set[str] = field(default_factory=set)

    @staticmethod
    def extensions(text: str) -> set[str]:
        """File extensions as people write them ("HEIC, .txt *.md") -> {"heic", "txt", "md"}."""
        words = text.replace(",", " ").split()
        return {w.lstrip("*.").lower() for w in words if w.lstrip("*.")}

    @property
    def fingerprint(self) -> str:
        if self.code:
            return f"code1 lines{self.code_stretch}"
        return (
            f"photo{self.photo_budget} shot{self.screenshot_budget} page{self.page_budget} "
            f"frame{self.frame_budget}/{self.frame_interval}s audio{self.audio_window}/{self.audio_hop}s"
        )


class Prompts:
    """EmbeddingGemma 2's task prompts (model card). Media inputs take no prompt."""

    @staticmethod
    def query(text: str) -> str:
        return f"task: search result | query: {text}"

    @staticmethod
    def code_query(text: str) -> str:
        return f"task: code retrieval | query: {text}"

    @staticmethod
    def document(title: str | None, text: str) -> str:
        return f"title: {title or 'none'} | text: {text}"


class CodeQuery:
    """"code: retry with backoff" searches the code index, and nothing else."""

    PREFIX = "code:"

    @staticmethod
    def text_of(query: str) -> str | None:
        trimmed = query.lstrip()
        if len(trimmed) >= 5 and trimmed[:5].lower() == CodeQuery.PREFIX:
            return trimmed[5:].strip()
        return None


@dataclass
class WorkerInfo:
    """What the embedding runtime reports; part of the index fingerprint."""

    model: str
    revision: str
    runtime: str
    dtype: str
    dim: int = DIM
    text_only: bool = False
    load_seconds: float | None = None

    @property
    def vector_source(self) -> str:
        return f"{self.model}@{self.revision[:7]} {self.runtime} {self.dtype}"


@dataclass
class Embedding:
    vector: list[float] | None
    error: str | None = None


class WorkerError(Exception):
    """The embedder is missing or died: indexing stops, and pending files stay pending."""
