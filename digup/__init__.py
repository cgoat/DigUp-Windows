"""DigUp for Windows: search the files on your PC by what's in them, with EmbeddingGemma 2.

A Windows port of https://github.com/ARahim3/DigUp (macOS). The engine (crawler, readers, index, search) follows the
original's design and measured choices; the Mac-only pieces (Metal, PDFKit, AVFoundation, Vision, AppKit) are replaced
with llama.cpp's Windows builds (CUDA, Vulkan or CPU), PDFium, ffmpeg, Windows.Media.Ocr and Qt.
"""

__version__ = "0.1.0"
