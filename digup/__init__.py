"""DigUp for Windows: search the files on your PC by what's in them, with EmbeddingGemma 2.

A Windows port of https://github.com/ARahim3/DigUp (macOS). The engine (crawler, readers, index, search) follows the
original's design and measured choices; the Mac-only pieces (Metal, PDFKit, AVFoundation, Vision, AppKit) are replaced
with llama.cpp's Windows builds (CUDA, Vulkan or CPU), PDFium, ffmpeg, Windows.Media.Ocr and Qt.
"""

import os as _os

__version__ = "0.1.0"

# NumPy's BLAS reserves buffers for every core when it loads (750 MB on a 32-core PC) and again per calling thread.
# DigUp spreads its own work over a few threads (vectorcache.py), so BLAS gets one. Set before NumPy is imported.
# (Only OpenBLAS: llama.cpp's CPU backend uses OpenMP, in the worker process, with its own thread count.)
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
