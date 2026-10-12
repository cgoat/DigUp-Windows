"""What DigUp downloads, once: the EmbeddingGemma 2 GGUF files and llama.cpp's Windows DLLs, both at pinned versions.

Downloads resume (HTTP ranges), are checked (sizes; SHA-256 for the model), and only a verified file gets its final
name, so being there is enough.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import paths

USER_AGENT = "DigUp-Windows/0.1 (+https://github.com/cgoat/DigUp-Windows)"

# MARK: The model

REPOSITORY = "ggml-org/embeddinggemma-2-GGUF"
REVISION = "bfcd298762cc34d0357ece5ebdd31791a3a374d8"


@dataclass(frozen=True)
class ModelFile:
    name: str
    bytes: int
    sha256: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{self.name}"


# Never the F16 files: the model's activations overflow float16.
TEXT = ModelFile("embeddinggemma-2-Q8_0.gguf", 309_855_456,
                 "2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135")
MEDIA = ModelFile("mmproj-embeddinggemma-2-Q8_0.gguf", 554_821_024,
                  "c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f")
MODEL_FILES = [TEXT, MEDIA]


def model_folder() -> Path:
    return paths.models()


def huggingface_snapshot() -> Path:
    hub = os.environ.get("HF_HUB_CACHE") or (
        os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME")
        else str(Path.home() / ".cache" / "huggingface" / "hub"))
    return Path(hub) / f"models--{REPOSITORY.replace('/', '--')}" / "snapshots" / REVISION


def locate_models() -> tuple[Path, Path | None] | None:
    """(text, media or None) from DigUp's folder or the Hugging Face cache; None without the text part."""
    for folder in (model_folder(), huggingface_snapshot()):
        text = folder / TEXT.name
        if text.exists():
            media = folder / MEDIA.name
            return text, media if media.exists() else None
    return None


def models_complete() -> bool:
    found = locate_models()
    return bool(found and found[1])


def missing_models() -> list[ModelFile]:
    return [f for f in MODEL_FILES if not (model_folder() / f.name).exists()
            and not (huggingface_snapshot() / f.name).exists()]


Progress = Callable[[int, int], None]  # (bytes so far, total)


def fetch(url: str, target: Path, total: int | None, progress: Progress | None = None,
          should_stop: Callable[[], bool] = lambda: False, attempts: int = 6) -> Path:
    """Downloads `url` to `target`.part, resuming what's there, then renames it. Returns the final path. A dropped
    connection is retried from where it stopped."""
    for attempt in range(attempts):
        try:
            return _fetch_once(url, target, total, progress, should_stop)
        except (urllib.error.URLError, ConnectionError, TimeoutError, IOError) as error:
            if isinstance(error, InterruptedError) or attempt == attempts - 1:
                raise
            time.sleep(min(2 ** attempt, 30))
    raise IOError("unreachable")


def _fetch_once(url: str, target: Path, total: int | None, progress: Progress | None,
                should_stop: Callable[[], bool]) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if total is not None and have > total:
        part.unlink()
        have = 0
    if total is None or have < total:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        if have:
            request.add_header("Range", f"bytes={have}-")
        with urllib.request.urlopen(request, timeout=60) as response:
            if have and response.status != 206:
                have = 0  # the server ignored the range: start over
            length = int(response.headers.get("Content-Length") or 0)
            total = total or (have + length if length else None)
            with open(part, "ab" if have else "wb") as out:
                while True:
                    if should_stop():
                        raise InterruptedError("download stopped")
                    block = response.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    have += len(block)
                    if progress:
                        progress(have, total or 0)
    if total is not None and part.stat().st_size != total:
        raise IOError(f"{target.name}: got {part.stat().st_size} of {total} bytes; try again")
    os.replace(part, target)
    return target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def download_models(progress: Progress | None = None, should_stop: Callable[[], bool] = lambda: False) -> None:
    """Gets whatever model file is missing (the text part first: search works as soon as it's there)."""
    todo = missing_models()
    grand = sum(f.bytes for f in todo)
    done = 0
    for file in todo:
        cached = huggingface_snapshot() / file.name
        target = model_folder() / file.name
        if cached.exists() and cached.stat().st_size == file.bytes:
            continue
        base = done
        fetch(file.url, target.with_suffix(".unverified"), file.bytes,
              (lambda n, _t, b=base: progress(b + n, grand)) if progress else None, should_stop)
        unverified = target.with_suffix(".unverified")
        if sha256(unverified) != file.sha256:
            unverified.unlink()
            raise IOError(f"{file.name} didn't match its checksum; it was removed, try again")
        os.replace(unverified, target)
        done += file.bytes


# MARK: llama.cpp

LLAMA_TAG = "b11461"  # the first tags with EmbeddingGemma 2 (text + vision + audio) are b11452+
RELEASES = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}"
# Backend -> the release's zips (the CUDA build needs its runtime DLLs, shipped separately).
BACKENDS = {
    "cuda": [f"llama-{LLAMA_TAG}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"],
    "vulkan": [f"llama-{LLAMA_TAG}-bin-win-vulkan-x64.zip"],
    "cpu": [f"llama-{LLAMA_TAG}-bin-win-cpu-x64.zip"],
}
# Only the libraries: no tools, no server.
_SKIP_DLL = ("-impl.dll",)


def _has_dll(name: str) -> bool:
    try:
        ctypes.WinDLL(name)
        return True
    except OSError:
        return False


def best_backend() -> str:
    """CUDA on an NVIDIA GPU, Vulkan on any other GPU with a Vulkan driver (AMD, Intel), else the CPU.
    $DIGUP_BACKEND chooses."""
    chosen = os.environ.get("DIGUP_BACKEND")
    if chosen in BACKENDS:
        return chosen
    if os.name != "nt":
        return "cpu"
    if _has_dll("nvcuda.dll"):
        return "cuda"
    if _has_dll("vulkan-1.dll"):
        return "vulkan"
    return "cpu"


def runtime_folder(backend: str) -> Path:
    return paths.runtime() / f"{LLAMA_TAG}-{backend}"


def runtime_ready(backend: str) -> bool:
    folder = runtime_folder(backend)
    return (folder / "llama.dll").exists() and (folder / "mtmd.dll").exists() and (folder / "READY").exists()


def runtime_bytes(backend: str) -> int:
    """About how much a backend downloads (zips)."""
    return {"cuda": 656_000_000, "vulkan": 60_000_000, "cpu": 40_000_000}.get(backend, 0)


def download_runtime(backend: str, progress: Progress | None = None,
                     should_stop: Callable[[], bool] = lambda: False) -> Path:
    folder = runtime_folder(backend)
    if runtime_ready(backend):
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    zips = BACKENDS[backend]
    estimate = runtime_bytes(backend)
    done = 0
    for name in zips:
        archive = paths.runtime() / "downloads" / name
        if not archive.exists():
            base = done
            fetch(f"{RELEASES}/{name}", archive, None,
                  (lambda n, _t, b=base: progress(min(b + n, estimate), estimate)) if progress else None, should_stop)
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                base_name = os.path.basename(member)
                if base_name.lower().endswith(".dll") and not base_name.endswith(_SKIP_DLL):
                    with z.open(member) as src, open(folder / base_name, "wb") as dst:
                        shutil.copyfileobj(src, dst, 8 << 20)
        done += archive.stat().st_size
        archive.unlink()
    (folder / "READY").write_text(LLAMA_TAG)
    return folder
