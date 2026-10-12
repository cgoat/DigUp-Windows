"""Audio and video through ffmpeg: 30-second windows of 16 kHz mono sound, and one frame per shot."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .extractors import ExtractError

SAMPLE_RATE = 16_000
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def ffmpeg() -> str:
    """$DIGUP_FFMPEG, the ffmpeg imageio-ffmpeg ships, or one on PATH."""
    chosen = os.environ.get("DIGUP_FFMPEG")
    if chosen:
        return chosen
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        found = shutil.which("ffmpeg")
        if found:
            return found
    raise ExtractError("ffmpeg not found")


@dataclass
class MediaPiece:
    path: str
    start: float
    end: float


@dataclass
class Probe:
    duration: float
    has_audio: bool
    has_video: bool


def probe(path: str) -> Probe:
    out = subprocess.run([ffmpeg(), "-hide_banner", "-i", path], capture_output=True, creationflags=NO_WINDOW)
    text = out.stderr.decode("utf-8", "replace")
    duration = 0.0
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if m:
        duration = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    has_audio = re.search(r"Stream #\S+.*: Audio:", text) is not None
    has_video = any("attached pic" not in line for line in re.findall(r"Stream #\S+.*: Video:.*", text))
    if not (has_audio or has_video) and "Invalid data" in text:
        raise ExtractError("unreadable: not audio or video")
    return Probe(duration, has_audio, has_video)


def write_wav(samples: np.ndarray, path: str) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(samples.astype("<i2").tobytes())


def read_wav(path: str) -> np.ndarray:
    """A WAV written by `write_wav` as float32 in [-1, 1]."""
    with wave.open(path, "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return data.astype(np.float32) / 32768.0


def audio_windows(path: str, window: float, hop: float, folder: Path, minimum: float = 2.0) -> tuple[list[MediaPiece], float]:
    """Decodes audio (a file, or a video's soundtrack) to 16 kHz mono and writes `window`-second WAV pieces `hop`
    seconds apart, skipping silent ones. Streams: memory stays at one window, even for a 3-hour podcast."""
    process = subprocess.Popen(
        [ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW,
    )
    window_n, hop_n = int(window * SAMPLE_RATE), int(hop * SAMPLE_RATE)
    buffer = np.zeros(0, dtype=np.int16)
    offset = 0
    pieces: list[MediaPiece] = []

    def emit(count: int) -> None:
        part = buffer[:count]
        rms = float(np.sqrt(np.mean((part.astype(np.float64) / 32768) ** 2))) if count else 0.0
        if rms <= 0.003:  # about -50 dBFS: silence isn't worth a vector
            return
        out = str(folder / (uuid.uuid4().hex + ".wav"))
        write_wav(part, out)
        pieces.append(MediaPiece(out, offset / SAMPLE_RATE, (offset + count) / SAMPLE_RATE))

    try:
        while True:
            data = process.stdout.read(SAMPLE_RATE * 2 * 4)
            if not data:
                break
            buffer = np.concatenate([buffer, np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2")])
            while len(buffer) >= window_n:
                emit(window_n)
                buffer = buffer[hop_n:]
                offset += hop_n
    finally:
        process.stdout.close()
        process.wait()
    covered = 0 if offset == 0 else window_n - hop_n
    if len(buffer) - covered >= int(minimum * SAMPLE_RATE):
        emit(len(buffer))
    duration = (offset + len(buffer)) / SAMPLE_RATE
    return pieces, duration


def average_hash(image: Image.Image) -> int:
    small = np.asarray(image.convert("L").resize((8, 8), Image.Resampling.BILINEAR), dtype=np.int32).ravel()
    mean = small.sum() // small.size
    value = 0
    for index, pixel in enumerate(small):
        if pixel > mean:
            value |= 1 << index
    return value


def keyframes(path: str, every: float, max_pixel: int, folder: Path, duration: float,
              max_frames: int = 1200) -> list[MediaPiece]:
    """A frame every `every` seconds, keeping one per shot: a frame that looks like the previous kept one (8x8 average
    hash within 4 bits) only extends it. Screen recordings and lectures shrink a lot."""
    if duration <= 0:
        return []
    step = max(every, duration / max_frames)
    frames_dir = folder / ("frames-" + uuid.uuid4().hex[:8])
    frames_dir.mkdir(parents=True, exist_ok=True)
    scale = f"scale='if(gt(iw,ih),min({max_pixel},iw),-2)':'if(gt(iw,ih),-2,min({max_pixel},ih))'"
    subprocess.run(
        [ffmpeg(), "-hide_banner", "-loglevel", "error", "-ss", f"{min(0.5, duration / 2):.3f}", "-i", path,
         "-an", "-sn", "-vf", f"fps=1/{step:.4f},{scale}", "-q:v", "3", str(frames_dir / "f%06d.jpg")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW,
    )
    start0 = min(0.5, duration / 2)
    frames: list[MediaPiece] = []
    last_hash = None
    for index, file in enumerate(sorted(frames_dir.glob("f*.jpg"))):
        t = start0 + index * step
        try:
            with Image.open(file) as image:
                h = average_hash(image)
        except Exception:
            continue
        if last_hash is not None and bin(h ^ last_hash).count("1") <= 4:
            file.unlink(missing_ok=True)
            continue
        last_hash = h
        frames.append(MediaPiece(str(file), t, duration))
    for a, b in zip(frames, frames[1:]):
        a.end = b.start
    return frames
