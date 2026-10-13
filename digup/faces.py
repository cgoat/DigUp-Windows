"""Faces: who is in a photo, so a search can find a person rather than people who look alike.

EmbeddingGemma 2 knows what a picture shows, not who: two people with the same hair and setting are neighbours to
it. Faces need a model trained on identities. This is InsightFace's buffalo_l pack, two of its models:

- SCRFD-10GF (`det_10g.onnx`) finds faces and five landmarks (eyes, nose, mouth corners);
- ArcFace ResNet-50 trained on WebFace600K (`w600k_r50.onnx`) turns each face, aligned to the landmarks, into 512
  numbers: faces of one person land close together whatever the pose, light or age.

They run on ONNX Runtime with DirectML (any GPU on Windows), or the CPU. The pre- and post-processing follows
InsightFace's own (scrfd.py, face_align.py, arcface_onnx.py), so the vectors are the ones its benchmarks measure.

The models are licensed for non-commercial research only (InsightFace's terms), so DigUp never downloads them by
itself: the person asks for them in Settings → People, after reading that.
"""

from __future__ import annotations

import hashlib

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import paths

PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
PACK_BYTES = 288_621_354
PACK_SHA256 = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
DETECTOR = ("det_10g.onnx", "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91")
RECOGNIZER = ("w600k_r50.onnx", "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43")
LICENSE = (
    "InsightFace's pretrained models (buffalo_l: SCRFD and ArcFace, trained on WebFace600K) are available for "
    "non-commercial research purposes only, under InsightFace's terms (https://github.com/deepinsight/insightface). "
    "DigUp runs them on this PC only, on your own photos; nothing is uploaded."
)
FACE_DIM = 512
# Two faces of one person: cosine of their ArcFace vectors. 0.4 is InsightFace's usual cut for these models; grouping
# into people is stricter (`people.py`) so that a group is one person.
SAME_PERSON = 0.40
MIN_FACE = 36  # pixels across in the photo: smaller faces are too blurry to tell people apart
DET_SIZE = 640
DET_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
# Where ArcFace expects the five landmarks in its 112x112 input (InsightFace's arcface_dst).
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def folder() -> Path:
    return paths.home() / "Faces"


def models_ready() -> bool:
    return all((folder() / name).exists() for name, _ in (DETECTOR, RECOGNIZER))


def download(progress=None, should_stop=lambda: False) -> None:
    """Gets buffalo_l (289 MB), checks it, and keeps the two models DigUp uses. Only when the person asked."""
    from .downloads import fetch, sha256

    target = folder()
    target.mkdir(parents=True, exist_ok=True)
    archive = fetch(PACK_URL, target / "buffalo_l.zip", PACK_BYTES, progress, should_stop)
    if sha256(archive) != PACK_SHA256:
        archive.unlink()
        raise IOError("the face models didn't match their checksum; they were removed, try again")
    with zipfile.ZipFile(archive) as z:
        for name, digest in (DETECTOR, RECOGNIZER):
            data = z.read(name)
            if hashlib.sha256(data).hexdigest() != digest:
                raise IOError(f"{name} didn't match its checksum")
            (target / name).write_bytes(data)
    archive.unlink()


def remove() -> None:
    for name, _ in (DETECTOR, RECOGNIZER):
        (folder() / name).unlink(missing_ok=True)


@dataclass
class Face:
    box: tuple[float, float, float, float]  # x, y, width, height, as fractions of the photo
    score: float  # how sure the detector is that it's a face
    vector: np.ndarray  # 512 float32, L2-normalized
    size: int  # pixels across in the photo


def _session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    providers = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    try:
        return ort.InferenceSession(str(path), options, providers=providers)
    except Exception:  # noqa: BLE001 - a GPU without DirectML support: the CPU then
        return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class FaceModels:
    """Both models, loaded once (in the embedding worker: a process of its own, as the big model is)."""

    def __init__(self):
        if not models_ready():
            raise FileNotFoundError("the face models aren't downloaded (Settings → People)")
        self.detector = _session(folder() / DETECTOR[0])
        self.recognizer = _session(folder() / RECOGNIZER[0])
        self.det_input = self.detector.get_inputs()[0].name
        self.det_outputs = [o.name for o in self.detector.get_outputs()]
        self.rec_input = self.recognizer.get_inputs()[0].name
        self._anchors: dict[tuple[int, int, int], np.ndarray] = {}

    @property
    def provider(self) -> str:
        return self.detector.get_providers()[0]

    # MARK: Detection (SCRFD)

    def detect(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Faces in a BGR image: (N x 5 boxes with scores, N x 5 x 2 landmarks), in the image's pixels."""
        import cv2

        h, w = bgr.shape[:2]
        if h / w > 1:
            nh, nw = DET_SIZE, int(DET_SIZE * w / h)
        else:
            nw, nh = DET_SIZE, int(DET_SIZE * h / w)
        scale = nh / h
        canvas = np.zeros((DET_SIZE, DET_SIZE, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (max(nw, 1), max(nh, 1)))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (DET_SIZE, DET_SIZE), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.detector.run(self.det_outputs, {self.det_input: blob})
        boxes, points, scores = [], [], []
        for i, stride in enumerate((8, 16, 32)):
            score, box, kps = outs[i][:, 0], outs[i + 3] * stride, outs[i + 6] * stride
            centers = self._centers(DET_SIZE // stride, DET_SIZE // stride, stride)
            keep = np.where(score >= DET_THRESHOLD)[0]
            if not keep.size:
                continue
            c = centers[keep]
            b = box[keep]
            boxes.append(np.stack([c[:, 0] - b[:, 0], c[:, 1] - b[:, 1], c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]], 1))
            k = kps[keep].reshape(-1, 5, 2) + c[:, None, :]
            points.append(k)
            scores.append(score[keep])
        if not boxes:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        boxes = np.concatenate(boxes) / scale
        points = np.concatenate(points) / scale
        scores = np.concatenate(scores)
        order = scores.argsort()[::-1]
        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)[order]
        points = points[order]
        keep = _nms(dets, NMS_THRESHOLD)
        return dets[keep], points[keep]

    def _centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        if key not in self._anchors:
            grid = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            centers = (grid * stride).reshape(-1, 2)
            self._anchors[key] = np.stack([centers, centers], axis=1).reshape(-1, 2)  # two anchors a place
        return self._anchors[key]

    # MARK: Recognition (ArcFace)

    def embed(self, bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """ArcFace vectors (N x 512, L2-normalized) for the faces whose five landmarks are given."""
        import cv2

        crops = [_align(bgr, k) for k in landmarks]
        if not crops:
            return np.zeros((0, FACE_DIM), np.float32)
        blob = cv2.dnn.blobFromImages(crops, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        vectors = np.concatenate([self.recognizer.run(None, {self.rec_input: blob[i:i + 1]})[0]
                                  for i in range(len(crops))])
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-6)

    def faces(self, path: str, max_side: int = 1600) -> list[Face]:
        """Every face worth telling apart in the photo at `path`."""
        bgr = load_bgr(path, max_side)
        if bgr is None:
            return []
        h, w = bgr.shape[:2]
        dets, landmarks = self.detect(bgr)
        sizes = np.maximum(dets[:, 2] - dets[:, 0], dets[:, 3] - dets[:, 1]) if len(dets) else np.zeros(0)
        keep = np.where(sizes >= MIN_FACE * max(1.0, max(h, w) / max_side))[0] if len(dets) else []
        if not len(keep):
            return []
        vectors = self.embed(bgr, landmarks[keep])
        out = []
        for j, i in enumerate(keep):
            x1, y1, x2, y2, score = dets[i]
            out.append(Face((max(0.0, x1 / w), max(0.0, y1 / h), (x2 - x1) / w, (y2 - y1) / h), float(score),
                            vectors[j].astype(np.float32), int(sizes[i])))
        return out


def _nms(dets: np.ndarray, threshold: float) -> list[int]:
    x1, y1, x2, y2, scores = dets.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(overlap <= threshold)[0] + 1]
    return keep


def _align(bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """The face turned and scaled so its landmarks sit where ArcFace expects them (a similarity transform)."""
    import cv2

    matrix, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), ARCFACE_DST, method=cv2.LMEDS)
    if matrix is None:
        matrix, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), ARCFACE_DST)
    return cv2.warpAffine(bgr, matrix, (112, 112), borderValue=0.0)


def load_bgr(path: str, max_side: int = 1600) -> np.ndarray | None:
    """A photo as BGR pixels, turned upright (EXIF) and at most `max_side` across (HEIC and RAW through Pillow)."""
    from PIL import Image

    from .extractors import open_image, to_rgb

    try:
        image = to_rgb(open_image(path))
    except Exception:  # noqa: BLE001 - unreadable: no faces
        return None
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.ascontiguousarray(np.asarray(image)[:, :, ::-1])


def crop(path: str, box: tuple[float, float, float, float], size: int = 160) -> "np.ndarray | None":
    """A square around a face, a little wider than the box, for showing it (RGB)."""
    bgr = load_bgr(path, 2000)
    if bgr is None:
        return None
    import cv2

    h, w = bgr.shape[:2]
    x, y, bw, bh = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    side = max(bw, bh) * 1.5
    cx, cy = x + bw / 2, y + bh / 2
    x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x2, y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(bgr[y1:y2, x1:x2], (size, size))[:, :, ::-1].copy()


def half(vector) -> bytes:
    return np.asarray(vector, dtype=np.float32).astype(np.float16).tobytes()


__all__ = ["FaceModels", "Face", "models_ready", "download", "LICENSE", "SAME_PERSON", "FACE_DIM", "crop", "half"]
