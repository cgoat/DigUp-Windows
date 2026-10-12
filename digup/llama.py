"""EmbeddingGemma 2 on llama.cpp's Windows DLLs (CUDA, Vulkan or CPU), through ctypes.

The same pipeline as the Mac app's LlamaEmbedder: the image and audio encoders (mtmd) feed the text model, and its
mean-pooled, L2-normalized output is the vector. Inputs are framed as the reference frames them: `<bos> text <eos>`,
`<bos> <|image> soft tokens <image|> <eos>` and the same for audio. The model is non-causal with no KV cache, so each
input, token ids and encoder rows together, goes through one `llama_process` call.

Images are decoded and resized here with Pillow's bicubic filter (what the reference uses, and what the Mac app ports
bit for bit), so mtmd takes them as they are. Audio arrives as 16 kHz mono and is cut at 30 s, as in the reference.

One thread at a time; `close()` frees the model (about 1 GB, or 0.3 GB text-only).
"""

from __future__ import annotations

import ctypes as C
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import downloads
from .model import DIM, Embedding, Prompts, WorkerError, WorkerInfo

BUDGETS = (70, 140, 280, 560, 1120)
# Which vectors this pipeline makes. Indexes record it; a change embeds every index again.
VECTOR_VERSION = 1
MAX_SEQUENCES = 16
# The DLLs reserve n_vocab floats (1 MiB) of pinned memory per token in a batch (the Mac build patches this out):
# texts are packed into batches of at most this many tokens, while one input may still use the whole context.
TEXT_BATCH_TOKENS = 512


def model_info(backend: str, text_only: bool = False, load_seconds: float | None = None) -> WorkerInfo:
    return WorkerInfo("google/embeddinggemma-2", downloads.REVISION, f"llama.cpp-win-vectors{VECTOR_VERSION}",
                      "text-q8_0+mmproj-q8_0", DIM, text_only, load_seconds)


# MARK: The C API (llama.h and mtmd.h at b11461)

class llama_model_params(C.Structure):
    _fields_ = [
        ("devices", C.c_void_p),
        ("tensor_buft_overrides", C.c_void_p),
        ("n_gpu_layers", C.c_int32),
        ("split_mode", C.c_int),
        ("load_mode", C.c_int),
        ("lazy_mode", C.c_int),
        ("main_gpu", C.c_int32),
        ("tensor_split", C.c_void_p),
        ("progress_callback", C.c_void_p),
        ("progress_callback_user_data", C.c_void_p),
        ("kv_overrides", C.c_void_p),
        ("vocab_only", C.c_bool),
        ("check_tensors", C.c_bool),
        ("use_extra_bufts", C.c_bool),
        ("no_host", C.c_bool),
        ("no_alloc", C.c_bool),
        ("load_mtp", C.c_bool),
    ]


class llama_context_params(C.Structure):
    _fields_ = [
        ("n_ctx", C.c_uint32),
        ("n_batch", C.c_uint32),
        ("n_ubatch", C.c_uint32),
        ("n_seq_max", C.c_uint32),
        ("n_rs_seq", C.c_uint32),
        ("n_outputs_max", C.c_uint32),
        ("n_outputs_max_per_seq", C.c_uint32),
        ("n_threads", C.c_int32),
        ("n_threads_batch", C.c_int32),
        ("ctx_type", C.c_int),
        ("rope_scaling_type", C.c_int),
        ("pooling_type", C.c_int),
        ("attention_type", C.c_int),
        ("flash_attn_type", C.c_int),
        ("rope_freq_base", C.c_float),
        ("rope_freq_scale", C.c_float),
        ("yarn_ext_factor", C.c_float),
        ("yarn_attn_factor", C.c_float),
        ("yarn_beta_fast", C.c_float),
        ("yarn_beta_slow", C.c_float),
        ("yarn_orig_ctx", C.c_uint32),
        ("defrag_thold", C.c_float),
        ("cb_eval", C.c_void_p),
        ("cb_eval_user_data", C.c_void_p),
        ("type_k", C.c_int),
        ("type_v", C.c_int),
        ("abort_callback", C.c_void_p),
        ("abort_callback_data", C.c_void_p),
        ("embeddings", C.c_bool),
        ("offload_kqv", C.c_bool),
        ("no_perf", C.c_bool),
        ("op_offload", C.c_bool),
        ("swa_full", C.c_bool),
        ("kv_unified", C.c_bool),
        ("samplers", C.c_void_p),
        ("n_samplers", C.c_size_t),
        ("ctx_other", C.c_void_p),
    ]


class mtmd_context_params(C.Structure):
    _fields_ = [
        ("use_gpu", C.c_bool),
        ("device", C.c_void_p),
        ("print_timings", C.c_bool),
        ("n_threads", C.c_int),
        ("image_marker", C.c_char_p),
        ("media_marker", C.c_char_p),
        ("flash_attn_type", C.c_int),
        ("warmup", C.c_bool),
        ("image_min_tokens", C.c_int),
        ("image_max_tokens", C.c_int),
        ("cb_eval", C.c_void_p),
        ("cb_eval_user_data", C.c_void_p),
        ("batch_max_tokens", C.c_int32),
        ("progress_callback", C.c_void_p),
        ("progress_callback_user_data", C.c_void_p),
    ]


class mtmd_input_text(C.Structure):
    _fields_ = [("text", C.c_char_p), ("text_len", C.c_size_t), ("add_special", C.c_bool),
                ("parse_special", C.c_bool)]


class llama_embd(C.Structure):
    _fields_ = [("data", C.POINTER(C.c_float)), ("n_rows", C.c_size_t), ("n_embd", C.c_size_t)]


LLAMA_POOLING_TYPE_MEAN = 1
LLAMA_PROCESS_TYPE_ENCODE = 0
MTMD_INPUT_CHUNK_TYPE_TEXT = 0
GGML_LOG_LEVEL_WARN, GGML_LOG_LEVEL_CONT = 3, 5

LOG_CALLBACK = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_void_p)


class Library:
    """llama.dll, mtmd.dll and ggml.dll from one runtime folder, loaded once per process."""

    _shared: "Library | None" = None
    _lock = threading.Lock()

    @classmethod
    def load(cls, folder: Path) -> "Library":
        with cls._lock:
            if cls._shared is None:
                cls._shared = Library(folder)
            return cls._shared

    def __init__(self, folder: Path):
        self.folder = folder
        if hasattr(os, "add_dll_directory"):
            self._dll_dir = os.add_dll_directory(str(folder))
        os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")
        self.ggml = C.CDLL(str(folder / "ggml.dll"))
        self.llama = C.CDLL(str(folder / "llama.dll"))
        self.mtmd = C.CDLL(str(folder / "mtmd.dll"))
        self._log_level = 2
        self._log_file = None
        self._callback = LOG_CALLBACK(self._log)
        self._declare()
        self.llama.llama_log_set(self._callback, None)
        self.mtmd.mtmd_helper_log_set(self._callback, None)
        # The release builds load their backends (CUDA, Vulkan, the best CPU variant) as DLLs at run time.
        self.ggml.ggml_backend_load_all_from_path(str(folder).encode())
        self.llama.llama_backend_init()

    def log_to(self, path: Path | None) -> None:
        self._log_file = open(path, "a", encoding="utf-8") if path else None

    def _log(self, level: int, text: bytes, _user) -> None:
        if level != GGML_LOG_LEVEL_CONT:
            self._log_level = level
        if self._log_level >= GGML_LOG_LEVEL_WARN and text:
            out = self._log_file or sys.stderr
            try:
                out.write(text.decode("utf-8", "replace"))
                out.flush()
            except Exception:
                pass

    def _declare(self) -> None:
        L, M, G = self.llama, self.mtmd, self.ggml
        vp, i32, sz, b = C.c_void_p, C.c_int32, C.c_size_t, C.c_bool

        def f(lib, name, res, *args):
            fn = getattr(lib, name)
            fn.restype = res
            fn.argtypes = list(args)

        f(G, "ggml_backend_load_all_from_path", None, C.c_char_p)
        f(L, "llama_log_set", None, LOG_CALLBACK, vp)
        f(M, "mtmd_helper_log_set", None, LOG_CALLBACK, vp)
        f(L, "llama_backend_init", None)
        f(L, "llama_model_default_params", llama_model_params)
        f(L, "llama_context_default_params", llama_context_params)
        f(L, "llama_model_load_from_file", vp, C.c_char_p, llama_model_params)
        f(L, "llama_model_free", None, vp)
        f(L, "llama_init_from_model", vp, vp, llama_context_params)
        f(L, "llama_free", None, vp)
        f(L, "llama_model_get_vocab", vp, vp)
        f(L, "llama_model_n_embd_out", i32, vp)
        f(L, "llama_model_n_embd_inp", i32, vp)
        f(L, "llama_vocab_eos", i32, vp)
        f(L, "llama_tokenize", i32, vp, C.c_char_p, i32, C.POINTER(i32), i32, b, b)
        f(L, "llama_batch_ext_init", vp, vp)
        f(L, "llama_batch_ext_free", None, vp)
        f(L, "llama_batch_ext_clear", None, vp)
        f(L, "llama_batch_ext_add_token", i32, vp, i32, i32)
        f(L, "llama_batch_ext_add_embd", i32, vp, i32, llama_embd)
        f(L, "llama_batch_ext_set_pos", b, vp, i32, C.POINTER(i32))
        f(L, "llama_batch_ext_set_output_embd", b, vp, i32, b)
        f(L, "llama_process", i32, vp, C.c_int, vp)
        f(L, "llama_get_embeddings_seq", C.POINTER(C.c_float), vp, i32)
        f(M, "mtmd_context_params_default", mtmd_context_params)
        f(M, "mtmd_init_from_file", vp, C.c_char_p, vp, mtmd_context_params)
        f(M, "mtmd_free", None, vp)
        f(M, "mtmd_default_marker", C.c_char_p)
        f(M, "mtmd_get_audio_sample_rate", C.c_int, vp)
        f(M, "mtmd_bitmap_init", vp, C.c_uint32, C.c_uint32, C.POINTER(C.c_ubyte))
        f(M, "mtmd_bitmap_init_from_audio", vp, sz, C.POINTER(C.c_float))
        f(M, "mtmd_bitmap_free", None, vp)
        f(M, "mtmd_input_chunks_init", vp)
        f(M, "mtmd_input_chunks_free", None, vp)
        f(M, "mtmd_input_chunks_size", sz, vp)
        f(M, "mtmd_input_chunks_get", vp, vp, sz)
        f(M, "mtmd_input_chunk_get_type", C.c_int, vp)
        f(M, "mtmd_input_chunk_get_tokens_text", C.POINTER(i32), vp, C.POINTER(sz))
        f(M, "mtmd_input_chunk_get_n_tokens", sz, vp)
        f(M, "mtmd_tokenize", i32, vp, vp, C.POINTER(mtmd_input_text), C.POINTER(vp), sz)
        f(M, "mtmd_encode_chunk", i32, vp, vp)
        f(M, "mtmd_get_output_embd", C.POINTER(C.c_float), vp)


class InputError(Exception):
    """A problem with one input: it gets no vector, and its neighbours go on."""


def target_size(height: int, width: int, budget: int) -> tuple[int, int] | None:
    """transformers' Gemma image processor: scale to fill `budget` soft tokens (each 3x3 patches of 16 px) keeping
    the aspect ratio, each side rounded down to a multiple of 48 px."""
    patch, pooling = 16, 3
    max_patches = budget * pooling * pooling
    factor = math.sqrt(max_patches * patch * patch / (height * width))
    side = pooling * patch
    th = int(math.floor(factor * height / side)) * side
    tw = int(math.floor(factor * width / side)) * side
    if th == 0 and tw == 0:
        return None
    max_side = (max_patches // (pooling * pooling)) * side
    if th == 0:
        th, tw = side, min(width // height * side, max_side)
    elif tw == 0:
        tw, th = side, min(height // width * side, max_side)
    return th, tw


class LlamaEmbedder:
    def __init__(self, text_only: bool, backend: str | None = None, log: Path | None = None):
        started = time.time()
        self.backend = backend or downloads.best_backend()
        folder = downloads.runtime_folder(self.backend)
        if not downloads.runtime_ready(self.backend):
            raise WorkerError(f"llama.cpp {downloads.LLAMA_TAG} ({self.backend}) isn't downloaded yet")
        found = downloads.locate_models()
        if not found:
            raise WorkerError(f"{downloads.TEXT.name} isn't downloaded yet")
        text_path, media_path = found
        if not text_only and media_path is None:
            raise WorkerError(f"{downloads.MEDIA.name} (images and audio) isn't downloaded yet")
        self.lib = lib = Library.load(folder)
        lib.log_to(log)
        L, M = lib.llama, lib.mtmd

        model_params = L.llama_model_default_params()
        model_params.n_gpu_layers = 0 if self.backend == "cpu" else 999
        self.model = L.llama_model_load_from_file(str(text_path).encode("utf-8"), model_params)
        if not self.model:
            raise WorkerError(f"can't load {text_path}")
        # One input must fit one micro-batch: 2048 tokens (the reference's cut), which also holds a 1120-token image or
        # a 750-token 30 s clip. Queries are short, so the query encoder reserves less and cuts at 512.
        self.max_tokens = 512 if text_only else 2048
        params = L.llama_context_default_params()
        params.n_ctx = params.n_batch = params.n_ubatch = self.max_tokens
        params.n_seq_max = MAX_SEQUENCES
        params.kv_unified = True
        params.embeddings = True
        params.pooling_type = LLAMA_POOLING_TYPE_MEAN
        threads = max(1, (os.cpu_count() or 4) // 2)
        params.n_threads = params.n_threads_batch = threads
        self.context = L.llama_init_from_model(self.model, params)
        if not self.context:
            L.llama_model_free(self.model)
            raise WorkerError("can't create a llama.cpp context")
        self.media = None
        if not text_only:
            mp = M.mtmd_context_params_default()
            mp.use_gpu = self.backend != "cpu"
            mp.print_timings = False
            mp.warmup = True
            mp.n_threads = threads
            mp.image_min_tokens = 1  # never rescale: images arrive at their final size
            mp.image_max_tokens = 1120
            self.media = M.mtmd_init_from_file(str(media_path).encode("utf-8"), self.model, mp)
            if not self.media:
                self.close()
                raise WorkerError(f"can't load {media_path}")
            self.sample_rate = M.mtmd_get_audio_sample_rate(self.media) or 16000
        self.batch = L.llama_batch_ext_init(self.context)
        self.vocab = L.llama_model_get_vocab(self.model)
        self.embedding_size = L.llama_model_n_embd_out(self.model)
        self.row_size = L.llama_model_n_embd_inp(self.model)
        self.sequences = 0
        self.tokens = 0
        self.position = 0
        self.text_only = text_only
        for _ in range(2):  # the first calls build the GPU pipelines: pay that now
            self.embed_texts([Prompts.query("warm up")])
        self.info = model_info(self.backend, text_only, round(time.time() - started, 2))

    def close(self) -> None:
        L, M = self.lib.llama, self.lib.mtmd
        if getattr(self, "batch", None):
            L.llama_batch_ext_free(self.batch)
            self.batch = None
        if getattr(self, "media", None):
            M.mtmd_free(self.media)
            self.media = None
        if getattr(self, "context", None):
            L.llama_free(self.context)
            self.context = None
        if getattr(self, "model", None):
            L.llama_model_free(self.model)
            self.model = None

    # MARK: Embedder

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self._reset()
        vectors: list[list[float]] = []
        for text in texts:
            tokens = self._tokenize(text)
            if self.sequences and (self.sequences == MAX_SEQUENCES
                                   or self.tokens + len(tokens) > min(TEXT_BATCH_TOKENS, self.max_tokens)):
                vectors += self._run()
            self._begin()
            for token in tokens:
                self._add_token(token)
        if self.sequences:
            vectors += self._run()
        return vectors

    def embed_images(self, paths: list[str], budget: int) -> list[Embedding]:
        if not self.media:
            raise WorkerError("this embedder was started text-only")
        if budget not in BUDGETS:
            raise WorkerError(f"budget must be one of {BUDGETS}")
        return [self._embedding(lambda p=p: self._embed_media(self._image_bitmap(p, budget), p)) for p in paths]

    def embed_audio(self, paths: list[str]) -> list[Embedding]:
        if not self.media:
            raise WorkerError("this embedder was started text-only")
        return [self._embedding(lambda p=p: self._embed_media(self._audio_bitmap(p), p)) for p in paths]

    @staticmethod
    def _embedding(make) -> Embedding:
        try:
            return Embedding(make())
        except (InputError, OSError, ValueError) as error:
            return Embedding(None, str(error)[:300])

    # MARK: Media

    def _image_bitmap(self, path: str, budget: int):
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
        except Exception as error:
            raise InputError(f"can't read {path}: {error}") from error
        target = target_size(rgb.height, rgb.width, budget)
        if target is None:
            raise InputError(f"image too thin to embed: {path}")
        if target != (rgb.height, rgb.width):
            rgb = rgb.resize((target[1], target[0]), Image.Resampling.BICUBIC)
        data = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        bitmap = self.lib.mtmd.mtmd_bitmap_init(rgb.width, rgb.height, data.ctypes.data_as(C.POINTER(C.c_ubyte)))
        if not bitmap:
            raise InputError(f"can't make a bitmap of {path}")
        return bitmap

    def _audio_bitmap(self, path: str):
        samples = load_audio(path, self.sample_rate)[: 30 * self.sample_rate]
        if samples.size == 0:
            raise InputError(f"no sound in {path}")
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        bitmap = self.lib.mtmd.mtmd_bitmap_init_from_audio(samples.size, samples.ctypes.data_as(C.POINTER(C.c_float)))
        if not bitmap:
            raise InputError(f"can't make audio of {path}")
        return bitmap

    def _embed_media(self, bitmap, path: str) -> list[float]:
        M = self.lib.mtmd
        chunks = M.mtmd_input_chunks_init()
        try:
            marker = M.mtmd_default_marker()
            text = mtmd_input_text(marker, len(marker), True, True)
            bitmaps = (C.c_void_p * 1)(bitmap)
            status = M.mtmd_tokenize(self.media, chunks, C.byref(text), bitmaps, 1)
            if status != 0:
                raise InputError(f"can't tokenize {path} ({status})")
            self._reset()
            self._begin()
            for index in range(M.mtmd_input_chunks_size(chunks)):
                chunk = M.mtmd_input_chunks_get(chunks, index)
                if M.mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT:
                    count = C.c_size_t(0)
                    tokens = M.mtmd_input_chunk_get_tokens_text(chunk, C.byref(count))
                    for i in range(count.value):
                        self._add_token(tokens[i])
                else:
                    if M.mtmd_encode_chunk(self.media, chunk) != 0:
                        raise InputError(f"the encoder failed on {path}")
                    rows = M.mtmd_get_output_embd(self.media)
                    if not rows:
                        raise InputError(f"the encoder failed on {path}")
                    self._add_rows(rows, M.mtmd_input_chunk_get_n_tokens(chunk))
            return self._run()[0]
        except InputError:
            self._reset()
            raise
        finally:
            M.mtmd_input_chunks_free(chunks)
            M.mtmd_bitmap_free(bitmap)

    # MARK: Text model

    def _tokenize(self, text: str) -> list[int]:
        """`<bos> text <eos>`, cut to `max_tokens` the way transformers cuts (keeping both special tokens)."""
        L = self.lib.llama
        data = text.encode("utf-8")
        capacity = len(data) + 8
        buffer = (C.c_int32 * capacity)()
        count = L.llama_tokenize(self.vocab, data, len(data), buffer, capacity, True, True)
        if count < 0:
            capacity = -count
            buffer = (C.c_int32 * capacity)()
            count = L.llama_tokenize(self.vocab, data, len(data), buffer, capacity, True, True)
        tokens = list(buffer[: max(count, 0)])
        if len(tokens) > self.max_tokens:
            tokens = tokens[: self.max_tokens - 1] + [L.llama_vocab_eos(self.vocab)]
        return tokens

    def _begin(self) -> None:
        self.sequences += 1
        self.position = 0

    def _reset(self) -> None:
        self.lib.llama.llama_batch_ext_clear(self.batch)
        self.sequences = 0
        self.tokens = 0

    def _add_token(self, token: int) -> None:
        self._place(self.lib.llama.llama_batch_ext_add_token(self.batch, self.sequences - 1, token))

    def _add_rows(self, rows, count: int) -> None:
        base = C.cast(rows, C.c_void_p).value
        step = self.row_size * C.sizeof(C.c_float)
        for row in range(count):
            embd = llama_embd(C.cast(base + row * step, C.POINTER(C.c_float)), 1, self.row_size)  # copied by the batch
            self._place(self.lib.llama.llama_batch_ext_add_embd(self.batch, self.sequences - 1, embd))

    def _place(self, index: int) -> None:
        if index < 0:
            raise InputError(f"batch full ({index})")
        pos = (C.c_int32 * 4)(self.position, 0, 0, 0)
        L = self.lib.llama
        if not (L.llama_batch_ext_set_pos(self.batch, index, pos) and
                L.llama_batch_ext_set_output_embd(self.batch, index, True)):
            raise InputError(f"can't place token {index}")
        self.position += 1
        self.tokens += 1

    def _run(self) -> list[list[float]]:
        """Runs the text model over the batch: one L2-normalized vector per input, in order."""
        L = self.lib.llama
        try:
            status = L.llama_process(self.context, LLAMA_PROCESS_TYPE_ENCODE, self.batch)
            if status != 0:
                raise WorkerError(f"llama_process failed ({status})")
            out = []
            for sequence in range(self.sequences):
                pooled = L.llama_get_embeddings_seq(self.context, sequence)
                if not pooled:
                    raise WorkerError(f"no pooled embedding for input {sequence}")
                vector = np.ctypeslib.as_array(pooled, shape=(self.embedding_size,)).astype(np.float64)
                norm = float(np.sqrt((vector * vector).sum()))
                out.append((vector / norm if norm > 0 else vector).astype(np.float32).tolist())
            return out
        finally:
            self._reset()


def load_audio(path: str, rate: int = 16000) -> np.ndarray:
    """Mono float32 at `rate`: our own 16 kHz WAVs directly, anything else through ffmpeg."""
    if path.lower().endswith(".wav"):
        import wave

        try:
            with wave.open(path, "rb") as w:
                if w.getframerate() == rate and w.getnchannels() == 1 and w.getsampwidth() == 2:
                    return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        except wave.Error:
            pass
    from .media import ffmpeg

    out = subprocess.run(
        [ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(rate), "-t", "30",
         "-f", "f32le", "-"],
        capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if out.returncode != 0:
        raise InputError(f"can't read audio {path}")
    return np.frombuffer(out.stdout, dtype="<f4").copy()
