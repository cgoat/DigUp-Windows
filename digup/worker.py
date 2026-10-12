"""The embedding worker: the model lives in its own process, so its memory (about 1 GB, and the GPU's) goes back to
the system when it exits. The protocol is the Mac app's (worker/digup_worker.py there):

  {"id": 1, "op": "info"}
  {"id": 2, "op": "text",  "inputs": ["task: search result | query: cat", ...]}
  {"id": 3, "op": "image", "inputs": ["C:\\\\...\\\\a.jpg", ...], "budget": 280}
  {"id": 4, "op": "audio", "inputs": ["C:\\\\...\\\\a.wav", ...]}
  {"id": 5, "op": "quit"}

The first line the worker prints is {"ready": true, "info": {...}} once the model is loaded. Replies:
{"id": 2, "ok": true, "dim": 768, "count": N, "vectors": "<base64 float32 LE, N x dim>", "errors": [...]}.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

import numpy as np

from . import paths
from .model import DIM, Embedding, WorkerError, WorkerInfo

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)


def serve(text_only: bool, backend: str | None = None) -> int:
    """`digup worker`: reads requests on stdin, answers on stdout."""
    out = sys.stdout.buffer
    stdin = sys.stdin.buffer

    def reply(obj) -> None:
        out.write(json.dumps(obj).encode() + b"\n")
        out.flush()

    try:
        from .llama import LlamaEmbedder

        embedder = LlamaEmbedder(text_only=text_only, backend=backend)
    except Exception as error:  # noqa: BLE001 - reported to the client
        reply({"ready": False, "error": str(error)})
        return 1
    info = embedder.info
    reply({"ready": True, "info": info.__dict__})
    for line in stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except ValueError:
            reply({"ok": False, "error": "bad request"})
            continue
        op, rid, inputs = request.get("op"), request.get("id"), request.get("inputs") or []
        if op == "quit":
            break
        try:
            if op == "info":
                reply({"id": rid, "ok": True, "info": info.__dict__})
                continue
            if op == "text":
                vectors = [Embedding(v) for v in embedder.embed_texts(inputs)]
            elif op == "image":
                vectors = embedder.embed_images(inputs, int(request.get("budget") or 280))
            elif op == "audio":
                vectors = embedder.embed_audio(inputs)
            else:
                reply({"id": rid, "ok": False, "error": f"unknown op {op}"})
                continue
            matrix = np.zeros((len(vectors), DIM), dtype="<f4")
            for i, e in enumerate(vectors):
                if e.vector is not None:
                    matrix[i] = e.vector
            reply({"id": rid, "ok": True, "dim": DIM, "count": len(vectors),
                   "vectors": base64.b64encode(matrix.tobytes()).decode(), "errors": [e.error for e in vectors]})
        except Exception as error:  # noqa: BLE001
            reply({"id": rid, "ok": False, "error": str(error)})
    embedder.close()
    return 0


def worker_command(text_only: bool) -> list[str]:
    """How to start a worker: the frozen app's own exe, or this Python."""
    if getattr(sys, "frozen", False):
        base = [sys.executable]
        helper = Path(sys.executable).with_name("digup.exe")
        if helper.exists():
            base = [str(helper)]
        args = base + ["worker"]
    else:
        args = [sys.executable, "-m", "digup.cli", "worker"]
    return args + (["--text-only"] if text_only else [])


class EmbeddingWorker:
    """Talks to a worker process over stdin/stdout, one JSON line each way."""

    def __init__(self, text_only: bool, background: bool = False, log: Path | None = None):
        log = log or paths.logs() / ("query-worker.log" if text_only else "index-worker.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log = log
        self._stderr = open(log, "ab")
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        self.process = subprocess.Popen(
            worker_command(text_only), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr, env=env,
            creationflags=NO_WINDOW | (BELOW_NORMAL if background else 0),
        )
        self._next = 1
        self._lock = threading.Lock()
        ready = self._read()
        if not ready.get("ready"):
            self.close()
            raise WorkerError(ready.get("error") or f"didn't start{self._tail()}")
        self.info = WorkerInfo(**ready["info"])

    def _read(self) -> dict:
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise WorkerError(f"worker stopped{self._tail()}")
            if line.strip():
                return json.loads(line)

    def _tail(self) -> str:
        try:
            self._stderr.flush()
            lines = deque(Path(self.log).read_text("utf-8", "replace").splitlines(), maxlen=6)
            return f"; worker log ({self.log}):\n  " + "\n  ".join(lines) if lines else ""
        except OSError:
            return ""

    def _send(self, op: str, inputs: list[str], budget: int | None = None) -> list[Embedding]:
        if not inputs:
            return []
        with self._lock:
            rid = self._next
            self._next += 1
            request = {"id": rid, "op": op, "inputs": inputs}
            if budget is not None:
                request["budget"] = budget
            try:
                self.process.stdin.write(json.dumps(request).encode() + b"\n")
                self.process.stdin.flush()
            except OSError as error:
                raise WorkerError(f"can't write to the worker{self._tail()}") from error
            reply = self._read()
        if reply.get("id") != rid:
            raise WorkerError("reply out of order")
        if not reply.get("ok"):
            raise WorkerError(reply.get("error") or "no vectors")
        dim = reply.get("dim", DIM)
        floats = np.frombuffer(base64.b64decode(reply["vectors"]), dtype="<f4").reshape(-1, dim)
        errors = reply.get("errors") or [None] * len(inputs)
        return [Embedding(None if errors[i] else floats[i].tolist(), errors[i]) for i in range(len(inputs))]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [e.vector or [] for e in self._send("text", texts)]

    def embed_images(self, paths_: list[str], budget: int) -> list[Embedding]:
        return self._send("image", paths_, budget)

    def embed_audio(self, paths_: list[str]) -> list[Embedding]:
        return self._send("audio", paths_)

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.process.stdin.write(b'{"id": 0, "op": "quit"}\n')
                self.process.stdin.flush()
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
        try:
            self._stderr.close()
        except Exception:
            pass
