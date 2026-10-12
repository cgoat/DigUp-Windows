"""`digup`: the command-line tool. Downloads the model, indexes, searches, reports, and runs as the app's embedding
worker (`digup worker`)."""

from __future__ import annotations

import argparse
import os
import sys
import time

from . import downloads, paths
from .model import CodeQuery, FileKind, Prompts
from .settings import Settings


def _store(code: bool):
    from .store import CODE_FILE, MAIN_FILE, IndexStore

    return IndexStore(paths.index(), CODE_FILE if code else MAIN_FILE)


def cmd_download(args) -> int:
    backend = args.backend or downloads.best_backend()
    last = [0.0]

    def progress(label):
        def report(n, total):
            if time.time() - last[0] > 1 or n == total:
                last[0] = time.time()
                pct = f"{100 * n / total:.0f}%" if total else ""
                print(f"\r{label}: {n / 1e6:.0f} MB {pct}   ", end="", flush=True)
        return report

    if not downloads.runtime_ready(backend):
        downloads.download_runtime(backend, progress(f"llama.cpp {downloads.LLAMA_TAG} ({backend})"))
        print()
    print(f"llama.cpp {downloads.LLAMA_TAG} ({backend}): {downloads.runtime_folder(backend)}")
    if downloads.missing_models():
        downloads.download_models(progress("EmbeddingGemma 2"))
        print()
    print(f"model: {downloads.locate_models()[0].parent}")
    return 0


def cmd_worker(args) -> int:
    from .worker import serve

    return serve(args.text_only, args.backend)


def _make_embedder(args):
    if getattr(args, "in_process", False):
        from .llama import LlamaEmbedder

        return lambda: LlamaEmbedder(text_only=False, backend=args.backend)
    from .worker import EmbeddingWorker

    return lambda: EmbeddingWorker(text_only=False)


def cmd_index(args) -> int:
    from .indexer import Indexer

    settings = Settings.load()
    code = args.code
    roots = [paths.normalize(f) for f in (args.folders or (settings.code_folders if code else settings.folders))]
    if not roots:
        print("no folders: pass some, or pick them in the app", file=sys.stderr)
        return 2
    store = _store(code)
    options = settings.options(code)
    if args.reindex:
        store.reset_all()
    indexer = Indexer(store, options, _make_embedder(args))
    started = time.time()
    summary = indexer.sync(roots)
    c = summary.crawl
    import json

    store.set_meta("last_sync", json.dumps({
        "time": time.time(), "added": summary.added, "changed": summary.changed, "moved": summary.moved,
        "removed": summary.removed, "unchanged": summary.unchanged, "skipped": c.skipped,
        "skipped_folders": c.skipped_folders, "datasets": c.datasets, "missing": [r for r in roots if not os.path.isdir(r)],
    }))
    print(f"sync: {summary.added} new, {summary.changed} changed, {summary.moved} moved, {summary.removed} removed, "
          f"{summary.unchanged} unchanged ({time.time() - started:.1f} s)")
    if c.skipped:
        print("  skipped:", ", ".join(f"{k} {v}" for k, v in sorted(c.skipped.items(), key=lambda kv: -kv[1])))
    for folder, n in c.datasets.items():
        print(f"  dataset left out: {folder} ({n} files)")
    if args.sync_only:
        return 0

    def show(event):
        note = f"  {event.note}" if event.note else ""
        print(f"[{event.done}/{event.total}] {event.state:8} {event.kind.value:10} {event.seconds:5.2f}s  "
              f"{os.path.basename(event.path)}{note}", flush=True)

    try:
        states = indexer.run(limit=args.limit, on_event=show)
        if args.limit is None:
            states2 = indexer.finish_long_files(on_step=show)
            for k, v in states2.items():
                states[k] = states.get(k, 0) + v
    finally:
        indexer.finish()
    print("done:", ", ".join(f"{k} {v}" for k, v in states.items()) or "nothing to do",
          f"in {time.time() - started:.1f} s")
    return 0


def cmd_search(args) -> int:
    from .searcher import Searcher

    if args.image:
        return search_image(args)
    if not args.query:
        print("give some words, or --image PATH", file=sys.stderr)
        return 2
    text = " ".join(args.query)
    code_text = CodeQuery.text_of(text)
    code = code_text is not None
    query_text = code_text if code else text
    store = _store(code)
    searcher = Searcher(store)
    vector = None
    if query_text.strip():
        started = time.time()
        if args.in_process:
            from .llama import LlamaEmbedder

            embedder = LlamaEmbedder(text_only=True, backend=args.backend)
        else:
            from .worker import EmbeddingWorker

            embedder = EmbeddingWorker(text_only=True)
        prompt = Prompts.code_query(query_text) if code else Prompts.query(query_text)
        vector = embedder.embed_texts([prompt])[0]
        embedder.close()
        if args.verbose:
            print(f"(query encoded in {time.time() - started:.2f} s, {searcher.vector_count} vectors)")
    kinds = {FileKind(k) for k in args.kind} if args.kind else None
    started = time.time()
    hits = searcher.search(query_text, vector, kinds=kinds, limit=args.limit)
    from .searcher import strong_count

    strong = strong_count(hits)
    for i, hit in enumerate(hits):
        if i == strong:
            print("  -- weaker matches --")
        where = ""
        if hit.loc is not None:
            if hit.kind in (FileKind.AUDIO, FileKind.VIDEO):
                where = f" @ {int(hit.loc // 60)}:{int(hit.loc % 60):02d}"
            elif hit.kind == FileKind.CODE:
                where = f":{int(hit.focus or hit.loc)}"
            elif hit.kind == FileKind.PDF:
                where = f" p.{int(hit.loc)}"
        z = f"z={hit.z:.1f}" if hit.z is not None else ""
        kw = f"kw#{hit.keyword_rank}" if hit.keyword_rank else ""
        print(f"{i + 1:2}. {hit.score:5.2f} [{hit.kind.value}] {hit.path}{where}  {z} {kw}")
        if hit.excerpt and args.verbose:
            print(f"      {hit.excerpt[:200]}")
    if not hits:
        print("nothing found")
    if args.verbose:
        print(f"(searched in {(time.time() - started) * 1000:.0f} ms)")
    return 0


def search_image(args) -> int:
    """Pictures, video moments and pages that look like an image (any image, indexed or not)."""
    from .searcher import Searcher, strong_count

    path = paths.normalize(args.image)
    if args.in_process:
        from .llama import LlamaEmbedder

        embedder = LlamaEmbedder(text_only=False, backend=args.backend)
    else:
        from .worker import EmbeddingWorker

        embedder = EmbeddingWorker(text_only=False)
    result = embedder.embed_images([path], 280)[0]
    embedder.close()
    if result.vector is None:
        print(f"can't read {path}: {result.error}", file=sys.stderr)
        return 1
    kinds = {FileKind(k) for k in args.kind} if args.kind else None
    hits = [h for h in Searcher(_store(False)).search("", result.vector, kinds=kinds, limit=args.limit + 1)
            if h.path.lower() != path.lower()][:args.limit]
    strong = strong_count(hits)
    for i, hit in enumerate(hits):
        if i == strong:
            print("  -- weaker matches --")
        where = ""
        if hit.loc is not None and hit.kind in (FileKind.AUDIO, FileKind.VIDEO):
            where = f" @ {int(hit.loc // 60)}:{int(hit.loc % 60):02d}"
        elif hit.loc is not None and hit.kind == FileKind.PDF:
            where = f" p.{int(hit.loc)}"
        print(f"{i + 1:2}. {hit.score:5.2f} [{hit.kind.value}] {hit.path}{where}  z={hit.z:.1f}")
    if not hits:
        print("nothing looks like that")
    return 0


def cmd_status(args) -> int:
    settings = Settings.load()
    backend = downloads.best_backend()
    print(f"data:     {paths.home()}")
    print(f"runtime:  llama.cpp {downloads.LLAMA_TAG} {backend} "
          f"({'ready' if downloads.runtime_ready(backend) else 'not downloaded: digup download'})")
    found = downloads.locate_models()
    print(f"model:    {'complete' if downloads.models_complete() else 'missing: digup download'}"
          + (f" ({found[0].parent})" if found else ""))
    print(f"folders:  {', '.join(settings.folders) or '(none)'}")
    if settings.code_search:
        print(f"code:     {', '.join(settings.code_folders) or '(none)'}")
    from .store import CODE_FILE, MAIN_FILE, IndexStore

    for name in (MAIN_FILE, CODE_FILE):
        file = paths.index() / name
        if not file.exists():
            continue
        store = IndexStore(paths.index(), name)
        counts = store.counts()
        print(f"{name}: {IndexStore.size_of(file) / 1e6:.1f} MB, {counts['vectors']} vectors, "
              f"{counts['segments']} segments" + (f", {counts['unfinished']} long files to finish"
                                                   if counts['unfinished'] else ""))
        for kind, states in sorted(counts["by_kind_and_state"].items()):
            print(f"  {kind:10} " + ", ".join(f"{s} {n}" for s, n in sorted(states.items())))
        store.close()
    return 0


def cmd_folders(args) -> int:
    settings = Settings.load()
    target = settings.code_folders if args.code else settings.folders
    for folder in args.add or []:
        folder = paths.normalize(folder)
        if folder not in target:
            target.append(folder)
    for folder in args.remove or []:
        folder = paths.normalize(folder)
        target[:] = [f for f in target if f.lower() != folder.lower()]
    if args.code and args.add:
        settings.code_search = True
    settings.save()
    print("\n".join(target) or "(none)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="digup", description="Search the files on your PC by what's in them.")
    parser.add_argument("--backend", choices=list(downloads.BACKENDS), help="llama.cpp backend (default: best found)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download llama.cpp and the model (once)")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("index", help="index folders (default: the app's)")
    p.add_argument("folders", nargs="*")
    p.add_argument("--code", action="store_true", help="the code index")
    p.add_argument("--reindex", action="store_true", help="embed everything again")
    p.add_argument("--limit", type=int)
    p.add_argument("--sync-only", action="store_true")
    p.add_argument("--in-process", action="store_true", help="run the model in this process")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="search with words (start with 'code:' for code) or --image PATH")
    p.add_argument("query", nargs="*")
    p.add_argument("--image", help="search with a picture: things that look like it")
    p.add_argument("--kind", action="append", choices=[k.value for k in FileKind])
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--in-process", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("status", help="what's indexed")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("folders", help="list, add or remove the folders to search")
    p.add_argument("--add", nargs="*")
    p.add_argument("--remove", nargs="*")
    p.add_argument("--code", action="store_true")
    p.set_defaults(func=cmd_folders)

    p = sub.add_parser("worker", help="the embedding worker (JSON lines on stdin/stdout)")
    p.add_argument("--text-only", action="store_true")
    p.set_defaults(func=cmd_worker)

    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
