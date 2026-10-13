# DigUp for Windows: design

This port keeps the Mac app's engine, its measured choices and its index layout, and replaces the pieces that only
exist on macOS. Where a number or a rule below has a reason, the reason was measured in the original
([ARahim3/DigUp](https://github.com/ARahim3/DigUp)) and is cited in the code as it is there.

## Pieces

| | macOS (Swift) | Windows (Python) |
|---|---|---|
| Embedding runtime | llama.cpp b11461 built from source, Metal, linked into the `digup` helper | llama.cpp b11461 **release DLLs** (CUDA 12.4, Vulkan or CPU), downloaded on first run, loaded with ctypes ([llama.py](digup/llama.py)) |
| Model process | `digup worker`, JSON lines on stdin/stdout | the same protocol ([worker.py](digup/worker.py)), `digup-cli.exe worker` |
| Images | ImageIO | Pillow (+ pillow-heif) |
| Image resize for the encoder | a Swift port of Pillow's bicubic | Pillow's bicubic itself |
| OCR | Apple Vision | Windows.Media.Ocr (built into Windows 10/11, the user's languages) |
| PDFs | PDFKit | PDFium (pypdfium2); scanned pages are read by OCR too |
| Audio, video | AVFoundation | ffmpeg (bundled through imageio-ffmpeg) |
| Documents | Cocoa text system | own readers for docx, odt, rtf, html, txt/md |
| Index | SQLite + FTS5, fp16 vectors | the same schema and tokenizer, Python's sqlite3 |
| Vector search | Accelerate (vDSP) | NumPy |
| App | SwiftUI/AppKit menu bar app | PySide6 (Qt) tray app |
| Hotkey | Carbon | `RegisterHotKey` |
| File watching | FSEvents | ReadDirectoryChangesW (watchdog) |
| Updates | Sparkle | not yet |

## The embedding pipeline

Unchanged from the Mac app, and checked against it: [tests/test_reference_vectors.py](tests/test_reference_vectors.py)
embeds the Mac repo's reference inputs and compares with the vectors the Mac app made (Metal, b11461). On an RTX 4090
with CUDA every cosine is ≥ 0.9997 (the bar is 0.999, what agreeing runtimes measure).

- Text: `<bos> text <eos>`, mean-pooled, L2-normalized, cut at 2,048 tokens (512 for queries) the way transformers
  cuts. Prompts from the model card: `task: search result | query: …`, `task: code retrieval | query: …`,
  `title: … | text: …`.
- Images: resized to fill the token budget (transformers' target size, Pillow's bicubic), then mtmd's encoder rows go
  into the text model with their `<|image>` framing in one `llama_process` call. Budgets: photos 140, screenshots and
  PDF pages 280, video frames 70.
- Audio: 16 kHz mono, at most 30 s per input (the model ignores the rest), windows of 30 s every 25 s.

One difference: the Mac build patches llama.cpp so a non-causal embedding context doesn't reserve logits
(`scripts/llama.cpp-no-logits.patch` there). The release DLLs aren't patched, so each token in a batch costs 1 MiB of
pinned host memory. Text inputs are therefore packed into batches of at most 512 tokens (`TEXT_BATCH_TOKENS`); one
long input still gets the full context. The buffer grows to the largest batch seen and is freed when the worker
quits.

Backends: `nvcuda.dll` present → CUDA; else `vulkan-1.dll` → Vulkan (AMD, Intel); else CPU. Settings can force one.

## Indexing

The same as the Mac app ([indexer.py](digup/indexer.py)): a sync reconciles the folders with the index (new files
queued, changed ones queued again, moves and renames recognized by NTFS file id, size and date, so nothing is embedded
again); then pending files are read on one thread while the previous batch is embedded, screenshots first, newest
first; long PDFs and documents get their first step and are finished after everything else. The index is fingerprinted
with the model, runtime and settings; vectors from different settings never mix.

Windows-specific rules ([rules.py](digup/rules.py), [crawler.py](digup/crawler.py)):

- Hidden and system files and folders, reparse points (junctions, symlinks) and `AppData` are skipped.
- OneDrive online-only files (`RECALL_ON_DATA_ACCESS`, `OFFLINE`) are skipped: reading one downloads it.
- There is no screenshot flag on Windows, so a screenshot is a file named the way the Snipping Tool, Print Screen,
  ShareX and others name them, or anything in a `Screenshots` folder.
- `.sln`/`.csproj` mark a code project, alongside `.git`, `package.json` and the rest.

## Memory and big drives

Indexing a whole data drive (2.2 million files, a million vectors) shaped these:

- **Vectors are a memory-mapped file**, not Python memory ([vectorcache.py](digup/vectorcache.py)): 768 int8 values
  and a scale per vector (cosines within 0.0015 of fp16; top 200 results unchanged), beside the index as
  `index.vectors/`. Mapped pages belong to the file system cache, so the search process owns about 300 MB instead of
  1.5 GB (3 GB while loading), and the next start has them at once. Indexing passes append what they added. A trigger
  logs deleted vectors (`vec_gone`): SQLite can give a new segment a deleted one's id, and the log keeps the cache
  from ever serving the old vector. Mostly-dead caches are written anew in a new generation folder (Windows can't
  replace a mapped file).
- **Scoring** runs on six threads, a 4,096-row slice at a time (0.2 s for a million vectors); each file's best row
  is picked with NumPy, not a Python loop; the word search runs beside it. NumPy's BLAS gets one thread
  (`OPENBLAS_NUM_THREADS=1`): it otherwise reserves buffers for every core (750 MB on 32 cores).
- **A scan streams** ([crawler.py](digup/crawler.py) `walk`): files come a folder at a time, the code-project and
  dataset rules are decided per folder, and the comparison with the index happens in SQLite temporary tables
  ([indexer.py](digup/indexer.py) `sync`). Holding both sides as Python objects took gigabytes.
- **Datasets and thumbnails** are left out by folder: hundreds of machine-named media files, or 100+ pictures nearly
  all under 8 KB (2.1 million 50x50 tiles on one drive). Windows backup folders (File History, system images,
  `$`-folders) are never walked.
- **The UI thread** only paints: status updates are capped at four a second, the tray icon is redrawn when its arc
  moves, Settings reads its numbers on a thread, and the preview has its own read-only connection.

## Search

Hybrid, as on the Mac ([searcher.py](digup/searcher.py)): every vector's cosine is put on a per-query scale (minus its
modality's mean, over the standard deviation of all), a file qualifies by standing out (z ≥ 2) or by holding at least
half the query's words, and the score is meaning z + a word-coverage bonus + 1 for a kind the query names. Codes and
numbers weigh words double. Results below half the top score fold under "Weaker matches".

**Searching with a picture** uses the same space: an image query is embedded by the full model (a second worker,
started on first use and stopped with the query encoder) at the screenshots' budget (280), and searched by meaning
alone. **Find similar** needs no model: it searches with the vector the index already holds for the result (its
segment's, else the file's picture, else any of its vectors). The query's own file is left out of the results.

The app shows keyword matches as you type (prefix match on the last word), then the full results once the query
encoder returns the vector. The encoder is a text-only worker (~300 MB) started when the panel opens and stopped five
minutes after the last search; vectors are loaded on the first meaning search and released with it.

## The app

- **Window** ([panel.py](digup/app/panel.py)): an ordinary window (title bar, resizable, in the taskbar; it stays
  when focus moves elsewhere, and closing it hides it to the notification area). The hotkey brings it up or
  minimizes it. Results with thumbnails on the left; a preview on the right (the picture, the PDF page, the
  video frame at its moment with Ctrl+Space to play there, or the passage with the words marked).
- **Opening at the place** ([opener.py](digup/app/opener.py)): Windows' default handlers can't be told a time or a
  page, so DigUp looks for apps that can (VLC, mpv, MPC-HC, PotPlayer; SumatraPDF, Edge; VS Code, Cursor, Zed,
  Sublime, Notepad++, JetBrains) and falls back to the default app.
- **Engine** ([engine.py](digup/app/engine.py)): one background thread; syncs at start and when a watched folder
  settles (2 s; code folders 60 s), indexes through a below-normal-priority worker that quits after a minute idle, and
  pauses on request, on battery saver, or on battery when asked.
- **Overview** ([overview.py](digup/app/overview.py)): Settings' first tab; per-folder counts by kind and state from
  the index, the files that failed or were left out with why, live status from the engine, and each index's last
  scan (stored by the engine and the CLI as `last_sync` in the index's meta table).
- **Single instance**: a second launch asks the running one to open the panel (a local socket).
- Data lives in `%LOCALAPPDATA%\DigUp` (`Models`, `Runtime`, `Index`, `Logs`, `settings.json`); `DIGUP_HOME` moves
  it (tests, portable use).

## Not ported yet

- Updates (Sparkle on the Mac): Releases on GitHub for now.
- The larger search window with a picture grid (⌘O on the Mac); the panel's filters cover most of it.
- Evals (`evals/` in the Mac repo) through the CLI.
