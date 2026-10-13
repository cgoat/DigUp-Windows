"""What gets indexed: media and documents by type, never keys, caches, app folders, code projects or datasets."""

from __future__ import annotations

import os
import re

from .model import FileKind, IndexOptions

# Folders never worth walking into. Hidden and system folders are skipped by the crawler itself.
SKIPPED_FOLDER_NAMES = {
    "node_modules", "bower_components", "venv", ".venv", "site-packages", "__pycache__", "Pods", "Carthage",
    "$RECYCLE.BIN", "System Volume Information", "AppData", "Windows", "Program Files", "Program Files (x86)",
    "ProgramData", "OneDriveTemp", "Temporary Internet Files", "INetCache",
}
# Backups and system folders Windows keeps on data drives: copies of files (File History keeps millions of tiny
# thumbnails), not your things. A folder whose name starts with "$" is a system one ($OF, $WINDOWS.~BT, $SysReset).
BACKUP_FOLDER_NAMES = {
    "FileHistory", "WindowsImageBackup", "Recovery", "Config.Msi", "MSOCache", "PerfLogs", "Windows.old",
}

PROJECT_MARKERS = [
    ".git", "package.json", "pyproject.toml", "setup.py", "requirements.txt", "Cargo.toml", "go.mod", "pom.xml",
    "build.gradle", "build.gradle.kts", "Gemfile", "composer.json", "CMakeLists.txt", "Package.swift",
]
PROJECT_EXTENSIONS = {"sln", "csproj", "vcxproj", "xcodeproj", "xcworkspace"}

DOC_EXTENSIONS = {"txt", "text", "md", "markdown", "rtf", "docx", "odt", "html", "htm"}
PDF_EXTENSIONS = {"pdf"}
IMAGE_EXTENSIONS = {
    "jpg", "jpeg", "jpe", "jfif", "png", "gif", "bmp", "dib", "tif", "tiff", "webp", "heic", "heif", "avif",
    "ico", "jxr", "wdp", "dng", "cr2", "cr3", "nef", "arw", "orf", "rw2", "raf",
}
AUDIO_EXTENSIONS = {"mp3", "m4a", "aac", "wav", "flac", "ogg", "oga", "opus", "wma", "aiff", "aif", "amr", "m4b", "caf"}
VIDEO_EXTENSIONS = {"mp4", "m4v", "mov", "mkv", "avi", "wmv", "webm", "flv", "3gp", "mpg", "mpeg", "m2ts", "mts", "ts"}

CODE_EXTENSIONS = {
    "py", "ipynb", "c", "h", "cc", "cpp", "cxx", "hpp", "hh", "m", "mm", "swift", "tsx", "cts", "js", "jsx", "mjs",
    "cjs", "go", "rs", "java", "kt", "kts", "scala", "rb", "php", "cs", "fs", "sh", "bash", "zsh", "fish", "ps1",
    "psm1", "bat", "cmd", "lua", "pl", "r", "jl", "dart", "sql", "vue", "svelte", "css", "scss", "sass", "less",
    "json", "jsonl", "yaml", "yml", "toml", "xml", "plist", "gradle", "cmake", "mk", "lock", "log", "csv", "tsv",
    "ini", "cfg", "conf", "proto", "graphql", "tf", "nix", "zig", "hs", "ex", "exs", "erl", "clj", "elm", "ml",
    "vim", "el", "asm", "metal", "glsl", "hlsl", "wgsl", "cu", "pyi", "pyc", "reg", "vbs", "csproj", "sln",
}

SCREENSHOT_PREFIXES = ("Screenshot", "Screen Shot", "Screen shot", "ScreenClip", "Snip", "CleanShot", "SCR-",
                       "Capture", "ShareX")

# Code search (the code index)
SOURCE_EXTENSIONS = {
    "py", "pyi", "pyx", "c", "h", "cc", "cpp", "cxx", "hpp", "hh", "hxx", "m", "mm", "swift", "ts", "tsx", "mts",
    "cts", "js", "jsx", "mjs", "cjs", "go", "rs", "java", "kt", "kts", "scala", "groovy", "rb", "php", "cs", "fs",
    "sh", "bash", "zsh", "fish", "ps1", "psm1", "bat", "cmd", "lua", "pl", "pm", "r", "jl", "dart", "sql", "vue",
    "svelte", "astro", "html", "htm", "css", "scss", "sass", "less", "gradle", "cmake", "mk", "proto", "graphql",
    "gql", "tf", "hcl", "nix", "zig", "hs", "ex", "exs", "erl", "clj", "cljs", "elm", "ml", "mli", "vim", "el",
    "lisp", "asm", "s", "metal", "glsl", "hlsl", "wgsl", "cu", "cuh", "ino", "v", "sv", "tex", "vb", "xaml",
}
CODE_TEXT_EXTENSIONS = {"md", "markdown", "rst", "adoc", "org", "txt", "yaml", "yml", "toml"}
CODE_FILE_NAMES = {
    "Makefile", "makefile", "GNUmakefile", "Dockerfile", "Containerfile", "Justfile", "justfile", "Rakefile",
    "Gemfile", "Podfile", "Brewfile", "Procfile", "Vagrantfile", "Jenkinsfile",
}
CODE_SKIPPED_FOLDER_NAMES = {
    "vendor", "Vendor", "vendored", "third_party", "third-party", "thirdparty", "3rdparty", "external", "extern",
    "deps", "build", "Build", "dist", "out", "target", "obj", "bin", "Debug", "Release", "x64", "coverage",
    "htmlcov", "generated", "__generated__", "wandb", "mlruns", "lightning_logs", "packages",
}

SECRET_TYPES = {"pem", "key", "p12", "pfx", "keystore", "jks", "ppk", "kdbx", "gpg", "asc", "crt", "cer", "der", "p8",
                "rdp", "ovpn"}
SECRET_PREFIXES = ("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", ".env", "secret", "credential", ".netrc", ".npmrc",
                   ".pypirc", "htpasswd", ".htpasswd", "_netrc")


def ext_of(path: str) -> str:
    return os.path.splitext(path)[1][1:].lower()


def is_secret(name: str) -> bool:
    lower = name.lower()
    return ext_of(lower) in SECRET_TYPES or lower.startswith(SECRET_PREFIXES)


def is_mpeg_transport_stream(path: str) -> bool:
    """MPEG-TS packets start with 0x47 every 188 bytes (or 192 with a timecode). Otherwise a .ts is TypeScript."""
    try:
        with open(path, "rb") as f:
            data = f.read(600)
    except OSError:
        return False
    if len(data) < 400:
        return False
    return (data[0] == 0x47 and data[188] == 0x47 and data[376] == 0x47) or (
        data[4] == 0x47 and data[196] == 0x47 and data[388] == 0x47
    )


def is_screenshot(path: str) -> bool:
    """Windows has no screenshot marker: the Snipping Tool's and Print Screen's names, and a "Screenshots" folder."""
    name = os.path.basename(path)
    if name.startswith(SCREENSHOT_PREFIXES):
        return True
    return os.path.basename(os.path.dirname(path)).lower() in ("screenshots", "screen captures", "screen clips")


def classify(path: str, options: IndexOptions) -> tuple[str, FileKind | str | None]:
    """("index", kind), ("skip", reason) or ("ignore", None)."""
    ext = ext_of(path)
    if ext in options.excluded_extensions:
        return "skip", "excluded type"
    if is_secret(os.path.basename(path)):
        return "skip", "key or password file"
    if ext in ("ts", "mts", "m2ts"):
        if is_mpeg_transport_stream(path):
            return "index", FileKind.VIDEO
        return ("index", FileKind.DOC) if options.include_code else ("skip", "code or data file")
    if ext in DOC_EXTENSIONS:
        return "index", FileKind.DOC
    if ext in CODE_EXTENSIONS:
        return ("index", FileKind.DOC) if options.include_code else ("skip", "code or data file")
    if ext in PDF_EXTENSIONS:
        return "index", FileKind.PDF
    if ext in IMAGE_EXTENSIONS:
        return "index", FileKind.SCREENSHOT if is_screenshot(path) else FileKind.IMAGE
    if ext in VIDEO_EXTENSIONS:
        return "index", FileKind.VIDEO
    if ext in AUDIO_EXTENSIONS:
        return "index", FileKind.AUDIO
    return "ignore", None


def classify_code(path: str, options: IndexOptions) -> tuple[str, FileKind | str | None]:
    name = os.path.basename(path)
    ext = ext_of(path)
    lower = name.lower()
    if ext in options.excluded_extensions:
        return "skip", "excluded type"
    if is_secret(name):
        return "skip", "key or password file"
    if lower.endswith((".min.js", ".min.mjs", ".min.css")):
        return "skip", "minified"
    if lower.startswith(("license", "licence", "copying", "notice")):
        return "skip", "license"
    if ext in ("ts", "mts") and is_mpeg_transport_stream(path):
        return "ignore", None
    if (ext in SOURCE_EXTENSIONS or ext in CODE_TEXT_EXTENSIONS or ext == "ipynb" or name in CODE_FILE_NAMES
            or name.startswith("Dockerfile")):
        return "index", FileKind.CODE
    if ext in CODE_EXTENSIONS:
        return "skip", "data file"
    return "ignore", None


def skip_reason_for_folder(path: str, excluded: list[str]) -> str | None:
    name = os.path.basename(path)
    if name in SKIPPED_FOLDER_NAMES:
        return "system or developer folder"
    if name in BACKUP_FOLDER_NAMES or name.startswith("$"):
        return "backup or system folder"
    if name.startswith(".") or name.endswith(".noindex"):
        return "hidden folder"
    low = path.lower()
    for folder in excluded:
        f = folder.lower().rstrip("\\")
        if low == f or low.startswith(f + "\\"):
            return "excluded folder"
    return None


def code_skip_reason_for_folder(path: str) -> str | None:
    if os.path.basename(path) in CODE_SKIPPED_FOLDER_NAMES:
        return "vendored or built code"
    if os.path.isfile(os.path.join(path, ".git")):
        return "submodule or worktree"
    return None


def is_project_root(folder: str) -> bool:
    try:
        names = os.listdir(folder)
    except OSError:
        return False
    found = set(names)
    return any(m in found for m in PROJECT_MARKERS) or any(ext_of(n) in PROJECT_EXTENSIONS for n in names)


CAMERA_PREFIXES = ("img", "dsc", "pxl", "mvimg", "vid", "gopr", "dji", "photo", "whatsapp", "screenshot", "wp_")


def looks_machine_named(stem: str) -> bool:
    """"clip_000123", "frame-0042", "3f9a1c0e-…" but not camera names like "IMG_1234"."""
    name = stem.lower()
    if name.startswith(CAMERA_PREFIXES):
        return False
    if re.fullmatch(r"[0-9a-f]{8,}([-_][0-9a-f]{4,})*", name):
        return True
    return re.fullmatch(r"[a-z]{0,16}[-_. ]?\d{3,}([-_.]\d+)*", name) is not None


TINY_IMAGE_BYTES = 8 * 1024
TINY_FOLDER_MINIMUM = 100


def dataset_folders(files: list[tuple[str, str, FileKind, int]], roots: set[str], minimum: int = 300) -> dict[str, int]:
    """Folders of training data or thumbnails, not your stuff:
    - hundreds of same-kind, machine-named media files ("clip_000123", "3f9a1c0e-…"), or
    - a hundred pictures or more, nearly all of them tiny files (under 8 KB: 50x50 dataset tiles, cached thumbnails).
      Picked by the folder, not the file: on a real drive 15% of the pictures worth searching are under 8 KB too, but
      they don't come a hundred to a folder. (On one drive this left out 2.1 million 6 KB tiles, copied six times.)"""
    groups: dict[tuple[str, str], list[tuple[str, int]]] = {}
    lower_roots = {r.lower() for r in roots}
    for folder, stem, kind, size in files:
        if kind in (FileKind.IMAGE, FileKind.AUDIO, FileKind.VIDEO) and folder.lower() not in lower_roots:
            groups.setdefault((folder, kind.value), []).append((stem, size))
    datasets: dict[str, int] = {}
    for (folder, kind), items in groups.items():
        machine = len(items) >= minimum and sum(looks_machine_named(s) for s, _ in items) / len(items) >= 0.7
        tiny = (kind == FileKind.IMAGE.value and len(items) >= TINY_FOLDER_MINIMUM
                and sum(size < TINY_IMAGE_BYTES for _, size in items) / len(items) >= 0.8)
        if machine or tiny:
            datasets[folder] = datasets.get(folder, 0) + len(items)
    return datasets
