"""Walks the chosen folders and applies `rules`. Reads metadata only, never file contents (except the 600-byte sniff
that tells a real MPEG-TS video from a TypeScript file).

For the code index it walks the code folders instead: code files only, in a git repo only those git keeps, and none
in vendored or built folders.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from dataclasses import dataclass, field

from . import paths, rules
from .model import FileKind, IndexOptions
from .store import Candidate

HIDDEN = stat.FILE_ATTRIBUTE_HIDDEN | stat.FILE_ATTRIBUTE_SYSTEM if hasattr(stat, "FILE_ATTRIBUTE_HIDDEN") else 0x6
REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
# OneDrive "online-only" files: reading one downloads it.
IN_CLOUD = getattr(stat, "FILE_ATTRIBUTE_OFFLINE", 0x1000) | 0x40000 | 0x400000
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class CrawlReport:
    candidates: list[Candidate] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    skipped_folders: dict[str, int] = field(default_factory=dict)
    datasets: dict[str, int] = field(default_factory=dict)
    complete: bool = False  # the walk reached the end (not stopped partway)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


def _attributes(st) -> int:
    return getattr(st, "st_file_attributes", 0)


def crawl(roots: list[str], options: IndexOptions, should_stop=lambda: False) -> CrawlReport:
    """Every file to index under `roots`, as a list in the report (for a few folders; `walk` streams them)."""
    report = CrawlReport()
    report.candidates = list(walk(roots, options, report, should_stop))
    return report


def walk(roots: list[str], options: IndexOptions, report: CrawlReport, should_stop=lambda: False):
    """Yields the files to index under `roots`, a folder at a time, and counts what it leaves out in `report`.
    `report.complete` is set when the walk reached the end. Memory stays at one folder's files, so a drive with
    millions of them can be walked (holding them all took gigabytes).

    The rules that look at more than one file are decided per folder as it goes: inside a code project only
    screenshots count (a folder is in one when it or a folder above it has a project marker), and a folder of
    hundreds of machine-named or tiny media files is a dataset (`rules.dataset_folders`)."""
    repos = GitRepos()
    normalized = sorted({paths.normalize(r) for r in roots}, key=len)
    # A root inside another root is walked with it, once.
    root_paths = [r for i, r in enumerate(normalized) if not any(paths.is_inside(r, o) for o in normalized[:i])]
    all_roots = set(normalized)
    media = (FileKind.IMAGE, FileKind.AUDIO, FileKind.VIDEO)

    for root in root_paths:
        if not os.path.isdir(root):
            continue
        if options.code:
            repos.enter_root(root)
        skips = [f for f in options.excluded_folders if paths.is_inside(f, root) and not paths.is_inside(root, f)]
        stack = [(root, (not options.code) and rules.is_project_root(root))]
        while stack:
            if should_stop():
                return
            folder, in_project = stack.pop()
            try:
                entries = list(os.scandir(folder))
            except OSError:
                continue
            found: list[Candidate] = []
            for entry in entries:
                try:
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                attributes = _attributes(st)
                if entry.is_dir(follow_symlinks=False):
                    if attributes & (HIDDEN | REPARSE):
                        report.skipped_folders["hidden, system or linked folder"] = (
                            report.skipped_folders.get("hidden, system or linked folder", 0) + 1)
                        continue
                    reason = rules.skip_reason_for_folder(entry.path, skips) or (
                        rules.code_skip_reason_for_folder(entry.path) if options.code else None)
                    if reason:
                        report.skipped_folders[reason] = report.skipped_folders.get(reason, 0) + 1
                        continue
                    if options.code:
                        repos.enter(entry.path)
                        stack.append((entry.path, False))
                    else:
                        stack.append((entry.path, in_project or rules.is_project_root(entry.path)))
                    continue
                if attributes & HIDDEN or entry.name.startswith("."):
                    continue
                verdict, value = (rules.classify_code if options.code else rules.classify)(entry.path, options)
                if verdict == "ignore":
                    report.skip("other type")
                    continue
                if verdict == "skip":
                    report.skip(value)
                    continue
                if attributes & REPARSE or not entry.is_file(follow_symlinks=False):
                    report.skip("link or special file")
                    continue
                if attributes & IN_CLOUD:
                    report.skip("in the cloud, not downloaded")
                    continue
                if options.code:
                    limit = options.max_notebook_bytes if rules.ext_of(entry.name) == "ipynb" else options.max_code_bytes
                    if st.st_size > limit:
                        report.skip("too big to be written by hand")
                        continue
                    if repos.ignores(entry.path):
                        report.skip("ignored by git")
                        continue
                elif in_project and value != FileKind.SCREENSHOT:
                    report.skip("inside a code project")
                    continue
                found.append(Candidate(entry.path, value, st.st_size, st.st_mtime, 0, 0))
            if not found:
                continue
            if not options.include_datasets and not options.code:
                datasets = rules.dataset_folders(
                    [(folder, os.path.splitext(os.path.basename(f.path))[0], f.kind, f.size) for f in found],
                    roots=all_roots,
                )
                if datasets:
                    report.datasets[folder] = report.datasets.get(folder, 0) + datasets[folder]
                    kept = [f for f in found if f.kind not in media]
                    report.skipped["dataset"] = report.skipped.get("dataset", 0) + len(found) - len(kept)
                    found = kept
            for file in found:
                try:
                    full = os.stat(file.path)  # DirEntry leaves st_ino at 0 on Windows: moves are found by it
                    file.inode, file.device = full.st_ino, full.st_dev
                except OSError:
                    pass
                yield file
    report.complete = True


class GitRepos:
    """The git repos a code crawl walks through, and the files each keeps: a file git ignores isn't code search's."""

    def __init__(self):
        self.files: dict[str, set[str] | None] = {}
        self.repo_of: dict[str, str | None] = {}

    def enter_root(self, path: str) -> None:
        folder = path
        while True:
            if os.path.exists(os.path.join(folder, ".git")):
                self.files[folder.lower()] = git_files(folder)
                self.repo_of[path.lower()] = folder
                return
            parent = os.path.dirname(folder)
            if parent == folder:
                break
            folder = parent
        self.repo_of[path.lower()] = None

    def enter(self, path: str) -> None:
        if os.path.isdir(os.path.join(path, ".git")):
            self.files[path.lower()] = git_files(path)
            self.repo_of[path.lower()] = path

    def _repo(self, folder: str) -> str | None:
        key = folder.lower()
        if key in self.repo_of:
            return self.repo_of[key]
        parent = os.path.dirname(folder)
        answer = self._repo(parent) if parent != folder else None
        self.repo_of[key] = answer
        return answer

    def ignores(self, path: str) -> bool:
        repo = self._repo(os.path.dirname(path))
        if not repo:
            return False
        kept = self.files.get(repo.lower())
        if kept is None:
            return False
        return path[len(repo) + 1:].replace("\\", "/").lower() not in kept


def git_files(repo: str) -> set[str] | None:
    """Paths (relative, forward slashes, lower-case) git keeps in `repo`, or None when git can't say."""
    git = shutil.which("git")
    if not git:
        return None
    try:
        out = subprocess.run(
            [git, "-C", repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, timeout=120, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            creationflags=NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return {p.decode("utf-8", "replace").lower() for p in out.stdout.split(b"\0") if p}
