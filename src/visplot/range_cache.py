"""Axis-range results saved to a directory the user names, so a later run
over the same selection skips the range pass.

Conventions, so what is written is easy to find and remove:
- Nothing is written unless a directory is named (visplot's --cache-dir).
- Every file is `visplot-cache_<FITS file stem>_<key>.npz`; a file being
  written is `<that name>.<process id>.partial`, renamed when complete, so a
  crash leaves only `.partial` files. The next run removes those whose
  process is gone.
- `clear_cache(directory)` (visplot's --clear-cache) removes every
  `visplot-cache_*` file there, and nothing else.

A result is keyed by everything it depends on: the file (path, size,
modification time), the rows, the channel/Stokes selection, the quantity,
how flags are handled, mirroring, and whether a log axis leaves out
non-positive values. Any change gives a different key.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

CACHE_PREFIX = "visplot-cache_"
PARTIAL_SUFFIX = ".partial"
# Bumped when what a cached range means changes, so older files stop matching.
FORMAT_VERSION = 1


class RangeCache:
    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.written: list[Path] = []

    def key(self, fits_path, row_indices: np.ndarray, axis_selection, quantity: str, *,
            apply_flags: bool, show_flagged: bool, mirror: bool, log_axis: bool) -> str:
        stat = os.stat(fits_path)
        h = hashlib.sha256()
        h.update(json.dumps({
            "version": FORMAT_VERSION,
            "file": str(Path(fits_path).resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "quantity": quantity,
            "flags": [apply_flags, show_flagged],
            "mirror": mirror,
            "log_axis": log_axis,
            "axes": sorted((axis, np.asarray(idx).tolist()) for axis, idx in (axis_selection or {}).items()),
        }, sort_keys=True).encode())
        h.update(np.ascontiguousarray(row_indices, dtype=np.int64).tobytes())
        return h.hexdigest()[:20]

    def path(self, fits_path, key: str) -> Path:
        return self.directory / f"{CACHE_PREFIX}{Path(fits_path).stem}_{key}.npz"

    def load(self, fits_path, key: str):
        """(lo, hi, histogram counts) for `key`, or None if not cached."""
        path = self.path(fits_path, key)
        if not path.exists():
            return None
        with np.load(path) as f:
            return float(f["lo"]), float(f["hi"]), f["histogram"].copy()

    def save(self, fits_path, key: str, lo: float, hi: float, histogram: np.ndarray, description: dict) -> Path:
        """Write one result: under a `.partial` name first, renamed when complete."""
        path = self.path(fits_path, key)
        partial = path.with_name(f"{path.name}.{os.getpid()}{PARTIAL_SUFFIX}")
        try:
            with open(partial, "wb") as f:
                np.savez_compressed(f, lo=lo, hi=hi, histogram=histogram,
                                    description=json.dumps(description, sort_keys=True))
            os.replace(partial, path)
        finally:
            if partial.exists():
                partial.unlink()
        self.written.append(path)
        return path

    def remove_stale_partials(self) -> list[Path]:
        """Remove `.partial` files left by processes that no longer run."""
        removed = []
        for partial in self.directory.glob(f"{CACHE_PREFIX}*{PARTIAL_SUFFIX}"):
            pid = partial.name[: -len(PARTIAL_SUFFIX)].rsplit(".", 1)[-1]
            if pid.isdigit() and _process_alive(int(pid)):
                continue
            partial.unlink(missing_ok=True)
            removed.append(partial)
        return removed

    def usage(self) -> tuple[int, int]:
        """(number of files, total bytes) of this directory's cache files."""
        files = list(self.directory.glob(f"{CACHE_PREFIX}*"))
        return len(files), sum(f.stat().st_size for f in files)


def clear_cache(directory: Path | str) -> list[Path]:
    """Remove every visplot cache file (complete or partial) in `directory`,
    and nothing else. Returns what was removed."""
    removed = []
    for path in sorted(Path(directory).glob(f"{CACHE_PREFIX}*")):
        if path.is_file():
            path.unlink()
            removed.append(path)
    return removed


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    return True
