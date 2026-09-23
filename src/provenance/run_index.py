"""Append-only run index -- the queryable "lab notebook" across every stage.

One line of JSON per run (JSON-lines), appended by ``provenance.manifest.
RunManifest`` on exit. Never rewritten in place: each run adds exactly one
line, so this file is safe to grow across a pipeline's whole lifetime
without ever needing to read-modify-write what's already there. Meant to
be queried directly with standard tools (``grep``/``jq``), not through a
bespoke API -- see docs/user/GWB_USER_GUIDE.md#finding-past-runs-the-run-index
for example queries. Distinct from the per-run manifest (full detail for
one run) and the curated gh-pages report (final, shared results).
"""

from __future__ import annotations

import json
from pathlib import Path


def append_to_run_index(work_dir: Path, entry: dict) -> Path:
    index_path = run_index_path(work_dir)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return index_path


def run_index_path(work_dir: Path) -> Path:
    return Path(work_dir) / "runs_index.jsonl"
