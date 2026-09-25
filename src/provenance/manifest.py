"""Per-invocation provenance manifest for GWB pipeline stages.

Every stage writes one manifest per run: git state, the fully resolved parameter
set, an identity fingerprint for each input file, every output the stage produced,
and how the run ended. See docs/dev/GWB_PIPELINE_REFACTOR_PLAN.md (standing rule 4) for
why this exists: a later run -- by this user or a collaborator -- must be able to
reproduce exactly what an earlier run did without relying on memory of what flags
were passed.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import logging

from data_io.raw_data_access import guard_output_path
from provenance.logging_setup import close_logger, setup_stage_logger, stage_log_path
from provenance.run_index import append_to_run_index


class RunManifest:
    """Context manager that records one pipeline-stage invocation.

    Usage:
        with RunManifest(stage="primary_calibration", work_dir=WORK_DIR,
                          parameters=resolved_config, inputs=[raw_fits_path]) as manifest:
            manifest.logger.info("starting bandpass solve")  # not print()
            manifest.add_output(bandpass_path)

    On exit, writes ``<work_dir>/provenance/<stage>/<run_id>.json`` (and, if the
    repo has uncommitted changes, a sibling ``<run_id>.diff``) whether the block
    succeeded or raised. Exceptions are recorded, never swallowed -- the ``with``
    block re-raises exactly as it would without the manifest.

    ``manifest.logger`` is a stage-scoped logger sharing this run's ``run_id``
    with its log file (``<work_dir>/logs/<stage>/<run_id>.log``), so the log
    and the manifest can be cross-referenced by filename alone.
    """

    def __init__(
        self,
        stage: str,
        work_dir: Path,
        parameters: dict,
        inputs: list[Path | str],
        repo_root: Path | None = None,
        console_log_level: int = logging.INFO,
    ) -> None:
        self.stage = stage
        self.work_dir = Path(work_dir)
        self.parameters = parameters
        self.run_id = _new_run_id()
        self.repo_root = repo_root or _find_repo_root(Path(__file__).resolve())
        self.git_state = _capture_git_state(self.repo_root)
        self._input_paths = [Path(p) for p in inputs]
        self.input_fingerprints = [_describe_input_file(p) for p in self._input_paths]
        self.output_paths: list[str] = []
        self.started_at_utc = _utc_now_iso()
        self.manifest_path, self.diff_path = _manifest_paths(self.work_dir, self.stage, self.run_id)
        self.logger = setup_stage_logger(self.stage, self.work_dir, self.run_id, console_log_level)

        # Written immediately, before any of the stage's own work runs -- a
        # process killed mid-stage (SIGKILL, session teardown) leaves this
        # "started" record behind instead of no manifest at all. __exit__
        # overwrites the same path with the final outcome; a manifest whose
        # outcome is still "started" means the run never reached __exit__.
        self._write_manifest_record(
            outcome={"status": "started", "error_type": None, "error_message": None, "traceback": None},
            finished_at_utc=None,
        )

    def add_output(self, path: Path | str) -> None:
        """Record one output file this run produced.

        Checks the output path against every input this run read from first
        (see data_io.raw_data_access.guard_output_path) -- a stage cannot
        register an output that would overwrite one of its own raw inputs.
        """
        guard_output_path(path, self._input_paths)
        self.output_paths.append(str(path))

    def _write_manifest_record(self, outcome: dict, finished_at_utc: str | None) -> dict:
        record = {
            "run_id": self.run_id,
            "stage": self.stage,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": finished_at_utc,
            "host": socket.gethostname(),
            "python_version": platform.python_version(),
            "git": {k: v for k, v in self.git_state.items() if k != "diff_text"},
            "parameters": self.parameters,
            "inputs": self.input_fingerprints,
            "outputs": self.output_paths,
            "outcome": outcome,
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.manifest_path, json.dumps(record, indent=2, sort_keys=True) + "\n")
        return record

    def __enter__(self) -> "RunManifest":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        finished_at_utc = _utc_now_iso()
        if exc_type is None:
            outcome = {"status": "success", "error_type": None, "error_message": None, "traceback": None}
            self.logger.info("stage %s completed", self.stage)
        else:
            outcome = {
                "status": "failed",
                "error_type": exc_type.__name__,
                "error_message": str(exc_val),
                "traceback": "".join(traceback.format_exception(exc_type, exc_val, exc_tb)),
            }
            self.logger.error("stage %s failed: %s: %s", self.stage, exc_type.__name__, exc_val)

        record = self._write_manifest_record(outcome, finished_at_utc)

        if self.git_state["dirty"] and self.git_state["diff_text"]:
            _atomic_write_text(self.diff_path, self.git_state["diff_text"])
            record["git"]["diff_file"] = self.diff_path.name
            _atomic_write_text(self.manifest_path, json.dumps(record, indent=2, sort_keys=True) + "\n")

        self.index_entry = {
            "run_id": self.run_id,
            "stage": self.stage,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": finished_at_utc,
            "git_commit": self.git_state["commit"],
            "git_dirty": self.git_state["dirty"],
            "parameters": self.parameters,
            "outcome_status": outcome["status"],
            "error_type": outcome["error_type"],
            "manifest_path": str(self.manifest_path),
            "log_path": str(stage_log_path(self.work_dir, self.stage, self.run_id)),
        }
        append_to_run_index(self.work_dir, self.index_entry)

        close_logger(self.logger)
        return False  # never suppress the exception


def _manifest_paths(work_dir: Path, stage: str, run_id: str) -> tuple[Path, Path]:
    stage_dir = Path(work_dir) / "provenance" / stage
    return stage_dir / f"{run_id}.json", stage_dir / f"{run_id}.diff"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text to `path` atomically: write to a temp file in the same
    directory, then rename into place. A process killed mid-write never
    leaves a truncated or corrupt file at `path` itself -- `path` either has
    its previous complete contents, or the new complete contents, never a
    partial write."""
    path = Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp_path.write_text(text)
    os.replace(tmp_path, path)


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}_{suffix}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _find_repo_root(start: Path) -> Path:
    current = start if start.is_dir() else start.parent
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"could not find a .git directory above {start}")


def _capture_git_state(repo_root: Path) -> dict:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status_porcelain = run("status", "--porcelain")
    dirty = bool(status_porcelain)
    diff_text = run("diff", "HEAD") if dirty else ""

    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "diff_text": diff_text,
    }


def _describe_input_file(path: Path) -> dict:
    stat = os.stat(path)
    mtime_utc = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_utc": mtime_utc,
    }
