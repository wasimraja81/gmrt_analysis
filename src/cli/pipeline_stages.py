"""Stage-gated GWB pipeline: a fixed, ordered list of stages, each independently
switched on or off by the config.

Adding a future stage (split, primary calibration, ...) means writing one
function here and adding one line to `STAGE_ORDER` -- never restructuring
the driver itself. Each stage wraps its own work in a `RunManifest`, reusing
T0-T4's provenance/logging exactly as already built; this module only wires
already-tested pieces together, no new scientific logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import yaml

from data_io.antenna_table import read_antenna_table, read_time_reference
from data_io.row_index import default_row_index_path, load_row_index, save_row_index
from data_io.source_table import read_source_table
from data_io.timestamp_check import check_timestamps
from instruments.gmrt.row_index import build_gmrt_row_index
from provenance.manifest import RunManifest


def check_file_timestamps(fits_path: Path, index, logger) -> None:
    """Log how recorded time becomes UTC (the file's time keywords and the
    rule) and the timestamps checked against the file's own u, v, w
    (`data_io.timestamp_check`): a warning when they disagree, cannot be
    measured, or the keywords give no offset."""
    reference = read_time_reference(fits_path)
    try:
        declared = reference.recorded_minus_utc_s
        logger.info("time system: %s", reference.describe())
    except ValueError as err:
        declared = float("nan")
        logger.warning("time system: %s", err)
    check = check_timestamps(index, read_antenna_table(fits_path), read_source_table(fits_path), declared)
    (logger.info if check.agrees else logger.warning)("%s", check.summary())


def run_build_index_stage(config: dict, on_chunk=None) -> Path:
    """Build the row index for config['fits_path'] and persist it adjacent to
    the raw file it indexes (see `default_row_index_path`). Returns the path
    the index was saved to.

    `on_chunk(rows_done, rows_total)` follows the full-file scan and can stop
    it (returning False raises ScanStopped: nothing is saved, and the stage's
    record says it failed); `build_index.max_chunk_bytes` sets the scan's
    block size (default: a share of the host's RAM), small for frequent
    progress (the visplot GUI's Build index).

    Idempotent by default: if an index already exists at that path, the
    expensive full-file scan is skipped -- a fresh manifest is still written,
    recording that this run skipped rather than rebuilt, so the provenance
    trail shows every invocation, not just the ones that did new work. Set
    `build_index.force_rebuild: true` in the config to rebuild anyway (e.g.
    after the raw file's DUD/antenna situation is known to have changed).

    Either way, the file's timestamps are checked against its u, v, w
    (`check_file_timestamps`), in the stage's log.
    """
    fits_path = Path(config["fits_path"])
    work_dir = Path(config["work_dir"])
    build_index_config = config.get("build_index", {})
    strict = build_index_config.get("strict", True)
    force_rebuild = build_index_config.get("force_rebuild", False)
    max_chunk_bytes = build_index_config.get("max_chunk_bytes")
    idx_path = default_row_index_path(fits_path)

    with RunManifest(
        stage="build_index",
        work_dir=work_dir,
        parameters={"fits_path": str(fits_path), "strict": strict, "force_rebuild": force_rebuild,
                    "max_chunk_bytes": max_chunk_bytes},
        inputs=[fits_path],
    ) as manifest:
        if idx_path.exists() and not force_rebuild:
            manifest.logger.info(
                "index already exists at %s, skipping rebuild (pass build_index.force_rebuild: "
                "true to rebuild anyway)",
                idx_path,
            )
            check_file_timestamps(fits_path, load_row_index(idx_path), manifest.logger)
            manifest.add_output(idx_path)
            return idx_path

        manifest.logger.info("building row index for %s", fits_path)
        index, resolution = build_gmrt_row_index(
            fits_path, strict=strict, max_chunk_bytes=max_chunk_bytes, verbose=True, logger=manifest.logger,
            on_chunk=on_chunk,
        )
        manifest.logger.info(
            "index built: %d rows, %d active antennas, %d dead-this-observation: %s",
            index.gcount,
            len(resolution.active_antennas),
            len(resolution.dead_this_observation_antennas),
            [a.name for a in resolution.dead_this_observation_antennas],
        )

        save_row_index(index, idx_path)
        manifest.add_output(idx_path)
        manifest.logger.info("saved index to %s", idx_path)
        check_file_timestamps(fits_path, index, manifest.logger)

    return idx_path


# Ordered list of (config key under `stages:`, stage function). A stage runs
# only if its key is true; skipped otherwise. Order here is execution order.
STAGE_ORDER: list[tuple[str, Callable[[dict], object]]] = [
    ("build_index", run_build_index_stage),
]


def load_pipeline_config(config_path: Path | str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict) -> None:
    """Run every enabled stage, in `STAGE_ORDER`, skipping disabled ones."""
    enabled = config.get("stages", {})
    for stage_name, stage_fn in STAGE_ORDER:
        if enabled.get(stage_name, False):
            print(f"[pipeline] running stage: {stage_name}", flush=True)
            stage_fn(config)
        else:
            print(f"[pipeline] skipping stage (disabled): {stage_name}", flush=True)
