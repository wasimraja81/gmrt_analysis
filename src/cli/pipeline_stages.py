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

from data_io.row_index import default_row_index_path, save_row_index
from instruments.gmrt.row_index import build_gmrt_row_index
from provenance.manifest import RunManifest


def run_build_index_stage(config: dict) -> Path:
    """Build the row index for config['fits_path'] and persist it adjacent to
    the raw file it indexes (see `default_row_index_path`). Returns the path
    the index was saved to.

    Idempotent by default: if an index already exists at that path, the
    expensive full-file scan is skipped -- a fresh manifest is still written,
    recording that this run skipped rather than rebuilt, so the provenance
    trail shows every invocation, not just the ones that did new work. Set
    `build_index.force_rebuild: true` in the config to rebuild anyway (e.g.
    after the raw file's DUD/antenna situation is known to have changed).
    """
    fits_path = Path(config["fits_path"])
    work_dir = Path(config["work_dir"])
    build_index_config = config.get("build_index", {})
    strict = build_index_config.get("strict", True)
    force_rebuild = build_index_config.get("force_rebuild", False)
    idx_path = default_row_index_path(fits_path)

    with RunManifest(
        stage="build_index",
        work_dir=work_dir,
        parameters={"fits_path": str(fits_path), "strict": strict, "force_rebuild": force_rebuild},
        inputs=[fits_path],
    ) as manifest:
        if idx_path.exists() and not force_rebuild:
            manifest.logger.info(
                "index already exists at %s, skipping rebuild (pass build_index.force_rebuild: "
                "true to rebuild anyway)",
                idx_path,
            )
            manifest.add_output(idx_path)
            return idx_path

        manifest.logger.info("building row index for %s", fits_path)
        index, resolution = build_gmrt_row_index(
            fits_path, strict=strict, verbose=True, logger=manifest.logger
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
