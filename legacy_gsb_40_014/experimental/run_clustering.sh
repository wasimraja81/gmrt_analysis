#!/usr/bin/env bash
# run_clustering.sh — Phase-2 clustering detection
#
# Convenience wrapper for run_clustering.py.  Activates the gmrt venv,
# sets a sensible MPLBACKEND (MacOSX for interactive plots), then delegates
# all arguments to run_clustering.py.
#
# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT: DRY-RUN.  Clustering flags are detected and plotted but NOT written
# to disk.  Safe for exploring CLUSTERING_THRESHOLD_JY multiple times.
#
# TO WRITE CLUSTERING FLAGS TO DISK:
#   ./run_clustering.sh --no-dry-run
#   Flags will be appended to FLAG_TABLE_SESSION (from .cfg).
# ─────────────────────────────────────────────────────────────────────────────
#
# Usage:
#   ./run_clustering.sh
#   ./run_clustering.sh --set "CLUSTERING_THRESHOLD_JY=3.0"
#   ./run_clustering.sh --set "SOURCE='3C286'" --set "CLUSTERING_THRESHOLD_JY=4.0"
#   ./run_clustering.sh --set "SOLVE_ELEVATION_MIN_DEG=25.0"
#   ./run_clustering.sh --config /path/to/other.cfg
#   ./run_clustering.sh --no-dry-run      # write new flags to FLAG_TABLE_SESSION
#   ./run_clustering.sh --refit           # re-solve bandpass with clustering flags
#   ./run_clustering.sh --save-plots      # save 6 PNGs to WORK_DIR
#
# Passthrough --set examples (all config keys accepted):
#   --set "CLUSTERING_THRESHOLD_JY=3.0"
#   --set "CLUSTERING_CORR='V'"
#   --set "PLOT_CHAN_RANGE=(100,180)"
#   --set "SOLVE_ELEVATION_MIN_DEG=25.0"
#   --set "SOURCE='3C286'"
#   --set "CAL_FITS=Path('/data/3c286.FITS')"
#   --set "WORK_DIR=Path('/data/3c286/work')"
#
# For the full option reference:
#   ./run_clustering.sh --help

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="${SCRIPT_DIR}/gmrt/bin/activate"

# ── activate venv (if present) ────────────────────────────────────────────────
if [[ -f "${VENV_ACTIVATE}" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_ACTIVATE}"
fi

# ── matplotlib backend ────────────────────────────────────────────────────────
# Use MacOSX so plot windows open interactively (blocking until closed).
# The caller can override by setting MPLBACKEND in the environment.
if [[ -z "${MPLBACKEND:-}" ]]; then
    export MPLBACKEND=MacOSX
fi

exec python "${SCRIPT_DIR}/pipeline_cli.py" clustering "$@"
