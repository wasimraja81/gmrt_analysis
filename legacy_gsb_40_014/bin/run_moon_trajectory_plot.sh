#!/usr/bin/env bash
# Run moon_trajectory_plot.py using the CASA/GMRT Python environment.
# Outputs to diagnostics_out/moon_imaging/moon0520_trajectory.png (in WORK_DIR).
#
# Usage:
#   WORK_DIR=/path/to/work  bash bin/run_moon_trajectory_plot.sh
#   WORK_DIR=/path/to/work  MS_PATH=/path/to/full.ms  bash bin/run_moon_trajectory_plot.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WORK_DIR="${WORK_DIR:-$(pwd)}"
MS_PATH="${MS_PATH:-$WORK_DIR/casa_selfcal/moon0520_dev/moon0520_full.ms}"
OUT_PNG="${OUT_PNG:-$WORK_DIR/diagnostics_out/moon_imaging/moon0520_trajectory.png}"
SCAN="${SCAN:-MOON0520}"

PYTHON="${PYTHON:-$REPO_ROOT/gmrt/bin/python3}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "[moon_trajectory] WORK_DIR  : $WORK_DIR"
echo "[moon_trajectory] MS_PATH   : $MS_PATH"
echo "[moon_trajectory] OUT_PNG   : $OUT_PNG"
echo "[moon_trajectory] Python    : $PYTHON"

if [ ! -d "$MS_PATH" ]; then
    echo "[moon_trajectory] WARNING: MS not found: $MS_PATH"
    echo "[moon_trajectory] WARNING: selfcal products not ready yet; skipping trajectory plot."
    exit 0
fi

exec "$PYTHON" "$REPO_ROOT/src/moon_trajectory_plot.py" \
    --ms    "$MS_PATH"  \
    --out   "$OUT_PNG"  \
    --scan  "$SCAN"
