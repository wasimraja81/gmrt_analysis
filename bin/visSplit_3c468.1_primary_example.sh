#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# PYTHON_CMD is set in the machine-specific block below

cd "$REPO_ROOT"

# Pre-secondary split for 3C468.1
# Produces primary-calibrated + primary-flagged visibilities used for diagnostics and clustering context.

# ── Machine-specific settings ─────────────────────────────────────────────────
_HOSTNAME="$(hostname -s)"
if [[ "$_HOSTNAME" == "wasim-desktop" ]]; then
    FITS=/data1/gmrt/40_014_25JUL2021/40_014_25jul2021_2.6s_gwb.FITS
    INDEX=/data1/gmrt/40_014_25JUL2021/40_014_25jul2021_2.6s_gwb.index.npz
    WORK_DIR=/data1/gmrt/40_014/work
    CHAN_START=1731; CHAN_END=1901
    _DATA_TAG=gwb
    PYTHON_CMD="${PYTHON_CMD:-${REPO_ROOT}/gmrt/bin/python}"
else
    FITS=~/DATA/gmrt_40_014/data/40_014_25jul2021_gsb.FITS
    INDEX=~/DATA/gmrt_40_014/work/40_014_25jul2021_gsb.index.npz
    WORK_DIR=~/DATA/gmrt_40_014/work
    CHAN_START=64; CHAN_END=191
    _DATA_TAG=gsb
    PYTHON_CMD="${PYTHON_CMD:-python}"
fi
# ───────────────────────────────────────────────────────────────────────────────
SOURCE=3C468.1
SRC_TAG="$(printf '%s' "$SOURCE" | tr '[:upper:]' '[:lower:]')"
PRIMARY="$WORK_DIR/primary_calibration/bandpass/3c48_bandpass_25jul_${_DATA_TAG}_iterfinal_clustering.npz"
PRIMARY_FLAG="$WORK_DIR/primary_calibration/flag/3c48_flag_table_session.json"
CONFIG="$REPO_ROOT/preprocess_ugmrt.cfg"
STOKES=(RR LL RL LR)

OUTDIR="$WORK_DIR/split/${SRC_TAG}"
mkdir -p "$OUTDIR"
OUTPUT="$OUTDIR/${SRC_TAG}_primary_calibrated.uvfits"
LOG_DIR="$WORK_DIR/logs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

if [[ -f "$OUTPUT" ]]; then
    echo "[visSplit-3c468.1-primary] WARNING: existing output will be overwritten: $OUTPUT"
fi

if [[ ! -f "$PRIMARY" ]]; then
    echo "Primary table not found: $PRIMARY"
    exit 1
fi

if [[ ! -f "$PRIMARY_FLAG" ]]; then
    echo "Primary flag table not found: $PRIMARY_FLAG"
    exit 1
fi

echo "[visSplit-3c468.1-primary] applying primary calibration + primary flags → $OUTPUT"

CMD=(
    "$PYTHON_CMD" "$REPO_ROOT/src/visSplit.py"
    --config "$CONFIG"
    --fits "$FITS"
    --index-cache "$INDEX"
    --source "$SOURCE"
    --chan-range "$CHAN_START" "$CHAN_END"
    --stokes "${STOKES[@]}"
    --elevation-min 25
    --tables "$PRIMARY"
    --flag-tables "$PRIMARY_FLAG"
    --time-interp-scheme nearest
    --time-extrapolation hold
    --out "$OUTPUT"
    --overwrite
)

CMD_FILE="$LOG_DIR/visSplit_3c468.1_primary_example_${RUN_TS}.cmd"
LOG_FILE="$LOG_DIR/visSplit_3c468.1_primary_example_${RUN_TS}.log"
{
    echo "# timestamp=$RUN_TS"
    echo "# cwd=$PWD"
    printf '%q ' "${CMD[@]}"
    printf '\n'
} > "$CMD_FILE"

echo "[visSplit-3c468.1-primary] cmd manifest: $CMD_FILE"
echo "[visSplit-3c468.1-primary] log: $LOG_FILE"

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo "[visSplit-3c468.1-primary] done. output: $OUTPUT"
