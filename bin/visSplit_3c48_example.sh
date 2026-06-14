#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# PYTHON_CMD is set in the machine-specific block below

cd "$REPO_ROOT"

# ── Machine/data-profile settings ───────────────────────────────────────
# Data profile can be forced on any machine:
#   GMRT_DATA_PROFILE=gsb  or  GMRT_DATA_PROFILE=gwb
# Optional path overrides:
#   GMRT_BASE_DIR, GMRT_DATA_DIR, GMRT_WORK_DIR, GMRT_CAL_FITS, GMRT_INDEX_CACHE
_HOSTNAME="$(hostname -s)"
if [[ "$_HOSTNAME" == "wasim-desktop" ]]; then
    _BASE_DIR_DEFAULT=/data1/gmrt/40_014
    _DATA_DIR_DEFAULT=/data1/gmrt/40_014_25JUL2021
    PYTHON_CMD="${PYTHON_CMD:-${REPO_ROOT}/gmrt/bin/python}"
else
    _BASE_DIR_DEFAULT="$HOME/DATA/gmrt_40_014"
    _DATA_DIR_DEFAULT="${_BASE_DIR_DEFAULT}/data"
    PYTHON_CMD="${PYTHON_CMD:-python}"
fi

BASE_DIR="${GMRT_BASE_DIR:-${_BASE_DIR_DEFAULT}}"
DATA_DIR="${GMRT_DATA_DIR:-${_DATA_DIR_DEFAULT}}"
WORK_DIR="${GMRT_WORK_DIR:-${BASE_DIR}/work}"

_GMRT_DATA_PROFILE="${GMRT_DATA_PROFILE:-}"
if [[ -z "$_GMRT_DATA_PROFILE" ]]; then
    if [[ "$_HOSTNAME" == "wasim-desktop" ]]; then
        _GMRT_DATA_PROFILE=gwb
    else
        _GMRT_DATA_PROFILE=gsb
    fi
fi
_GMRT_DATA_PROFILE="$(printf '%s' "$_GMRT_DATA_PROFILE" | tr '[:upper:]' '[:lower:]')"
if [[ "$_GMRT_DATA_PROFILE" != "gwb" && "$_GMRT_DATA_PROFILE" != "gsb" ]]; then
    echo "Invalid GMRT_DATA_PROFILE='$_GMRT_DATA_PROFILE' (expected 'gwb' or 'gsb')" >&2
    exit 2
fi

if [[ "$_GMRT_DATA_PROFILE" == "gwb" ]]; then
    _fits_stem='40_014_25jul2021_2.6s_gwb'
    CHAN_START=1731; CHAN_END=1901
else
    _fits_stem='40_014_25jul2021_gsb'
    CHAN_START=64; CHAN_END=191
fi

_DATA_TAG="$_GMRT_DATA_PROFILE"
FITS="${GMRT_CAL_FITS:-${DATA_DIR}/${_fits_stem}.FITS}"
_fits_root="$FITS"
_fits_root="${_fits_root%.FITS}"
_fits_root="${_fits_root%.fits}"
INDEX="${GMRT_INDEX_CACHE:-${_fits_root}.index.npz}"
_RAW_INDEX="$INDEX"
_CHAN_START_RAW="$CHAN_START"
_CHAN_END_RAW="$CHAN_END"
export GMRT_DATA_PROFILE="$_GMRT_DATA_PROFILE"
# ───────────────────────────────────────────────────────────────────────────
FLAG="$WORK_DIR/primary_calibration/flag/3c48_flag_table_session.json"
SOURCE=3C48
SRC_TAG="$(printf '%s' "$SOURCE" | tr '[:upper:]' '[:lower:]')"
PRIMARY="$WORK_DIR/primary_calibration/bandpass/3c48_bandpass_25jul_${_DATA_TAG}.npz"
CONFIG="$REPO_ROOT/preprocess_ugmrt.cfg"
STOKES=(RR LL RL LR)

OUTDIR="$WORK_DIR/split/${SRC_TAG}"  # lower-case source name for directory
mkdir -p "$OUTDIR"
OUTPUT="$OUTDIR/${SRC_TAG}_calibrated_flagged.uvfits"
LOG_DIR="$WORK_DIR/logs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

if [[ -f "$OUTPUT" ]]; then
  echo "[visSplit-3c48] WARNING: existing output will be overwritten: $OUTPUT"
fi

if [[ ! -f "$PRIMARY" ]]; then
    echo "Primary table not found: $PRIMARY"
    exit 1
fi

if [[ ! -f "$FLAG" ]]; then
    echo "Flag table not found: $FLAG"
    exit 1
fi

echo "[visSplit-3c48] applying primary calibration + flag table → $OUTPUT"

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
  --time-interp-scheme nearest
  --time-extrapolation hold
  --flag-tables "$FLAG"
  --out "$OUTPUT"
  --overwrite
)

CMD_FILE="$LOG_DIR/visSplit_3c48_example_${RUN_TS}.cmd"
LOG_FILE="$LOG_DIR/visSplit_3c48_example_${RUN_TS}.log"
{
  echo "# timestamp=$RUN_TS"
  echo "# cwd=$PWD"
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "$CMD_FILE"

echo "[visSplit-3c48] cmd manifest: $CMD_FILE"
echo "[visSplit-3c48] log: $LOG_FILE"

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
  # --drop-zero-weight-rows   # to use: add \ to the line above and uncomment this

echo "[visSplit-3c48] done. output: $OUTPUT"
