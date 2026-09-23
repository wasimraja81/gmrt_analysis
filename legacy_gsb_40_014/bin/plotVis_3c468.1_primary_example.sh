#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# PYTHON_CMD is set in the machine-specific block below

cd "$REPO_ROOT"

# Plot diagnostics for 3C468.1 primary-only calibrated split data
# Run this after visSplit_3c468.1_primary_example.sh

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

if [[ "$_DATA_TAG" == "gwb" ]]; then
  _CHAN_END_CALIB=170
else
  _CHAN_END_CALIB=127
fi

FITS="$WORK_DIR/split/3c468.1/3c468.1_primary_calibrated.uvfits"
INDEX="$WORK_DIR/split/3c468.1/3c468.1_primary_calibrated.uvfits.row_index_cache.npz"
SOURCE=3C468.1
CONFIG="$REPO_ROOT/preprocess_ugmrt.cfg"

CHAN_START=0
CHAN_END="$_CHAN_END_CALIB"
OUTROOT="$WORK_DIR/diagnostics_out/secondary/3c468.1/transfer"

LOG_DIR="$WORK_DIR/logs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTROOT" "$LOG_DIR"

if [[ ! -f "$FITS" ]]; then
  echo "Primary-only split file not found: $FITS"
  echo "Run visSplit_3c468.1_primary_example.sh first."
  exit 1
fi

if (( CHAN_START > CHAN_END )); then
  echo "Invalid channel range: CHAN_START ($CHAN_START) > CHAN_END ($CHAN_END)"
  exit 1
fi

if [[ -f "$OUTROOT/plotvis_3c468.1.pdf" ]]; then
  echo "[plotVis-3c468.1-primary] WARNING: existing plot outputs will be overwritten under: $OUTROOT"
fi

echo "[plotVis-3c468.1-primary] run config:"
echo "  FITS=$FITS"
echo "  INDEX=$INDEX"
echo "  SOURCE=$SOURCE"
echo "  CHAN_RANGE=$CHAN_START..$CHAN_END"
echo "  OUTROOT=$OUTROOT"
echo "[plotVis-3c468.1-primary] mode=calibrated: skipping --bpcal/--flag"

CMD=(
  "$PYTHON_CMD" "$REPO_ROOT/src/plotVis.py"
  --config "$CONFIG"
  --fits "$FITS"
  --outdir "$OUTROOT"
  --index-cache "$INDEX"
  --source "$SOURCE"
  --chan-range "$CHAN_START" "$CHAN_END"
  --elevation-min 25
  --products RR,LL
  --panels amp_uvdist,phase_uvdist,real_uvdist,imag_uvdist,amp_time,phase_time,real_time,imag_time,az_time,el_time,amp_freq,phase_freq,real_freq,imag_freq,ri_scatter,vector_avg,uv_sampling
  --sample-frac 0.01
  --overlay-flags
  --multipage both
)

CMD_FILE="$LOG_DIR/plotVis_3c468.1_primary_example_${RUN_TS}.cmd"
LOG_FILE="$LOG_DIR/plotVis_3c468.1_primary_example_${RUN_TS}.log"
{
  echo "# timestamp=$RUN_TS"
  echo "# cwd=$PWD"
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "$CMD_FILE"

echo "[plotVis-3c468.1-primary] cmd manifest: $CMD_FILE"
echo "[plotVis-3c468.1-primary] log: $LOG_FILE"

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo "[plotVis-3c468.1-primary] done. outputs under: $OUTROOT"
