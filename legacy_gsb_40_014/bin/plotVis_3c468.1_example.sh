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
_RAW_FITS="$FITS"
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

# Choose data mode:
#   raw        -> apply --bpcal and --flag tables
#   calibrated -> do NOT apply tables again

# Reproducible default for this example run:
DATA_MODE=calibrated
#DATA_MODE=raw
if [[ "$DATA_MODE" == "raw" ]]; then
  FITS="$_RAW_FITS"
  INDEX="$_RAW_INDEX"
  CHAN_START="$_CHAN_START_RAW"
  CHAN_END="$_CHAN_END_RAW"
  OUTROOT="$WORK_DIR/diagnostics_out/secondary/3c468.1/final_qa"
elif [[ "$DATA_MODE" == "calibrated" ]]; then
  FITS="$WORK_DIR/split/3c468.1/3c468.1_primary_secondary_calibrated_flagged.uvfits"
  INDEX="$WORK_DIR/split/3c468.1/3c468.1_primary_secondary_calibrated_flagged.uvfits.row_index_cache.npz"
  CHAN_START=0
  CHAN_END="$_CHAN_END_CALIB"
  OUTROOT="$WORK_DIR/diagnostics_out/secondary/3c468.1/final_qa"
else
  echo "Invalid DATA_MODE='$DATA_MODE' (use: raw or calibrated)"
  exit 1
fi

FLAG="$WORK_DIR/secondary_calibration/flag/3c468.1_split_clustering_flag_table_session.json"
SOURCE=3C468.1
PRIMARY="$WORK_DIR/primary_calibration/bandpass/3c48_bandpass_25jul_${_DATA_TAG}_iterfinal_clustering.npz"
CONFIG="$REPO_ROOT/preprocess_ugmrt.cfg"

if (( CHAN_START > CHAN_END )); then
  echo "Invalid channel range: CHAN_START ($CHAN_START) > CHAN_END ($CHAN_END)"
  exit 1
fi

mkdir -p "$OUTROOT"
LOG_DIR="$WORK_DIR/logs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "[plotVis-3c468.1] run config:"
echo "  DATA_MODE=$DATA_MODE"
echo "  FITS=$FITS"
echo "  INDEX=$INDEX"
echo "  SOURCE=$SOURCE"
echo "  CHAN_RANGE=$CHAN_START..$CHAN_END"
echo "  OUTROOT=$OUTROOT"
if [[ "$DATA_MODE" == "raw" ]]; then
  echo "  PRIMARY=$PRIMARY"
  echo "  FLAG=$FLAG"
fi

shopt -s nullglob
SECONDARY_TABLES=("$WORK_DIR"/secondary_calibration/3c468.1_secondary_phase_only_scan*.npz)
shopt -u nullglob

CMD=(
  "$PYTHON_CMD" "$REPO_ROOT/src/plotVis.py"
  --config "$CONFIG" \
  --fits "$FITS" \
  --outdir "$OUTROOT" \
  --index-cache "$INDEX" \
  --source "$SOURCE" \
  --chan-range "$CHAN_START" "$CHAN_END" \
  --elevation-min 25 \
  --products RR,LL \
  --panels amp_uvdist,phase_uvdist,real_uvdist,imag_uvdist,amp_time,phase_time,real_time,imag_time,az_time,el_time,amp_freq,phase_freq,real_freq,imag_freq,ri_scatter,vector_avg,uv_sampling \
  --sample-frac 0.01 \
  --overlay-flags \
  --multipage both
)

case "$DATA_MODE" in
  raw)
    if [[ ! -f "$PRIMARY" ]]; then
      echo "Primary table not found: $PRIMARY"
      exit 1
    fi
    if [[ ! -f "$FLAG" ]]; then
      echo "[plotVis-3c468.1] WARNING: flag table not found ($FLAG) — proceeding without flags"
      FLAG=""
    fi
    if [[ ${#SECONDARY_TABLES[@]} -eq 0 ]]; then
      echo "No secondary scan tables found under $WORK_DIR/secondary_calibration/3c468.1_secondary_phase_only_scan*.npz"
      exit 1
    fi

    TABLES=("$PRIMARY" "${SECONDARY_TABLES[@]}")

    CMD+=(
      --bpcal "${TABLES[@]}"
      --time-interp-scheme nearest
      --time-extrapolation hold
    )
    [[ -n "$FLAG" ]] && CMD+=(--flag "$FLAG")
    FLAG_MSG="(no flag table)"; [[ -n "$FLAG" ]] && FLAG_MSG="and flag table"
    echo "[plotVis-3c468.1] mode=raw: applying 1 primary + ${#SECONDARY_TABLES[@]} secondary table(s) ${FLAG_MSG}"
    ;;
  calibrated)
    echo "[plotVis-3c468.1] mode=calibrated: skipping --bpcal/--flag to avoid double application"
    ;;
  *)
    echo "Invalid DATA_MODE='$DATA_MODE' (use: raw or calibrated)"
    exit 1
    ;;
esac

echo "[plotVis-3c468.1] explicit --chan-range ${CHAN_START} ${CHAN_END} (config CHAN_RANGE ignored)"

CMD_FILE="$LOG_DIR/plotVis_3c468.1_example_${RUN_TS}.cmd"
LOG_FILE="$LOG_DIR/plotVis_3c468.1_example_${RUN_TS}.log"
{
  echo "# timestamp=$RUN_TS"
  echo "# cwd=$PWD"
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "$CMD_FILE"

echo "[plotVis-3c468.1] cmd manifest: $CMD_FILE"
echo "[plotVis-3c468.1] log: $LOG_FILE"

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo "[plotVis-3c468.1] done. outputs under: $OUTROOT"
