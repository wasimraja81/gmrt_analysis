#!/usr/bin/env bash

set -euo pipefail

FITS=~/DATA/gmrt_40_014/data/40_014_25jul2021_gsb.FITS
INDEX=~/DATA/gmrt_40_014/work/40_014_25jul2021_gsb.index.npz
FLAG=~/DATA/gmrt_40_014/work/primary_calibration/flag/3c48_flag_table_session.json
SOURCE=3C48
SRC_TAG="$(printf '%s' "$SOURCE" | tr '[:upper:]' '[:lower:]')"
PRIMARY=~/DATA/gmrt_40_014/work/primary_calibration/bandpass/3c48_bandpass_25jul_gsb.npz
CONFIG=./preprocess_ugmrt.cfg
CHAN_START=64
CHAN_END=191
STOKES=(RR LL)
#STOKES=(RR LL RL LR)

OUTDIR=~/DATA/gmrt_40_014/work/split/"${SRC_TAG}"  # lower-case source name for directory
mkdir -p "$OUTDIR"
OUTPUT="$OUTDIR/${SRC_TAG}_calibrated_flagged.uvfits"
LOG_DIR=~/DATA/gmrt_40_014/work/logs
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
  python visSplit.py
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
