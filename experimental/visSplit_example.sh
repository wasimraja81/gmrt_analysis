#!/usr/bin/env bash

set -euo pipefail

FITS=~/DATA/gmrt_40_014/data/40_014_25jul2021_gsb.FITS
INDEX=~/DATA/gmrt_40_014/work/40_014_25jul2021_gsb.index.npz
FLAG=~/DATA/gmrt_40_014/work/3c468.1_flag_table_session.json
SOURCE=3C48
PRIMARY=~/DATA/gmrt_40_014/work/3c48_bandpass_25jul_gsb.npz
CONFIG=./preprocess_ugmrt.cfg
CHAN_START=64
CHAN_END=191
STOKES=(RR LL)
#STOKES=(RR LL RL LR)

OUTDIR=~/DATA/gmrt_40_014/work/split
mkdir -p "$OUTDIR"
OUTPUT="$OUTDIR/3c468.1_calibrated.uvfits"

shopt -s nullglob
#SECONDARY_TABLES=(~/DATA/gmrt_40_014/work/3c468.1_secondary_phase_only_scan*.npz)
SECONDARY_TABLES=()
shopt -u nullglob

if [[ ! -f "$PRIMARY" ]]; then
    echo "Primary table not found: $PRIMARY"
    exit 1
fi

TABLES=("$PRIMARY")
if [[ ${#SECONDARY_TABLES[@]} -gt 0 ]]; then
    TABLES+=("${SECONDARY_TABLES[@]}")
fi

echo "[visSplit-example] applying 1 primary + ${#SECONDARY_TABLES[@]} secondary scan table(s) → $OUTPUT"

python visSplit.py \
  --config "$CONFIG" \
  --fits "$FITS" \
  --index-cache "$INDEX" \
  --source "$SOURCE" \
  --chan-range "$CHAN_START" "$CHAN_END" \
  --stokes "${STOKES[@]}" \
  --elevation-min 25 \
  --tables "${TABLES[@]}" \
  --time-interp-scheme nearest \
  --time-extrapolation hold \
  --flag-tables "$FLAG" \
  --out "$OUTPUT" \
  --overwrite
  # --drop-zero-weight-rows   # to use: add \ to the line above and uncomment this

echo "[visSplit-example] done. output: $OUTPUT"
