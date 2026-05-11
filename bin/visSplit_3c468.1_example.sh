#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-python}"

cd "$REPO_ROOT"

# Post-secondary split for 3C468.1
# Run this after secondaryCalibration_example.sh to apply primary + secondary solutions.

FITS=~/DATA/gmrt_40_014/data/40_014_25jul2021_gsb.FITS
INDEX=~/DATA/gmrt_40_014/work/40_014_25jul2021_gsb.index.npz
FLAG=~/DATA/gmrt_40_014/work/secondary_calibration/flag/3c468.1_split_clustering_flag_table_session.json
SOURCE=3C468.1
SRC_TAG="$(printf '%s' "$SOURCE" | tr '[:upper:]' '[:lower:]')"
PRIMARY=~/DATA/gmrt_40_014/work/primary_calibration/bandpass/3c48_bandpass_25jul_gsb_iterfinal_clustering.npz
CONFIG="$REPO_ROOT/preprocess_ugmrt.cfg"
CHAN_START=64
CHAN_END=191
STOKES=(RR LL)
#STOKES=(RR LL RL LR)

OUTDIR=~/DATA/gmrt_40_014/work/split/"${SRC_TAG}"
mkdir -p "$OUTDIR"
OUTPUT="$OUTDIR/${SRC_TAG}_primary_secondary_calibrated_flagged.uvfits"
LOG_DIR=~/DATA/gmrt_40_014/work/logs
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

if [[ -f "$OUTPUT" ]]; then
    echo "[visSplit-3c468.1] WARNING: existing output will be overwritten: $OUTPUT"
fi

shopt -s nullglob
SECONDARY_TABLES=(~/DATA/gmrt_40_014/work/secondary_calibration/3c468.1_secondary_phase_only_scan*.npz)
shopt -u nullglob

if [[ ! -f "$PRIMARY" ]]; then
    echo "Primary table not found: $PRIMARY"
    exit 1
fi

if [[ ! -f "$FLAG" ]]; then
    echo "[visSplit-3c468.1] WARNING: flag table not found ($FLAG) — proceeding without flags"
    FLAG=""
fi

if [[ ${#SECONDARY_TABLES[@]} -eq 0 ]]; then
    echo "No secondary scan tables found under ~/DATA/gmrt_40_014/work/secondary_calibration/3c468.1_secondary_phase_only_scan*.npz"
    exit 1
fi

TABLES=("$PRIMARY" "${SECONDARY_TABLES[@]}")

FLAG_MSG="(no flag table)"; [[ -n "$FLAG" ]] && FLAG_MSG="+ flag table"
echo "[visSplit-3c468.1] applying 1 primary + ${#SECONDARY_TABLES[@]} secondary scan table(s) ${FLAG_MSG} → $OUTPUT"

CMD=(
    "$PYTHON_CMD" "$REPO_ROOT/src/visSplit.py"
    --config "$CONFIG"
    --fits "$FITS"
    --index-cache "$INDEX"
    --source "$SOURCE"
    --chan-range "$CHAN_START" "$CHAN_END"
    --stokes "${STOKES[@]}"
    --elevation-min 25
    --tables "${TABLES[@]}"
    --time-interp-scheme nearest
    --time-extrapolation hold
    --out "$OUTPUT"
    --overwrite
)
[[ -n "$FLAG" ]] && CMD+=(--flag-tables "$FLAG")

CMD_FILE="$LOG_DIR/visSplit_3c468.1_example_${RUN_TS}.cmd"
LOG_FILE="$LOG_DIR/visSplit_3c468.1_example_${RUN_TS}.log"
{
    echo "# timestamp=$RUN_TS"
    echo "# cwd=$PWD"
    printf '%q ' "${CMD[@]}"
    printf '\n'
} > "$CMD_FILE"

echo "[visSplit-3c468.1] cmd manifest: $CMD_FILE"
echo "[visSplit-3c468.1] log: $LOG_FILE"

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
  # --drop-zero-weight-rows   # to use: add \ to the line above and uncomment this

echo "[visSplit-3c468.1] done. output: $OUTPUT"
