#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-python}"

cd "$REPO_ROOT"

# Split all MOON* scans from the raw GSB FITS into three calibration variants:
#   raw_flagged                        — no cal, both primary+secondary flags
#   primary_calibrated_flagged         — primary bandpass/gain, both flags
#   primary_secondary_calibrated_flagged — primary + per-scan secondary phase, both flags
#
# All variants are split from the original raw FITS (no pre-split file used).
# Both the primary flag table and the secondary clustering flag table are always applied.

FITS=~/DATA/gmrt_40_014/data/40_014_25jul2021_gsb.FITS
INDEX=~/DATA/gmrt_40_014/work/40_014_25jul2021_gsb.index.npz
PRIMARY=~/DATA/gmrt_40_014/work/primary_calibration/bandpass/3c48_bandpass_25jul_gsb_iterfinal_clustering.npz
PRIMARY_FLAG=~/DATA/gmrt_40_014/work/primary_calibration/flag/3c48_flag_table_session.json
SECONDARY_FLAG=~/DATA/gmrt_40_014/work/secondary_calibration/flag/3c468.1_split_clustering_flag_table_session.json
CONFIG="$REPO_ROOT/preprocess_ugmrt.cfg"
CHAN_START=64
CHAN_END=191
STOKES=(RR LL)

MOON_SOURCES=(MOON0520 MOON0545 MOON0605 MOON0625 MOON0635)

OUTDIR=~/DATA/gmrt_40_014/work/split/moon
LOG_DIR=~/DATA/gmrt_40_014/work/logs
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR" "$LOG_DIR"

# Guard: primary bandpass table must exist
if [[ ! -f "$PRIMARY" ]]; then
    echo "[visSplit-moon] ERROR: primary table not found: $PRIMARY"
    exit 1
fi
if [[ ! -f "$PRIMARY_FLAG" ]]; then
    echo "[visSplit-moon] ERROR: primary flag table not found: $PRIMARY_FLAG"
    exit 1
fi
if [[ ! -f "$SECONDARY_FLAG" ]]; then
    echo "[visSplit-moon] ERROR: secondary flag table not found: $SECONDARY_FLAG"
    exit 1
fi

shopt -s nullglob
SECONDARY_TABLES=(~/DATA/gmrt_40_014/work/secondary_calibration/3c468.1_secondary_phase_only_scan*.npz)
shopt -u nullglob

if [[ ${#SECONDARY_TABLES[@]} -eq 0 ]]; then
    echo "[visSplit-moon] ERROR: no secondary scan tables found under ~/DATA/gmrt_40_014/work/secondary_calibration/3c468.1_secondary_phase_only_scan*.npz"
    exit 1
fi

run_split() {
    local label="$1"
    local source="$2"
    local output="$3"
    shift 3
    local extra_args=("$@")

    local cmd_file="$LOG_DIR/visSplit_moon_${label}_${RUN_TS}.cmd"
    local log_file="$LOG_DIR/visSplit_moon_${label}_${RUN_TS}.log"

    local cmd=(
        "$PYTHON_CMD" "$REPO_ROOT/src/visSplit.py"
        --config "$CONFIG"
        --fits "$FITS"
        --index-cache "$INDEX"
        --source "$source"
        --chan-range "$CHAN_START" "$CHAN_END"
        --stokes "${STOKES[@]}"
        --elevation-min 25
        "${extra_args[@]}"
        --flag-tables "$PRIMARY_FLAG" "$SECONDARY_FLAG"
        --out "$output"
        --overwrite
    )

    {
        echo "# timestamp=$RUN_TS"
        echo "# cwd=$PWD"
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } > "$cmd_file"

    echo "[visSplit-moon] cmd manifest: $cmd_file"
    echo "[visSplit-moon] log: $log_file"
    "${cmd[@]}" 2>&1 | tee "$log_file"
    echo "[visSplit-moon] done: $output"
}

for SOURCE in "${MOON_SOURCES[@]}"; do
    TAG="$(printf '%s' "$SOURCE" | tr '[:upper:]' '[:lower:]')"  # e.g. moon0520

    OUT_RAW="$OUTDIR/${TAG}_raw_flagged.uvfits"
    OUT_PRIMARY="$OUTDIR/${TAG}_primary_calibrated_flagged.uvfits"
    OUT_FINAL="$OUTDIR/${TAG}_primary_secondary_calibrated_flagged.uvfits"

    echo ""
    echo "=== $SOURCE ==="

    # Variant 1: raw — no calibration tables, both flags
    echo "[visSplit-moon] $SOURCE: raw_flagged → $OUT_RAW"
    run_split "${TAG}_raw" "$SOURCE" "$OUT_RAW"

    # Variant 2: primary-only — primary bandpass table + both flags
    echo "[visSplit-moon] $SOURCE: primary_calibrated_flagged → $OUT_PRIMARY"
    run_split "${TAG}_primary" "$SOURCE" "$OUT_PRIMARY" \
        --tables "$PRIMARY" \
        --time-interp-scheme nearest \
        --time-extrapolation hold

    # Variant 3: primary + secondary — all scan tables + both flags
    echo "[visSplit-moon] $SOURCE: primary_secondary_calibrated_flagged → $OUT_FINAL"
    run_split "${TAG}_final" "$SOURCE" "$OUT_FINAL" \
        --tables "$PRIMARY" "${SECONDARY_TABLES[@]}" \
        --time-interp-scheme nearest \
        --time-extrapolation hold

done

echo ""
echo "[visSplit-moon] all moon sources split complete"
