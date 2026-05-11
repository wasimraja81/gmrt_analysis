#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-python}"

cd "$REPO_ROOT"

# Plot visibility diagnostics for all Moon split variants.
# All three variants (raw_flagged, primary_calibrated_flagged,
# primary_secondary_calibrated_flagged) are already-split UVFITS files;
# calibration and flags have been baked in at split time, so we use
# DATA_MODE=calibrated (no table re-application).
#
# Output layout:
#   diagnostics_out/target/<tag>/raw/         ← raw flagged plots
#   diagnostics_out/target/<tag>/transfer/    ← primary-only calibrated plots
#   diagnostics_out/target/<tag>/final_qa/    ← primary+secondary calibrated plots

SPLIT_DIR=~/DATA/gmrt_40_014/work/split/moon
DIAG_BASE=~/DATA/gmrt_40_014/work/diagnostics_out/target
CONFIG="$REPO_ROOT/preprocess_ugmrt.cfg"
CHAN_START=0
CHAN_END=127

MOON_SOURCES=(MOON0520 MOON0545 MOON0605 MOON0625 MOON0635)

LOG_DIR=~/DATA/gmrt_40_014/work/logs
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

run_plot() {
    local label="$1"
    local fits="$2"
    local source="$3"
    local outroot="$4"

    local index="${fits}.row_index_cache.npz"
    mkdir -p "$outroot"

    local cmd_file="$LOG_DIR/plotVis_moon_${label}_${RUN_TS}.cmd"
    local log_file="$LOG_DIR/plotVis_moon_${label}_${RUN_TS}.log"

    local cmd=(
        "$PYTHON_CMD" "$REPO_ROOT/src/plotVis.py"
        --config "$CONFIG"
        --fits "$fits"
        --outdir "$outroot"
        --index-cache "$index"
        --source "$source"
        --chan-range "$CHAN_START" "$CHAN_END"
        --elevation-min 25
        --products RR,LL
        --panels amp_uvdist,phase_uvdist,real_uvdist,imag_uvdist,amp_time,phase_time,real_time,imag_time,az_time,el_time,amp_freq,phase_freq,real_freq,imag_freq,ri_scatter,vector_avg,uv_sampling
        --sample-frac 0.01
        --overlay-flags
        --multipage both
    )

    echo "[plotVis-moon] run config:"
    echo "  label=$label"
    echo "  FITS=$fits"
    echo "  SOURCE=$source"
    echo "  OUTROOT=$outroot"

    {
        echo "# timestamp=$RUN_TS"
        echo "# cwd=$PWD"
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } > "$cmd_file"

    echo "[plotVis-moon] cmd manifest: $cmd_file"
    echo "[plotVis-moon] log: $log_file"
    "${cmd[@]}" 2>&1 | tee "$log_file"
    echo "[plotVis-moon] done: outputs under $outroot"
}

for SOURCE in "${MOON_SOURCES[@]}"; do
    TAG="$(printf '%s' "$SOURCE" | tr '[:upper:]' '[:lower:]')"  # e.g. moon0520

    RAW_FITS="$SPLIT_DIR/${TAG}_raw_flagged.uvfits"
    PRIMARY_FITS="$SPLIT_DIR/${TAG}_primary_calibrated_flagged.uvfits"
    FINAL_FITS="$SPLIT_DIR/${TAG}_primary_secondary_calibrated_flagged.uvfits"

    echo ""
    echo "=== $SOURCE ==="

    if [[ ! -f "$RAW_FITS" ]]; then
        echo "[plotVis-moon] WARNING: missing $RAW_FITS — skipping raw"
    else
        run_plot "${TAG}_raw" "$RAW_FITS" "$SOURCE" "$DIAG_BASE/${TAG}/raw"
    fi

    if [[ ! -f "$PRIMARY_FITS" ]]; then
        echo "[plotVis-moon] WARNING: missing $PRIMARY_FITS — skipping transfer"
    else
        run_plot "${TAG}_primary" "$PRIMARY_FITS" "$SOURCE" "$DIAG_BASE/${TAG}/transfer"
    fi

    if [[ ! -f "$FINAL_FITS" ]]; then
        echo "[plotVis-moon] WARNING: missing $FINAL_FITS — skipping final_qa"
    else
        run_plot "${TAG}_final" "$FINAL_FITS" "$SOURCE" "$DIAG_BASE/${TAG}/final_qa"
    fi

done

echo ""
echo "[plotVis-moon] all moon sources plotted"
