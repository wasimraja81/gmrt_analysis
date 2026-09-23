#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# PYTHON_CMD is set in the machine-specific block below

cd "$REPO_ROOT"

# Split all MOON* scans from the raw GSB FITS into three calibration variants:
#   raw_flagged                        — no cal, both primary+secondary flags
#   primary_calibrated_flagged         — primary bandpass/gain, both flags
#   primary_secondary_calibrated_flagged — primary + per-scan secondary phase, both flags
#
# All variants are split from the original raw FITS (no pre-split file used).
# Both the primary flag table and the secondary clustering flag table are always applied.

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
PRIMARY="$WORK_DIR/primary_calibration/bandpass/3c48_bandpass_25jul_${_DATA_TAG}_iterfinal_clustering.npz"
PRIMARY_FLAG="$WORK_DIR/primary_calibration/flag/3c48_flag_table_session.json"
SECONDARY_FLAG="$WORK_DIR/secondary_calibration/flag/3c468.1_split_clustering_flag_table_session.json"
CONFIG="$REPO_ROOT/preprocess_ugmrt.cfg"
STOKES=(RR LL RL LR)

MOON_SOURCES=(MOON0520 MOON0545 MOON0605 MOON0625 MOON0635)

OUTDIR="$WORK_DIR/split/moon"
LOG_DIR="$WORK_DIR/logs"
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
SECONDARY_TABLES=("$WORK_DIR"/secondary_calibration/3c468.1_secondary_phase_only_scan*.npz)
shopt -u nullglob

if [[ ${#SECONDARY_TABLES[@]} -eq 0 ]]; then
    echo "[visSplit-moon] ERROR: no secondary scan tables found under $WORK_DIR/secondary_calibration/3c468.1_secondary_phase_only_scan*.npz"
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
        ${extra_args[@]+"${extra_args[@]}"}
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
