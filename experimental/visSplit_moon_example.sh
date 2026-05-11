#!/usr/bin/env bash
# visSplit_moon_example.sh
# Split all MOON* scans from the raw GSB FITS into individual calibrated
# UVFITS files.  For each Moon source the script:
#   1. Applies the primary bandpass table.
#   2. Appends any Moon-specific secondary phase tables found by glob.
#   3. Applies the Moon-specific flag table if one exists, otherwise
#      falls back to the session flag table shared with 3C468.1.
# Run with:  bash visSplit_moon_example.sh

set -euo pipefail

if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "Please run this script with bash, not sh."
    exit 1
fi

# ── Input files ───────────────────────────────────────────────────────────────
FITS=~/DATA/gmrt_40_014/data/40_014_25jul2021_gsb.FITS
INDEX=~/DATA/gmrt_40_014/work/40_014_25jul2021_gsb.index.npz
PRIMARY=~/DATA/gmrt_40_014/work/3c48_bandpass_25jul_gsb.npz
CONFIG=./preprocess_ugmrt.cfg
CHAN_START=64
CHAN_END=191
STOKES=(RR LL)
#STOKES=(RR LL RL LR)

# Secondary phase tables toggle:
#   0 = primary-only calibration
#   1 = primary + secondary scan phase tables
USE_SECONDARY=0

if [[ "$USE_SECONDARY" -eq 1 ]]; then
    CAL_MODE_TAG="primary_secondary"
else
    CAL_MODE_TAG="primaryonly"
fi

# Fallback flag table (used when no Moon-specific table exists)
FALLBACK_FLAG=~/DATA/gmrt_40_014/work/3c468.1_flag_table_session.json

# ── Output directory ──────────────────────────────────────────────────────────
OUTDIR=~/DATA/gmrt_40_014/work/split/moon
mkdir -p "$OUTDIR"

# ── Source list ───────────────────────────────────────────────────────────────
# All Moon scan identifiers found in the observation (id_to_name from index).
MOON_SOURCES=(MOON0520 MOON0545 MOON0605 MOON0625 MOON0635)

# ── Guard ─────────────────────────────────────────────────────────────────────
if [[ ! -f "$PRIMARY" ]]; then
    echo "Primary bandpass table not found: $PRIMARY"
    exit 1
fi

# ── Loop over Moon sources ────────────────────────────────────────────────────
for SOURCE in "${MOON_SOURCES[@]}"; do

    # Convert source name to a safe filename fragment (lower-case, no special chars)
    TAG="$(echo "$SOURCE" | tr '[:upper:]' '[:lower:]')"  # e.g. moon0520

    OUTPUT="$OUTDIR/${TAG}_${CAL_MODE_TAG}_calibrated.uvfits"

    # Optional 3C468.1 secondary phase tables (same field → same phase solutions)
    SECONDARY_TABLES=()
    if [[ "$USE_SECONDARY" -eq 1 ]]; then
        shopt -s nullglob
        SECONDARY_TABLES=(~/DATA/gmrt_40_014/work/3c468.1_secondary_phase_only_scan*.npz)
        shopt -u nullglob
    fi

    TABLES=("$PRIMARY")
    if [[ ${#SECONDARY_TABLES[@]} -gt 0 ]]; then
        TABLES+=("${SECONDARY_TABLES[@]}")
    fi

    # Choose flag table: Moon-specific if it exists, otherwise fallback
    MOON_FLAG=~/DATA/gmrt_40_014/work/"${TAG}"_flag_table_session.json
    if [[ -f "$MOON_FLAG" ]]; then
        FLAG="$MOON_FLAG"
    else
        FLAG="$FALLBACK_FLAG"
    fi

    echo "[visSplit-moon] source=$SOURCE  mode=$CAL_MODE_TAG use_secondary=$USE_SECONDARY tables=1+${#SECONDARY_TABLES[@]}  flag=$(basename "$FLAG")  → $OUTPUT"

    python visSplit.py \
      --config   "$CONFIG" \
      --fits     "$FITS" \
      --index-cache "$INDEX" \
      --source   "$SOURCE" \
      --chan-range "$CHAN_START" "$CHAN_END" \
      --stokes   "${STOKES[@]}" \
      --elevation-min 10 \
      --tables   "${TABLES[@]}" \
      --time-interp-scheme nearest \
      --time-extrapolation hold \
      --flag-tables "$FLAG" \
      --out      "$OUTPUT" \
      --overwrite
      # --drop-zero-weight-rows   # to use: add \\ to the line above and uncomment this

    echo "[visSplit-moon] done: $OUTPUT"
    echo

done

echo "[visSplit-moon] all Moon scans written to $OUTDIR"
