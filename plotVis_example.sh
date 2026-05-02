#!/usr/bin/env bash

set -euo pipefail

FITS=~/DATA/gmrt_40_014/data/40_014_25jul2021_gsb.FITS
INDEX=~/DATA/gmrt_40_014/work/40_014_25jul2021_gsb.index.npz
FLAG=~/DATA/gmrt_40_014/work/3c468.1_flag_table_session.json
SOURCE=3c468.1
PRIMARY=~/DATA/gmrt_40_014/work/3c48_bandpass_25jul_gsb_iterfinal_clustering.npz
CONFIG=./preprocess_ugmrt.cfg
CHAN_START=64
CHAN_END=191

OUTROOT=./diagnostics_out/primary_tables
#OUTROOT=./diagnostics_out/secondary_scan_tables
mkdir -p "$OUTROOT"

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

echo "[plotVis-example] running time-aware multi-table apply with 1 primary + ${#SECONDARY_TABLES[@]} secondary scan table(s)"

python plotVis.py \
--config "$CONFIG" \
--fits "$FITS" \
--bpcal "${TABLES[@]}" \
--time-interp-scheme nearest \
--time-extrapolation hold \
--flag "$FLAG" \
--outdir "$OUTROOT" \
--index-cache "$INDEX" \
--source "$SOURCE" \
--chan-range "$CHAN_START" "$CHAN_END" \
--elevation-min 25 \
--products RR,LL,V \
--panels amp_uvdist,phase_uvdist,real_uvdist,imag_uvdist,amp_time,phase_time,real_time,imag_time,amp_freq,phase_freq,real_freq,imag_freq,ri_scatter,vector_avg \
--sample-frac 0.02 \
--overlay-flags \
--multipage both

echo "[plotVis-example] done. outputs under: $OUTROOT"
