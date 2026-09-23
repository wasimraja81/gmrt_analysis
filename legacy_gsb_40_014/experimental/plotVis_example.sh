#!/usr/bin/env bash

set -euo pipefail

# Choose data mode:
#   raw        -> apply --bpcal and --flag tables
#   calibrated -> do NOT apply tables again

# Raw preset (use with DATA_MODE=raw):
DATA_MODE=raw
FITS=~/DATA/gmrt_40_014/data/40_014_25jul2021_gsb.FITS
INDEX=~/DATA/gmrt_40_014/work/40_014_25jul2021_gsb.index.npz
# Explicit channel selection passed to plotVis.py (overrides CHAN_RANGE in config).
# Raw preset (original 256-ch file central band):
CHAN_START=64
CHAN_END=191
OUTROOT=./diagnostics_out/primary_tables

## Calibrated preset (use with DATA_MODE=calibrated):
#DATA_MODE=calibrated
#FITS=/Users/raj030/DATA/gmrt_40_014/work/split/3c468.1_calibrated.uvfits
#INDEX=/Users/raj030/DATA/gmrt_40_014/work/split/3c468.1_calibrated.uvfits.row_index_cache.npz
## Calibrated split preset (128-ch file all channels):
#CHAN_START=0
#CHAN_END=127
#OUTROOT=./diagnostics_out/primary_and_secondary_tables

FLAG=~/DATA/gmrt_40_014/work/3c468.1_flag_table_session.json
SOURCE=3C468.1
PRIMARY=~/DATA/gmrt_40_014/work/3c48_bandpass_25jul_gsb_iterfinal_clustering.npz
CONFIG=./preprocess_ugmrt.cfg

if (( CHAN_START > CHAN_END )); then
  echo "Invalid channel range: CHAN_START ($CHAN_START) > CHAN_END ($CHAN_END)"
  exit 1
fi

mkdir -p "$OUTROOT"

shopt -s nullglob
#SECONDARY_TABLES=(~/DATA/gmrt_40_014/work/3c468.1_secondary_phase_only_scan*.npz)
SECONDARY_TABLES=()
shopt -u nullglob

CMD=(
  python plotVis.py
  --config "$CONFIG" \
  --fits "$FITS" \
  --outdir "$OUTROOT" \
  --index-cache "$INDEX" \
  --source "$SOURCE" \
  --chan-range "$CHAN_START" "$CHAN_END" \
  --elevation-min 25 \
  --products RR,LL \
  --panels amp_uvdist,phase_uvdist,real_uvdist,imag_uvdist,amp_time,phase_time,real_time,imag_time,amp_freq,phase_freq,real_freq,imag_freq,ri_scatter,vector_avg,uv_sampling \
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
      echo "Flag table not found: $FLAG"
      exit 1
    fi

    TABLES=("$PRIMARY")
    if [[ ${#SECONDARY_TABLES[@]} -gt 0 ]]; then
      TABLES+=("${SECONDARY_TABLES[@]}")
    fi

    CMD+=(
      --bpcal "${TABLES[@]}"
      --time-interp-scheme nearest
      --time-extrapolation hold
      --flag "$FLAG"
    )
    echo "[plotVis-example] mode=raw: applying 1 primary + ${#SECONDARY_TABLES[@]} secondary table(s) and flag table"
    ;;
  calibrated)
    echo "[plotVis-example] mode=calibrated: skipping --bpcal/--flag to avoid double application"
    ;;
  *)
    echo "Invalid DATA_MODE='$DATA_MODE' (use: raw or calibrated)"
    exit 1
    ;;
esac

echo "[plotVis-example] explicit --chan-range ${CHAN_START} ${CHAN_END} (config CHAN_RANGE ignored)"

"${CMD[@]}"

echo "[plotVis-example] done. outputs under: $OUTROOT"
