#!/usr/bin/env bash

set -euo pipefail

# Choose data mode:
#   raw        -> apply --bpcal and --flag tables
#   calibrated -> do NOT apply tables again

# Reproducible default for this example run (can override: DATA_MODE=raw bash plotVis_3c48_example.sh):
DATA_MODE="${DATA_MODE:-calibrated}"
#DATA_MODE=raw
if [[ "$DATA_MODE" == "raw" ]]; then
  FITS=~/DATA/gmrt_40_014/data/40_014_25jul2021_gsb.FITS
  INDEX=~/DATA/gmrt_40_014/work/40_014_25jul2021_gsb.index.npz
  CHAN_START=64
  CHAN_END=191
  OUTROOT=~/DATA/gmrt_40_014/work/diagnostics_out/3c48_primary_tables
elif [[ "$DATA_MODE" == "calibrated" ]]; then
  FITS=~/DATA/gmrt_40_014/work/split/3c48/3c48_calibrated_flagged.uvfits
  INDEX=~/DATA/gmrt_40_014/work/split/3c48/3c48_calibrated_flagged.uvfits.row_index_cache.npz
  CHAN_START=0
  CHAN_END=127
  OUTROOT=~/DATA/gmrt_40_014/work/diagnostics_out/3c48_split_calibrated
else
  echo "Invalid DATA_MODE='$DATA_MODE' (use: raw or calibrated)"
  exit 1
fi

FLAG=~/DATA/gmrt_40_014/work/primary_calibration/flag/3c48_flag_table_session.json
SOURCE=3C48
PRIMARY=~/DATA/gmrt_40_014/work/primary_calibration/bandpass/3c48_bandpass_25jul_gsb.npz
CONFIG=./preprocess_ugmrt.cfg

if (( CHAN_START > CHAN_END )); then
  echo "Invalid channel range: CHAN_START ($CHAN_START) > CHAN_END ($CHAN_END)"
  exit 1
fi

mkdir -p "$OUTROOT"
LOG_DIR=~/DATA/gmrt_40_014/work/logs
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "[plotVis-3c48] run config:"
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
      echo "Flag table not found: $FLAG"
      exit 1
    fi

    CMD+=(
      --bpcal "$PRIMARY"
      --time-interp-scheme nearest
      --time-extrapolation hold
      --flag "$FLAG"
    )
    echo "[plotVis-3c48] mode=raw: applying primary table and flag table"
    ;;
  calibrated)
    echo "[plotVis-3c48] mode=calibrated: skipping --bpcal/--flag to avoid double application"
    ;;
  *)
    echo "Invalid DATA_MODE='$DATA_MODE' (use: raw or calibrated)"
    exit 1
    ;;
esac

echo "[plotVis-3c48] explicit --chan-range ${CHAN_START} ${CHAN_END} (config CHAN_RANGE ignored)"

CMD_FILE="$LOG_DIR/plotVis_3c48_example_${RUN_TS}.cmd"
LOG_FILE="$LOG_DIR/plotVis_3c48_example_${RUN_TS}.log"
{
  echo "# timestamp=$RUN_TS"
  echo "# cwd=$PWD"
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "$CMD_FILE"

echo "[plotVis-3c48] cmd manifest: $CMD_FILE"
echo "[plotVis-3c48] log: $LOG_FILE"

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo "[plotVis-3c48] done. outputs under: $OUTROOT"
