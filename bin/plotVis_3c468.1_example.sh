#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_CMD="${PYTHON_CMD:-python}"

cd "$REPO_ROOT"

# Choose data mode:
#   raw        -> apply --bpcal and --flag tables
#   calibrated -> do NOT apply tables again

# Reproducible default for this example run:
DATA_MODE=calibrated
#DATA_MODE=raw
if [[ "$DATA_MODE" == "raw" ]]; then
  FITS=~/DATA/gmrt_40_014/data/40_014_25jul2021_gsb.FITS
  INDEX=~/DATA/gmrt_40_014/work/40_014_25jul2021_gsb.index.npz
  CHAN_START=64
  CHAN_END=191
  OUTROOT=~/DATA/gmrt_40_014/work/diagnostics_out/3c468.1_primary_and_secondary_tables
elif [[ "$DATA_MODE" == "calibrated" ]]; then
  FITS=~/DATA/gmrt_40_014/work/split/3c468.1/3c468.1_primary_secondary_calibrated_flagged.uvfits
  INDEX=~/DATA/gmrt_40_014/work/split/3c468.1/3c468.1_primary_secondary_calibrated_flagged.uvfits.row_index_cache.npz
  CHAN_START=0
  CHAN_END=127
  OUTROOT=~/DATA/gmrt_40_014/work/diagnostics_out/3c468.1_split_calibrated
else
  echo "Invalid DATA_MODE='$DATA_MODE' (use: raw or calibrated)"
  exit 1
fi

FLAG=~/DATA/gmrt_40_014/work/secondary_calibration/flag/3c468.1_split_clustering_flag_table_session.json
SOURCE=3C468.1
PRIMARY=~/DATA/gmrt_40_014/work/primary_calibration/bandpass/3c48_bandpass_25jul_gsb_iterfinal_clustering.npz
CONFIG="$REPO_ROOT/preprocess_ugmrt.cfg"

if (( CHAN_START > CHAN_END )); then
  echo "Invalid channel range: CHAN_START ($CHAN_START) > CHAN_END ($CHAN_END)"
  exit 1
fi

mkdir -p "$OUTROOT"
LOG_DIR=~/DATA/gmrt_40_014/work/logs
RUN_TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "[plotVis-3c468.1] run config:"
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

shopt -s nullglob
SECONDARY_TABLES=(~/DATA/gmrt_40_014/work/secondary_calibration/3c468.1_secondary_phase_only_scan*.npz)
shopt -u nullglob

CMD=(
  "$PYTHON_CMD" "$REPO_ROOT/src/plotVis.py"
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
      echo "[plotVis-3c468.1] WARNING: flag table not found ($FLAG) — proceeding without flags"
      FLAG=""
    fi
    if [[ ${#SECONDARY_TABLES[@]} -eq 0 ]]; then
      echo "No secondary scan tables found under ~/DATA/gmrt_40_014/work/secondary_calibration/3c468.1_secondary_phase_only_scan*.npz"
      exit 1
    fi

    TABLES=("$PRIMARY" "${SECONDARY_TABLES[@]}")

    CMD+=(
      --bpcal "${TABLES[@]}"
      --time-interp-scheme nearest
      --time-extrapolation hold
    )
    [[ -n "$FLAG" ]] && CMD+=(--flag "$FLAG")
    FLAG_MSG="(no flag table)"; [[ -n "$FLAG" ]] && FLAG_MSG="and flag table"
    echo "[plotVis-3c468.1] mode=raw: applying 1 primary + ${#SECONDARY_TABLES[@]} secondary table(s) ${FLAG_MSG}"
    ;;
  calibrated)
    echo "[plotVis-3c468.1] mode=calibrated: skipping --bpcal/--flag to avoid double application"
    ;;
  *)
    echo "Invalid DATA_MODE='$DATA_MODE' (use: raw or calibrated)"
    exit 1
    ;;
esac

echo "[plotVis-3c468.1] explicit --chan-range ${CHAN_START} ${CHAN_END} (config CHAN_RANGE ignored)"

CMD_FILE="$LOG_DIR/plotVis_3c468.1_example_${RUN_TS}.cmd"
LOG_FILE="$LOG_DIR/plotVis_3c468.1_example_${RUN_TS}.log"
{
  echo "# timestamp=$RUN_TS"
  echo "# cwd=$PWD"
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "$CMD_FILE"

echo "[plotVis-3c468.1] cmd manifest: $CMD_FILE"
echo "[plotVis-3c468.1] log: $LOG_FILE"

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo "[plotVis-3c468.1] done. outputs under: $OUTROOT"
